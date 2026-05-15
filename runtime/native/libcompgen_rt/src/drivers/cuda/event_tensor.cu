/* Event Tensor primitives — Phase 4 paper §3.3 minimal runtime.
 *
 * Implements the four ops the static + dynamic schedulers emit calls
 * to: notify, wait, update, trigger. Every operation is a single
 * device-side atomic on an int64 cell of a global-memory tensor.
 * No host involvement; no auxiliary runtime state.
 *
 * Paper correspondence:
 *   - notify(E[i])  ⟶ atomicSub(&E[i], decrement) + threadfence
 *   - wait(E[i])    ⟶ spin while atomicAdd(&E[i], 0) > 0
 *   - update(E[i])  ⟶ atomicExch(&E[i], new_count)        (Fig. 5b)
 *   - trigger(E[i]) ⟶ atomicExch(&E[i], consumer_count)   (Fig. 5b)
 *
 * The semantics are identical to compgen.runtime.event_tensor's
 * Python reference; only the implementation differs (hardware atomics
 * + cooperative spin instead of threading.Condition).
 *
 * Why _system atomics: cluster-launch (paper Fig. 6) places event
 * tensors in distributed shared memory accessible across SMs in the
 * cluster. The _system suffix selects an atomic that's coherent
 * across the entire device, which is what we need when an SM other
 * than the owner reads/writes the cell. atomicSub_system is
 * available on SM_60+; we target SM_90+ in this build so it's
 * always present.
 *
 * Why __nanosleep in wait: the paper acknowledges spin-wait can burn
 * an SM cycle indefinitely. __nanosleep is a real hardware sleep on
 * SM_70+; we yield 64 ns between checks. Sub-microsecond turnaround
 * with no host involvement.
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "../../../include/compgen_rt/compgen_rt.h"

namespace cg = cooperative_groups;

extern "C" {

/* ----- device-side primitives ------------------------------------------ */

/* notify: atomically decrement E[idx] by `decrement`. Caller is the
 * producer task whose completion is being announced; the corresponding
 * wait() on E[idx] in the consumer task unblocks once the cell hits
 * zero (or below). Decrement >1 models grouped completion, matching
 * the paper's Fig. 7 RS task that waits on two MM tasks via
 * decrement=1 each (the dual: wait_count=2, two notifies of 1 each).
 */
__device__ __forceinline__
void cg_rt_cuda_etensor_notify_d(long long *E, int idx, int decrement) {
    /* atomicSub on int64 isn't a built-in in older CUDA; use atomicAdd
     * with negative. atomicAdd_system on int64 is SM_60+. Threadfence
     * keeps the producer's prior writes (the actual data the consumer
     * will read) ordered before the counter decrement.
     */
    __threadfence_system();
    atomicAdd_system((unsigned long long *)&E[idx],
                     (unsigned long long)(-(long long)decrement));
}

/* wait: spin until E[idx] reaches <= 0. Yields 64 ns between probes
 * via __nanosleep; on SM_70+ this is a real hardware sleep, not a
 * busy-loop. The paper's persistent-kernel pattern means the SM is
 * already pinned for the duration of the launch, so spinning here
 * doesn't deadlock other tasks — they're on other SMs.
 */
__device__ __forceinline__
void cg_rt_cuda_etensor_wait_d(long long *E, int idx) {
    /* Use atomicAdd of 0 to read with system-coherent semantics —
     * a plain load might be cached and miss the producer's update.
     */
    while (atomicAdd_system((unsigned long long *)&E[idx], 0ULL) > 0) {
        __nanosleep(64);
    }
    /* threadfence after wait so the consumer's subsequent reads of
     * the producer's data are correctly ordered.
     */
    __threadfence_system();
}

/* update: atomically store a new counter value at E[idx]. Used by
 * the data-dependent UpdateOp (paper Fig. 5b) when a runtime tensor
 * (e.g. topk) determines per-cell wait counts at MoE-routing time.
 * Returns void — over-writers don't care about the previous value.
 */
__device__ __forceinline__
void cg_rt_cuda_etensor_update_d(long long *E, int idx, long long new_count) {
    atomicExch_system((unsigned long long *)&E[idx],
                      (unsigned long long)new_count);
    __threadfence_system();
}

/* trigger: identical to update at the device level — same atomicExch.
 * Kept as a separate symbol so the schedule pass + emitter can record
 * which sites are TriggerOp (consumer-count materialisation per paper
 * Fig. 5b second half) vs UpdateOp. Useful for trace + debugging.
 */
__device__ __forceinline__
void cg_rt_cuda_etensor_trigger_d(long long *E, int idx, long long consumer_count) {
    atomicExch_system((unsigned long long *)&E[idx],
                      (unsigned long long)consumer_count);
    __threadfence_system();
}

/* ----- host-callable shims for testing the device primitives --------- */
/* Phase 4 ships the device primitives as inlined headers (above).
 * These shim kernels exist so the conformance harness + smoke tests
 * can exercise notify/wait/update/trigger without a full megakernel
 * pipeline in place. Removed in Phase 5 once the emitter inlines the
 * device functions directly into per-task bodies.
 */

__global__ void cg_rt_cuda_etensor_notify_kernel(
    long long *E, int idx, int decrement
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cg_rt_cuda_etensor_notify_d(E, idx, decrement);
    }
}

__global__ void cg_rt_cuda_etensor_wait_kernel(long long *E, int idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cg_rt_cuda_etensor_wait_d(E, idx);
    }
}

__global__ void cg_rt_cuda_etensor_update_kernel(
    long long *E, int idx, long long new_count
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cg_rt_cuda_etensor_update_d(E, idx, new_count);
    }
}

__global__ void cg_rt_cuda_etensor_trigger_kernel(
    long long *E, int idx, long long consumer_count
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cg_rt_cuda_etensor_trigger_d(E, idx, consumer_count);
    }
}

/* Public C API — the launcher invokes these to set initial wait
 * counts before the persistent kernel starts, and tests use them to
 * validate atomic correctness independently of the megakernel.
 */

cg_rt_status_t cg_rt_cuda_etensor_alloc(
    long long **out_ptr,
    int        num_cells,
    long long  initial_wait_count
) {
    if (out_ptr == NULL || num_cells <= 0) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    long long *dev = NULL;
    cudaError_t rc = cudaMalloc(&dev, sizeof(long long) * (size_t)num_cells);
    if (rc != cudaSuccess) {
        return CG_RT_ERR_OUT_OF_MEMORY;
    }
    /* Fill every cell with the initial wait count. cuMemsetD32 doesn't
     * suit int64; do it via a tiny launch.
     */
    extern __global__ void cg_rt_cuda_etensor_fill_kernel(
        long long *E, int n, long long val
    );
    int threads = 256;
    int blocks  = (num_cells + threads - 1) / threads;
    cg_rt_cuda_etensor_fill_kernel<<<blocks, threads>>>(dev, num_cells, initial_wait_count);
    rc = cudaDeviceSynchronize();
    if (rc != cudaSuccess) {
        cudaFree(dev);
        return CG_RT_ERR_UNKNOWN;
    }
    *out_ptr = dev;
    return CG_RT_OK;
}

void cg_rt_cuda_etensor_free(long long *ptr) {
    if (ptr != NULL) {
        cudaFree(ptr);
    }
}

cg_rt_status_t cg_rt_cuda_etensor_load(
    long long *E,
    int        idx,
    long long *out_value
) {
    if (E == NULL || out_value == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    long long host_val = 0;
    cudaError_t rc = cudaMemcpy(
        &host_val, &E[idx], sizeof(long long), cudaMemcpyDeviceToHost
    );
    if (rc != cudaSuccess) {
        return CG_RT_ERR_UNKNOWN;
    }
    *out_value = host_val;
    return CG_RT_OK;
}

}  /* extern "C" */

/* The fill kernel needs C++ linkage so we keep it outside extern "C".
 * cudaMalloc memory is uninitialized; this fills with the supplied
 * wait count.
 */
__global__ void cg_rt_cuda_etensor_fill_kernel(
    long long *E, int n, long long val
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        E[tid] = val;
    }
}
