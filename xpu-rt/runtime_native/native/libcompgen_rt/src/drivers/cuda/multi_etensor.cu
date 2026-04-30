/* Phase-4b cross-GPU Event Tensor primitives.
 *
 * Mirrors event_tensor.cu's notify/wait/update/trigger but operates
 * on **peer-mapped** event tensor pointers — pointers a remote rank
 * passes to us via the comm setup, valid for cross-GPU atomicAdd_system
 * after cuCtxEnablePeerAccess (REMOTE bridge probe #047 confirmed
 * peer access GREEN over PCIe NODE).
 *
 * Same atomic + threadfence pattern as the local primitives. The
 * differences are:
 *
 * 1. The pointer passed in points into a different GPU's memory
 *    space. atomicAdd_system + the PCIe-coherent fence make this
 *    work.
 * 2. The notify here must be paired with a wait on the *remote*
 *    GPU's host wrapper code (the producer notifies the consumer
 *    GPU's event tensor, the consumer GPU's tasks wait on it via
 *    the local primitives from event_tensor.cu).
 *
 * Phase-5 emitter inlines these for cross-GPU edges in the same
 * way it inlines the local ones — extern "C" stubs here exist only
 * so a future compile path that links against libcompgen_rt-cuda.so
 * gets the symbols. The megakernel itself bakes the bodies in via
 * NVRTC.
 */

#include <cuda_runtime.h>

#include "../../../include/compgen_rt/compgen_rt.h"

extern "C" {

__device__ __forceinline__
void cg_rt_cuda_etensor_peer_notify_d(
    long long *E_remote, int idx, int decrement
) {
    /* Same as the local notify but the pointer is peer-mapped.
     * atomicAdd_system on PCIe-mapped int64 is supported on
     * sm_60+. The system-coherent fence orders the producer's
     * data writes before the counter decrement on the remote
     * side. */
    __threadfence_system();
    atomicAdd_system((unsigned long long *)&E_remote[idx],
                     (unsigned long long)(-(long long)decrement));
}

/* Host-callable shim for unit tests. Same shape as event_tensor.cu's
 * notify_kernel but takes a peer pointer. */
__global__ void cg_rt_cuda_etensor_peer_notify_kernel(
    long long *E_remote, int idx, int decrement
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cg_rt_cuda_etensor_peer_notify_d(E_remote, idx, decrement);
    }
}

}  /* extern "C" */
