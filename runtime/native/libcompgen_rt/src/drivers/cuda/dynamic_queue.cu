/* On-GPU ready queue for the dynamic scheduler — Paper §3.2.
 *
 * A circular buffer of int32 task descriptors with atomic head/tail
 * indices. SMs `pop` ready tasks; producers `push` dependents when
 * an event tensor reaches zero. Single shared queue across all SMs
 * in the cluster — cheap (one cache line of state) but contends on
 * the head/tail atomics; the paper accepts that for the load-balance
 * benefit.
 *
 * Layout (allocated as one cudaMalloc by the launcher):
 *     int32_t head;            // atomic — next slot to pop
 *     int32_t tail;            // atomic — next slot to push
 *     int32_t pad[14];         // 64-byte cache-line padding
 *     int32_t slots[capacity]; // task_id ring buffer; -1 == empty
 *
 * Consumer (each SM):
 *     while (true) {
 *         int task_id = pop();
 *         if (task_id == NO_TASK) {
 *             if (all_tasks_done()) break;
 *             __nanosleep(64);
 *             continue;
 *         }
 *         execute(task_id);
 *         for each successor of task_id:
 *             if last predecessor → push(successor);
 *     }
 *
 * Producer (notify path that satisfied a task's last predecessor):
 *     push(successor_id);
 *
 * Sentinel: NO_TASK = -1. The slots array is initialised to -1 by
 * cg_rt_cuda_queue_alloc; pop returns -1 when head == tail (empty).
 */

#include <cuda_runtime.h>

#include "../../../include/compgen_rt/compgen_rt.h"

extern "C" {

#define CG_RT_CUDA_QUEUE_NO_TASK (-1)
#define CG_RT_CUDA_QUEUE_HEADER_INTS 16  /* head, tail, + pad to 64 B */

/* Header layout so device code can index without struct alignment
 * surprises. Slots start at offset CG_RT_CUDA_QUEUE_HEADER_INTS.
 */
__device__ __forceinline__ int  cg_rt_cuda_queue_capacity_d(const int *q) { return q[2]; }

/* push: append task_id at tail; returns 1 on success, 0 if the queue
 * is full (caller decides whether to spin-retry or fail loudly).
 *
 * Wraparound: tail is modulo capacity. When (tail+1) % cap == head,
 * the queue is full. We don't reserve a sentinel slot so capacity-1
 * pushes max — accept the off-by-one in exchange for code simplicity.
 */
__device__ __forceinline__
int cg_rt_cuda_queue_push_d(int *q, int task_id) {
    int capacity = cg_rt_cuda_queue_capacity_d(q);
    int slots_offset = CG_RT_CUDA_QUEUE_HEADER_INTS;

    /* Atomic CAS loop: read tail, compute next, CAS to claim the slot.
     * If full (next == head), bail.
     */
    while (true) {
        int tail = atomicAdd(&q[1], 0);   /* atomic load */
        int head = atomicAdd(&q[0], 0);
        int next = (tail + 1) % capacity;
        if (next == head) {
            return 0;  /* full */
        }
        int observed = atomicCAS(&q[1], tail, next);
        if (observed == tail) {
            /* We own slot `tail`. Store the task_id; threadfence so
             * the consumer sees the slot value before the new tail.
             * (Pop reads tail first then slots; this ordering matters.)
             */
            q[slots_offset + tail] = task_id;
            __threadfence_system();
            return 1;
        }
        /* CAS lost; retry. */
    }
}

/* pop: dequeue from head. Returns CG_RT_CUDA_QUEUE_NO_TASK on empty.
 *
 * Race: between "read head" and "read slot" another consumer could
 * have advanced head past our slot. We use atomicCAS on head to
 * claim a unique slot before reading it.
 */
__device__ __forceinline__
int cg_rt_cuda_queue_pop_d(int *q) {
    int capacity = cg_rt_cuda_queue_capacity_d(q);
    int slots_offset = CG_RT_CUDA_QUEUE_HEADER_INTS;

    while (true) {
        int head = atomicAdd(&q[0], 0);
        int tail = atomicAdd(&q[1], 0);
        if (head == tail) {
            return CG_RT_CUDA_QUEUE_NO_TASK;  /* empty */
        }
        int next = (head + 1) % capacity;
        int observed = atomicCAS(&q[0], head, next);
        if (observed == head) {
            /* We own slot `head`. The producer's threadfence ensures
             * the slot's task_id is visible.
             */
            int task_id = q[slots_offset + head];
            /* Optional: clear the slot to NO_TASK so debug tools can
             * tell live entries from stale ones.
             */
            q[slots_offset + head] = CG_RT_CUDA_QUEUE_NO_TASK;
            return task_id;
        }
        /* CAS lost; retry. */
    }
}

/* ----- host-side allocator ------------------------------------------- */

cg_rt_status_t cg_rt_cuda_queue_alloc(
    int **out_ptr,
    int   capacity
) {
    if (out_ptr == NULL || capacity < 2) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    int total_ints = CG_RT_CUDA_QUEUE_HEADER_INTS + capacity;
    int *dev = NULL;
    cudaError_t rc = cudaMalloc(&dev, sizeof(int) * (size_t)total_ints);
    if (rc != cudaSuccess) {
        return CG_RT_ERR_OUT_OF_MEMORY;
    }
    /* Initialise: head=0, tail=0, capacity=N, slots=-1. */
    int *host = (int *)malloc(sizeof(int) * (size_t)total_ints);
    if (host == NULL) {
        cudaFree(dev);
        return CG_RT_ERR_OUT_OF_MEMORY;
    }
    host[0] = 0;            /* head */
    host[1] = 0;            /* tail */
    host[2] = capacity;     /* capacity */
    for (int i = 3; i < CG_RT_CUDA_QUEUE_HEADER_INTS; ++i) host[i] = 0;
    for (int i = CG_RT_CUDA_QUEUE_HEADER_INTS; i < total_ints; ++i) {
        host[i] = CG_RT_CUDA_QUEUE_NO_TASK;
    }
    rc = cudaMemcpy(dev, host, sizeof(int) * (size_t)total_ints, cudaMemcpyHostToDevice);
    free(host);
    if (rc != cudaSuccess) {
        cudaFree(dev);
        return CG_RT_ERR_UNKNOWN;
    }
    *out_ptr = dev;
    return CG_RT_OK;
}

void cg_rt_cuda_queue_free(int *ptr) {
    if (ptr != NULL) {
        cudaFree(ptr);
    }
}

cg_rt_status_t cg_rt_cuda_queue_seed_initial(
    int       *q,
    const int *initial_task_ids,
    int        num_initial
) {
    if (q == NULL || initial_task_ids == NULL || num_initial < 0) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    /* Direct host-side memcpy into the slot array + set tail=N.
     * Precondition: queue is freshly allocated (head=tail=0).
     */
    cudaError_t rc = cudaMemcpy(
        &q[CG_RT_CUDA_QUEUE_HEADER_INTS],
        initial_task_ids,
        sizeof(int) * (size_t)num_initial,
        cudaMemcpyHostToDevice
    );
    if (rc != cudaSuccess) {
        return CG_RT_ERR_UNKNOWN;
    }
    rc = cudaMemcpy(
        &q[1], &num_initial, sizeof(int), cudaMemcpyHostToDevice
    );
    if (rc != cudaSuccess) {
        return CG_RT_ERR_UNKNOWN;
    }
    return CG_RT_OK;
}

}  /* extern "C" */
