/* Phase-4b NCCL bridge — single-process multi-device bring-up.
 *
 * Wraps ``ncclCommInitAll`` for the simple case the workstation
 * Blackwell setup (REMOTE bridge probe #047) supports: 2 GPUs, same
 * NUMA node, peer access GREEN both directions over PCIe NODE.
 * Multi-process bootstrap (ncclGetUniqueId + ncclCommInitRank) is
 * a v2 expansion when we move to multi-host runs.
 *
 * Linkage: NCCL header from the system CUDA toolkit (12.6's nccl.h
 * is ABI-compatible with the pip nvidia-nccl-cu13 the process
 * actually loads at runtime via the SONAME ``libnccl.so.2``). See
 * the CMakeLists.txt comment for why we don't tie to a specific
 * NCCL version at compile time.
 *
 * The peer-access setup happens inside ``cg_rt_cuda_comm_init_local``:
 * for every (i, j) pair of devices in the comm, both directions of
 * ``cuCtxEnablePeerAccess`` are called. Without it, the megakernel's
 * cross-GPU peer-notify atomics would fail with
 * ``CUDA_ERROR_INVALID_ADDRESS_SPACE``.
 */

#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "../../../include/compgen_rt/compgen_rt.h"

struct cg_rt_cuda_comm {
    int           num_devices;
    int          *device_indices;   /* size num_devices */
    ncclComm_t   *ncomms;           /* size num_devices */
    CUcontext    *contexts;         /* size num_devices, primary contexts */
};

cg_rt_status_t cg_rt_cuda_comm_init_local(
    int                  num_devices,
    const int           *device_indices,
    cg_rt_cuda_comm_t  **out
) {
    if (num_devices < 1 || device_indices == NULL || out == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }

    cg_rt_cuda_comm_t *comm = (cg_rt_cuda_comm_t *)calloc(1, sizeof(*comm));
    if (comm == NULL) return CG_RT_ERR_OUT_OF_MEMORY;
    comm->num_devices = num_devices;
    comm->device_indices = (int *)calloc((size_t)num_devices, sizeof(int));
    comm->ncomms = (ncclComm_t *)calloc((size_t)num_devices, sizeof(ncclComm_t));
    comm->contexts = (CUcontext *)calloc((size_t)num_devices, sizeof(CUcontext));
    if (comm->device_indices == NULL || comm->ncomms == NULL || comm->contexts == NULL) {
        cg_rt_cuda_comm_destroy(comm);
        return CG_RT_ERR_OUT_OF_MEMORY;
    }
    memcpy(comm->device_indices, device_indices, (size_t)num_devices * sizeof(int));

    /* Retain primary contexts for each device so peer-access enable
     * has a target context to bind to. cuInit is idempotent on
     * re-call. */
    if (cuInit(0) != CUDA_SUCCESS) {
        cg_rt_cuda_comm_destroy(comm);
        return CG_RT_ERR_UNKNOWN;
    }
    for (int i = 0; i < num_devices; ++i) {
        CUdevice cu_dev;
        if (cuDeviceGet(&cu_dev, device_indices[i]) != CUDA_SUCCESS) {
            cg_rt_cuda_comm_destroy(comm);
            return CG_RT_ERR_UNKNOWN;
        }
        if (cuDevicePrimaryCtxRetain(&comm->contexts[i], cu_dev) != CUDA_SUCCESS) {
            cg_rt_cuda_comm_destroy(comm);
            return CG_RT_ERR_UNKNOWN;
        }
    }

    /* Enable peer access pairwise. cuCtxEnablePeerAccess fails with
     * CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED if torch initialised
     * peer access before us — that's fine, we treat it as success. */
    for (int i = 0; i < num_devices; ++i) {
        if (cuCtxSetCurrent(comm->contexts[i]) != CUDA_SUCCESS) {
            cg_rt_cuda_comm_destroy(comm);
            return CG_RT_ERR_UNKNOWN;
        }
        for (int j = 0; j < num_devices; ++j) {
            if (i == j) continue;
            CUresult rc = cuCtxEnablePeerAccess(comm->contexts[j], 0);
            if (rc != CUDA_SUCCESS &&
                rc != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
                cg_rt_cuda_comm_destroy(comm);
                return CG_RT_ERR_UNKNOWN;
            }
        }
    }

    /* Initialise NCCL communicators. ncclCommInitAll is the
     * single-process multi-device path: NCCL handles bootstrap
     * internally without a unique_id exchange. */
    ncclResult_t nrc = ncclCommInitAll(
        comm->ncomms, num_devices, comm->device_indices);
    if (nrc != ncclSuccess) {
        cg_rt_cuda_comm_destroy(comm);
        return CG_RT_ERR_UNKNOWN;
    }

    *out = comm;
    return CG_RT_OK;
}

cg_rt_status_t cg_rt_cuda_comm_destroy(cg_rt_cuda_comm_t *comm) {
    if (comm == NULL) return CG_RT_OK;
    if (comm->ncomms != NULL) {
        for (int i = 0; i < comm->num_devices; ++i) {
            if (comm->ncomms[i] != NULL) {
                ncclCommDestroy(comm->ncomms[i]);
            }
        }
        free(comm->ncomms);
    }
    if (comm->contexts != NULL) {
        for (int i = 0; i < comm->num_devices; ++i) {
            if (comm->contexts[i] != NULL && comm->device_indices != NULL) {
                CUdevice cu_dev;
                if (cuDeviceGet(&cu_dev, comm->device_indices[i]) == CUDA_SUCCESS) {
                    cuDevicePrimaryCtxRelease(cu_dev);
                }
            }
        }
        free(comm->contexts);
    }
    if (comm->device_indices != NULL) free(comm->device_indices);
    free(comm);
    return CG_RT_OK;
}

int cg_rt_cuda_comm_size(cg_rt_cuda_comm_t *comm) {
    return comm == NULL ? 0 : comm->num_devices;
}

cg_rt_status_t cg_rt_cuda_comm_allreduce_fp32_sum(
    cg_rt_cuda_comm_t *comm,
    const void *const *inputs_per_rank,
    void *const       *outputs_per_rank,
    size_t             count
) {
    if (comm == NULL || inputs_per_rank == NULL ||
        outputs_per_rank == NULL || count == 0) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }

    /* NCCL requires a group call when issuing one collective per
     * rank from a single process. Each ncclAllReduce binds to the
     * device's current context — set them before the calls. */
    ncclGroupStart();
    for (int i = 0; i < comm->num_devices; ++i) {
        if (cuCtxSetCurrent(comm->contexts[i]) != CUDA_SUCCESS) {
            ncclGroupEnd();
            return CG_RT_ERR_UNKNOWN;
        }
        ncclResult_t rc = ncclAllReduce(
            inputs_per_rank[i], outputs_per_rank[i], count,
            ncclFloat32, ncclSum, comm->ncomms[i],
            /* stream = */ 0);
        if (rc != ncclSuccess) {
            ncclGroupEnd();
            return CG_RT_ERR_UNKNOWN;
        }
    }
    ncclGroupEnd();

    /* Sync all devices so the caller can read the outputs back. */
    for (int i = 0; i < comm->num_devices; ++i) {
        if (cuCtxSetCurrent(comm->contexts[i]) != CUDA_SUCCESS) {
            return CG_RT_ERR_UNKNOWN;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            return CG_RT_ERR_UNKNOWN;
        }
    }

    return CG_RT_OK;
}
