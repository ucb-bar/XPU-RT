/* Persistent megakernel launcher — Phase 4 host glue.
 *
 * Sets up the cooperative-launch parameters from a Phase-2 / Phase-3
 * `LaunchConfig` (lowered to C from the Python dataclass via the
 * launcher's argument struct), allocates event tensors + the dynamic
 * ready queue if needed, then issues `cuLaunchCooperativeKernel`.
 *
 * Cluster launch (paper Fig. 6): when ``cluster_dim`` is nonzero we
 * attach a ``cudaLaunchAttributeClusterDimension`` so the launch
 * runs as a cluster — this is the path that gives us cluster.sync()
 * + DSMEM. Confirmed available on sm_120 per the bwell probe
 * (REMOTE bridge block #011).
 *
 * Phase 5 emits the persistent kernel function itself (one big
 * `__global__ void megakernel_forward(...)` that loops popping
 * tasks). This launcher just configures the launch and issues
 * cuLaunchCooperativeKernel — it doesn't know what the kernel
 * does.
 *
 * Multi-GPU (Phase 4b): not yet wired here. When Phase 4b lands,
 * this launcher will accept an optional `ncclComm_t` and, after
 * launch, drive the cooperative completion barrier across ranks.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "../../../include/compgen_rt/compgen_rt.h"

extern "C" {

cg_rt_status_t cg_rt_cuda_megakernel_launch(
    cg_rt_device_t                          *device,
    const cg_rt_cuda_megakernel_launch_t    *config,
    void                                   **kernel_args
) {
    if (device == NULL || config == NULL || config->kernel_handle == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }

    CUfunction kernel = (CUfunction)config->kernel_handle;

    /* Build launch attributes. We always set cooperative because
     * the persistent-kernel pattern needs it; cluster is opt-in.
     */
    CUlaunchAttribute attrs[2];
    int num_attrs = 0;

    attrs[num_attrs].id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
    attrs[num_attrs].value.cooperative = 1;
    num_attrs++;

    if (config->cluster_dim_x > 0
        && config->cluster_dim_y > 0
        && config->cluster_dim_z > 0) {
        attrs[num_attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        attrs[num_attrs].value.clusterDim.x = config->cluster_dim_x;
        attrs[num_attrs].value.clusterDim.y = config->cluster_dim_y;
        attrs[num_attrs].value.clusterDim.z = config->cluster_dim_z;
        num_attrs++;
    }

    CUlaunchConfig launch_cfg = {0};
    launch_cfg.gridDimX  = (unsigned int)config->grid_dim_x;
    launch_cfg.gridDimY  = (unsigned int)config->grid_dim_y;
    launch_cfg.gridDimZ  = (unsigned int)config->grid_dim_z;
    launch_cfg.blockDimX = (unsigned int)config->block_dim_x;
    launch_cfg.blockDimY = (unsigned int)config->block_dim_y;
    launch_cfg.blockDimZ = (unsigned int)config->block_dim_z;
    launch_cfg.sharedMemBytes = (unsigned int)config->shared_mem_bytes;
    launch_cfg.hStream    = NULL;  /* default stream */
    launch_cfg.attrs      = attrs;
    launch_cfg.numAttrs   = num_attrs;

    /* If the kernel needs more than the default 48 KB shared memory,
     * opt in via cuFuncSetAttribute. Phase 5's emitter records the
     * required smem in config->shared_mem_bytes; if it exceeds the
     * default we set the opt-in. Workstation Blackwell (sm_120)
     * caps this at 99 KiB per the probe.
     */
    if (config->shared_mem_bytes > 49152) {
        CUresult rc_attr = cuFuncSetAttribute(
            kernel,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            (int)config->shared_mem_bytes
        );
        if (rc_attr != CUDA_SUCCESS) {
            const char *_name = NULL;
            cuGetErrorName(rc_attr, &_name);
            fprintf(stderr,
                "compgen_rt: cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES=%u) failed: %s\n",
                (unsigned)config->shared_mem_bytes, _name ? _name : "?");
            return CG_RT_ERR_UNKNOWN;
        }
    }

    CUresult rc = cuLaunchKernelEx(
        &launch_cfg,
        kernel,
        kernel_args,
        NULL  /* extras */
    );
    if (rc != CUDA_SUCCESS) {
        const char *_name = NULL;
        cuGetErrorName(rc, &_name);
        fprintf(stderr,
            "compgen_rt: cuLaunchKernelEx failed: %s "
            "(grid=(%u,%u,%u) block=(%u,%u,%u) cluster=(%u,%u,%u) shmem=%u cooperative=1)\n",
            _name ? _name : "?",
            (unsigned)launch_cfg.gridDimX, (unsigned)launch_cfg.gridDimY, (unsigned)launch_cfg.gridDimZ,
            (unsigned)launch_cfg.blockDimX, (unsigned)launch_cfg.blockDimY, (unsigned)launch_cfg.blockDimZ,
            (unsigned)config->cluster_dim_x, (unsigned)config->cluster_dim_y, (unsigned)config->cluster_dim_z,
            (unsigned)launch_cfg.sharedMemBytes);
        return CG_RT_ERR_UNKNOWN;
    }

    /* Synchronise so the caller observes the persistent kernel's
     * outputs. The whole point of "one cooperative launch" is that
     * by the time this returns, every task in the megakernel has
     * completed and notified its successors — there's no second
     * launch to coordinate.
     */
    cudaError_t rc_sync = cudaDeviceSynchronize();
    if (rc_sync != cudaSuccess) {
        fprintf(stderr,
            "compgen_rt: cudaDeviceSynchronize after megakernel launch failed: %s\n",
            cudaGetErrorName(rc_sync));
        return CG_RT_ERR_UNKNOWN;
    }
    return CG_RT_OK;
}

}  /* extern "C" */
