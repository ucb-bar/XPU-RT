/* Live device-traits probe — Phase 6 native HAL backend.
 *
 * Replaces compgen.runtime.probe.probe_via_torch when this CUDA
 * driver build is loaded on a real CUDA host. Returns the same
 * field set the Python probe collects, plus the higher-fidelity
 * `cudaDevAttr*` values the C side can read directly without going
 * through cuda-python bindings.
 *
 * The launcher fills a caller-allocated `cg_rt_cuda_probe_t` struct
 * (declared in the public header). Python's ctypes wrapper marshals
 * it back into a dict shape compatible with the torch path so
 * downstream code in compgen.runtime.traits.DeviceTraits.with_probe
 * doesn't need to branch.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>

#include "../../../include/compgen_rt/compgen_rt.h"

extern "C" {

static int safe_attr(cudaDeviceAttr attr, int device) {
    int value = 0;
    cudaError_t rc = cudaDeviceGetAttribute(&value, attr, device);
    return (rc == cudaSuccess) ? value : 0;
}

cg_rt_status_t cg_rt_cuda_probe_device(
    int                  device_index,
    cg_rt_cuda_probe_t  *out
) {
    if (out == NULL) return CG_RT_ERR_INVALID_ARGUMENT;

    int device_count = 0;
    cudaError_t rc = cudaGetDeviceCount(&device_count);
    if (rc != cudaSuccess) return CG_RT_ERR_UNKNOWN;
    if (device_index < 0 || device_index >= device_count) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }

    cudaDeviceProp props;
    rc = cudaGetDeviceProperties(&props, device_index);
    if (rc != cudaSuccess) return CG_RT_ERR_UNKNOWN;

    memset(out, 0, sizeof(*out));

    /* Identity. */
    strncpy(out->device_name, props.name, sizeof(out->device_name) - 1);
    out->compute_capability_major = props.major;
    out->compute_capability_minor = props.minor;
    out->num_visible_devices       = device_count;

    /* Counts + sizes. */
    out->sm_count                            = props.multiProcessorCount;
    out->max_threads_per_block               = props.maxThreadsPerBlock;
    out->max_threads_per_multiprocessor      = props.maxThreadsPerMultiProcessor;
    out->warp_size                           = props.warpSize;
    out->max_grid_dim_x                      = props.maxGridSize[0];
    out->max_grid_dim_y                      = props.maxGridSize[1];
    out->max_grid_dim_z                      = props.maxGridSize[2];
    out->max_device_memory_bytes             = (long long)props.totalGlobalMem;
    out->l2_cache_bytes                      = props.l2CacheSize;

    /* Higher-fidelity attributes. */
    out->max_shared_memory_per_block_optin_bytes =
        safe_attr(cudaDevAttrMaxSharedMemoryPerBlockOptin, device_index);
    out->max_blocks_per_cluster =
        safe_attr(cudaDevAttrMaxBlocksPerMultiprocessor, device_index);
    out->cluster_launch =
        safe_attr(cudaDevAttrClusterLaunch, device_index);
    out->cooperative_launch =
        safe_attr(cudaDevAttrCooperativeLaunch, device_index);
    out->concurrent_kernels =
        safe_attr(cudaDevAttrConcurrentKernels, device_index);
    out->concurrent_managed_access =
        safe_attr(cudaDevAttrConcurrentManagedAccess, device_index);

    /* CC-derived booleans (callers can also derive these themselves
     * from compute_capability_major; we materialise them so the
     * probe shape matches the Python torch-path probe).
     */
    int cc_major = props.major;
    out->supports_tma                  = (cc_major >= 9) ? 1 : 0;
    out->supports_clusters             = (cc_major >= 9 && out->cluster_launch) ? 1 : 0;
    out->supports_fp8                  = (cc_major >= 9) ? 1 : 0;
    out->supports_fp4                  = (cc_major >= 10) ? 1 : 0;
    out->supports_ondevice_scheduler   = (cc_major >= 9) ? 1 : 0;

    /* Driver + runtime versions. */
    int driver_version = 0;
    int runtime_version = 0;
    cuDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);
    out->driver_version  = driver_version;
    out->runtime_version = runtime_version;

    return CG_RT_OK;
}

}  /* extern "C" */
