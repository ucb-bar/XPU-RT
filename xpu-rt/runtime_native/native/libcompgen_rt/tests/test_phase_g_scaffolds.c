/*
 * Scaffold-driver tests for HIP, Vulkan, and FireSim.
 *
 * Header-smoke verification that the new HIP, Vulkan, and FireSim
 * driver vtables compile, register correctly with
 * ``cg_rt_instance_create``, and route unsupported calls cleanly.
 *
 * Coverage:
 * - ``cg_rt_instance_create`` resolves "hip" / "vulkan" / "firesim"
 *   to the registered vtables (CG_RT_OK, instance pointer set).
 * - ``cg_rt_device_open(... 0 ...)`` returns CG_RT_ERR_NOT_FOUND when
 *   device_count is 0 (the scaffold-tier contract on builds without
 *   the matching CG_RT_WITH_<NAME> flag).
 *
 * When the real HIP / Vulkan / FireSim primitives are wired in behind
 * the build flags, the functional tests run from the same harness with
 * CG_RT_ERR_NOT_FOUND swapped for CG_RT_OK on hosts with the matching
 * SDK.
 */

#include "compgen_rt/compgen_rt.h"
#include "test_harness.h"

#include <stdint.h>

static void check_scaffold(const char *driver_name, int *failed) {
    cg_rt_instance_t *instance = NULL;
    cg_rt_status_t st = cg_rt_instance_create(driver_name, &instance);
    EXPECT_EQ(st, CG_RT_OK);
    EXPECT_TRUE(instance != NULL);

    /* On builds without the matching CG_RT_WITH_<NAME> flag the
     * scaffold's query_device_count returns 0, so device_open with
     * index 0 should be rejected with NOT_FOUND. */
    cg_rt_device_t *device = NULL;
    st = cg_rt_device_open(instance, 0, &device);
    EXPECT_EQ(st, CG_RT_ERR_NOT_FOUND);

    cg_rt_instance_destroy(instance);
}

TEST_CASE(phase_g_hip_scaffold_resolves, "hip driver registers in the driver table") {
    check_scaffold("hip", failed);
}

TEST_CASE(phase_g_vulkan_scaffold_resolves, "vulkan driver registers in the driver table") {
    check_scaffold("vulkan", failed);
}

TEST_CASE(phase_g_firesim_scaffold_resolves, "firesim driver registers in the driver table") {
    check_scaffold("firesim", failed);
}

TEST_CASE(phase_g_unknown_driver_returns_not_found, "unknown driver names route to NOT_FOUND") {
    cg_rt_instance_t *instance = NULL;
    cg_rt_status_t st = cg_rt_instance_create("definitely_not_a_real_driver", &instance);
    EXPECT_EQ(st, CG_RT_ERR_NOT_FOUND);
    EXPECT_TRUE(instance == NULL);
}

int main(void) {
    return run_tests();
}
