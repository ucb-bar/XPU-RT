/*
 * Semaphore behaviour: monotonicity, idempotence, timeout, failure.
 */

#include "compgen_rt/compgen_rt.h"
#include "test_harness.h"

#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

static cg_rt_instance_t *g_instance = NULL;
static cg_rt_device_t   *g_device   = NULL;

static void setup_device(void) {
    if (g_instance != NULL) return;
    cg_rt_instance_create("cpu_sync", &g_instance);
    cg_rt_device_open(g_instance, 0, &g_device);
}

TEST_CASE(sem_create_destroy, "semaphore create/destroy with initial value") {
    setup_device();
    cg_rt_semaphore_t *sem = NULL;
    REQUIRE(cg_rt_semaphore_create(g_device, 5, &sem));
    uint64_t v = 0;
    REQUIRE(cg_rt_semaphore_query(sem, &v));
    EXPECT_EQ(v, 5);
    cg_rt_semaphore_destroy(sem);
}

TEST_CASE(sem_monotonic, "signal advances monotonically; lower values are no-ops") {
    setup_device();
    cg_rt_semaphore_t *sem = NULL;
    REQUIRE(cg_rt_semaphore_create(g_device, 0, &sem));
    REQUIRE(cg_rt_semaphore_signal(sem, 3));
    REQUIRE(cg_rt_semaphore_signal(sem, 2));  /* backward — no-op */
    uint64_t v = 0;
    REQUIRE(cg_rt_semaphore_query(sem, &v));
    EXPECT_EQ(v, 3);
    REQUIRE(cg_rt_semaphore_signal(sem, 10));
    REQUIRE(cg_rt_semaphore_query(sem, &v));
    EXPECT_EQ(v, 10);
    cg_rt_semaphore_destroy(sem);
}

TEST_CASE(sem_wait_immediate, "wait on satisfied semaphore returns immediately") {
    setup_device();
    cg_rt_semaphore_t *sem = NULL;
    REQUIRE(cg_rt_semaphore_create(g_device, 100, &sem));
    EXPECT_EQ(cg_rt_semaphore_wait(sem, 50, CG_RT_TIMEOUT_POLL), CG_RT_OK);
    cg_rt_semaphore_destroy(sem);
}

TEST_CASE(sem_poll_timeout, "poll on unmet semaphore returns TIMED_OUT") {
    setup_device();
    cg_rt_semaphore_t *sem = NULL;
    REQUIRE(cg_rt_semaphore_create(g_device, 0, &sem));
    EXPECT_EQ(cg_rt_semaphore_wait(sem, 1, CG_RT_TIMEOUT_POLL), CG_RT_ERR_TIMED_OUT);
    cg_rt_semaphore_destroy(sem);
}

struct signaller_args {
    cg_rt_semaphore_t *sem;
    uint64_t           value;
    uint32_t           delay_us;
};

static void *signal_after_delay(void *arg) {
    struct signaller_args *a = arg;
    usleep(a->delay_us);
    cg_rt_semaphore_signal(a->sem, a->value);
    return NULL;
}

TEST_CASE(sem_wait_blocks_then_wakes, "wait blocks until a background signal arrives") {
    setup_device();
    cg_rt_semaphore_t *sem = NULL;
    REQUIRE(cg_rt_semaphore_create(g_device, 0, &sem));

    pthread_t t;
    struct signaller_args args = { .sem = sem, .value = 7, .delay_us = 10000 };
    pthread_create(&t, NULL, signal_after_delay, &args);

    /* 1s timeout — plenty of headroom over the 10ms delay. */
    EXPECT_EQ(cg_rt_semaphore_wait(sem, 7, 1000000000ULL), CG_RT_OK);

    pthread_join(t, NULL);
    cg_rt_semaphore_destroy(sem);
}

TEST_CASE(sem_wait_times_out, "wait returns TIMED_OUT when no signal arrives") {
    setup_device();
    cg_rt_semaphore_t *sem = NULL;
    REQUIRE(cg_rt_semaphore_create(g_device, 0, &sem));
    /* 10ms timeout. */
    EXPECT_EQ(cg_rt_semaphore_wait(sem, 1, 10000000ULL), CG_RT_ERR_TIMED_OUT);
    cg_rt_semaphore_destroy(sem);
}

static void *fail_after_delay(void *arg) {
    cg_rt_semaphore_t *sem = arg;
    usleep(10000);
    cg_rt_semaphore_fail(sem, CG_RT_ERR_ABORTED);
    return NULL;
}

TEST_CASE(sem_failure_aborts_waiters, "cg_rt_semaphore_fail aborts pending waiters") {
    setup_device();
    cg_rt_semaphore_t *sem = NULL;
    REQUIRE(cg_rt_semaphore_create(g_device, 0, &sem));

    pthread_t t;
    pthread_create(&t, NULL, fail_after_delay, sem);
    EXPECT_EQ(cg_rt_semaphore_wait(sem, 1, 1000000000ULL), CG_RT_ERR_ABORTED);
    pthread_join(t, NULL);
    cg_rt_semaphore_destroy(sem);
}

int main(void) {
    setup_device();
    int rc = run_tests();
    cg_rt_device_close(g_device);
    cg_rt_instance_destroy(g_instance);
    return rc;
}
