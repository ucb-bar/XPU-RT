/*
 * Event tensor — atomic notify + blocking wait. Paper primitive.
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

TEST_CASE(et_create_num_cells, "num_cells = product of shape") {
    setup_device();
    int64_t shape[] = {2, 3, 4};
    cg_rt_event_tensor_t *et = NULL;
    REQUIRE(cg_rt_event_tensor_create(g_device, 3, shape,
                                      CG_RT_EVENT_DTYPE_I64, 5, &et));
    EXPECT_EQ(cg_rt_event_tensor_num_cells(et), 24);
    cg_rt_event_tensor_destroy(et);
}

TEST_CASE(et_notify_decrements, "notify decrements the counter atomically") {
    setup_device();
    int64_t shape[] = {4};
    cg_rt_event_tensor_t *et = NULL;
    REQUIRE(cg_rt_event_tensor_create(g_device, 1, shape,
                                      CG_RT_EVENT_DTYPE_I32, 10, &et));
    REQUIRE(cg_rt_event_tensor_notify(et, 2, 3));
    int64_t v = 0;
    REQUIRE(cg_rt_event_tensor_query(et, 2, &v));
    EXPECT_EQ(v, 7);
    /* Unrelated cells untouched. */
    REQUIRE(cg_rt_event_tensor_query(et, 0, &v));
    EXPECT_EQ(v, 10);
    cg_rt_event_tensor_destroy(et);
}

TEST_CASE(et_wait_satisfied, "wait returns OK when counter already <= 0") {
    setup_device();
    int64_t shape[] = {1};
    cg_rt_event_tensor_t *et = NULL;
    REQUIRE(cg_rt_event_tensor_create(g_device, 1, shape,
                                      CG_RT_EVENT_DTYPE_I64, 0, &et));
    EXPECT_EQ(cg_rt_event_tensor_wait(et, 0, CG_RT_TIMEOUT_POLL), CG_RT_OK);
    cg_rt_event_tensor_destroy(et);
}

TEST_CASE(et_poll_unsatisfied, "poll on positive counter returns TIMED_OUT") {
    setup_device();
    int64_t shape[] = {1};
    cg_rt_event_tensor_t *et = NULL;
    REQUIRE(cg_rt_event_tensor_create(g_device, 1, shape,
                                      CG_RT_EVENT_DTYPE_I64, 5, &et));
    EXPECT_EQ(cg_rt_event_tensor_wait(et, 0, CG_RT_TIMEOUT_POLL),
              CG_RT_ERR_TIMED_OUT);
    cg_rt_event_tensor_destroy(et);
}

struct notify_args {
    cg_rt_event_tensor_t *et;
    size_t                idx;
    int64_t               decrement;
    uint32_t              delay_us;
};

static void *notify_after_delay(void *arg) {
    struct notify_args *a = arg;
    usleep(a->delay_us);
    cg_rt_event_tensor_notify(a->et, a->idx, a->decrement);
    return NULL;
}

TEST_CASE(et_wait_wakes_on_notify, "wait blocks until background notify drops counter") {
    setup_device();
    int64_t shape[] = {2};
    cg_rt_event_tensor_t *et = NULL;
    REQUIRE(cg_rt_event_tensor_create(g_device, 1, shape,
                                      CG_RT_EVENT_DTYPE_I64, 3, &et));

    pthread_t t1, t2, t3;
    struct notify_args a1 = { .et = et, .idx = 1, .decrement = 1, .delay_us = 5000  };
    struct notify_args a2 = { .et = et, .idx = 1, .decrement = 1, .delay_us = 10000 };
    struct notify_args a3 = { .et = et, .idx = 1, .decrement = 1, .delay_us = 15000 };
    pthread_create(&t1, NULL, notify_after_delay, &a1);
    pthread_create(&t2, NULL, notify_after_delay, &a2);
    pthread_create(&t3, NULL, notify_after_delay, &a3);

    EXPECT_EQ(cg_rt_event_tensor_wait(et, 1, 1000000000ULL), CG_RT_OK);
    int64_t v = 0;
    REQUIRE(cg_rt_event_tensor_query(et, 1, &v));
    EXPECT_EQ(v, 0);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    pthread_join(t3, NULL);
    cg_rt_event_tensor_destroy(et);
}

TEST_CASE(et_reset_rewinds_counters, "reset restores all cells to the given value") {
    setup_device();
    int64_t shape[] = {3};
    cg_rt_event_tensor_t *et = NULL;
    REQUIRE(cg_rt_event_tensor_create(g_device, 1, shape,
                                      CG_RT_EVENT_DTYPE_I64, 1, &et));
    REQUIRE(cg_rt_event_tensor_notify(et, 0, 1));
    REQUIRE(cg_rt_event_tensor_notify(et, 1, 1));
    REQUIRE(cg_rt_event_tensor_notify(et, 2, 1));
    REQUIRE(cg_rt_event_tensor_reset(et, 7));

    int64_t v = 0;
    for (size_t i = 0; i < 3; ++i) {
        REQUIRE(cg_rt_event_tensor_query(et, i, &v));
        EXPECT_EQ(v, 7);
    }
    cg_rt_event_tensor_destroy(et);
}

int main(void) {
    setup_device();
    int rc = run_tests();
    cg_rt_device_close(g_device);
    cg_rt_instance_destroy(g_instance);
    return rc;
}
