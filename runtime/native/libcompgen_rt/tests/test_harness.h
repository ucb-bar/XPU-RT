/*
 * Tiny zero-dependency test harness for libcompgen_rt C tests.
 *
 * Usage:
 *   TEST_CASE("semaphore advances monotonically") {
 *       EXPECT_EQ(cg_rt_semaphore_signal(sem, 1), CG_RT_OK);
 *       ...
 *   }
 *   int main(void) { return run_tests(); }
 */

#ifndef COMPGEN_RT_TEST_HARNESS_H_
#define COMPGEN_RT_TEST_HARNESS_H_

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct test_registration {
    const char *name;
    void (*fn)(int *failed);
    struct test_registration *next;
} test_registration_t;

static test_registration_t *test_head = NULL;
static test_registration_t *test_tail = NULL;

static void register_test(test_registration_t *reg) {
    if (test_head == NULL) {
        test_head = reg;
    } else {
        test_tail->next = reg;
    }
    test_tail = reg;
}

/*
 * Pass a C identifier AND a display string, because macro concatenation
 * with __LINE__ does not expand cleanly across all compilers. Keep the
 * identifier unique per file.
 */
#define TEST_CASE(ident, display_name)                                       \
    static void ident##_body(int *failed);                                   \
    static test_registration_t ident##_reg = {                               \
        .name = display_name, .fn = ident##_body, .next = NULL };            \
    __attribute__((constructor)) static void ident##_register(void) {        \
        register_test(&ident##_reg);                                         \
    }                                                                        \
    static void ident##_body(int *failed)

#define EXPECT_EQ(a, b) do {                                                 \
    long long _a = (long long)(a);                                           \
    long long _b = (long long)(b);                                           \
    if (_a != _b) {                                                          \
        fprintf(stderr, "  [%s:%d] EXPECT_EQ failed: %s (%lld) != %s (%lld)\n",\
                __FILE__, __LINE__, #a, _a, #b, _b);                         \
        *failed = 1;                                                         \
    }                                                                        \
} while (0)

#define EXPECT_NE(a, b) do {                                                 \
    long long _a = (long long)(a);                                           \
    long long _b = (long long)(b);                                           \
    if (_a == _b) {                                                          \
        fprintf(stderr, "  [%s:%d] EXPECT_NE failed: both %lld\n",           \
                __FILE__, __LINE__, _a);                                     \
        *failed = 1;                                                         \
    }                                                                        \
} while (0)

#define EXPECT_TRUE(x) do {                                                  \
    if (!(x)) {                                                              \
        fprintf(stderr, "  [%s:%d] EXPECT_TRUE failed: %s\n",                \
                __FILE__, __LINE__, #x);                                     \
        *failed = 1;                                                         \
    }                                                                        \
} while (0)

#define REQUIRE(expr) do {                                                   \
    cg_rt_status_t _rc = (expr);                                             \
    if (_rc != CG_RT_OK) {                                                   \
        fprintf(stderr, "  [%s:%d] REQUIRE failed: %s -> %s\n",              \
                __FILE__, __LINE__, #expr, cg_rt_status_string(_rc));        \
        *failed = 1;                                                         \
        return;                                                              \
    }                                                                        \
} while (0)

static int run_tests(void) {
    int total = 0, passed = 0;
    for (test_registration_t *r = test_head; r != NULL; r = r->next) {
        printf("[ RUN      ] %s\n", r->name);
        int local_failed = 0;
        r->fn(&local_failed);
        if (local_failed) {
            printf("[  FAILED  ] %s\n", r->name);
        } else {
            printf("[       OK ] %s\n", r->name);
            ++passed;
        }
        ++total;
    }
    printf("[==========] %d passed out of %d\n", passed, total);
    return (passed == total) ? 0 : 1;
}

#endif /* COMPGEN_RT_TEST_HARNESS_H_ */
