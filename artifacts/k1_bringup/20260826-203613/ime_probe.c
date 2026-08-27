/* Per-core probe: is RVV usable, and does the SpaceMiT IME op (smt.vmadot) execute?
 *
 * /proc/cpuinfo does not enumerate the vendor IME extension, so the only honest
 * test is to execute the instruction and see whether it traps.
 * Encoding from merlin/benchmarks/SpacemiTX60/compile_matmul_xsmt_i8_ukernel_all.sh:37
 *   .insn r 0x2b, 0x3, 0x71   (opcode=custom-1, funct3=3, funct7=0x71)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <signal.h>
#include <setjmp.h>
#include <sched.h>
#include <unistd.h>
#include <string.h>

static sigjmp_buf jb;
static void onsig(int s) { (void)s; siglongjmp(jb, 1); }

/* returns 0 = executed, 1 = trapped */
static int try_rvv(void) {
    if (sigsetjmp(jb, 1) == 0) {
        unsigned long vl;
        __asm__ volatile("vsetvli %0, zero, e8, m1, ta, ma" : "=r"(vl));
        return 0;
    }
    return 1;
}

static int try_ime(void) {
    if (sigsetjmp(jb, 1) == 0) {
        unsigned long vl;
        /* enable the vector unit first so a trap can only come from the IME op */
        __asm__ volatile("vsetvli %0, zero, e8, m1, ta, ma" : "=r"(vl));
        __asm__ volatile(".insn r 0x2b, 0x3, 0x71, x0, x0, x0");
        return 0;
    }
    return 1;
}

int main(void) {
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = onsig;
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS,  &sa, NULL);

    printf("cpu  cluster  rvv        ime(smt.vmadot)\n");
    for (long c = 0; c < ncpu; c++) {
        cpu_set_t m;
        CPU_ZERO(&m);
        CPU_SET(c, &m);
        if (sched_setaffinity(0, sizeof m, &m) != 0) {
            printf("%3ld  %-7s  AFFINITY-FAILED\n", c, "?");
            continue;
        }
        sched_yield();
        int where = sched_getcpu();
        int rvv = try_rvv();
        int ime = try_ime();
        printf("%3ld  %-7d  %-9s  %s%s\n", c, (where < 4 ? 0 : 1),
               rvv ? "TRAP" : "ok",
               ime ? "TRAP (unsupported)" : "EXECUTED (supported)",
               (where == (int)c) ? "" : "   [!! ran on another cpu]");
        fflush(stdout);
    }
    return 0;
}
