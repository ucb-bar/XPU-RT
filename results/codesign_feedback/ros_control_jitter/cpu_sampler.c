/* Per-core busy fraction on the K1, sampled from /proc/stat.
 *
 *   cpu_sampler <interval_ms> <out.csv>
 *
 * One row per cpu per sample: cpu,t_ms,busy_pct,epoch_ms,rdtime_ticks. busy = everything but
 * idle and iowait, as a percentage of the interval. The last two columns are what aligns a
 * sample to a run: ROS runs record their rdtime origin and wall-clock start in manifest.json,
 * XPU-RT runs print theirs at start-up.
 *
 * USER_HZ is 100, so a 100 ms sample has 10 jiffies per core and a single sample is quantised
 * to 10 %. Report the mean over the run window, never a single sample. /proc/loadavg is not
 * usable on this board (two kernel threads hold it at 2.00 permanently).
 */
#define _GNU_SOURCE
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t stop = 0;
static void on_sig(int s) { (void)s; stop = 1; }
static inline uint64_t rdtime(void) { uint64_t t; __asm__ volatile("rdtime %0" : "=r"(t)); return t; }
static int64_t epoch_ms(void) { struct timeval tv; gettimeofday(&tv, NULL); return (int64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000; }

#define NCPU 8
static int read_stat(uint64_t busy[NCPU], uint64_t total[NCPU]) {
  FILE *f = fopen("/proc/stat", "r"); if (!f) return -1;
  char line[512]; int n = 0;
  while (fgets(line, sizeof line, f)) {
    int c; unsigned long long u, ni, s, id, io, irq, sirq, st;
    if (sscanf(line, "cpu%d %llu %llu %llu %llu %llu %llu %llu %llu", &c, &u, &ni, &s, &id, &io, &irq, &sirq, &st) == 9 && c < NCPU) {
      total[c] = u + ni + s + id + io + irq + sirq + st;
      busy[c] = total[c] - id - io; n++;
    }
  }
  fclose(f); return n;
}

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s <interval_ms> <out.csv>\n", argv[0]); return 2; }
  long interval = atol(argv[1]);
  FILE *out = fopen(argv[2], "w"); if (!out) { perror(argv[2]); return 1; }
  signal(SIGTERM, on_sig); signal(SIGINT, on_sig);
  fprintf(out, "cpu,t_ms,busy_pct,epoch_ms,rdtime_ticks\n");
  uint64_t b0[NCPU] = {0}, t0[NCPU] = {0}, b1[NCPU], t1[NCPU];
  if (read_stat(b0, t0) < 0) return 1;
  const int64_t e0 = epoch_ms();
  while (!stop) {
    struct timespec ts = { interval / 1000, (interval % 1000) * 1000000L }; nanosleep(&ts, NULL);
    if (read_stat(b1, t1) < 0) break;
    int64_t e = epoch_ms(); uint64_t r = rdtime();
    for (int c = 0; c < NCPU; c++) {
      uint64_t dt = t1[c] - t0[c], db = b1[c] - b0[c];
      fprintf(out, "%d,%lld,%.1f,%lld,%llu\n", c, (long long)(e - e0), dt ? 100.0 * db / dt : 0.0,
              (long long)e, (unsigned long long)r);
      b0[c] = b1[c]; t0[c] = t1[c];
    }
    fflush(out);
  }
  fclose(out); return 0;
}
