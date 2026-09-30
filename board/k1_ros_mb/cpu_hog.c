/* A CPU-bound background process (telemetry, logging, compression stand-in): spins until killed,
 * unpinned, so the OS places it. Used to measure how each runtime's cadence survives a busy
 * neighbour. */
#include <signal.h>
static volatile int stop = 0; static void h(int s){(void)s; stop = 1;}
int main(void){ signal(15,h); volatile unsigned long x = 0; while(!stop){ x += x * 3 + 1; } return 0; }
