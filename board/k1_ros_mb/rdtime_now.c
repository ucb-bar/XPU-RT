/* Prints the current rdtime tick so several processes can share one trace origin (--t0). */
#include <stdint.h>
#include <stdio.h>
int main(void) { uint64_t t; __asm__ volatile("rdtime %0" : "=r"(t)); printf("%llu\n", (unsigned long long)t); return 0; }
