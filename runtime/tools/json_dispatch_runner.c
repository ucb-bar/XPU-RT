#include "xpurt_scheduler_core.h"

#include <stdio.h>

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <dispatch_graph.json> [driver]\n", argv[0]);
    return 1;
  }

  const char* json_path = argv[1];
  const char* driver_name = (argc >= 3) ? argv[2] : "local-task";

  xpurt_status_t st = xpurt_run_dispatch_graph(json_path, driver_name);
  if (st != XPURT_STATUS_OK) {
    fprintf(stderr, "xpurt_run_dispatch_graph failed for %s (driver=%s)\n",
            json_path, driver_name);
    return 1;
  }

  return 0;
}

