#include "xpu-rt/baseline_runner.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
	if (argc < 2) {
		fprintf(stderr,
			"Usage: %s <dispatch_graph.json> [driver] [graph_iters] "
			"[dispatch_iters]\n",
			argv[0]);
		return 1;
	}

	baseline_runner_config_t cfg;
	memset(&cfg, 0, sizeof(cfg));
	cfg.graph_json_path = argv[1];
	cfg.driver_name = (argc >= 3) ? argv[2] : "local-task";
	cfg.graph_iters = (argc >= 4) ? atoi(argv[3]) : 1;
	cfg.dispatch_iters = (argc >= 5) ? atoi(argv[4]) : 1;
	cfg.parallelism = 1;

	return baseline_runner_run(&cfg);
}
