// runtime/tools/xpurt_scheduler_runner.c
//
// Target-and-model-parameterized consumer of the xpu-rt scheduler runner
// shipped inside libxpurt_standalone.a (built by Merlin). Mirrors
// /scratch2/agustin/merlin/samples/SpacemiTX60/dispatch_scheduler/main.c
// but with a built-in target preset table so the same binary can drive
// SpacemiT X60, QRB5165 (Snapdragon 865), or a host x86_64 build.
//
// "Different model" = different schedule.json + --vmfb_dir pair. The
// scheduler runner discovers each dispatch's module by name out of the
// schedule and resolves the matching VMFB under --vmfb_dir.
//
// Build:
//   ./runtime/build_runtime.sh --target host \
//       --xpurt-lib <merlin>/build/host-merlin-release/runtime/src/iree/runtime/libxpurt_standalone.a
//   ./runtime/build_runtime.sh --target spacemit \
//       --xpurt-lib <merlin>/build/spacemit-merlin-perf/runtime/src/iree/runtime/libxpurt_standalone.a
//
// Usage:
//   ./xpurt_scheduler_runner <schedule.json> --target=spacemit_x60 \
//       --vmfb_dir=/path/on/board/dispatches \
//       --cpu_p_cpu_ids=4,5,6,7 --cpu_e_cpu_ids=0,1,2,3
//
// The --target flag is the only thing that distinguishes this from the
// SpacemiT-specific main.c in Merlin: we pick the {target_platform,
// variant_p_dir, variant_e_dir, elf_marker} tuple from a preset table
// rather than hard-coding it. CLI flags can override any preset field.

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"

#include "core/cli_utils.h"
#include "xpu-rt/scheduler_runner.h"

//------------------------------------------------------------------------------
// Target presets
//------------------------------------------------------------------------------
//
// Each preset packages the four target-specific scheduler_runner_config_t
// strings plus a few sensible CPU-pinning defaults. Users can override
// any field via CLI flags. To add a new target, append a row to
// kPresets[] — no other changes needed.

typedef struct {
	const char *name; // selected via --target=<name>
	const char *target_platform; // VMFB path component
	const char *variant_p_dir; // ISA variant dir for CPU_P
	const char *variant_e_dir; // ISA variant dir for CPU_E
	const char *elf_marker; // architecture ELF marker (NULL = skip)
	// Sensible CPU-pinning defaults; CLI flags still win.
	const char *default_cpu_p_cpu_ids;
	const char *default_cpu_e_cpu_ids;
	const char *default_cpu_cpu_ids; // unified CPU device; "" = unset
	int default_visible_cores;
	int default_qnn_gpu_enabled;
	int default_qnn_hta_enabled;
} target_preset_t;

static const target_preset_t kPresets[] = {
	// SpacemiT X60: 8 RVV cores, no E cluster split by default. The
	// existing SpacemiT main.c in Merlin uses 0,1,2,3 for P and 4,5 for
	// E — kept here for parity with that runner's behavior.
	{
		.name = "spacemit_x60",
		.target_platform = "spacemit_x60",
		.variant_p_dir = "RVV",
		.variant_e_dir = "scalar",
		.elf_marker = "_embedded_elf_riscv_64",
		.default_cpu_p_cpu_ids = "0,1,2,3",
		.default_cpu_e_cpu_ids = "4,5",
		.default_cpu_cpu_ids = "",
		.default_visible_cores = 8,
		.default_qnn_gpu_enabled = 0,
		.default_qnn_hta_enabled = 0,
	},
	// QRB5165 (Snapdragon 865): 4 P-cores (Cortex-A77) + 4 E-cores
	// (Cortex-A55), Adreno GPU + Hexagon HTA accessible via QNN.
	// Conventional pinning here: P on cores 4-7, E on 0-3 (matches
	// run_full_loop.py defaults).
	{
		.name = "qrb5165",
		.target_platform = "qrb5165",
		.variant_p_dir = "armv8a",
		.variant_e_dir = "armv8a",
		.elf_marker = "_embedded_elf_arm_64",
		.default_cpu_p_cpu_ids = "4,5,6,7",
		.default_cpu_e_cpu_ids = "0,1,2,3",
		.default_cpu_cpu_ids = "",
		.default_visible_cores = 8,
		// Auto-on when the schedule names a QNN target; the scheduler
		// runner already does that detection. Default to off so a
		// pure-CPU schedule doesn't pull in libQnn*.so unnecessarily.
		.default_qnn_gpu_enabled = 0,
		.default_qnn_hta_enabled = 0,
	},
	// Host x86_64: useful for unit-testing the runtime / hot-swap path
	// without crossing to a board. No ELF marker — the resolver skips
	// stem matching and relies on full module names.
	{
		.name = "host",
		.target_platform = "host",
		.variant_p_dir = "x86_64",
		.variant_e_dir = "x86_64",
		.elf_marker = NULL,
		.default_cpu_p_cpu_ids = "0,1",
		.default_cpu_e_cpu_ids = "2,3",
		.default_cpu_cpu_ids = "",
		.default_visible_cores = 0, // 0 = no CPU-id range validation
		.default_qnn_gpu_enabled = 0,
		.default_qnn_hta_enabled = 0,
	},
};

static const target_preset_t *find_preset(const char *name) {
	if (!name || !name[0])
		return &kPresets[0]; // first preset is the default
	for (size_t i = 0; i < sizeof(kPresets) / sizeof(kPresets[0]); ++i) {
		if (strcmp(kPresets[i].name, name) == 0)
			return &kPresets[i];
	}
	return NULL;
}

static void print_targets(FILE *f) {
	for (size_t i = 0; i < sizeof(kPresets) / sizeof(kPresets[0]); ++i) {
		fprintf(f, "    %-15s platform=%s variant_p=%s variant_e=%s "
			"elf_marker=%s\n",
			kPresets[i].name, kPresets[i].target_platform,
			kPresets[i].variant_p_dir, kPresets[i].variant_e_dir,
			kPresets[i].elf_marker ? kPresets[i].elf_marker : "(none)");
	}
}

static void print_usage(const char *argv0) {
	fprintf(stderr,
		"Usage:\n"
		"  %s <dispatch_schedule.json> [driver] [graph_iters] "
		"[dispatch_iters] [report_every] [--flags]\n"
		"\n"
		"Defaults:\n"
		"  driver          = local-task\n"
		"  graph_iters     = 1\n"
		"  dispatch_iters  = 1\n"
		"  report_every    = 0 (final only)\n"
		"\n"
		"Required-ish flags:\n"
		"  --target=<name>             Target preset (default: spacemit_x60).\n"
		"                              Available presets:\n",
		argv0);
	print_targets(stderr);
	fprintf(stderr,
		"  --vmfb_dir=<path>           Directory containing per-dispatch .vmfb.\n"
		"\n"
		"Override flags (any preset field can be overridden):\n"
		"  --target_platform=<name>    Override preset's target_platform.\n"
		"  --variant_p_dir=<name>      Override preset's variant_p_dir.\n"
		"  --variant_e_dir=<name>      Override preset's variant_e_dir.\n"
		"  --elf_marker=<name>         Override preset's elf_marker.\n"
		"  --cpu_p_cpu_ids=...         Logical CPUs for CPU_P.\n"
		"  --cpu_e_cpu_ids=...         Logical CPUs for CPU_E.\n"
		"  --cpu_cpu_ids=...           Unified CPU device (all cores).\n"
		"  --visible_cores=N           Validate IDs are in [0, N).\n"
		"  --qnn_gpu_enabled[=1|0]     Force-create the QNN_GPU device.\n"
		"  --qnn_hta_enabled[=1|0]     Force-create the QNN_HTA device.\n"
		"\n"
		"Output flags:\n"
		"  --out_json=<path>           Summary JSON.\n"
		"  --out_dot=<path>            DOT graph.\n"
		"  --trace_csv=<path>          Trace CSV.\n"
		"  --telemetry_jsonl=<path>    Per-dispatch JSON-Lines for XPU-RT\n"
		"                              hardware-in-the-loop feedback.\n"
		"  --schedule_next=<path>      Watch this path for hot-swapped\n"
		"                              schedules (atomic at epoch boundary).\n"
		"\n"
		"Example (spacemit_x60):\n"
		"  %s schedule.json --target=spacemit_x60 \\\n"
		"     --vmfb_dir=/root/dronet/breakdowns \\\n"
		"     --cpu_p_cpu_ids=0,1,2,3 --cpu_e_cpu_ids=4,5\n"
		"\n"
		"Example (qrb5165 + telemetry + hot-swap):\n"
		"  %s schedule.json --target=qrb5165 \\\n"
		"     --vmfb_dir=/root/iree_run/dronet/breakdowns \\\n"
		"     --telemetry_jsonl=/tmp/telemetry.jsonl \\\n"
		"     --schedule_next=/tmp/schedule_next.json\n",
		argv0, argv0);
}

int main(int argc, char **argv) {
	// IREE global flags (e.g. --task_topology_cpu_ids=...) are consumed
	// here so our parser sees only its own flags.
	iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK, &argc, &argv);

	if (argc < 2) {
		print_usage(argv[0]);
		return 1;
	}

	const char *json_path = argv[1];
	const char *driver = (argc >= 3) ? argv[2] : "local-task";
	const int graph_iters =
		(argc >= 4) ? parse_int_or_default(argv[3], 1) : 1;
	const int dispatch_iters =
		(argc >= 5) ? parse_int_or_default(argv[4], 1) : 1;
	const int report_every =
		(argc >= 6) ? parse_int_or_default(argv[5], 0) : 0;

	// Pre-pass: find --target=<name> so we can apply preset defaults
	// before processing the rest of the flag list. Anything else (CPU
	// ids, qnn toggles) wins over the preset.
	const char *target_name = NULL;
	for (int i = 6; i < argc; ++i) {
		const char *v = get_flag_value(argv[i], "--target");
		if (v) {
			target_name = v;
			break;
		}
	}
	const target_preset_t *preset = find_preset(target_name);
	if (preset == NULL) {
		fprintf(stderr, "Unknown --target=%s\nAvailable:\n",
			target_name ? target_name : "(none)");
		print_targets(stderr);
		return 1;
	}

	// Initialize from preset; CLI overrides applied below.
	const char *vmfb_dir = NULL;
	const char *cpu_p_cpu_ids = preset->default_cpu_p_cpu_ids;
	const char *cpu_e_cpu_ids = preset->default_cpu_e_cpu_ids;
	const char *cpu_cpu_ids = (preset->default_cpu_cpu_ids
		&& preset->default_cpu_cpu_ids[0])
		? preset->default_cpu_cpu_ids
		: NULL;
	int visible_cores = preset->default_visible_cores;
	int qnn_gpu_enabled = preset->default_qnn_gpu_enabled;
	int qnn_hta_enabled = preset->default_qnn_hta_enabled;
	const char *target_platform = preset->target_platform;
	const char *variant_p_dir = preset->variant_p_dir;
	const char *variant_e_dir = preset->variant_e_dir;
	const char *elf_marker = preset->elf_marker;

	const char *out_json = NULL;
	const char *out_dot = NULL;
	const char *trace_csv = NULL;
	const char *telemetry_jsonl = NULL;
	const char *schedule_next = NULL;

	for (int i = 6; i < argc; ++i) {
		const char *v = NULL;
		// --target was already resolved above; skip it here.
		if (get_flag_value(argv[i], "--target") != NULL) {
			continue;
		}
		if ((v = get_flag_value(argv[i], "--target_platform"))) {
			target_platform = v;
		} else if ((v = get_flag_value(argv[i], "--variant_p_dir"))) {
			variant_p_dir = v;
		} else if ((v = get_flag_value(argv[i], "--variant_e_dir"))) {
			variant_e_dir = v;
		} else if ((v = get_flag_value(argv[i], "--elf_marker"))) {
			// Allow explicit empty to disable stem matching.
			elf_marker = v[0] ? v : NULL;
		} else if ((v = get_flag_value(argv[i], "--vmfb_dir"))) {
			vmfb_dir = v;
		} else if ((v = get_flag_value(argv[i], "--cpu_p_cpu_ids"))) {
			cpu_p_cpu_ids = v;
		} else if ((v = get_flag_value(argv[i], "--cpu_e_cpu_ids"))) {
			cpu_e_cpu_ids = v;
		} else if ((v = get_flag_value(argv[i], "--cpu_cpu_ids"))) {
			cpu_cpu_ids = v;
		} else if ((v = get_flag_value(argv[i], "--visible_cores"))) {
			visible_cores = parse_int_or_default(v, visible_cores);
		} else if ((v = get_flag_value(argv[i], "--qnn_gpu_enabled"))) {
			qnn_gpu_enabled = parse_int_or_default(v, 1);
		} else if (strcmp(argv[i], "--qnn_gpu_enabled") == 0) {
			qnn_gpu_enabled = 1;
		} else if ((v = get_flag_value(argv[i], "--qnn_hta_enabled"))) {
			qnn_hta_enabled = parse_int_or_default(v, 1);
		} else if (strcmp(argv[i], "--qnn_hta_enabled") == 0) {
			qnn_hta_enabled = 1;
		} else if ((v = get_flag_value(argv[i], "--out_json"))) {
			out_json = v;
		} else if ((v = get_flag_value(argv[i], "--out_dot"))) {
			out_dot = v;
		} else if ((v = get_flag_value(argv[i], "--trace_csv"))) {
			trace_csv = v;
		} else if ((v = get_flag_value(argv[i], "--telemetry_jsonl"))) {
			telemetry_jsonl = v;
		} else if ((v = get_flag_value(argv[i], "--schedule_next"))) {
			schedule_next = v;
		} else {
			fprintf(stderr, "Unknown arg: %s\n\n", argv[i]);
			print_usage(argv[0]);
			return 1;
		}
	}

	scheduler_runner_config_t cfg;
	memset(&cfg, 0, sizeof(cfg));
	cfg.graph_json_path = json_path;
	cfg.driver_name = driver;
	cfg.graph_iters = graph_iters;
	cfg.dispatch_iters = dispatch_iters;
	cfg.report_every = report_every;
	cfg.vmfb_root_dir = vmfb_dir;
	cfg.cpu_p_cpu_ids = cpu_p_cpu_ids;
	cfg.cpu_e_cpu_ids = cpu_e_cpu_ids;
	cfg.cpu_cpu_ids = cpu_cpu_ids;
	cfg.visible_cores = visible_cores;
	cfg.qnn_gpu_enabled = qnn_gpu_enabled;
	cfg.qnn_hta_enabled = qnn_hta_enabled;
	cfg.out_json_path = out_json;
	cfg.out_dot_path = out_dot;
	cfg.trace_csv_path = trace_csv;
	cfg.telemetry_jsonl_path = telemetry_jsonl;
	cfg.telemetry_fd = -1; // path-based; this binary owns no fd
	cfg.schedule_next_path = schedule_next;
	cfg.target_platform = target_platform;
	cfg.variant_p_dir = variant_p_dir;
	cfg.variant_e_dir = variant_e_dir;
	cfg.elf_marker = elf_marker;

	fprintf(stdout, "[xpurt] target=%s platform=%s variant_p=%s "
		"variant_e=%s elf_marker=%s\n",
		preset->name, target_platform, variant_p_dir, variant_e_dir,
		elf_marker ? elf_marker : "(none)");
	fflush(stdout);

	return scheduler_runner_run(&cfg);
}
