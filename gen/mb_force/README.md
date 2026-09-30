# gen/mb_force/ — YOLO with the matrix engine forced on

`gen/mb_shard/` with YOLO's tables taken from runs that force every eligible op onto the matrix engine
(`scripts/make_ime_profile.py --from-runs …ime_force_shard4…`). See `docs/K1/partitioned_schedule.md`.
