source /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/env.sh
python latency_eval.py --latency-ms 0     --pipeline --init-rng 0 --n 24 --tag _PIPEVAL
python latency_eval.py --latency-ms 283.4 --pipeline --init-rng 0 --n 24
python latency_eval.py --latency-ms 283.4 --pipeline --init-rng 2 --n 24
python latency_eval.py --latency-ms 283.4 --pipeline --init-rng 4 --n 24
python latency_eval.py --latency-ms 555   --pipeline --init-rng 0 --n 24
python latency_eval.py --latency-ms 555   --pipeline --init-rng 2 --n 24
python latency_eval.py --latency-ms 555   --pipeline --init-rng 4 --n 24
