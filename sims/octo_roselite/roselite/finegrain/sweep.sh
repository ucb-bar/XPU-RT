#!/bin/bash
# Fine-grain latency sweep. All latencies MEASURED on the QRB5165 this session.
#   name        latency_ms  cadence_ms   provenance
#   pipe110     117.7       117.6        pipelined 110ms cadence, 8.50 inf/s
#   pipe200     231.8       203.0        pipelined 200ms cadence, fresh result ~203ms
#   serial283   283.4       283.4        3-way serial chain, one in flight
#   fp32_555    555.0       555.0        fp32 numerically-valid path
#   cpu685      684.8       684.8        CPU-only monolith
cd /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain
run3 () {  # name lat cadence  -> 3 seeds concurrently
  for s in 0 2 4; do ./run_arm.sh "$1" "$2" "$3" $s & done
  wait
  echo "=== arm $1 complete ==="
}
run3 serial283 283.4 283.4
run3 pipe200   231.8 203.0
run3 pipe110   117.7 117.6
run3 cpu685    684.8 684.8
run3 fp32_555  555.0 555.0
echo "SWEEP COMPLETE"
