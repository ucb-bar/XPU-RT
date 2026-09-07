import glob, re, json, collections
import numpy as np
out = {}
for f in glob.glob('schedules/scheduled_networks_octo_pareto_GW_p*_w*_cpsat_profiled.json'):
    m = re.search(r'GW_p(\d+)_w(\d+)_cpsat', f)
    if not m: continue
    key = f"{int(m.group(1))}_{int(m.group(2))}"
    try: d = json.load(open(f))
    except Exception: continue
    st = collections.defaultdict(lambda: 1e18); en = collections.defaultdict(lambda: -1e18)
    for v in d['dispatches'].values():
        j = v['job_name']; s = v['start_time']
        st[j] = min(st[j], s); en[j] = max(en[j], s + v['duration'])
    jobs = sorted(st, key=lambda x: int(re.sub(r'\D', '', x) or 0))
    span = [en[j] - st[j] for j in jobs]
    starts = sorted(st[j] for j in jobs)
    out[key] = dict(lat_med=float(np.median(span)), lat_max=float(max(span)),
                    cadence=float(np.median(np.diff(starts))) if len(starts) > 1 else float('nan'),
                    makespan=d['metadata']['makespan'])
print(json.dumps(out))
