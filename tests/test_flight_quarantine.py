import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import flight_quarantine as FQ  # noqa: E402

COLS = ["seed", "outcome", "steps", "gates_passed", *FQ.FIELDS]


def _write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); w.writerows(rows)


def _row(seed, cruise, outcome="crash", steps="700", trace="/abs/ctrl_traces/xpu_x.csv"):
    return dict(seed=seed, outcome=outcome, steps=steps, gates_passed="1", ctrl_trace=trace, percep_latency_ms="36.0",
                percep_hold_ms="0", course="a", prop_density="0.20", person_h="2.4", walk_speed="0.0",
                cruise_speed=cruise, moment_scale="0.0055")


@pytest.fixture
def camp(tmp_path, monkeypatch):
    res = tmp_path / "res"; (res / "campaign_t").mkdir(parents=True)
    monkeypatch.setattr(FQ, "RES", str(res)); monkeypatch.setattr(FQ, "_ENTRIES", None)
    monkeypatch.setattr(FQ, "REGISTRY", str(res / "flight_quarantine.csv"))
    p = res / "campaign_t" / "campaign.csv"
    frozen = [_row(1000 + i, "1.2", "timeout", "1800") for i in range(3)]
    good = [_row(1000 + i, "1.4") for i in range(3)]
    _write(p, frozen + good)
    e = dict(csv="campaign_t/campaign.csv", **{f: frozen[0][f] for f in FQ.FIELDS}, expected_n=3, reason="t",
             evidence_log="x.log", evidence_line=1)
    e["ctrl_trace"] = "xpu_x.csv"
    with open(res / "flight_quarantine.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(e)); w.writeheader(); w.writerow(e)
    return p


def test_drops_only_the_listed_batch(camp):
    rows = FQ.flight_rows(str(camp))
    assert len(rows) == 3 and {r["cruise_speed"] for r in rows} == {"1.4"}


def test_number_formatting_does_not_hide_a_batch(camp):
    rows = list(csv.DictReader(open(camp)))
    for r in rows:
        r["prop_density"] = "0.2"; r["percep_hold_ms"] = "0.0"
    _write(camp, rows)
    assert len(FQ.flight_rows(str(camp))) == 3


def test_refuses_when_the_key_matches_a_different_count(camp):
    rows = list(csv.DictReader(open(camp))) + [_row(1003, "1.2")]
    _write(camp, rows)
    with pytest.raises(SystemExit, match="re-flown"):
        FQ.flight_rows(str(camp))


def test_unlisted_csv_passes_through(tmp_path, camp):
    other = tmp_path / "res" / "campaign_t" / "other.csv"
    _write(other, [_row(1000, "1.2", "timeout", "1800")])
    assert len(FQ.flight_rows(str(other))) == 1
