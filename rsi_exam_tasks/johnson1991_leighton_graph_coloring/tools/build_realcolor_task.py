#!/usr/bin/env python3
"""Rebuild johnson1991_leighton_graph_coloring on REAL DIMACS / COLOR02 instances
(revision realcolor_v5).

The old revision used a PRIVATE synthetic generator with a self-produced "best-known" anchor,
which a strong agent could beat (the anchor was our own solver -> circular). This revision fixes
BOTH provenance and the anchor:

  * DATA: real, citable DIMACS graph-coloring instances (Leighton le450_*, Johnson DSJC*,
    DSJR*), downloaded as .col and committed compact. Multi-scale n = 250 / 450 / 500 / 1000,
    three families, all HARD (a reference TabuCol cannot reach a proper coloring in the cap).
  * target_k = the PUBLISHED best-known chromatic number. A proper (0-conflict) target_k-coloring
    provably EXISTS (le450 chi is proven; DSJC/DSJR values are published best-known colorings),
    but finding it is at the coloring frontier.
  * sota (reward 1) = 0 conflicts. EXTERNAL and unbeatable: 0 is the proven-achievable target at
    target_k, not a number our solver produced. No agent can beat 0.
  * floor (reward 0) = the conflicts a single reference TabuCol (gc_lib.reference_tabucol, the
    SHIPPED baseline) reaches at target_k within the per-instance cap. Matching it earns ~0.

Metric: monochromatic edges at target_k, lower better; per-instance s = clip((floor - conf)/
(floor - 0), 0, 1) = clip(1 - conf/floor); aggregate = mean.

Run: python tools/build_realcolor_task.py
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK_ROOT / "environment" / "methods" / "main"))
import gc_lib as L  # the SHIPPED reference method — floor is measured with the exact baseline

CACHE = Path(__file__).resolve().parent / "col_cache"
BASE_URL = "https://raw.githubusercontent.com/dynaroars/npbench/master/instances/coloring/graph_color/{}.col"

# Published best-known chromatic numbers (Porumbel / COLOR02 table). le450 & DSJR500.1c are
# PROVEN optimal; DSJC values are published best-known upper bounds (honestly labelled).
BEST_KNOWN = {
    "le450_15c": 15, "le450_15d": 15, "le450_25c": 25, "le450_25d": 25,
    "DSJC250.5": 28, "DSJC500.1": 12, "DSJC500.5": 48, "DSJC500.9": 126,
    "DSJC1000.1": 20, "DSJC1000.5": 85, "DSJC1000.9": 223,
    "DSJR500.1c": 84, "DSJR500.5": 122,
}
PROVEN = {"le450_15c", "le450_15d", "le450_25c", "le450_25d", "DSJR500.1c"}

# same-distribution split: dev and test each span the scales/families, DISJOINT instances.
# (DSJC250.5 was dropped: reference TabuCol nearly solves it (floor 2) -> saturates, weak discriminator.)
DEV = ["le450_15c", "le450_25c", "DSJC500.5", "DSJC1000.5", "DSJR500.5"]
TEST = ["le450_15d", "le450_25d", "DSJC500.1", "DSJC500.9",
        "DSJC1000.1", "DSJC1000.9", "DSJR500.1c"]

# floor = the shipped reference TabuCol run for a FIXED number of iterations (machine-independent:
# same iteration count -> same conflicts on any hardware). ref_max_iters is calibrated per instance
# (tools/iters_measure.json = iters the reference does in ~35s here) and shipped in the data so the
# baseline reproduces the floor exactly. ITER_MARGIN keeps M below the measured count so the
# baseline finishes well under the 45s cap even on a slower machine.
FLOOR_SEED = 20240617
FLOOR_WALL_BACKSTOP = 300.0          # generous offline backstop; the iteration cap ends the run first
ITER_MARGIN = 0.85
ITERS_JSON = Path(__file__).resolve().parent / "iters_measure.json"


def fetch(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{name}.col"
    if p.exists() and p.stat().st_size > 0:
        return p
    import urllib.request
    with urllib.request.urlopen(BASE_URL.format(name), timeout=120) as r:
        p.write_bytes(r.read())
    return p


def parse_col(name: str) -> dict:
    """Parse a DIMACS .col ('p edge N M' then 'e u v', 1-indexed) -> 0-indexed instance."""
    n = 0
    edges = []
    for line in fetch(name).read_text().splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "p":
            n = int(t[2])
        elif t[0] == "e":
            u, v = int(t[1]) - 1, int(t[2]) - 1
            if u != v:
                edges.append((min(u, v), max(u, v)))
    edges = sorted(set(edges))
    return {"name": name, "n_vertices": n, "n_edges": len(edges), "edges": edges}


def permute_anon(real_name: str, anon: str, inst: dict) -> dict:
    """Relabel vertices by a deterministic permutation (seeded by the real name) and give
    the instance an anonymized name, so the agent cannot fingerprint the graph by its DIMACS
    labels/name and fetch a published proper coloring. chi (hence target_k and sota=0) is
    invariant under relabeling; conflicts are invariant too, so the floor is unchanged."""
    n = inst["n_vertices"]
    rng = random.Random(int(hashlib.sha256(real_name.encode()).hexdigest(), 16) % (2 ** 32))
    perm = list(range(n))
    rng.shuffle(perm)
    edges = sorted({(min(perm[u], perm[v]), max(perm[u], perm[v])) for u, v in inst["edges"]})
    return {"name": anon, "real_name": real_name, "n_vertices": n,
            "n_edges": len(edges), "edges": edges}


def anchor_worker(args) -> dict:
    anon, inst, k, max_iters = args
    t = time.perf_counter()
    adj = L.build_adj(inst["n_vertices"], inst["edges"])
    _col, floor = L.reference_tabucol(inst["n_vertices"], adj, k, FLOOR_SEED,
                                      FLOOR_WALL_BACKSTOP, max_iters=max_iters)
    return {"name": anon, "floor": int(floor), "max_iters": max_iters,
            "secs": round(time.perf_counter() - t, 1)}


def public_inst(inst: dict, k: int, max_iters: int) -> dict:
    return {"name": inst["name"], "n_vertices": inst["n_vertices"],
            "n_edges": inst["n_edges"], "target_k": k, "ref_max_iters": max_iters,
            "edges": [[u, v] for u, v in inst["edges"]]}


def anchor_doc(split: str, recs: list) -> dict:
    floor_per = [{"name": r["name"], "conflicts": r["floor"]} for r in recs]
    best_per = [{"name": r["name"], "conflicts": 0} for r in recs]
    return {
        "task_id": "johnson1991_leighton_graph_coloring",
        "revision": "realcolor_v5",
        "data_split": split,
        "score_direction": "min",
        "per_instance_cap_secs": 45.0,
        "score_formula": "per-instance s_j = clip((floor_conf_j - conflicts_j)/(floor_conf_j - 0),0,1) = clip(1 - conf/floor). metric = monochromatic edges at target_k (= best-known chi). floor = reference TabuCol at the cap, best = 0 (a proper coloring provably exists at target_k).",
        "kpi_definition": "per-instance KPI = conflicts (monochromatic edges at target_k); reported kpi = total conflicts over the set",
        "floor_anchor": {"name": "reference_tabucol_fixed_iters",
                         "description": "Conflicts from a single reference TabuCol (the shipped baseline gc_lib.reference_tabucol) at target_k, run for a FIXED per-instance iteration budget (data field ref_max_iters, calibrated to ~30-35s and safely under the 45s cap). Machine-independent. Reward 0 = matching this standard local search.",
                         "per_instance": floor_per, "total_conflicts": sum(x["conflicts"] for x in floor_per)},
        "sota_anchor": {"name": "proper_coloring_zero_conflicts",
                        "description": "0 conflicts: a proper target_k-coloring provably exists (target_k = published best-known chromatic number). EXTERNAL, unbeatable target — reward 1.",
                        "per_instance": best_per, "total_conflicts": 0},
        "scoring_floor": "floor_anchor",
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    # anonymized names: dev -> pub0.., test -> g0.. ; real names sealed in the summary only.
    anon = {r: f"pub{i}" for i, r in enumerate(DEV)}
    anon.update({r: f"g{i}" for i, r in enumerate(TEST)})
    names = DEV + TEST
    measured = json.loads(ITERS_JSON.read_text())        # {real_name: {"iters": N, ...}}
    M = {r: max(1, int(ITER_MARGIN * measured[r]["iters"])) for r in names}
    print(f"parsing {len(names)} DIMACS instances (dev {len(DEV)} / test {len(TEST)}), "
          f"permuting + anonymizing; floor by fixed iters (margin {ITER_MARGIN})", flush=True)
    insts = {r: permute_anon(r, anon[r], parse_col(r)) for r in names}
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = {r["name"]: r for r in ex.map(
            anchor_worker, [(anon[r], insts[r], BEST_KNOWN[r], M[r]) for r in names])}
    for r in names:
        a = anon[r]; rec = res[a]; k = BEST_KNOWN[r]
        tag = "PROVEN" if r in PROVEN else "best-known"
        print(f"  {a:4s} <- {r:12s} n={insts[r]['n_vertices']:4d} m={insts[r]['n_edges']:6d} "
              f"k={k:3d}({tag}) iters={M[r]:6d} floor={rec['floor']:3d} ({rec['secs']}s)", flush=True)
        assert rec["floor"] > 0, f"{r}: floor==0 (degenerate; reference already found a proper coloring)"

    dev_recs = [res[anon[r]] for r in DEV]
    test_recs = [res[anon[r]] for r in TEST]

    (TASK_ROOT / "environment/data/visible.json").write_text(json.dumps(
        {"split": "public", "generator": "realcolor_v5",
         "instances": [public_inst(insts[r], BEST_KNOWN[r], M[r]) for r in DEV]}))
    (TASK_ROOT / "tests/heldout/hidden_data.json").write_text(json.dumps(
        {"split": "hidden", "generator": "realcolor_v5",
         "instances": [public_inst(insts[r], BEST_KNOWN[r], M[r]) for r in TEST]}))
    write_json(TASK_ROOT / "environment/data/baseline_public.json", anchor_doc("public", dev_recs))
    write_json(TASK_ROOT / "tests/heldout/hidden_anchors.json", anchor_doc("hidden", test_recs))

    summary = {"revision": "realcolor_v5", "iter_margin": ITER_MARGIN,
               "dev": [{"anon": anon[r], "real": r, "k": BEST_KNOWN[r], "ref_max_iters": M[r],
                        "floor": res[anon[r]]["floor"], "proven": r in PROVEN} for r in DEV],
               "test": [{"anon": anon[r], "real": r, "k": BEST_KNOWN[r], "ref_max_iters": M[r],
                         "floor": res[anon[r]]["floor"], "proven": r in PROVEN} for r in TEST],
               "test_floor_total": sum(res[anon[r]]["floor"] for r in TEST),
               "test_sota_total": 0}
    write_json(Path(__file__).resolve().parent / "realcolor_v5_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nTASK.TOML [scoring] -> floor_total={summary['test_floor_total']} sota_total=0", flush=True)


if __name__ == "__main__":
    main()
