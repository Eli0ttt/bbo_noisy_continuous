#!/usr/bin/env python3
from __future__ import annotations
import collections,json,re,subprocess,sys
from pathlib import Path
def decode(p):
    r=subprocess.run(["zstd","-dc",str(p)],capture_output=True,text=True)
    if r.returncode: raise RuntimeError(r.stderr.strip() or "zstd failed")
    return r.stdout
def load(p):
    try:return json.loads(p.read_text(encoding="utf-8"))
    except Exception:return None
def rel(p,r):
    try:return str(p.relative_to(r))
    except Exception:return str(p)
def audit(job):
    trials=[p for p in job.iterdir() if p.is_dir()]
    if not trials:raise RuntimeError("trial directory not found")
    trial=trials[0]; agent=trial/"agent"; art=trial/"artifacts/logs/artifacts/bbo-split"; vers=art/"versions"
    history=load(agent/"history.json") or []; sel=load(agent/"final-selection.json") or {}
    hist={x.get("version"):x for x in history if isinstance(x,dict)}
    traces={}; totals=collections.Counter()
    if (art/"traces").exists():
      for p in sorted((art/"traces").glob("*.jsonl.zstd")):
        c=collections.Counter()
        for line in decode(p).splitlines():
          try:e=json.loads(line)
          except Exception:continue
          if e.get("type")=="tool/call":
            d=e.get("data") or {}; c[str(d.get("name","unknown"))]+=1
        key=p.name.removesuffix(".jsonl.zstd"); m=re.fullmatch(r"(v\d+)-([AB])([DR]?)",key)
        traces[key]={"version":m.group(1) if m else None,"role":m.group(2) if m else None,
          "phase":{"D":"delivery_recovery","R":"validation_repair"}.get(m.group(3),"research") if m else None,
          "call_count":sum(c.values()),"tool_counts":dict(c),"trace_file":rel(p,trial)}
        totals.update(c)
    rows=[]
    if vers.exists():
      def vn(p):
        m=re.fullmatch(r"v(\d+)",p.name);return int(m.group(1)) if m else 10**9
      for vd in sorted([p for p in vers.iterdir() if p.is_dir() and re.fullmatch(r"v\d+",p.name)],key=vn):
        h=hist.get(vd.name,{})
        row={k:h.get(k) for k in ("status","parent_version","parent_score","score","score_anytime","score_final",
                                  "score_delta_vs_parent","candidate_delivery","delivery_recovery",
                                  "validation_initial","role_validation","validation_final","repairs","repair")}
        row["version"]=vd.name
        fs={"A_candidate":vd/"A/candidate.py","A_handoff":vd/"A/handoff.json",
            "B_candidate":vd/"B/candidate.py","B_handoff":vd/"B/handoff.json",
            "merged_solver":vd/"merged/solver.py","merge_report":vd/"merge-report.json"}
        row["files"]={k:rel(p,trial) for k,p in fs.items() if p.exists()}
        mr=load(vd/"merge-report.json")
        if mr is not None:row["merge"]=mr
        for role in ("A","B"):
          rr={k:v for k,v in traces.items() if v["version"]==vd.name and v["role"]==role}
          if rr:row[role+"_traces"]=rr
        rows.append({k:v for k,v in row.items() if v is not None})
    out={"schema":3,"job":job.name,"build_id":sel.get("build_id"),"condition":sel.get("condition"),
      "research_time_limit_sec":sel.get("research_time_limit_sec"),"min_new_round_sec":sel.get("min_new_round_sec"),
      "stop_reason":sel.get("stop_reason"),
      "champion":{"version":sel.get("selected_version"),"visible_score":sel.get("selected_visible_score"),"sha256":sel.get("selected_sha256")},
      "latest_lineage_version":sel.get("latest_lineage_version"),"latest_scored_version":sel.get("latest_scored_version"),
      "latest_scored_score":sel.get("latest_scored_score"),"verifier_reward":load(trial/"verifier/reward.json"),
      "versions":rows,"tools":{"all_agents_by_tool":dict(totals),"trace_count":len(traces),"traces":traces}}
    agent.mkdir(parents=True,exist_ok=True); dst=agent/"audit.json"
    dst.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    # Cleanup only our redundant/intermediate files, after audit is complete.
    if vers.exists():
      for vd in vers.iterdir():
        if not vd.is_dir() or not re.fullmatch(r"v\d+",vd.name):continue
        for p in (vd/"parent.py",vd/"contract.json",vd/"A/role-validation.txt",vd/"A/role-validation-repair.txt",
                  vd/"B/role-validation.txt",vd/"B/role-validation-repair.txt"):
          if p.exists():p.unlink()
    for p in (agent/"history.json",agent/"final-selection.json",agent/"tool-audit-host.json"):
      if p.exists():p.unlink()
    for pat in ("*.stdout.log","*.stderr.log","*.error.log","*.selfcheck.error.log"):
      for p in agent.glob(pat):p.unlink()
    print("AUDIT_WRITTEN="+str(dst));print("VERSIONS="+str(len(rows)));print("TRACE_COUNT="+str(len(traces)))
    print("CHAMPION="+json.dumps(out["champion"],ensure_ascii=False));return 0
if __name__=="__main__":
    if len(sys.argv)!=2:raise SystemExit("usage: audit_split_job.py JOB_DIR")
    raise SystemExit(audit(Path(sys.argv[1])))
