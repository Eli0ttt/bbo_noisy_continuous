"""Parallel split-ownership collaboration for RSI BBO research."""
from __future__ import annotations
import asyncio, hashlib, json, math, re, shlex, time
from typing import Any
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

class BboSplitCollabAgent(BaseAgent):
    BUILD_ID="split-20260929-r8-diagnostics-repair-cleanup"
    ROOT="/logs/artifacts/bbo-split"
    WORK="/app/methods/main/.collab"
    SOLVER="/app/methods/main/solver.py"
    SESSION_ROOT="/tmp/bbo-split-sessions"
    NODE="/opt/node/bin/node"
    CLI="/opt/deepseek-harness/apps/cli/src/bin.ts"
    CFG="/opt/dsh-config"

    def __init__(self,*args:Any,condition="no-cordis",**kwargs:Any):
        if condition not in ("no-cordis","dynamic-cordis"): raise ValueError(condition)
        super().__init__(*args,**kwargs)
        self.condition=condition
        self.traces={}; self.session_error={}; self.session_output={}

    @staticmethod
    def name(): return "bbo-split-collab"
    def version(self): return "1.0.0"

    def _get_required(self,name):
        v=self._get_env(name)
        if not v: raise RuntimeError(f"missing {name}")
        return v

    def _env_vars(self,role):
        env={"DSH_HOME":f"{self.SESSION_ROOT}/{role}","DSH_TELEMETRY_DISABLED":"1",
             "DSH_PERMISSION_MODE":"danger-full-access","TSX_TSCONFIG_PATH":"/opt/deepseek-harness/tsconfig.json",
             "NO_COLOR":"1","SPLIT_ROLE":role}
        if role=="A":
            env["DEEPSEEK_API_KEY"]=self._get_required("DEEPSEEK_API_KEY")
            env["DEEPSEEK_BASE_URL"]=self._get_required("DEEPSEEK_BASE_URL")
            env["DEEPSEEK_MODEL"]=self._get_env("DEEPSEEK_MODEL") or self._get_env("MODEL") or "deepseek-v4-flash-s1"
        else:
            env["AGENT_B_API_KEY"]=self._get_required("AGENT_B_API_KEY")
            env["AGENT_B_BASE_URL"]=self._get_required("AGENT_B_BASE_URL")
            env["AGENT_B_MODEL"]=self._get_required("AGENT_B_MODEL")
        return env

    def _command(self,role,prompt):
        patches=[f"{self.CFG}/bbo-no-cordis.yml"]
        if self.condition=="dynamic-cordis": patches.append(f"{self.CFG}/bbo-cordis-extra.yml")
        patches += [f"{self.CFG}/bbo-split-guard.yml",f"{self.CFG}/bbo-split-agent-{role.lower()}.yml"]
        resolver="console.log(require.resolve('tsx/esm', { paths: ['/opt/deepseek-harness'] }))"
        args=["--profile","headless"]
        for p in patches: args += ["--patch",p]
        args.append(prompt)
        return ("set -eu\nL=$("+shlex.quote(self.NODE)+" -e "+shlex.quote(resolver)+")\nexec "+
                shlex.quote(self.NODE)+' --import "$L" '+shlex.quote(self.CLI)+" "+" ".join(shlex.quote(x) for x in args))

    async def _exec(self,env,cmd,timeout=60,cwd="/app",variables=None):
        return await env.exec(cmd,cwd=cwd,env=variables or {},timeout_sec=timeout)
    async def _must(self,env,cmd,timeout=60):
        r=await self._exec(env,cmd,timeout)
        if r.return_code: raise RuntimeError(f"exit={r.return_code} cmd={cmd!r} stdout={(r.stdout or '')[-800:]!r} stderr={(r.stderr or '')[-800:]!r}")
        return r.stdout or ""
    def _log(self,name,text):
        self.logs_dir.mkdir(parents=True,exist_ok=True); (self.logs_dir/name).write_text(text,encoding="utf-8")
    async def _hash(self,env,path): return (await self._must(env,"sha256sum "+shlex.quote(path))).split()[0]

    async def _trace_list(self,env,role):
        r=await self._exec(env,"find "+shlex.quote(f"{self.SESSION_ROOT}/{role}/sessions")+" -type f -name session.jsonl.zstd -print 2>/dev/null | sort",20)
        return set((r.stdout or "").splitlines()) if r.return_code==0 else set()

    async def _session(self,env,role,prompt,cwd,limit,tag):
        before=await self._trace_list(env,role)
        ok=False
        try:
            r=await self._exec(env,self._command(role,prompt),timeout=limit,cwd=cwd,variables=self._env_vars(role))
            self.session_output[tag]=r.stdout or ""
            if r.return_code: self.session_error[tag]=f"exit_code={r.return_code}: {(r.stderr or r.stdout or '')[-1000:]}"
            ok=r.return_code==0
        except Exception as exc:
            self.session_error[tag]=f"{type(exc).__name__}: {exc}"; self._log(tag+".error.log",self.session_error[tag]+"\n")
        finally:
            try:
                after=await self._trace_list(env,role); fresh=sorted(after-before); chosen=fresh[-1] if fresh else None
                self.traces[tag]=chosen
                if chosen:
                    dst=f"{self.ROOT}/traces/{tag}.jsonl.zstd"
                    await self._must(env,"cp "+shlex.quote(chosen)+" "+shlex.quote(dst)+"; chmod 644 "+shlex.quote(dst),30)
            except Exception as exc:
                self.session_error[tag]=(self.session_error.get(tag,"")+f"; trace_collection:{type(exc).__name__}").strip("; ")
        return ok

    async def _selfcheck(self,env,path,tag,timeout):
        try: r=await self._exec(env,"python3 /app/selfcheck.py --solver "+shlex.quote(path)+" --json",timeout)
        except Exception as exc: self._log(tag+".selfcheck.error.log",repr(exc)); return None
        self._log(tag+".selfcheck.stdout.log",r.stdout or "")
        if r.stderr: self._log(tag+".selfcheck.stderr.log",r.stderr)
        if r.return_code: return None
        for line in reversed((r.stdout or "").splitlines()):
            try: obj=json.loads(line)
            except Exception: continue
            if isinstance(obj,dict) and obj.get("metric")=="oracle_normalized_auc70_final30" and isinstance(obj.get("score"),(int,float)): return obj
        return None

    async def setup(self,environment:BaseEnvironment):
        checks=[
          "test -f /opt/dsh-config/bbo-no-cordis.yml",
          "test -f /opt/dsh-config/bbo-split-agent-a.yml",
          "test -f /opt/dsh-config/bbo-split-agent-b.yml",
          "test -f /opt/dsh-config/bbo-split-guard.yml",
          "test -f /opt/dsh-config/bbo-split-merge.py",
          "test -f /opt/dsh-config/bbo_candidate_preflight.py",
          "test -f /app/selfcheck.py",
          "test -f "+shlex.quote(self.SOLVER),
        ]
        for c in checks:
            r=await self._exec(environment,c,15)
            if r.return_code: raise RuntimeError("setup failed: "+c)
        await self._must(environment,"rm -rf "+shlex.quote(self.SESSION_ROOT)+" "+shlex.quote(self.WORK)+"; mkdir -p "+
                         shlex.quote(self.WORK+"/versions")+" "+shlex.quote(self.ROOT+"/traces")+" "+
                         shlex.quote(self.SESSION_ROOT+"/A")+" "+shlex.quote(self.SESSION_ROOT+"/B"),30)

    async def _role_validate(self,env,parent,candidate,role,contract,out):
        cmd=(f"python3 /opt/dsh-config/bbo-split-merge.py validate "
             f"--parent {shlex.quote(parent)} --candidate {shlex.quote(candidate)} "
             f"--role {role} --contract {shlex.quote(contract)}")
        r=await self._exec(env,cmd,60)
        text=(r.stdout or "")+(r.stderr or "")
        await self._exec(env,f"printf %s {shlex.quote(text)} > {shlex.quote(out)}",10)
        return r.return_code==0,text[-4000:]

    async def run(self,instruction,environment,context:AgentContext):
        if not isinstance(instruction,str) or not instruction.strip(): raise RuntimeError("missing task instruction")
        env=environment
        total=int(self._get_env("COLLAB_TOTAL_SEC") or "1800")
        total=min(total,43200)
        # This is only a "do not start a new round" threshold. It does not cap
        # A/B once a round has started. It prevents spending the final minutes
        # on a round that cannot reach merge/preflight/selfcheck.
        min_start=int(self._get_env("COLLAB_MIN_NEW_ROUND_SEC") or "600")
        deadline=time.monotonic()+total
        task_hash=hashlib.sha256(instruction.encode()).hexdigest()
        self._log("build.json",json.dumps({"build_id":self.BUILD_ID,"condition":self.condition,"total_sec":total},indent=2)+"\n")

        v0=f"{self.WORK}/versions/v0.py"
        await self._must(env,"cp "+shlex.quote(self.SOLVER)+" "+shlex.quote(v0),30)
        remain=max(0,math.ceil(deadline-time.monotonic()))
        if remain<1: raise RuntimeError("deadline before baseline selfcheck")
        base_metric=await self._selfcheck(env,v0,"v0",remain)
        if base_metric is None: raise RuntimeError("baseline selfcheck failed")

        lineage=v0; lineage_version="v0"; lineage_score=float(base_metric["score"])
        champion=v0; champion_version="v0"; champion_score=lineage_score
        latest_scored=v0; latest_scored_version="v0"; latest_scored_score=lineage_score
        history=[{"version":"v0","status":"baseline_scored","score":lineage_score,
                  "score_anytime":base_metric.get("score_anytime"),"score_final":base_metric.get("score_final")}]
        self._log("history.json",json.dumps(history,ensure_ascii=False,indent=2)+"\n")
        n=1; stop="time_limit"

        try:
            while time.monotonic()<deadline:
                remain=max(0,math.ceil(deadline-time.monotonic()))
                if remain < min_start:
                    stop="insufficient_time_for_new_round"
                    break
                tag=f"v{n}"; d=f"{self.WORK}/versions/{tag}"
                parent_version=lineage_version
                parent_score=lineage_score
                await self._must(env,"mkdir -p "+shlex.quote(d+"/A")+" "+shlex.quote(d+"/B")+" "+shlex.quote(d+"/merged")+
                                 "; cp "+shlex.quote(lineage)+" "+shlex.quote(d+"/parent.py")+
                                 "; cp "+shlex.quote(lineage)+" "+shlex.quote(d+"/A/candidate.py")+
                                 "; cp "+shlex.quote(lineage)+" "+shlex.quote(d+"/B/candidate.py")+
                                 "; printf '%s\n' '{\"summary\":\"\",\"peer_requests\":[],\"assumptions\":[]}' > "+shlex.quote(d+"/A/handoff.json")+
                                 "; printf '%s\n' '{\"summary\":\"\",\"peer_requests\":[],\"assumptions\":[]}' > "+shlex.quote(d+"/B/handoff.json")+
                                 "; chmod u+rw "+shlex.quote(d+"/A/candidate.py")+" "+shlex.quote(d+"/A/handoff.json")+" "+
                                                   shlex.quote(d+"/B/candidate.py")+" "+shlex.quote(d+"/B/handoff.json")+
                                 "; test -w "+shlex.quote(d+"/A/candidate.py")+" && test -w "+shlex.quote(d+"/B/candidate.py"),30)

                cr=await self._exec(env,"python3 /opt/dsh-config/bbo-split-merge.py contract --parent "+shlex.quote(d+"/parent.py")+" --out "+shlex.quote(d+"/contract.json"),30)
                if cr.return_code:
                    history.append({"version":tag,"stage":"contract_failed","error":(cr.stderr or cr.stdout or "")[-1200:]}); break

                row={"version":tag,"status":"parallel_agents_running","parent_version":parent_version,
                     "parent_score":parent_score,"started_with_remaining_sec":remain}
                history.append(row)
                self._log("history.json",json.dumps(history,ensure_ascii=False,indent=2)+"\n")

                recent=json.dumps(history[-4:],ensure_ascii=False,indent=2)
                prior_handoffs={"A":None,"B":None}
                if n>1:
                    for rr in ("A","B"):
                        hp=f"{self.WORK}/versions/v{n-1}/{rr}/handoff.json"
                        hr=await self._exec(env,"cat "+shlex.quote(hp),10)
                        if hr.return_code==0 and (hr.stdout or "").strip():
                            prior_handoffs[rr]=(hr.stdout or "")[-5000:]
                prior_handoffs_text=json.dumps(prior_handoffs,ensure_ascii=False,indent=2)
                scored=[x for x in history[:-1] if isinstance(x.get("score"),(int,float))]
                prev_scored=scored[-1] if scored else None
                best_scored=max(scored,key=lambda x:x["score"]) if scored else None
                prev_merge={}
                if n>1:
                    mrp=f"{self.WORK}/versions/v{n-1}/merge-report.json"
                    mrr=await self._exec(env,"cat "+shlex.quote(mrp),10)
                    if mrr.return_code==0 and (mrr.stdout or "").strip():
                        try: prev_merge=json.loads(mrr.stdout)
                        except Exception: prev_merge={}
                scientific_feedback={
                    "parent":{
                        "version":parent_version,"score":parent_score,
                        "score_anytime":prev_scored.get("score_anytime") if prev_scored else None,
                        "score_final":prev_scored.get("score_final") if prev_scored else None,
                        "delta_vs_its_parent":prev_scored.get("score_delta_vs_parent") if prev_scored else None},
                    "champion":{
                        "version":best_scored.get("version") if best_scored else "v0",
                        "score":best_scored.get("score") if best_scored else 0.0,
                        "score_anytime":best_scored.get("score_anytime") if best_scored else 0.0,
                        "score_final":best_scored.get("score_final") if best_scored else 0.0},
                    "previous_merge":{
                        "A_changed":prev_merge.get("A_changed",[]),
                        "B_changed":prev_merge.get("B_changed",[]),
                        "batch_changed":prev_merge.get("batch_changed")},
                    "interpretation_rule":"Treat score decomposition as noisy experimental evidence, not proof of causality. Prefer one coherent change so later deltas are interpretable."}
                scientific_feedback_text=json.dumps(scientific_feedback,ensure_ascii=False,indent=2)
                shared=f"""You are one of two parallel implementation agents in a controlled research experiment on noisy continuous BBO.
Version: {tag}. Both agents receive the exact same parent.py and contract.json. The coordinator will merge only disjoint AST-owned symbols and will reject unauthorized edits.
The final optimizer must obey the official task exactly: 120 objective queries per run, self-contained solver.py, NumPy/stdlib only, deterministic supplied rng, finite in-bounds ask output, noisy loss is minimized. You must not run selfcheck or inspect hidden/trusted evaluator assets.
Official task instruction:
---BEGIN OFFICIAL TASK---
{instruction}
---END OFFICIAL TASK---
Recent same-run outcomes:
{recent}
Previous round handoffs (requests are advisory; only use interfaces already present in this round's parent.py):
{prior_handoffs_text}
Structured experimental feedback:
{scientific_feedback_text}
Use score_anytime vs score_final to identify trade-offs, but do not overfit one noisy visible measurement. Make one coherent, testable change and explain its hypothesis briefly in handoff.json.
Your process working directory is your own role directory. `candidate.py` and `handoff.json` there are writable.
Prefer editing those RELATIVE filenames. Do not create a substitute candidate under /tmp; the coordinator only merges the official candidate.py.
Read {d}/parent.py and {d}/contract.json before editing. Your candidate.py is already an exact copy of parent.py, so a stalled/no-op session still leaves a valid artifact. Make at most one coherent algorithmic change in your owned surface. Preserve peer-visible interfaces. Run py_compile and a tiny synthetic ask/tell smoke check only; then stop. Do not edit /app/methods/main/ or the peer directory.
"""
                pa=shared+f"""
ROLE A — state/update owner.
Edit only {d}/A/candidate.py and optionally {d}/A/handoff.json.
Your owned surface is exactly contract.json ownership.A: __init__, tell, A-exclusive existing helpers, new _a_* helpers, and imports.
Every NEW module-level helper function/class must start with `_a_` (e.g. `_a_phi`, `_a_EliteArchive`).
Existing `_a_*` symbols remain permanently A-owned across rounds, even when ask calls them.
Never modify ask, class attribute batch, self.batch, or any `_b_*` symbol. You do NOT own query batch semantics. If the parent contains an old `self.batch = ...` assignment in an A-owned method, remove that assignment so B's class attribute controls the effective batch.
Primary responsibility: optimizer state, learning/update dynamics, noise handling, adaptation, and making state useful to the proposal policy.
If B will need a new state field, add it now but do not modify B's code. Record a short peer_requests item in handoff.json; B can consume it next round after this merged version becomes the parent.
Do not rewrite the whole solver. First inspect the parent/contract, then edit the pre-created candidate, validate, and stop."""
                pb=shared+f"""
ROLE B — proposal/query-schedule owner.
Edit only {d}/B/candidate.py and optionally {d}/B/handoff.json.
Your owned surface is exactly contract.json ownership.B: ask, B-exclusive existing helpers, new _b_* helpers, and class attribute batch.
Every NEW module-level helper function/class must start with `_b_`.
Existing `_b_*` symbols remain permanently B-owned across rounds.
You may call/read `_a_*` APIs from ask, but never modify them. Never modify __init__, tell, or `_a_*` symbols. You do NOT own __init__ or tell and you may not change imports.
Primary responsibility: candidate generation, exploration/exploitation schedule, within-batch ordering, boundary handling, and query allocation.
You exclusively control effective batch size through the class attribute `batch`; do not rely on A to set `self.batch`.
Use only state already present in parent.py. If you want A to expose new state, record it in handoff.json for the next round rather than inventing a same-round dependency. New B-local state must initialize safely inside B-owned methods.
Do not rewrite the whole solver. First inspect the parent/contract, then edit the pre-created candidate, validate, and stop."""

                started=time.monotonic()
                limit=max(1,math.ceil(deadline-time.monotonic()))
                ra,rb=await asyncio.gather(
                    self._session(env,"A",pa,d+"/A",limit,tag+"-A"),
                    self._session(env,"B",pb,d+"/B",limit,tag+"-B"),
                )

                delivery_recovery={}
                ph0=await self._hash(env,d+"/parent.py")
                for role,ok,cwd in [("A",ra,d+"/A"),("B",rb,d+"/B")]:
                    ch=await self._hash(env,cwd+"/candidate.py")
                    if ok or ch!=ph0:
                        continue
                    rremain=max(0,math.ceil(deadline-time.monotonic()))
                    if rremain < 120:
                        delivery_recovery[role]={"attempted":False,"reason":"insufficient_global_time"}
                        continue
                    base_tag=tag+"-"+role
                    partial=(self.session_output.get(base_tag) or "")[-3500:]
                    recovery_prompt=f"""Your previous session exited nonzero before changing the official candidate.py.
This is a delivery recovery, not a new research round and not peer review.
Finish and commit only the coherent change you had already started. Do not broaden or redesign it.
Your previous partial output:
---BEGIN PARTIAL OUTPUT---
{partial}
---END PARTIAL OUTPUT---
Edit the writable relative candidate.py (and handoff.json if useful), run py_compile and a tiny synthetic smoke test, then stop."""
                    rtag=tag+"-"+role+"D"
                    rok=await self._session(env,role,recovery_prompt,cwd,rremain,rtag)
                    ch2=await self._hash(env,cwd+"/candidate.py")
                    delivery_recovery[role]={"attempted":True,"session_ok":rok,
                                             "candidate_changed":ch2!=ph0,
                                             "trace":self.traces.get(rtag),
                                             "error":self.session_error.get(rtag)}
                    if role=="A": ra = ra or (rok and ch2!=ph0)
                    else: rb = rb or (rok and ch2!=ph0)
                if delivery_recovery:
                    row["delivery_recovery"]=delivery_recovery
                    self._log("history.json",json.dumps(history,ensure_ascii=False,indent=2)+"\n")

                va,ea=await self._role_validate(env,d+"/parent.py",d+"/A/candidate.py","A",d+"/contract.json",d+"/A/role-validation.txt")
                vb,eb=await self._role_validate(env,d+"/parent.py",d+"/B/candidate.py","B",d+"/contract.json",d+"/B/role-validation.txt")
                # Persist validator diagnostics in history BEFORE cleanup removes
                # the one-line/intermediate validation text files.
                validation_initial={"A":{"pass":va,"error":None if va else ea},
                                    "B":{"pass":vb,"error":None if vb else eb}}
                row["validation_initial"]=validation_initial
                self._log("history.json",json.dumps(history,ensure_ascii=False,indent=2)+"\n")
                repairs={}
                for role,ok,err,cwd in [("A",va,ea,d+"/A"),("B",vb,eb,d+"/B")]:
                    if ok: continue
                    rremain=max(0,math.ceil(deadline-time.monotonic()))
                    if rremain < 90:
                        repairs[role]={"attempted":False,"reason":"insufficient_global_time","error":err}
                        continue
                    repair_prompt=f"""Your candidate violates the mechanical ownership contract.
This is a STRICT MECHANICAL REPAIR, not a research step.
1. Preserve all intended changes on symbols owned by role {role}.
2. For every peer-owned or frozen AST node named by the validator, restore that node from parent.py exactly (including docstrings/decorators/defaults/body). Do not merely make it semantically equivalent.
3. Do not introduce any new algorithmic idea, do not review the peer, and do not change additional owned code unless required to compile after restoration.
4. If the violation is an accidental edit to a peer-owned method such as ask/tell, copy that complete method definition from parent.py over candidate.py.
Parent: {d}/parent.py
Contract: {d}/contract.json
Candidate: {cwd}/candidate.py
Validator output:
{err}
After the minimal restoration, run py_compile and the role validator if time permits, then stop."""
                    repair_tag=tag+"-"+role+"R"
                    rok=await self._session(env,role,repair_prompt,cwd,rremain,repair_tag)
                    vok,verr=await self._role_validate(env,d+"/parent.py",cwd+"/candidate.py",role,d+"/contract.json",cwd+"/role-validation-repair.txt")
                    repairs[role]={"attempted":True,"session_ok":rok,"valid_after":vok,"error_after":verr,
                                   "trace":self.traces.get(repair_tag)}
                    if role=="A": va=vok
                    else: vb=vok
                row.update({"role_validation":{"A":va,"B":vb},
                            "validation_final":{
                                "A":{"pass":va,"error":None if va else (
                                    repairs.get("A",{}).get("error_after") or ea)},
                                "B":{"pass":vb,"error":None if vb else (
                                    repairs.get("B",{}).get("error_after") or eb)}},
                            "repairs":repairs})
                self._log("history.json",json.dumps(history,ensure_ascii=False,indent=2)+"\n")
                if not (va and vb):
                    row.update({"status":"role_validation_failed"})
                    self._log("history.json",json.dumps(history,ensure_ascii=False,indent=2)+"\n")
                    n+=1
                    continue

                try:
                    ah=await self._hash(env,d+"/A/candidate.py")
                    bh=await self._hash(env,d+"/B/candidate.py")
                    ph=await self._hash(env,d+"/parent.py")
                    row["candidate_delivery"]={"parent_sha256":ph,"A_sha256":ah,"B_sha256":bh,
                                               "A_changed":ah!=ph,"B_changed":bh!=ph}
                    self._log("history.json",json.dumps(history,ensure_ascii=False,indent=2)+"\n")
                except Exception as exc:
                    row["candidate_delivery_error"]=f"{type(exc).__name__}: {exc}"

                merge_remain=max(0,math.ceil(deadline-time.monotonic()))
                if merge_remain<1:
                    row.update({"status":"parallel_agents_finished_no_merge_time",
                                "A_session_ok":ra,"B_session_ok":rb,
                                "A_error":self.session_error.get(tag+"-A"),
                                "B_error":self.session_error.get(tag+"-B"),
                                "A_trace":self.traces.get(tag+"-A"),"B_trace":self.traces.get(tag+"-B")})
                    self._log("history.json",json.dumps(history,ensure_ascii=False,indent=2)+"\n")
                    stop="time_limit_after_parallel_agents"; break
                mr=await self._exec(env,
                    "python3 /opt/dsh-config/bbo-split-merge.py merge "+
                    "--parent "+shlex.quote(d+"/parent.py")+" --a "+shlex.quote(d+"/A/candidate.py")+" --b "+shlex.quote(d+"/B/candidate.py")+
                    " --out "+shlex.quote(d+"/merged/solver.py")+" --report "+shlex.quote(d+"/merge-report.json"),
                    merge_remain)
                if mr.return_code:
                    row.update({"status":"merge_failed","A_session_ok":ra,"B_session_ok":rb,
                         "A_error":self.session_error.get(tag+"-A"),"B_error":self.session_error.get(tag+"-B"),
                         "merge_error":(mr.stderr or mr.stdout or "")[-2000:]})
                    self._log("history.json",json.dumps(history,ensure_ascii=False,indent=2)+"\n"); n+=1; continue

                try:
                    rr=await self._exec(env,"cat "+shlex.quote(d+"/merge-report.json"),10)
                    merge_report=json.loads(rr.stdout) if rr.return_code==0 and rr.stdout else {}
                except Exception:
                    merge_report={}
                row.update({"merge_status":"merged",
                            "namespace_normalization":merge_report.get("namespace_normalization")})
                self._log("history.json",json.dumps(history,ensure_ascii=False,indent=2)+"\n")
                pre_remain=max(0,math.ceil(deadline-time.monotonic()))
                if pre_remain<1:
                    row.update({"status":"merged_unscored_no_time"})
                    self._log("history.json",json.dumps(history,ensure_ascii=False,indent=2)+"\n")
                    stop="time_limit_after_merge"; break
                pr=await self._exec(env,"python3 /opt/dsh-config/bbo_candidate_preflight.py "+shlex.quote(d+"/merged/solver.py"),pre_remain)
                if pr.return_code:
                    row.update({"status":"preflight_failed",
                         "A_session_ok":ra,"B_session_ok":rb,
                         "A_error":self.session_error.get(tag+"-A"),"B_error":self.session_error.get(tag+"-B"),
                         "error":(pr.stderr or pr.stdout or "")[-2000:]})
                    self._log("history.json",json.dumps(history,ensure_ascii=False,indent=2)+"\n"); n+=1; continue

                # Continuous lineage: every structurally valid merged solver becomes
                # the next parent, even if its visible score later drops.
                lineage=d+"/merged/solver.py"; lineage_version=tag
                score_metric=None
                score_remain=max(0,math.ceil(deadline-time.monotonic()))
                if score_remain>0: score_metric=await self._selfcheck(env,lineage,tag,score_remain)
                score=float(score_metric["score"]) if score_metric else None
                if score is not None:
                    lineage_score=score; latest_scored=lineage; latest_scored_version=tag; latest_scored_score=score
                    if score>=champion_score:
                        champion=lineage; champion_version=tag; champion_score=score
                def read_json(path):
                    return path
                row.update({"status":"scored" if score is not None else "merged_unscored",
                     "parent_version":parent_version,"parent_score":parent_score,
                     "score":score,"score_delta_vs_parent":(score-parent_score) if score is not None and isinstance(parent_score,(int,float)) else None,
                     "score_anytime":score_metric.get("score_anytime") if score_metric else None,
                     "score_final":score_metric.get("score_final") if score_metric else None,
                     "lineage_next":tag,"champion_version_after":champion_version,"champion_score_after":champion_score,
                     "A_session_ok":ra,"B_session_ok":rb,"A_error":self.session_error.get(tag+"-A"),"B_error":self.session_error.get(tag+"-B"),
                     "A_trace":self.traces.get(tag+"-A"),"B_trace":self.traces.get(tag+"-B"),
                     "wall_sec":round(time.monotonic()-started,1)})
                self._log("history.json",json.dumps(history,ensure_ascii=False,indent=2)+"\n")
                n+=1
                if time.monotonic()>=deadline: stop="time_limit_after_selfcheck"; break
        finally:
            # Final submission uses best visible-scored solver; development lineage
            # remains continuous and is recorded separately. Export research-only
            # scratch to /logs, then remove it so /app/methods/main ends with solver.py.
            await self._must(env,
                "cp "+shlex.quote(champion)+" "+shlex.quote(self.SOLVER)+
                "; rm -rf "+shlex.quote(self.ROOT+"/versions")+
                "; cp -a "+shlex.quote(self.WORK+"/versions")+" "+shlex.quote(self.ROOT+"/versions")+
                "; rm -rf "+shlex.quote(self.WORK),60)

        selected_hash=await self._hash(env,self.SOLVER)
        selection={"selected_version":champion_version,"selected_visible_score":champion_score,
                   "selected_sha256":selected_hash,"selection_policy":"best visible-scored structurally valid merged solver; ties keep latest",
                   "development_lineage_policy":"next round always starts from latest structurally valid merged solver, even after a visible regression",
                   "latest_lineage_version":lineage_version,"latest_scored_version":latest_scored_version,"latest_scored_score":latest_scored_score,
                   "stop_reason":stop,"versions_attempted":len(history),"condition":self.condition,"build_id":self.BUILD_ID,
                   "task_instruction_sha256":task_hash,"research_time_limit_sec":total,"min_new_round_sec":min_start}
        self._log("final-selection.json",json.dumps(selection,ensure_ascii=False,indent=2)+"\n")
        context.metadata={**(context.metadata or {}),"build_id":self.BUILD_ID,"selected_version":champion_version,
                          "visible_score":champion_score,"versions_attempted":len(history),"condition":self.condition}
