from __future__ import annotations

import base64
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any, Sequence

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext


class DshBboAgent(BaseAgent):
    """Run one autonomous DeepSeek Harness rollout for RSI-Exam BBO.

    Harbor/Docker is the benchmark isolation boundary.  DSH's own nested shell
    sandbox is disabled for this task because the task container has no usable
    bubblewrap/Landlock backend; with ``workspace-write`` every bash call is
    rejected before it runs.  DSH is still confined to the Harbor task
    container, whose network/resource limits and separate verifier are kept.
    """

    _VALID_CONDITIONS = frozenset({"no-cordis", "dynamic-cordis"})
    _NODE = "/opt/node/bin/node"
    _CLI = "/opt/deepseek-harness/apps/cli/src/bin.ts"
    _NO_CORDIS_PATCH = "/opt/dsh-config/bbo-no-cordis.yml"
    _CORDIS_PATCH = "/opt/dsh-config/bbo-cordis-extra.yml"
    _CHECKPOINT_HELPER = "/opt/dsh-config/version_checkpoint.py"
    _SELFCHECK_GUARD = "/opt/dsh-config/bbo-selfcheck-guard.mjs"
    _VERSION = "0.8.2-official-autoresearch-complete-visible-ledger"
    _PROMPT_PLACEHOLDER = "{{ instruction }}"
    _BOOKKEEPING_ADDENDUM = r"""

---

## Harness transactional bookkeeping for the official version/log requirement

The official autoresearch instruction above requires visible selfchecks, an experiment log,
saved versions, and keep/rollback decisions. This harness routes every **scored visible
evaluation used for research** through a transactional helper so the source, score, and
decision remain one-to-one and auditable.

Evaluate the shipped v0 baseline, if you want its visible score, with:

```bash
python /opt/dsh-config/version_checkpoint.py baseline
```

For every candidate solver, use a new monotonically increasing version id:

```bash
python /opt/dsh-config/version_checkpoint.py evaluate --version vN --description "brief hypothesis/change"
```

The helper first snapshots the exact candidate from `/app/methods/main/solver.py`, then runs
the **unmodified official** `/app/selfcheck.py --json`. After the selfcheck it restores the
current canonical parent. The candidate is uncommitted until you explicitly choose:

```bash
python /opt/dsh-config/version_checkpoint.py keep --version vN --note "why this is kept"
```

or:

```bash
python /opt/dsh-config/version_checkpoint.py revert --version vN --note "why this is reverted"
```

`keep` is allowed only after a successful official selfcheck. A failed/timed-out candidate
must be reverted, fixed, and evaluated again as a new version before it can become canonical.

To deliberately branch from an older checkpoint after resolving the current candidate:

```bash
python /opt/dsh-config/version_checkpoint.py checkout --version vM
```

Only v0 or a successfully-selfchecked checkpoint may become canonical.

Important rules:

- Do **not** execute `/app/selfcheck.py` directly from Bash. The harness blocks direct
  model-facing selfcheck execution. You may read/inspect its source; scoring must use
  `baseline` or `evaluate` above.
- Do not edit `methods/main/solver.py` for the next experiment while a candidate is pending.
- Every visible score that influences research/version selection must therefore have a
  matching immutable helper checkpoint (v0 is the baseline exception).
- Version numbering, hypotheses, experiment selection, selfcheck frequency, keep/revert
  decisions, and optimization strategy remain yours.
- Do not manually overwrite version snapshots, the checkpoint manifest, or its log table.
- Finalization is owned by the outer harness only after the DeepSeek research process exits.
  If one candidate is still pending, it is an uncommitted transaction and is reverted at
  handoff; the last eligible canonical checkpoint is submitted without comparing scores.
- Missing/mutated snapshots, an unsuccessfully-evaluated committed version, broken lineage,
  uncheckpointed final edits, wrong finalization ownership, or final/submitted mismatch are
  hard handoff failures.

This policy changes only local research bookkeeping/tool routing. It does not alter the task,
visible data, official selfcheck implementation or metric, hidden verifier, scorer, resource
limits, information boundary, or your autonomous research choices.
"""



    _HANDOFF_FINALIZER_SOURCE = r"""from __future__ import annotations
import datetime as _dt, fcntl, hashlib, json, os, sys
from pathlib import Path

ROOT = Path("/app/methods")
STATE = ROOT / "version_checkpoints.json"
LOCK = ROOT / ".version-checkpoint.lock"
PROTOCOL = "outer-harness-last-explicitly-committed-canonical"

def utc_now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_write_json(path, payload):
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tmp.open("wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def fail(message):
    raise RuntimeError(message)

def successful_candidate_selfcheck(row):
    selfcheck = row.get("selfcheck")
    return (
        isinstance(selfcheck, dict)
        and selfcheck.get("return_code") == 0
        and not bool(selfcheck.get("timed_out"))
        and row.get("score") is not None
    )

try:
    if not STATE.is_file():
        fail("checkpoint manifest missing at handoff")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        state = json.loads(STATE.read_text(encoding="utf-8"))
        versions = state.get("versions")
        if not isinstance(versions, dict):
            fail("invalid checkpoint manifest: versions map missing")

        if state.get("finalized"):
            fail(
                "premature_finalized_state: research phase must not finalize; "
                f"finalized_by={state.get('finalized_by')!r} "
                f"final_version={state.get('final_version')!r}"
            )

        pre_submitted = [
            v for v, row in versions.items()
            if isinstance(row, dict) and row.get("status") == "submitted"
        ]
        if pre_submitted:
            fail("premature_submitted_labels=" + ",".join(sorted(pre_submitted)))

        canonical = state.get("canonical_version")
        if not isinstance(canonical, str) or canonical not in versions:
            fail(f"invalid canonical version at handoff: {canonical!r}")
        canonical_row = versions[canonical]
        canonical_sha = canonical_row.get("solver_sha256")
        if not isinstance(canonical_sha, str) or not canonical_sha:
            fail(f"canonical checkpoint {canonical} has no solver sha256")
        if canonical != "v0":
            if canonical_row.get("status") != "kept":
                fail(
                    f"canonical checkpoint {canonical} is not explicitly committed: "
                    f"status={canonical_row.get('status')!r}"
                )
            if not successful_candidate_selfcheck(canonical_row):
                fail(
                    f"canonical checkpoint {canonical} has no successful official visible selfcheck"
                )

        final_solver = ROOT / "main" / "solver.py"
        if not final_solver.is_file() or final_solver.is_symlink():
            fail("final /app/methods/main/solver.py must be a regular file")
        final_sha = sha256_file(final_solver)

        pending = state.get("pending_version")
        if pending is not None:
            pending_row = versions.get(pending)
            if not isinstance(pending_row, dict):
                fail(f"pending version {pending!r} is missing from manifest")
            if final_sha != canonical_sha:
                pending_sha = pending_row.get("solver_sha256")
                if isinstance(pending_sha, str) and final_sha == pending_sha:
                    fail(
                        f"pending version {pending} is uncommitted but the final solver "
                        "matches its snapshot; an explicit keep was required"
                    )
                fail(
                    "uncheckpointed final artifact while a candidate was pending: "
                    f"main does not match canonical checkpoint {canonical}"
                )
            pending_status = pending_row.get("status")
            if pending_status not in {"evaluated", "selfcheck_failed", "evaluating"}:
                fail(f"invalid pending status for {pending}: {pending_status!r}")
            pending_row["previous_status"] = pending_status
            pending_row["status"] = "reverted"
            pending_row["decided_at"] = utc_now()
            pending_row["reverted_to"] = canonical
            pending_row["decision_source"] = "outer_harness_auto_abort_uncommitted_at_handoff"
            pending_row["decision_note"] = (
                "uncommitted pending candidate reverted by the outer harness at handoff; "
                f"canonical checkpoint {canonical} remained live"
            )
            auto = state.setdefault("auto_reverted_pending_versions", [])
            if pending not in auto:
                auto.append(pending)
            state["pending_version"] = None
        elif final_sha != canonical_sha:
            fail(
                "uncheckpointed final artifact: final solver does not match "
                f"canonical checkpoint {canonical}; evaluate and keep the final candidate "
                "before finishing"
            )

        canonical_row["previous_status"] = canonical_row.get("status")
        canonical_row["status"] = "submitted"
        canonical_row["decided_at"] = utc_now()
        canonical_row["decision_source"] = "outer_harness_submit_canonical_at_handoff"
        canonical_row["decision_note"] = (
            "submitted by the outer harness from the last canonical checkpoint "
            "after the research process exited"
        )
        state["finalized"] = True
        state["finalized_at"] = utc_now()
        state["finalized_by"] = "outer_harness"
        state["finalization_protocol"] = PROTOCOL
        state["final_version"] = canonical
        atomic_write_json(STATE, state)

    print(json.dumps({
        "status": "finalized",
        "final_version": canonical,
        "finalized_by": "outer_harness",
        "finalization_protocol": PROTOCOL,
        "auto_reverted_pending_versions": state.get("auto_reverted_pending_versions", []),
    }, sort_keys=True))
except Exception as exc:
    print("HANDOFF_FINALIZE_ERROR: " + str(exc), file=sys.stderr)
    raise SystemExit(2)"""

    def __init__(
        self,
        *args: Any,
        condition: str = "no-cordis",
        prompt_template_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        if condition not in self._VALID_CONDITIONS:
            raise ValueError(
                f"condition must be one of {sorted(self._VALID_CONDITIONS)}, got {condition!r}"
            )

        # Preserve BaseAgent's normal initialization.  We additionally keep the
        # official Jinja template path because this custom adapter must render
        # it explicitly before passing a prompt to DSH.
        super().__init__(
            *args,
            prompt_template_path=prompt_template_path,
            **kwargs,
        )
        self.condition = condition
        self._prompt_template_path = (
            Path(prompt_template_path).expanduser().resolve()
            if prompt_template_path
            else None
        )
        self._task_instruction_on_disk: str | None = None
        self._bookkeeping_audit: dict[str, Any] = {}

    @staticmethod
    def name() -> str:
        return "dsh-bbo"

    def version(self) -> str:
        return self._VERSION

    def _patches(self) -> list[str]:
        patches = [self._NO_CORDIS_PATCH]
        if self.condition == "dynamic-cordis":
            patches.append(self._CORDIS_PATCH)
        return patches

    def _runtime_env(self) -> dict[str, str]:
        api_key = self._get_env("DEEPSEEK_API_KEY")
        base_url = self._get_env("DEEPSEEK_BASE_URL")
        missing = [
            key
            for key, value in {
                "DEEPSEEK_API_KEY": api_key,
                "DEEPSEEK_BASE_URL": base_url,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"missing required host environment variable(s): {missing}")

        return {
            "DEEPSEEK_API_KEY": api_key,
            "DEEPSEEK_BASE_URL": base_url,
            "DSH_HOME": "/logs/artifacts/dsh-home",
            "DSH_TELEMETRY_DISABLED": "1",
            # The historical workspace-write mode failed before *every* bash
            # command because this Docker task does not expose a nested DSH
            # sandbox backend.  danger-full-access is DSH-local only; Harbor's
            # Docker/no-network/resource boundary remains authoritative.
            "DSH_PERMISSION_MODE": "danger-full-access",
            "NO_COLOR": "1",
            "TSX_TSCONFIG_PATH": "/opt/deepseek-harness/tsconfig.json",
            "ARB_OUTPUT_TOKEN_LIMIT": "500000",
            "ARB_AGENT_TIMEOUT_SEC": "43200",
        }

    def _dsh_command(self, app_args: Sequence[str]) -> str:
        # Keep the exact CLI launch path that is already proven to work with
        # the user's checked-out DeepSeek Harness version.
        resolver = (
            "console.log(require.resolve('tsx/esm', { paths: "
            + json.dumps(["/opt/deepseek-harness"])
            + " }))"
        )
        args = ["--profile", "headless"]
        for patch in self._patches():
            args.extend(["--patch", patch])
        args.extend(app_args)
        return "\n".join(
            [
                f"TSX_LOADER=$({shlex.quote(self._NODE)} -e {shlex.quote(resolver)})",
                "exec "
                f"{shlex.quote(self._NODE)} --import \"$TSX_LOADER\" "
                f"{shlex.quote(self._CLI)} "
                + " ".join(shlex.quote(arg) for arg in args),
            ]
        )

    def _write_required_log(self, name: str, content: str | None) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / name).write_text(content or "", encoding="utf-8")

    def _write_optional_log(self, name: str, content: str | None) -> None:
        """Write diagnostic output only when it contains useful content.

        Older runs created many zero-byte stderr/handoff files.  Harbor already
        records trial status, so empty diagnostics add review clutter without
        adding evidence.
        """
        if not content:
            return
        if not content.strip():
            return
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / name).write_text(content, encoding="utf-8")

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.replace("\r\n", "\n").strip()

    def _load_official_template(self) -> str:
        if self._prompt_template_path is None:
            raise RuntimeError(
                "prompt_template_path is required; formal-run.sh must pass the official "
                "RSI-Exam infra/prompts/autoresearch.j2"
            )
        if not self._prompt_template_path.is_file():
            raise RuntimeError(
                f"official autoresearch template not found: {self._prompt_template_path}"
            )

        template = self._prompt_template_path.read_text(encoding="utf-8")
        if template.count(self._PROMPT_PLACEHOLDER) != 1:
            raise RuntimeError(
                "official autoresearch template must contain exactly one "
                f"{self._PROMPT_PLACEHOLDER!r} placeholder"
            )
        # Fail closed if the local template is no longer the official research
        # protocol that this experiment claims to use.
        for marker in (
            "LOOP FOREVER:",
            "/app/methods/experiment_log.md",
            "/app/methods/versions/v<N>",
            "/app/AUTORESEARCH.md",
            "/app/TASK.md",
        ):
            if marker not in template:
                raise RuntimeError(
                    f"official autoresearch template missing expected marker: {marker}"
                )
        return template

    def _render_official_prompt(self, instruction: str) -> str:
        if self._task_instruction_on_disk is None:
            raise RuntimeError("setup() did not cache /app/TASK.md")
        if self._normalize_text(instruction) != self._normalize_text(
            self._task_instruction_on_disk
        ):
            raise RuntimeError(
                "Harbor-provided instruction differs from /app/TASK.md; refusing "
                "to run a protocol-drifted formal rollout"
            )

        template = self._load_official_template()
        official_rendered = template.replace(
            self._PROMPT_PLACEHOLDER,
            instruction.rstrip("\n"),
        )
        rendered = official_rendered + self._BOOKKEEPING_ADDENDUM
        template_sha = hashlib.sha256(template.encode("utf-8")).hexdigest()
        task_sha = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        addendum_sha = hashlib.sha256(self._BOOKKEEPING_ADDENDUM.encode("utf-8")).hexdigest()
        rendered_sha = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        self._write_required_log("rendered-prompt.sha256", rendered_sha + "\n")
        self._write_required_log(
            "prompt-source.txt",
            "\n".join(
                [
                    f"template_path={self._prompt_template_path}",
                    f"template_sha256={template_sha}",
                    f"task_sha256={task_sha}",
                    f"bookkeeping_addendum_sha256={addendum_sha}",
                    f"rendered_sha256={rendered_sha}",
                    "rendering=official template literal replacement + complete-visible-ledger bookkeeping addendum",
                    "",
                ]
            ),
        )
        return rendered

    def _record_result(self, context: AgentContext, result: ExecResult) -> None:
        context.metadata = {
            **(context.metadata or {}),
            "condition": self.condition,
            "dsh_profile": "headless",
            "dsh_patches": self._patches(),
            "official_prompt_protocol": True,
            "rendered_autoresearch_prompt": True,
            "single_persistent_session": True,
            "bookkeeping_checkpoint_guard": True,
            "version_checkpoint_helper": self._CHECKPOINT_HELPER,
            "version_checkpoint_protocol": "transactional-complete-visible-ledger",
            "visible_selfcheck_protocol": "checkpoint-helper-only",
            "direct_selfcheck_guard": True,
            "version_decision_protocol": "resolve-before-next-version",
            "final_submission_protocol": "outer-harness-last-explicitly-committed-canonical",
            "finalization_owner": "outer_harness",
            "workspace": "/app",
            "dsh_permission_mode": "danger-full-access",
            "isolation_boundary": "harbor-docker-task-container",
            "return_code": result.return_code,
            "experiment_log_exists": self._bookkeeping_audit.get(
                "experiment_log_exists", False
            ),
            "logged_version_count": self._bookkeeping_audit.get(
                "logged_version_count", 0
            ),
            "snapshot_version_count": self._bookkeeping_audit.get(
                "snapshot_version_count", 0
            ),
            "snapshot_complete": self._bookkeeping_audit.get(
                "snapshot_complete", False
            ),
            "checkpoint_guard_complete": self._bookkeeping_audit.get(
                "checkpoint_guard_complete", False
            ),
            "decision_complete": self._bookkeeping_audit.get(
                "decision_complete", False
            ),
            "lineage_complete": self._bookkeeping_audit.get(
                "lineage_complete", False
            ),
            "canonical_eligible": self._bookkeeping_audit.get(
                "canonical_eligible", False
            ),
            "invalid_committed_versions": self._bookkeeping_audit.get(
                "invalid_committed_versions", []
            ),
            "finalized": self._bookkeeping_audit.get("finalized", False),
            "finalized_by": self._bookkeeping_audit.get("finalized_by"),
            "finalization_protocol": self._bookkeeping_audit.get("finalization_protocol"),
            "final_version": self._bookkeeping_audit.get("final_version"),
        }

    async def setup(self, environment: BaseEnvironment) -> None:
        # Verify the host-side source before paying for a model call.
        self._load_official_template()

        checks = [
            "set -eu",
            "mkdir -p /logs/artifacts/dsh-home /logs/artifacts/dsh",
            f"test -x {shlex.quote(self._NODE)}",
            f"test -f {shlex.quote(self._CLI)}",
            f"test -f {shlex.quote(self._NO_CORDIS_PATCH)}",
            f"test -f {shlex.quote(self._CHECKPOINT_HELPER)}",
            f"test -f {shlex.quote(self._SELFCHECK_GUARD)}",
            "test -f /app/AUTORESEARCH.md",
            "test -f /app/TASK.md",
            "test -f /app/budget.py",
            "test -f /app/selfcheck.py",
            "test -f /app/methods/main/solver.py",
            "sha256sum /app/AUTORESEARCH.md /app/TASK.md /app/budget.py",
        ]
        if self.condition == "dynamic-cordis":
            checks.append(f"test -f {shlex.quote(self._CORDIS_PATCH)}")

        result = await environment.exec(
            "\n".join(checks + [self._dsh_command(["--dump-config"])]),
            cwd="/app",
            env=self._runtime_env(),
            timeout_sec=300,
        )
        self._write_required_log("effective-config.yml", result.stdout)
        self._write_optional_log("setup.stderr.log", result.stderr)
        if result.return_code != 0:
            raise RuntimeError(
                "DeepSeek Harness setup/config audit failed; "
                f"see {self.logs_dir / 'setup.stderr.log'}"
            )
        config_text = result.stdout or ""
        for required in (
            "@deepseek-ai/dsh-tool-bash",
            "@deepseek-ai/dsh-tool-fs",
            "@deepseek-ai/dsh-tool-fs-search",
        ):
            if required not in config_text:
                raise RuntimeError(f"effective DSH config is missing required tool {required}")
        if "bbo-selfcheck-guard" not in config_text:
            raise RuntimeError("effective DSH config is missing bbo-selfcheck-guard")
        cordis_present = "@deepseek-ai/dsh-tool-cordis" in config_text
        if cordis_present != (self.condition == "dynamic-cordis"):
            raise RuntimeError(
                "effective DSH config does not match the requested Cordis treatment: "
                f"condition={self.condition!r}, cordis_present={cordis_present}"
            )

        # Cache the exact task text from inside the task container.  run() will
        # require Harbor's instruction argument to match it before rendering the
        # official autoresearch template.
        task_read = await environment.exec(
            "cat /app/TASK.md",
            cwd="/app",
            env=self._runtime_env(),
            timeout_sec=60,
        )
        self._write_optional_log("task-read.stderr.log", task_read.stderr)
        if task_read.return_code != 0:
            raise RuntimeError("failed to read /app/TASK.md during setup")
        self._task_instruction_on_disk = task_read.stdout or ""

        # Initialize an immutable v0 baseline snapshot and helper-managed
        # experiment log before the model starts.  This does not run a visible
        # selfcheck and does not provide any extra task information.
        checkpoint_init = await environment.exec(
            f"python3 {shlex.quote(self._CHECKPOINT_HELPER)} init",
            cwd="/app",
            env=self._runtime_env(),
            timeout_sec=60,
        )
        self._write_optional_log("checkpoint-init.stdout.log", checkpoint_init.stdout)
        self._write_optional_log("checkpoint-init.stderr.log", checkpoint_init.stderr)
        if checkpoint_init.return_code != 0:
            raise RuntimeError("failed to initialize transactional version checkpoint state machine")

    async def _finalize_research_bookkeeping(
        self,
        environment: BaseEnvironment,
    ) -> dict[str, Any]:
        # Research has fully exited before this lifecycle transition. The
        # agent-facing helper exposes no finalize operation.
        encoded = base64.b64encode(
            self._HANDOFF_FINALIZER_SOURCE.encode("utf-8")
        ).decode("ascii")
        finalizer_code = (
            "import base64;"
            "exec(base64.b64decode(" + repr(encoded) + ").decode('utf-8'))"
        )
        finalize = await environment.exec(
            "python3 -c " + shlex.quote(finalizer_code),
            cwd="/app",
            env=self._runtime_env(),
            timeout_sec=60,
        )
        self._write_optional_log("handoff-finalize.stdout.log", finalize.stdout)
        self._write_optional_log("handoff-finalize.stderr.log", finalize.stderr)
        if finalize.return_code != 0:
            detail = (
                finalize.stderr or finalize.stdout or
                "unknown handoff-finalization failure"
            ).strip()
            raise RuntimeError("outer-harness finalization failed: " + detail)

        audit = await environment.exec(
            f"python3 {shlex.quote(self._CHECKPOINT_HELPER)} audit --json",
            cwd="/app",
            env=self._runtime_env(),
            timeout_sec=60,
        )
        self._write_optional_log("autoresearch-audit.stderr.log", audit.stderr)
        self._write_required_log("autoresearch-audit.json", audit.stdout)
        try:
            parsed = json.loads(audit.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError("invalid JSON from version checkpoint audit") from exc
        self._bookkeeping_audit = parsed
        if audit.return_code != 0 or not parsed.get("checkpoint_guard_complete", False):
            reasons = parsed.get("guard_failure_reasons") or []
            detail = "; ".join(str(x) for x in reasons) if reasons else (
                "unspecified checkpoint state-machine failure"
            )
            raise RuntimeError(
                "version checkpoint state machine rejected final handoff: " + detail
            )
        if parsed.get("finalized_by") != "outer_harness":
            raise RuntimeError(
                "version checkpoint state machine rejected final handoff: "
                f"invalid finalization owner {parsed.get('finalized_by')!r}"
            )
        return parsed

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        # The historical custom adapter incorrectly forwarded only the task
        # instruction. Render the official autoresearch.j2 explicitly so DSH
        # receives the official research loop + task section, followed only by
        # a harness bookkeeping workflow that operationalizes the prompt's own
        # version/log/keep-or-revert requirement.
        rendered_prompt = self._render_official_prompt(instruction)

        # One persistent DSH conversation is one outer research rollout.  Do
        # NOT impose a fixed number of versions/self-checks here: the official
        # prompt tells the autonomous agent to iterate, while the task grants
        # up to 12 hours rather than a mandatory iteration count.
        result = await environment.exec(
            self._dsh_command([rendered_prompt]),
            cwd="/app",
            env=self._runtime_env(),
            timeout_sec=43200,
        )
        self._write_optional_log("dsh.stdout.log", result.stdout)
        self._write_optional_log("dsh.stderr.log", result.stderr)

        if result.return_code != 0:
            self._record_result(context, result)
            raise RuntimeError(
                "DeepSeek Harness research rollout exited unsuccessfully; "
                f"see {self.logs_dir / 'dsh.stderr.log'}"
            )

        # This is only an artifact sanity check; the separate Harbor verifier
        # remains authoritative and hidden from the research loop.
        handoff = await environment.exec(
            "set -eu; test -f /app/methods/main/solver.py; "
            "test ! -L /app/methods/main/solver.py",
            cwd="/app",
            env=self._runtime_env(),
            timeout_sec=120,
        )
        self._write_optional_log("handoff.stdout.log", handoff.stdout)
        self._write_optional_log("handoff.stderr.log", handoff.stderr)
        if handoff.return_code != 0:
            raise RuntimeError(
                "Agent did not leave a valid regular /app/methods/main/solver.py; "
                f"see {self.logs_dir / 'handoff.stderr.log'}"
            )

        try:
            self._bookkeeping_audit = await self._finalize_research_bookkeeping(environment)
        except Exception:
            self._record_result(context, result)
            raise
        self._record_result(context, result)
