from __future__ import annotations

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
    _VERSION = "0.5.0-official-autoresearch-review"
    _PROMPT_PLACEHOLDER = "{{ instruction }}"

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
        rendered = template.replace(
            self._PROMPT_PLACEHOLDER,
            instruction.rstrip("\n"),
        )
        template_sha = hashlib.sha256(template.encode("utf-8")).hexdigest()
        task_sha = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        rendered_sha = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        self._write_required_log("rendered-prompt.sha256", rendered_sha + "\n")
        self._write_required_log(
            "prompt-source.txt",
            "\n".join(
                [
                    f"template_path={self._prompt_template_path}",
                    f"template_sha256={template_sha}",
                    f"task_sha256={task_sha}",
                    f"rendered_sha256={rendered_sha}",
                    "rendering=literal replacement of official {{ instruction }} placeholder",
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

    async def _audit_research_bookkeeping(
        self,
        environment: BaseEnvironment,
    ) -> dict[str, Any]:
        command = r'''python3 - <<'PY'
import json
import re
from pathlib import Path

root = Path('/app/methods')
versions = root / 'versions'
all_snapshot_dirs = sorted(
    [p.name for p in versions.iterdir() if p.is_dir()]
) if versions.is_dir() else []
canonical_snapshot_dirs = sorted(
    [name for name in all_snapshot_dirs if re.fullmatch(r'v[0-9]+', name)],
    key=lambda x: int(x[1:]),
)
noncanonical_snapshot_dirs = sorted(
    [name for name in all_snapshot_dirs if name not in canonical_snapshot_dirs]
)
log = root / 'experiment_log.md'
solver = root / 'main' / 'solver.py'
log_text = log.read_text(encoding='utf-8', errors='replace') if log.is_file() else ''
logged_versions = []
for match in re.finditer(r'^##\s+(v[0-9]+)\b', log_text, flags=re.MULTILINE):
    version = match.group(1)
    if version not in logged_versions:
        logged_versions.append(version)
# v0 is the shipped baseline; snapshot completeness concerns agent-created versions.
logged_candidate_versions = [v for v in logged_versions if v != 'v0']
missing = [v for v in logged_candidate_versions if v not in canonical_snapshot_dirs]
extra = [v for v in canonical_snapshot_dirs if v not in logged_candidate_versions]
print(json.dumps({
    'final_solver_exists': solver.is_file(),
    'experiment_log_exists': log.is_file(),
    'experiment_log_nonempty': log.is_file() and log.stat().st_size > 0,
    'versions_dir_exists': versions.is_dir(),
    'logged_versions': logged_versions,
    'logged_candidate_versions': logged_candidate_versions,
    'logged_version_count': len(logged_candidate_versions),
    'snapshot_dirs': all_snapshot_dirs,
    'canonical_snapshot_versions': canonical_snapshot_dirs,
    'snapshot_version_count': len(canonical_snapshot_dirs),
    'noncanonical_snapshot_dirs': noncanonical_snapshot_dirs,
    'missing_snapshot_versions': missing,
    'extra_snapshot_versions': extra,
    'snapshot_complete': bool(logged_candidate_versions) and not missing and not noncanonical_snapshot_dirs,
}, sort_keys=True))
PY'''
        audit = await environment.exec(
            command,
            cwd="/app",
            env=self._runtime_env(),
            timeout_sec=60,
        )
        self._write_optional_log("autoresearch-audit.stderr.log", audit.stderr)
        self._write_required_log("autoresearch-audit.json", audit.stdout)
        if audit.return_code != 0:
            raise RuntimeError("failed to audit autoresearch bookkeeping artifacts")
        try:
            parsed = json.loads(audit.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError("invalid JSON from autoresearch bookkeeping audit") from exc
        return parsed

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        # The historical custom adapter incorrectly forwarded only the task
        # instruction.  Render the official autoresearch.j2 explicitly so DSH
        # receives the same research loop + task section expected by RSI-Exam.
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

        self._bookkeeping_audit = await self._audit_research_bookkeeping(environment)
        self._record_result(context, result)
