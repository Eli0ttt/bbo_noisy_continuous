from __future__ import annotations
import ast
import hashlib
import os
import re
from pathlib import Path
import yaml

root = Path(os.environ["BBO_ROOT"]).resolve()
base = root / "repro/work/EvoMAS-bbo"
pipeline = base / "adapter/evomas_bbo_pipeline.py"
runtime = base / "adapter/native_bbo_runtime.py"
source_pool = base / "pool/paper_bbeh_s1"
target_pool = base / "pool/paper_bbeh_s2"
s1 = "openai:deepseek-v4-flash-s1"
s2 = "openai:deepseek-v4-flash-s2"

old_p = pipeline.read_text(encoding="utf-8")
old_r = runtime.read_text(encoding="utf-8")
new_p = old_p.replace(s1, s2)
if s1 in new_p or s2 not in new_p:
    raise RuntimeError("Pipeline model routing is incomplete")
all_model_literals = {
    node.value for node in ast.walk(ast.parse(new_p))
    if isinstance(node, ast.Constant) and isinstance(node.value, str)
    and "deepseek-v4-flash-s" in node.value
}
if all_model_literals != {s2}:
    raise RuntimeError(f"Unexpected pipeline model literals: {all_model_literals}")

matches = list(re.finditer(r"(?m)^ALLOWED_MODELS\s*=\s*\{[^\n]*\}[ \t]*$", old_r))
if len(matches) != 1:
    raise RuntimeError(f"Expected one runtime model allowlist; found {len(matches)}")
new_r = old_r[:matches[0].start()] + f'ALLOWED_MODELS = {{"{s2}"}}' + old_r[matches[0].end():]
allowed = [node.value for node in ast.walk(ast.parse(new_r))
           if isinstance(node, ast.Assign)
           and any(isinstance(t, ast.Name) and t.id == "ALLOWED_MODELS" for t in node.targets)]
if len(allowed) != 1 or ast.literal_eval(allowed[0]) != {s2}:
    raise RuntimeError("Runtime does not enforce the s2-only allowlist")
compile(new_p, str(pipeline), "exec")
compile(new_r, str(runtime), "exec")

names = ("peer_review.yaml", "majority_vote.yaml", "debate.yaml")
converted = {}
for name in names:
    config = yaml.safe_load((source_pool / name).read_text(encoding="utf-8"))
    agents = config["agents"]
    if not agents or any(agent.get("tools") for agent in agents.values()):
        raise RuntimeError(f"Unexpected seed tools in {name}")
    for agent in agents.values():
        agent["model_id"] = s2
    converted[name] = config

if target_pool.exists():
    for name in names:
        if yaml.safe_load((target_pool / name).read_text(encoding="utf-8")) != converted[name]:
            raise RuntimeError(f"Existing s2 pool differs: {name}")
else:
    target_pool.mkdir(parents=True)
    for name, config in converted.items():
        (target_pool / name).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

for path, old, new in ((pipeline, old_p, new_p), (runtime, old_r, new_r)):
    if old != new:
        backup = path.with_name(path.name + ".before_s2_only")
        if not backup.exists():
            backup.write_text(old, encoding="utf-8")
        path.write_text(new, encoding="utf-8")
    print("[S2 PATCH]", path.name, hashlib.sha256(new.encode()).hexdigest())
print("[S2 POOL]", target_pool, "configs=3; all agents use s2; no tools")
print("[PASS] s2-only migration; no API calls")
