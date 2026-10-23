from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

EVOMAS_ROOT = (
    Path(__file__).resolve().parents[2]
    / "EvoMAS-openai"
)
sys.path.insert(0, str(EVOMAS_ROOT))

from src.agents.runners.smolagents import SmolagentsRunner
from src.agents.spec import AgentSpec


CONFIG_PATH = Path(sys.argv[1])
OUTPUT_DIR = Path(sys.argv[2])
INSTRUCTION_PATH = Path(sys.argv[3])

config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
instruction = INSTRUCTION_PATH.read_text(encoding="utf-8")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

runner = SmolagentsRunner()
results = {}


def run_agent(agent_id: str, task: str) -> dict:
    spec = AgentSpec(**config["agents"][agent_id])
    result = runner.run(spec, task)
    payload = result.model_dump()

    (OUTPUT_DIR / f"{agent_id}_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / f"{agent_id}_response.txt").write_text(
        payload.get("content", ""),
        encoding="utf-8",
    )

    if not result.success:
        raise RuntimeError(
            f"{agent_id} failed: {payload.get('error')}"
        )

    results[agent_id] = payload
    return payload


designer_task = f"""
You are the designer in an EvoMAS system for the official RSI
Query-Budgeted Noisy Black-Box Optimization task.

Task contract:
{instruction}

Use the python_interpreter tool at least once to check one small
numerical calculation relevant to the proposed optimizer.

Then produce:
1. A concrete optimizer design.
2. The query allocation strategy under the exact budget.
3. Noise handling and anytime-performance considerations.
4. A concise implementation plan.

Do not call the official selfcheck. Do not access the network.
"""

implementer_task = f"""
You are the implementer in an EvoMAS system for the official RSI
Query-Budgeted Noisy Black-Box Optimization task.

Task contract:
{instruction}

Produce a complete candidate solver in one fenced Python code block.
The solver must follow the required constructor, ask, and tell interface,
use only stdlib and numpy at runtime, and obey the exact query budget.

Do not call the official selfcheck. Do not access the network.
"""

run_agent("designer", designer_task)
run_agent("implementer", implementer_task)

reports = "\n\n".join(
    f"--- {agent_id} ---\n{results[agent_id]['content']}"
    for agent_id in ["designer", "implementer"]
)

aggregator_task = f"""
You are the aggregator in an EvoMAS system for the official RSI
Query-Budgeted Noisy Black-Box Optimization task.

Task contract:
{instruction}

Two upstream agents produced the following reports:

{reports}

Select or repair the strongest candidate. Return exactly one complete
candidate solver in a single fenced Python code block, followed by a short
rationale.

The solver must follow the official constructor, ask, and tell interface,
use only stdlib and numpy at runtime, and obey the exact query budget.
Do not call the official selfcheck. Do not access the network.
"""

run_agent("aggregator", aggregator_task)

summary = {
    "config": str(CONFIG_PATH),
    "agents_executed": list(results),
    "success": True,
    "tool_agent": "designer",
    "tool_name": "python_interpreter",
}

(OUTPUT_DIR / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(summary, ensure_ascii=False))
