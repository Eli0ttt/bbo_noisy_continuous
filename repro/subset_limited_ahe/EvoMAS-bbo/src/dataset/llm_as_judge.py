"""
LLM-as-Judge Evaluator

Uses an LLM to evaluate MAS execution quality by analyzing execution traces.
The judge scores outputs from 5 aspects (each 0-20, total 0-100) without
requiring ground truth, enabling evaluation of open-ended tasks.

Aspects:
    1. Correctness  (0-20): Is the final answer factually correct and logical?
    2. Reasoning     (0-20): Is the reasoning chain sound and well-structured?
    3. Collaboration (0-20): Do agents communicate and build on each other effectively?
    4. Efficiency    (0-20): Is the solution achieved without unnecessary steps?
    5. Completeness  (0-20): Are all aspects of the query addressed?

Usage:
    evaluator = LLMAsJudgeEvaluator(model_id="bedrock:...")
    score = evaluator.evaluate_correctness(output, ground_truth=None, query=query)
    # score is an int in [0, 100]
"""

import logging
import re
from typing import Any, Optional, Dict

from src.models.model import get_model

logger = logging.getLogger(__name__)

ASPECTS = [
    "Correctness",
    "Reasoning",
    "Collaboration",
    "Efficiency",
    "Completeness",
]

# ----------------------------------------------------------------------------
# Per-dataset judge prompts. get_judge_prompt() dispatches to these.
# ----------------------------------------------------------------------------

BBEH_JUDGE_PROMPT = """You are an expert evaluator assessing the quality of a multi-agent system's (MAS) execution on a given task.

Analyze the AGENT OUTPUT (which contains the execution trace and final answer) and score it on the following 5 aspects.

**Scoring rubric (total 0-100):**

1. **Correctness (0-40):** Is the final answer factually correct, logically sound, and directly addresses the query? 40 = perfectly correct; 0 = completely wrong or no answer. This is the most important aspect — a wrong answer with great reasoning should still score low overall.

2. **Reasoning (0-20):** Is the reasoning chain coherent, step-by-step, and free of logical errors? Does the system show clear problem decomposition? 20 = flawless reasoning; 0 = no reasoning or entirely flawed.

3. **Collaboration (0-10):** Do agents build on each other's outputs effectively? For single-agent systems, score based on how well the agent structures its own workflow. 10 = excellent; 0 = no structure.

4. **Efficiency (0-15):** Is the solution achieved without unnecessary repetition or wasted steps? 15 = concise and direct; 0 = extremely wasteful or stuck in loops.

5. **Completeness (0-15):** Are all parts of the query addressed? 15 = fully complete; 0 = mostly missing.

**QUERY:**
{query}

**AGENT OUTPUT (execution trace and final answer):**
{output}

**EVALUATION:**
IMPORTANT: Do NOT re-solve the problem or verify the answer yourself. Simply assess the quality of the agent's work based on the trace above. Output ONLY the scores in exactly this format (no other text before the scores):

Correctness: <score>
Reasoning: <score>
Collaboration: <score>
Efficiency: <score>
Completeness: <score>
Total: <sum>
Brief justification: <1-2 sentences>
"""

WORKBENCH_JUDGE_PROMPT = """You are an expert evaluator assessing the quality of a multi-agent system's (MAS) execution on a given task.

Analyze the AGENT OUTPUT (which contains the execution trace and final answer) and score it on the following 5 aspects. Each aspect is scored from 0 to 20.

**Scoring rubric:**

1. **Correctness (0-20):** Does the agent's final state/answer reflect actual successful tool executions visible in the trace? 20 = every required tool call returned success and produced the intended state; 10 = some tool calls succeeded but the final state is incomplete or partially wrong; 0 = no effective actions. IMPORTANT: If the agent merely claims a result without visible tool returns supporting it (e.g., claims an email was sent but no send_email success appears in the trace), Correctness must be capped at 5 regardless of narrative quality. HOWEVER: If the trace shows the correct tool calls were made with the correct arguments (matching the task requirements), and the agent summarizes the outcome, treat this as evidence of successful execution even if explicit success/return messages are not shown. The presence of the correct function call with correct parameters is sufficient evidence — do NOT penalize the agent for the trace format omitting return values.

2. **Reasoning (0-20):** Is the reasoning chain coherent, step-by-step, and does it correctly select which tools to call and in what order? 20 = flawless; 0 = no reasoning or entirely flawed. If tool calls failed repeatedly and the agent did not adapt, cap this at 10.

3. **Collaboration (0-20):** Do agents build on each other's outputs effectively? Is there meaningful division of labor across the tool-use workflow? 20 = excellent multi-agent synergy with clean cross-domain handoff; 10 = adequate handoff; 0 = no collaboration or harmful interference. For single-agent systems, score based on how well the agent structures its own workflow.

4. **Efficiency (0-20):** Is the solution achieved without redundant tool calls, unnecessary reads, or re-doing actions that already succeeded? 20 = concise and direct; 0 = extremely wasteful or stuck in loops.

5. **Completeness (0-20):** Are all sub-tasks in the query addressed? Does the output cover every requested action? 20 = fully complete; 0 = mostly missing or ignores key aspects.

**QUERY:**
{query}

**AGENT OUTPUT (execution trace and final answer):**
{output}

**EVALUATION:**
IMPORTANT: Do NOT re-solve the problem or verify the answer yourself. Simply assess the quality of the agent's work based on the trace above. Output ONLY the scores in exactly this format (no other text before the scores):

Correctness: <score>
Reasoning: <score>
Collaboration: <score>
Efficiency: <score>
Completeness: <score>
Total: <sum>
Brief justification: <1-2 sentences>
"""

SWEBENCH_JUDGE_PROMPT = """You are an expert evaluator assessing the quality of a multi-agent system's (MAS) execution on a given task.

Analyze the AGENT OUTPUT (which contains the execution trace and final answer) and score it on the following 5 aspects.

**Scoring rubric (total 0-100):**

1. **Correctness (0-40):** Does the produced patch fix the root cause described in the issue? Focus on whether the edits target the RIGHT file(s) at the RIGHT location(s) and whether the change actually addresses the failure mode. If the trace shows the agent localized the bug correctly and the patch's behavior would make the reported failing case pass while leaving unrelated behavior intact, score this highly. Do NOT penalize for stylistic differences that are semantically equivalent to a reference fix: variable/function names, import ordering, whitespace, blank lines, comment presence, inline-vs-helper factoring, defensive null-checks that don't change semantics, equivalent Python constructs (e.g., `isinstance(x, int)` vs `type(x) == int`), or placing the fix at a slightly different but functionally correct point in the same logical location. The oracle for SWE-bench is test outcome, not textual similarity to the gold patch. This is the most important aspect — a wrong or no-op patch must score low regardless of reasoning quality. 40 = a patch that would make the failing test pass without breaking others; 0 = no patch or a patch that cannot plausibly fix the issue.

2. **Reasoning (0-20):** Is the fault-localization and diagnosis chain coherent? Does the agent identify the right files, the right function/method, and the underlying cause of the bug before editing? 20 = clear localization → diagnosis → fix; 0 = edits made without understanding the problem.

3. **Collaboration (0-10):** Do agents build on each other's outputs effectively across the explore / diagnose / fix / verify phases? For single-agent systems, score based on how well the agent structures its own workflow into those phases. 10 = excellent synergy with clean phase handoff; 0 = no structure or harmful interference.

4. **Efficiency (0-15):** Is the fix achieved without reading unrelated files, broad shotgun edits, or re-doing work? 15 = minimal, targeted reads and edits; 0 = extremely wasteful or stuck in loops.

5. **Completeness (0-15):** Does the patch handle the edge cases mentioned in the issue and avoid leaving TODOs, partial fixes, or skipped cases? 15 = complete, self-contained patch; 0 = missing key parts of the fix.

**QUERY:**
{query}

**AGENT OUTPUT (execution trace and final answer):**
{output}

**EVALUATION:**
IMPORTANT: Do NOT re-solve the problem or verify the answer yourself. Evaluate based on the trace provided. When assessing Correctness, ask whether the produced patch would plausibly make the reported failing behavior pass while preserving unrelated behavior — NOT whether it textually resembles any specific reference patch. Semantically equivalent patches must be scored equally regardless of stylistic choice:
- Variable/function names: `tmp`, `result`, `_buf` can all be correct.
- Import ordering and style: grouped vs. alphabetized, absolute vs. relative.
- Whitespace, blank lines, trailing commas.
- Inline fix vs. introducing a small helper function.
- Adding vs. omitting defensive null/type checks that don't change the semantics.
- Equivalent Python constructs (e.g., `isinstance(x, int)` vs `type(x) == int`, list comprehension vs. loop).
- Fix placed at a slightly different — but functionally correct — point within the same logical block.

If the patch addresses the root cause and would pass the intended test(s), Correctness should reflect that, regardless of surface form. Output ONLY the scores in exactly this format (no other text before the scores):

Correctness: <score>
Reasoning: <score>
Collaboration: <score>
Efficiency: <score>
Completeness: <score>
Total: <sum>
Brief justification: <1-2 sentences>
"""


# ----------------------------------------------------------------------------
# Legacy building blocks — retained for historical ablation scripts under
# experiments/ (judge_prompt_optimization.py, judge_vs_groundtruth.py,
# evolution_with_judge_variants.py) that compose variants as
# JUDGE_PROMPT_TEMPLATE + *_ADDENDUM. NOT used on the live path.
# ----------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator assessing the quality of a multi-agent system's (MAS) execution on a given task.

Analyze the AGENT OUTPUT (which contains the execution trace and final answer) and score it on the following 5 aspects. Each aspect is scored from 0 to 20.

**Scoring rubric:**

1. **Correctness (0-20):** Is the final answer factually correct, logically sound, and directly addresses the query? 20 = perfectly correct; 0 = completely wrong or no answer.

2. **Reasoning (0-20):** Is the reasoning chain coherent, step-by-step, and free of logical errors? Does the system show clear problem decomposition? 20 = flawless reasoning; 0 = no reasoning or entirely flawed.

3. **Collaboration (0-20):** Do agents build on each other's outputs effectively? Is there meaningful division of labor or does a single agent do all the work? 20 = excellent multi-agent synergy; 10 = adequate handoff; 0 = no collaboration or harmful interference. For single-agent systems, score based on how well the agent structures its own workflow.

4. **Efficiency (0-20):** Is the solution achieved without unnecessary repetition, redundant computation, or wasted steps? 20 = concise and direct; 0 = extremely wasteful or stuck in loops.

5. **Completeness (0-20):** Are all parts of the query addressed? Does the output cover edge cases and provide a full answer? 20 = fully complete; 0 = mostly missing or ignores key aspects.

**QUERY:**
{query}

**AGENT OUTPUT (execution trace and final answer):**
{output}

**EVALUATION:**
IMPORTANT: Do NOT re-solve the problem or verify the answer yourself. Simply assess the quality of the agent's work based on the trace above. Output ONLY the scores in exactly this format (no other text before the scores):

Correctness: <score>
Reasoning: <score>
Collaboration: <score>
Efficiency: <score>
Completeness: <score>
Total: <sum>
Brief justification: <1-2 sentences>
"""

BBEH_ADDENDUM = """
**Task-specific guidance (BBEH — logical reasoning benchmark):**
- Correctness: Focus on the substantive content of the final answer. Treat semantically equivalent answers as equally correct — superficial formatting differences (parentheses, brackets, quotes, capitalization, trailing punctuation, wrapping text like "The answer is X", unit presence/absence, list delimiter style) must NOT reduce the score. Examples of equivalent pairs: "E" / "(E)" / "Option E"; "4" / "4." / "The answer is 4"; "['Dan', 'Kyle']" / "[Dan, Kyle]" / "Dan, Kyle"; "disproved" / "Disproved" / "DISPROVED". Reasoning that reaches the correct substantive answer should receive full Correctness credit even if presentation differs from the reference.
- Reasoning: Prioritize step-by-step logical deduction. Multi-step chains should be traceable.
- Collaboration: Evaluate whether agents divide subtasks (e.g., parsing, solving, verifying) effectively.
- Efficiency: Penalize excessive re-derivation or circular reasoning that doesn't converge.
- Completeness: The final answer must be fully specified (no ambiguity, no missing elements). Formatting variations are not ambiguity.
"""

WORKBENCH_ADDENDUM = """
**Task-specific guidance (WorkBench — multi-domain tool-use tasks):**
- Correctness: Focus on whether the agent made the RIGHT tool calls with the RIGHT arguments that would achieve the user's goal (correct email_id deleted, correct search performed, correct IDs used). Ignore minor formatting variations in function-call syntax — equivalent invocations that only differ in style (e.g., "email.delete_email.func(email_id=\"00000479\")" vs "email_delete_email(email_id=\"00000479\")") should be scored equally when intent and arguments are correct. Reward verified tool outcomes over narrative claims.
- Reasoning: Evaluate whether the agent correctly identifies which tools to call and in what order.
- Collaboration: Multi-agent workflows should hand off context cleanly across domains (e.g., email → calendar).
- Efficiency: Penalize redundant tool calls, unnecessary reads, or re-doing actions that already succeeded.
- Completeness: All sub-tasks in the query must be addressed — partial execution is heavily penalized.
"""

SWE_ADDENDUM = """
**Task-specific guidance (SWE-bench — software engineering patch generation):**
- Correctness: Score by whether the patch would plausibly make the reported failing behavior pass while preserving unrelated behavior — NOT by textual similarity to any specific gold patch. The SWE-bench oracle is test outcome. Semantically equivalent patches must be scored equally regardless of stylistic choice: variable/function names, import ordering and style, whitespace/blank lines, inline-vs-helper factoring, optional defensive null/type checks, equivalent Python constructs (e.g., `isinstance(x, int)` vs `type(x) == int`), or a fix placed at a slightly different but functionally correct point within the same logical location.
- Reasoning: Evaluate fault-localization quality — does the agent identify the right file(s), the right function/method, and the underlying cause before editing?
- Collaboration: Assess division of labor across the explore / diagnose / fix / verify phases.
- Efficiency: Penalize excessive file reads, shotgun edits to unrelated files, or overly broad patches.
- Completeness: The patch should handle edge cases mentioned in the issue and not leave TODO stubs.
"""


def get_judge_prompt(dataset_name: str = "") -> str:
    """Select the production per-dataset judge prompt.

    Dispatches to the full, self-contained winning prompt for each dataset.
    Unknown datasets fall back to the generic JUDGE_PROMPT_TEMPLATE.
    """
    ds = dataset_name.lower()
    if "bbeh" in ds or "aime" in ds:
        return BBEH_JUDGE_PROMPT
    elif "workbench" in ds:
        return WORKBENCH_JUDGE_PROMPT
    elif "swe" in ds:
        return SWEBENCH_JUDGE_PROMPT
    return JUDGE_PROMPT_TEMPLATE


class LLMAsJudgeEvaluator:
    """
    LLM-as-Judge evaluator for trace-based multi-aspect scoring.

    Scores MAS outputs on 5 aspects (Correctness, Reasoning, Collaboration,
    Efficiency, Completeness), each 0-20, totaling 0-100. Returns the raw
    score in [0, 100] for use as the Metrics term in the reward function.
    """

    def __init__(
        self,
        model_id: str = "bedrock:us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        dataset_name: str = ""
    ):
        """
        Initialize LLM-as-Judge evaluator.

        Args:
            model_id: Model to use for judging
            temperature: Sampling temperature (default: 0.0 for consistency)
            max_tokens: Max tokens for judge response
            dataset_name: Dataset name (reserved for future per-dataset prompts)
        """
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dataset_name = dataset_name
        self.prompt_template = get_judge_prompt(dataset_name)

        try:
            self.model = get_model(
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens
            )
            logger.info(f"Initialized LLM-as-Judge with {model_id} (dataset: {dataset_name or 'generic'})")
        except Exception as e:
            logger.error(f"Failed to initialize LLM-as-Judge: {e}")
            raise

        self.name = "LLM-as-Judge"
        # Store last evaluation details for inspection
        self.last_scores = None

    def evaluate_correctness(
        self,
        output: str,
        ground_truth: Any = None,
        query: Optional[str] = None
    ) -> Optional[float]:
        """
        Evaluate MAS output quality using multi-aspect trace analysis.

        Args:
            output: Agent output / execution trace to evaluate
            ground_truth: Unused (kept for interface compatibility)
            query: Original query/task

        Returns:
            Score in [0, 100], or None on error
        """
        if not output:
            logger.warning("Empty output, returning None")
            return None

        prompt = self.prompt_template.format(
            query=query if query else "[Query not provided]",
            output=output.strip()[:8000],
        )

        try:
            import asyncio
            if asyncio.iscoroutinefunction(self.model.__call__):
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                response = loop.run_until_complete(self.model(prompt))
            else:
                response = self.model(prompt)

            if not isinstance(response, str):
                response = str(response)

            scores = self._parse_scores(response)
            if scores is None:
                logger.warning(f"Could not parse judge scores: {response[:300]}")
                return None

            self.last_scores = scores
            total = sum(scores.values())

            logger.debug(
                f"LLM Judge scores: {scores}, total={total}/100"
            )

            return total

        except Exception as e:
            logger.error(f"LLM-as-Judge evaluation failed: {e}")
            return None

    def _parse_scores(self, response: str) -> Optional[Dict[str, int]]:
        """
        Parse the 5 aspect scores from the judge response.

        Args:
            response: Judge model response text

        Returns:
            Dict mapping aspect name to score, or None if parsing fails.
            Scores are clamped to [0, max_per_aspect] where max is inferred
            from the prompt (default 20, but supports custom scales).
        """
        scores = {}
        for aspect in ASPECTS:
            pattern = rf"{aspect}\s*:\s*(\d+)"
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                scores[aspect] = max(score, 0)  # clamp floor only; total is what matters

        if len(scores) < len(ASPECTS):
            # Try fallback: look for any 5 numbers on separate lines
            numbers = re.findall(r"^\s*\w+\s*:\s*(\d+)\s*$", response, re.MULTILINE)
            if len(numbers) >= len(ASPECTS):
                for i, aspect in enumerate(ASPECTS):
                    if aspect not in scores:
                        scores[aspect] = max(int(numbers[i]), 0)

        if len(scores) == len(ASPECTS):
            return scores

        logger.warning(f"Parsed only {len(scores)}/{len(ASPECTS)} aspects")
        return None

    def evaluate_with_metrics(
        self,
        output: str,
        ground_truth: Any = None,
        query: Optional[str] = None,
        error: str = ""
    ) -> Optional[Dict]:
        """
        Evaluate with comprehensive metrics (compatible with dataset evaluators).

        Args:
            output: Agent output / execution trace
            ground_truth: Unused (kept for interface compatibility)
            query: Original query (optional)
            error: Error message if any

        Returns:
            Dictionary with evaluation metrics
        """
        score = self.evaluate_correctness(output, ground_truth, query)

        return {
            "score": score,
            "correct": score is not None and score >= 50,
            "aspect_scores": self.last_scores,
            "evaluated_by": "llm_as_judge",
            "judge_model": self.model_id,
            "error": error if error else None
        }


def create_llm_judge_evaluator(model_id: str) -> LLMAsJudgeEvaluator:
    """
    Factory function to create LLM-as-Judge evaluator.

    Args:
        model_id: Model ID for the judge

    Returns:
        LLMAsJudgeEvaluator instance
    """
    return LLMAsJudgeEvaluator(model_id=model_id)
