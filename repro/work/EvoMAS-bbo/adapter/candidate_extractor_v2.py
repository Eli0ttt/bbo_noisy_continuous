import ast
import re

def extract_candidate(content: str) -> str:
    def source_with_optimizer(source: str) -> bool:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False
        return any(
            isinstance(node, ast.ClassDef)
            and node.name == "Optimizer"
            and {"__init__", "ask", "tell"}.issubset(
                {method.name for method in node.body if isinstance(method, ast.FunctionDef)}
            )
            for node in tree.body
        )

    def unwrap(source: str) -> str | None:
        source = source.strip() + "\n"
        if source_with_optimizer(source):
            return source
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        embedded = [
            node.value.value.strip() + "\n"
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith("solver_code")
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and source_with_optimizer(node.value.value)
        ]
        if len(embedded) == 1:
            return embedded[0]
        return None

    blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", content, re.I | re.S)
    # Some OpenAI-compatible models use XML-style code tags rather than
    # Markdown fences. Treat those as containers, never as Python source.
    blocks.extend(re.findall(r"<code>\s*(.*?)\s*</code>", content, re.I | re.S))
    for block in sorted(blocks, key=len, reverse=True):
        candidate = unwrap(block)
        if candidate is not None:
            return candidate
    # Smolagents may return executable source directly as its final answer.
    # Preserve support for the un-fenced form instead of treating it as a run failure.
    raw = content.strip()
    raw = re.sub(r"^Final answer:\s*", "", raw, flags=re.I).strip()
    candidate = unwrap(raw)
    if candidate is not None:
        return candidate
    raise ValueError("No top-level Optimizer candidate or unambiguous solver_code string in final report")

