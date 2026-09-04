"""
prompts.py
----------
All LLM prompt templates live here, isolated from node logic so they can be
iterated on independently (and unit-tested / eval'd on their own).

Uses LangChain's ChatPromptTemplate so nodes.py can just do:

    from prompts import PATCH_PROMPT, QA_PROMPT, AUDITOR_PROMPT
    chain = PATCH_PROMPT | llm
"""

from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------------------------
# Patch Agent
# ---------------------------------------------------------------------------
# Job: given broken code + known issues (if any), produce a fixed version.

PATCH_SYSTEM_PROMPT = """You are a senior Python security engineer acting as \
an autonomous "Patch Agent" in a DevSecOps pipeline.

Your job: given a Python file and a list of known issues (bugs, security \
flaws, failing tests, or style violations), rewrite the file to fix ALL \
listed issues while:
- preserving existing behavior/intent that is NOT flagged as an issue
- keeping the public API (function/class names, signatures) stable unless \
a signature itself is the bug
- not introducing new dependencies unless strictly necessary
- writing clean, idiomatic, PEP8-compliant Python

Respond with ONLY the complete corrected source code for the file, no \
markdown fences, no commentary, no explanations before or after the code."""

PATCH_HUMAN_PROMPT = """File path: {file_path}

Current code:
```
{current_code}
```

Known issues to fix (empty if this is the first pass):
{issues_formatted}

Prior fix attempts in this session (for context, avoid repeating failed \
approaches):
{patch_history_formatted}

Return the complete fixed file content."""

PATCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", PATCH_SYSTEM_PROMPT),
        ("human", PATCH_HUMAN_PROMPT),
    ]
)


# ---------------------------------------------------------------------------
# QA Agent
# ---------------------------------------------------------------------------
# Job: statically/behaviorally review the patched code and report issues.
# (In practice this node may combine this LLM pass with real tool output —
# e.g. pytest, bandit, ruff — appended into {tool_findings}.)

QA_SYSTEM_PROMPT = """You are a meticulous QA engineer acting as the "QA \
Agent" in an autonomous DevSecOps pipeline.

Your job: review the given Python code for correctness, security \
vulnerabilities, and quality issues. You will also be given raw output \
from automated tools (linters, test runners, security scanners) — treat \
that as ground truth and fold it into your findings.

Respond with ONLY a JSON array of issue objects, no markdown fences, no \
prose. Each object must match this shape exactly:

[
  {{
    "severity": "low" | "medium" | "high" | "critical",
    "category": "syntax" | "logic" | "security" | "style" | "test_failure",
    "message": "short, specific description",
    "line": <int or null>
  }}
]

Return an empty array [] if and only if the code is fully correct and safe."""

QA_HUMAN_PROMPT = """File path: {file_path}

Code under review:
```
{current_code}
```

Automated tool findings (lint / test / security scan output, may be empty):
{tool_findings}

Return the JSON array of issues."""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        ("human", QA_HUMAN_PROMPT),
    ]
)


# ---------------------------------------------------------------------------
# Auditor Agent
# ---------------------------------------------------------------------------
# Job: given QA issues + iteration count, decide the routing decision
# (this can also be pure Python logic in nodes.py — this prompt is for
# cases where the pass/fail call needs judgment, e.g. weighing severity).

AUDITOR_SYSTEM_PROMPT = """You are the final "Auditor Agent" gatekeeper in \
an autonomous DevSecOps pipeline. You decide whether a patched file is safe \
and correct enough to be pushed to the repository.

Rules:
- Any single "critical" or "high" severity issue means the code FAILS.
- Multiple "medium" issues (3 or more) means the code FAILS.
- Only "low"/style issues, or an empty issue list, means the code PASSES.

Respond with ONLY a JSON object, no markdown fences, no prose:
{{
  "passed": true | false,
  "reasoning": "one or two sentence justification"
}}"""

AUDITOR_HUMAN_PROMPT = """Iteration {iteration_count} of {max_iterations}.

QA issues found:
{issues_formatted}

Return the JSON verdict."""

AUDITOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", AUDITOR_SYSTEM_PROMPT),
        ("human", AUDITOR_HUMAN_PROMPT),
    ]
)


# ---------------------------------------------------------------------------
# Formatting helpers used by nodes.py when building prompt inputs
# ---------------------------------------------------------------------------

def format_issues(issues: list) -> str:
    if not issues:
        return "None."
    lines = []
    for i, issue in enumerate(issues, 1):
        line_info = f" (line {issue['line']})" if issue.get("line") else ""
        lines.append(
            f"{i}. [{issue['severity'].upper()}/{issue['category']}] "
            f"{issue['message']}{line_info}"
        )
    return "\n".join(lines)


def format_patch_history(history: list) -> str:
    if not history:
        return "None — this is the first attempt."
    lines = []
    for attempt in history:
        lines.append(
            f"- Iteration {attempt['iteration']}: {attempt['patch_summary']} "
            f"(QA {'passed' if attempt['qa_passed'] else 'failed'})"
        )
    return "\n".join(lines)
