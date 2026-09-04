"""
nodes.py
--------
One function per graph node. Every node has the signature:

    def node(state: AgentState) -> dict

and returns only the keys it wants to update (LangGraph merges this into
the running state). Keep nodes side-effect-light except where the job IS
the side effect (e.g. push_node running git commands).
"""

from __future__ import annotations

import json
import logging
import subprocess

from langchain_openai import ChatOpenAI  # swap for your provider of choice

from state import AgentState, QAIssue, PatchAttempt
from prompts import (
    PATCH_PROMPT,
    QA_PROMPT,
    AUDITOR_PROMPT,
    format_issues,
    format_patch_history,
)

logger = logging.getLogger(__name__)

# Centralize model config so it's easy to swap models per-node if needed.
llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ---------------------------------------------------------------------------
# Patch Agent
# ---------------------------------------------------------------------------

def patch_agent_node(state: AgentState) -> dict:
    """Generate (or regenerate) a fix for the current code."""
    logger.info("patch_agent_node: iteration %s", state["iteration_count"])

    chain = PATCH_PROMPT | llm
    response = chain.invoke(
        {
            "file_path": state["file_path"],
            "current_code": state["current_code"],
            "issues_formatted": format_issues(state["qa_issues"]),
            "patch_history_formatted": format_patch_history(state["patch_history"]),
        }
    )

    fixed_code = _strip_code_fences(response.content)

    return {
        "current_code": fixed_code,
        "status": "patching",
    }


# ---------------------------------------------------------------------------
# QA Agent
# ---------------------------------------------------------------------------

def qa_agent_node(state: AgentState) -> dict:
    """Run automated tooling + LLM review, produce structured qa_issues."""
    logger.info("qa_agent_node: iteration %s", state["iteration_count"])

    tool_findings = _run_static_tools(state["current_code"], state["file_path"])

    chain = QA_PROMPT | llm
    response = chain.invoke(
        {
            "file_path": state["file_path"],
            "current_code": state["current_code"],
            "tool_findings": tool_findings or "None.",
        }
    )

    issues: list[QAIssue] = _safe_json_parse(response.content, default=[])

    return {
        "qa_issues": issues,
        "status": "validating",
    }


# ---------------------------------------------------------------------------
# Auditor Agent
# ---------------------------------------------------------------------------

def auditor_agent_node(state: AgentState) -> dict:
    """Decide pass/fail, record history, bump iteration count."""
    logger.info("auditor_agent_node: iteration %s", state["iteration_count"])

    chain = AUDITOR_PROMPT | llm
    response = chain.invoke(
        {
            "iteration_count": state["iteration_count"],
            "max_iterations": state["max_iterations"],
            "issues_formatted": format_issues(state["qa_issues"]),
        }
    )

    verdict = _safe_json_parse(response.content, default={"passed": False, "reasoning": "parse_error"})
    qa_passed = bool(verdict.get("passed", False))

    attempt: PatchAttempt = {
        "iteration": state["iteration_count"],
        "code_before": state["original_code"] if not state["patch_history"] else state["patch_history"][-1]["code_after"],
        "code_after": state["current_code"],
        "patch_summary": verdict.get("reasoning", ""),
        "qa_passed": qa_passed,
        "issues_found": state["qa_issues"],
    }

    new_iteration_count = state["iteration_count"] + 1
    hit_cap = new_iteration_count >= state["max_iterations"]

    if qa_passed:
        new_status = "validated"
    elif hit_cap:
        new_status = "failed"
    else:
        new_status = "pending"  # routes back to patch_agent

    return {
        "qa_passed": qa_passed,
        "iteration_count": new_iteration_count,
        "patch_history": state["patch_history"] + [attempt],
        "status": new_status,
        "error_message": None if qa_passed or not hit_cap else "Max iterations reached without passing QA.",
    }


def route_after_audit(state: AgentState) -> str:
    """Conditional-edge function used by graph.py to pick the next node."""
    if state["status"] == "validated":
        return "push"
    if state["status"] == "failed":
        return "end_failed"
    return "patch"  # loop back


# ---------------------------------------------------------------------------
# Push Agent
# ---------------------------------------------------------------------------

def push_agent_node(state: AgentState) -> dict:
    """Commit and push the validated fix to the target repo/branch."""
    logger.info("push_agent_node: pushing validated fix")

    repo_info = state.get("repo_info")
    if not repo_info:
        return {"status": "failed", "error_message": "No repo_info provided; cannot push."}

    try:
        file_path = repo_info["file_path"]
        with open(file_path, "w") as f:
            f.write(state["current_code"])

        commit_message = repo_info.get(
            "commit_message", "fix: autonomous devsecops agent patch"
        )
        branch = repo_info["branch"]

        subprocess.run(["git", "checkout", "-B", branch], check=True)
        subprocess.run(["git", "add", file_path], check=True)
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "push", "origin", branch], check=True)

        return {"status": "pushed"}
    except subprocess.CalledProcessError as e:
        return {"status": "failed", "error_message": f"git push failed: {e}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def _safe_json_parse(text: str, default):
    text = _strip_code_fences(text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON from LLM response: %r", text[:200])
        return default


def _run_static_tools(code: str, file_path: str) -> str:
    """
    Placeholder for real tool integration (ruff, bandit, pytest, mypy...).
    Wire this up to member 2/3's tooling once available. Should return a
    human-readable string summary to feed into the QA prompt.
    """
    return ""
