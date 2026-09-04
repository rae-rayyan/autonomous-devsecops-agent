"""
state.py
--------
Defines the shared state object (AgentState) that flows through every node
in the LangGraph state machine. Every node reads from and returns a
(partial) AgentState dict.

Keep this file dependency-light: it should not import from nodes.py or
graph.py to avoid circular imports.
"""

from __future__ import annotations

from typing import TypedDict, List, Optional, Literal
from typing_extensions import NotRequired


# ---------------------------------------------------------------------------
# Supporting structures
# ---------------------------------------------------------------------------

class QAIssue(TypedDict):
    """A single problem found by the QA/auditor step."""
    severity: Literal["low", "medium", "high", "critical"]
    category: Literal["syntax", "logic", "security", "style", "test_failure"]
    message: str
    line: NotRequired[Optional[int]]


class PatchAttempt(TypedDict):
    """A single fix attempt, kept for history / debugging / LLM context."""
    iteration: int
    code_before: str
    code_after: str
    patch_summary: str
    qa_passed: bool
    issues_found: List[QAIssue]


class RepoInfo(TypedDict):
    """Everything needed to push the final validated code to GitHub."""
    repo_url: str
    branch: str
    file_path: str
    commit_message: NotRequired[str]
    base_branch: NotRequired[str]


# ---------------------------------------------------------------------------
# Main graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # --- input ---
    original_code: str
    file_path: str

    # --- working copy, mutated across the patch/QA loop ---
    current_code: str

    # --- QA / audit results ---
    qa_issues: List[QAIssue]
    qa_passed: bool

    # --- loop control ---
    iteration_count: int
    max_iterations: int

    # --- history for traceability / prompting with prior context ---
    patch_history: List[PatchAttempt]

    # --- overall pipeline status ---
    status: Literal[
        "pending",
        "patching",
        "validating",
        "validated",
        "failed",
        "pushing",
        "pushed",
    ]

    # --- push metadata (used only once validated) ---
    repo_info: NotRequired[RepoInfo]

    # --- free-text notes/errors surfaced to the user ---
    error_message: NotRequired[Optional[str]]


def make_initial_state(
    code: str,
    file_path: str,
    repo_info: Optional[RepoInfo] = None,
    max_iterations: int = 5,
) -> AgentState:
    """Convenience factory so nodes/graph don't hand-roll the initial dict."""
    state: AgentState = {
        "original_code": code,
        "file_path": file_path,
        "current_code": code,
        "qa_issues": [],
        "qa_passed": False,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "patch_history": [],
        "status": "pending",
        "error_message": None,
    }
    if repo_info is not None:
        state["repo_info"] = repo_info
    return state
