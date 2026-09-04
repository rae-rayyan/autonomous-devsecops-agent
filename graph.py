"""
graph.py
--------
Wires nodes.py functions into a LangGraph StateGraph:

    patch_agent --> qa_agent --> auditor_agent --(loop)--> patch_agent
                                              \--(pass)--> push_agent --> END
                                               \--(fail cap)--> END

Run with: python graph.py path/to/broken_file.py
"""

from __future__ import annotations

import sys
import logging

from langgraph.graph import StateGraph, END

from state import AgentState, make_initial_state, RepoInfo
from nodes import (
    patch_agent_node,
    qa_agent_node,
    auditor_agent_node,
    push_agent_node,
    route_after_audit,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("patch", patch_agent_node)
    workflow.add_node("qa", qa_agent_node)
    workflow.add_node("audit", auditor_agent_node)
    workflow.add_node("push", push_agent_node)

    workflow.set_entry_point("patch")

    workflow.add_edge("patch", "qa")
    workflow.add_edge("qa", "audit")

    workflow.add_conditional_edges(
        "audit",
        route_after_audit,
        {
            "patch": "patch",       # loop back for another fix attempt
            "push": "push",         # validated -> ship it
            "end_failed": END,      # hit iteration cap, give up
        },
    )

    workflow.add_edge("push", END)

    return workflow.compile()


def run(file_path: str, repo_info: RepoInfo | None = None, max_iterations: int = 5):
    with open(file_path, "r") as f:
        code = f.read()

    initial_state = make_initial_state(
        code=code,
        file_path=file_path,
        repo_info=repo_info,
        max_iterations=max_iterations,
    )

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    print(f"\nFinal status: {final_state['status']}")
    print(f"Iterations used: {final_state['iteration_count']}/{final_state['max_iterations']}")
    if final_state.get("error_message"):
        print(f"Error: {final_state['error_message']}")

    return final_state


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python graph.py <path_to_python_file> [github_repo_url] [branch]")
        sys.exit(1)

    target_file = sys.argv[1]
    repo_info = None
    if len(sys.argv) >= 4:
        repo_info = {
            "repo_url": sys.argv[2],
            "branch": sys.argv[3],
            "file_path": target_file,
        }

    run(target_file, repo_info=repo_info)
