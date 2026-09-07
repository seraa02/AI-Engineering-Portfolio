"""
LangGraph graph definition for the Multi-Agent Research Assistant.

Graph structure:
  START -> planner_node -> supervisor_validate_plan_node ->
    researcher_node (per subquestion, sequential) ->
    supervisor_check_complete_node -> writer_node ->
    supervisor_validate_report_node -> END

Conditional edge from supervisor_check_complete routes back to researcher
if not all subquestions are done.

State is persisted to Redis after every node.
"""

from __future__ import annotations

import logging
from typing import Optional, Literal

import anthropic
import redis as redis_lib
from langgraph.graph import StateGraph, END, START

from src.state import (
    ResearchState,
    save_state,
    get_budget_config,
    get_budget_usage,
    get_subquestions,
    get_findings,
    append_trace,
    add_token_usage,
    increment_search_count,
)
from src.schemas import (
    PlannerInput,
    ResearcherInput,
    WriterInput,
    SupervisorValidatePlanInput,
    SupervisorCheckCompleteInput,
    SupervisorValidateReportInput,
    BudgetUsage,
    Finding,
    utc_now,
)
from src.agents.planner import run_planner
from src.agents.researcher import run_researcher
from src.agents.writer import run_writer
from src.agents.supervisor import (
    supervisor_validate_plan,
    supervisor_check_complete,
    supervisor_validate_report,
)

logger = logging.getLogger(__name__)


class ResearchGraph:
    """
    Encapsulates the LangGraph graph for the research pipeline.
    
    Requires an Anthropic client, Tavily client, and Redis client.
    All these are injected at construction time.
    """

    def __init__(
        self,
        anthropic_client: anthropic.Anthropic,
        tavily_client,
        redis_client: redis_lib.Redis,
        planner_model: str = "claude-haiku-4-5",
        supervisor_model: str = "claude-haiku-4-5",
        writer_model: str = "claude-sonnet-4-6",
        researcher_model: str = "claude-haiku-4-5",
        redis_ttl: int = 86400,
    ):
        self.anthropic_client = anthropic_client
        self.tavily_client = tavily_client
        self.redis_client = redis_client
        self.planner_model = planner_model
        self.supervisor_model = supervisor_model
        self.writer_model = writer_model
        self.researcher_model = researcher_model
        self.redis_ttl = redis_ttl

        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _planner_node(self, state: ResearchState) -> ResearchState:
        """Run the Planner agent."""
        logger.info("Running planner node for run_id=%s", state["run_id"])
        state = dict(state)
        state["status"] = "planning"

        budget_config = get_budget_config(state)
        planner_input = PlannerInput(
            question=state["question"],
            max_subquestions=budget_config.max_subquestions,
        )

        try:
            output, trace_entry = run_planner(
                planner_input=planner_input,
                anthropic_client=self.anthropic_client,
                model=self.planner_model,
            )

            # Update state
            state["subquestions"] = [sq.model_dump() for sq in output.subquestions]
            state = append_trace(state, trace_entry)
            state = add_token_usage(
                state,
                trace_entry.token_usage.get("input_tokens", 0),
                trace_entry.token_usage.get("output_tokens", 0),
            )

        except Exception as exc:
            logger.error("Planner node failed: %s", exc)
            state["status"] = "failed"
            state["error"] = str(exc)

        save_state(self.redis_client, state, self.redis_ttl)
        return ResearchState(**state)

    def _supervisor_validate_plan_node(self, state: ResearchState) -> ResearchState:
        """Supervisor validates the plan and enforces budget."""
        logger.info("Supervisor validating plan for run_id=%s", state["run_id"])
        state = dict(state)

        subquestions = get_subquestions(state)
        budget_config = get_budget_config(state)
        budget_usage = get_budget_usage(state)

        plan_input = SupervisorValidatePlanInput(
            subquestions=subquestions,
            budget_config=budget_config,
            budget_usage=budget_usage,
        )

        output, trace_entry = supervisor_validate_plan(plan_input)

        if output.approved:
            state["subquestions"] = [sq.model_dump() for sq in output.trimmed_subquestions]
            # Update subquestion count in budget usage
            budget_usage.subquestion_count = len(output.trimmed_subquestions)
            state["budget_usage"] = budget_usage.model_dump()
            state["status"] = "researching"
        else:
            state["status"] = "failed"
            state["error"] = output.reason

        state = append_trace(state, trace_entry)
        save_state(self.redis_client, state, self.redis_ttl)
        return ResearchState(**state)

    def _researcher_node(self, state: ResearchState) -> ResearchState:
        """Research the next pending sub-question."""
        logger.info("Researcher node running for run_id=%s", state["run_id"])
        state = dict(state)

        subquestions = get_subquestions(state)
        budget_config = get_budget_config(state)
        budget_usage = get_budget_usage(state)

        # Find next pending subquestion
        target_sq = None
        for sq in subquestions:
            if sq.status in ("pending", "in_progress"):
                searches_done = budget_usage.searches_per_subquestion.get(sq.id, 0)
                if searches_done < budget_config.max_searches_per_subquestion:
                    target_sq = sq
                    break

        if target_sq is None:
            logger.info("No pending subquestions found — skipping researcher node")
            save_state(self.redis_client, state, self.redis_ttl)
            return ResearchState(**state)

        # Mark subquestion as in_progress
        updated_subquestions = []
        for sq in subquestions:
            if sq.id == target_sq.id:
                sq = sq.model_copy(update={"status": "in_progress"})
            updated_subquestions.append(sq)
        state["subquestions"] = [sq.model_dump() for sq in updated_subquestions]

        # Get how many searches already done for this subquestion
        searches_already_done = budget_usage.searches_per_subquestion.get(target_sq.id, 0)

        researcher_input = ResearcherInput(
            subquestion=target_sq,
            max_searches=budget_config.max_searches_per_subquestion,
        )

        output, trace_entry = run_researcher(
            researcher_input=researcher_input,
            anthropic_client=self.anthropic_client,
            tavily_client=self.tavily_client,
            model=self.researcher_model,
            searches_already_done=searches_already_done,
        )

        # Update findings
        findings_map = state["findings"].copy()
        findings_map[output.subquestion_id] = [f.model_dump() for f in output.findings]
        state["findings"] = findings_map

        # Increment search count
        for _ in range(output.searches_performed):
            state = increment_search_count(state, target_sq.id)

        # Mark subquestion as complete
        budget_usage_updated = get_budget_usage(state)
        updated_subquestions2 = []
        for sq_dict in state["subquestions"]:
            from src.schemas import SubQuestion
            sq = SubQuestion(**sq_dict)
            if sq.id == target_sq.id:
                sq = sq.model_copy(update={"status": "complete"})
            updated_subquestions2.append(sq.model_dump())
        state["subquestions"] = updated_subquestions2

        # Update token usage
        state = append_trace(state, trace_entry)
        state = add_token_usage(
            state,
            trace_entry.token_usage.get("input_tokens", 0),
            trace_entry.token_usage.get("output_tokens", 0),
        )

        save_state(self.redis_client, state, self.redis_ttl)
        return ResearchState(**state)

    def _supervisor_check_complete_node(self, state: ResearchState) -> ResearchState:
        """Supervisor checks if all subquestions have been researched."""
        logger.info("Supervisor check complete for run_id=%s", state["run_id"])
        state = dict(state)

        subquestions = get_subquestions(state)
        findings = get_findings(state)
        budget_config = get_budget_config(state)
        budget_usage = get_budget_usage(state)

        check_input = SupervisorCheckCompleteInput(
            subquestions=subquestions,
            findings={k: v for k, v in findings.items()},
            budget_config=budget_config,
            budget_usage=budget_usage,
            created_at=state["created_at"],
        )

        output, trace_entry = supervisor_check_complete(check_input)

        state = append_trace(state, trace_entry)
        save_state(self.redis_client, state, self.redis_ttl)
        return ResearchState(**state)

    def _writer_node(self, state: ResearchState) -> ResearchState:
        """Writer produces the final report."""
        logger.info("Writer node running for run_id=%s", state["run_id"])
        state = dict(state)
        state["status"] = "writing"

        subquestions = get_subquestions(state)
        findings = get_findings(state)

        writer_input = WriterInput(
            question=state["question"],
            subquestions=subquestions,
            findings=findings,
        )

        try:
            output, trace_entry = run_writer(
                writer_input=writer_input,
                anthropic_client=self.anthropic_client,
                model=self.writer_model,
            )

            state["report"] = output.report
            state = append_trace(state, trace_entry)
            state = add_token_usage(
                state,
                trace_entry.token_usage.get("input_tokens", 0),
                trace_entry.token_usage.get("output_tokens", 0),
            )

        except Exception as exc:
            logger.error("Writer node failed: %s", exc)
            state["report"] = f"Report generation failed: {exc}"
            state["status"] = "failed"
            state["error"] = str(exc)

        save_state(self.redis_client, state, self.redis_ttl)
        return ResearchState(**state)

    def _supervisor_validate_report_node(self, state: ResearchState) -> ResearchState:
        """Supervisor validates the final report."""
        logger.info("Supervisor validating report for run_id=%s", state["run_id"])
        state = dict(state)

        findings = get_findings(state)
        report = state.get("report") or ""

        report_input = SupervisorValidateReportInput(
            report=report,
            findings=findings,
        )

        output, trace_entry = supervisor_validate_report(report_input)

        if output.approved:
            state["status"] = "complete"
        else:
            # Report failed validation — still complete but note error
            state["status"] = "complete"
            state["error"] = output.reason

        state = append_trace(state, trace_entry)
        save_state(self.redis_client, state, self.redis_ttl)
        return ResearchState(**state)

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------

    def _route_after_check_complete(self, state: ResearchState) -> Literal["researcher_node", "writer_node"]:
        """Route to researcher if more work needed, else to writer."""
        subquestions = get_subquestions(state)
        budget_config = get_budget_config(state)
        budget_usage = get_budget_usage(state)

        # Check wall clock
        from src.agents.supervisor import _wall_clock_exceeded
        if _wall_clock_exceeded(state["created_at"], budget_config.wall_clock_timeout_seconds):
            logger.info("Routing to writer (wall clock timeout)")
            return "writer_node"

        # Check token budget
        if budget_usage.total_tokens_used >= budget_config.max_total_tokens:
            logger.info("Routing to writer (token budget exhausted)")
            return "writer_node"

        # Check if any subquestion is pending
        for sq in subquestions:
            if sq.status in ("pending", "in_progress"):
                searches_done = budget_usage.searches_per_subquestion.get(sq.id, 0)
                if searches_done < budget_config.max_searches_per_subquestion:
                    logger.info("Routing back to researcher for subquestion %s", sq.id)
                    return "researcher_node"

        logger.info("All subquestions complete — routing to writer")
        return "writer_node"

    def _route_after_validate_plan(self, state: ResearchState) -> Literal["researcher_node", "__end__"]:
        """Route to researcher if plan approved, else end."""
        if state["status"] == "failed":
            return "__end__"
        return "researcher_node"

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        """Build and compile the LangGraph graph."""
        graph = StateGraph(ResearchState)

        # Add nodes
        graph.add_node("planner_node", self._planner_node)
        graph.add_node("supervisor_validate_plan_node", self._supervisor_validate_plan_node)
        graph.add_node("researcher_node", self._researcher_node)
        graph.add_node("supervisor_check_complete_node", self._supervisor_check_complete_node)
        graph.add_node("writer_node", self._writer_node)
        graph.add_node("supervisor_validate_report_node", self._supervisor_validate_report_node)

        # Add edges
        graph.add_edge(START, "planner_node")
        graph.add_edge("planner_node", "supervisor_validate_plan_node")

        graph.add_conditional_edges(
            "supervisor_validate_plan_node",
            self._route_after_validate_plan,
            {
                "researcher_node": "researcher_node",
                "__end__": END,
            },
        )

        graph.add_edge("researcher_node", "supervisor_check_complete_node")

        graph.add_conditional_edges(
            "supervisor_check_complete_node",
            self._route_after_check_complete,
            {
                "researcher_node": "researcher_node",
                "writer_node": "writer_node",
            },
        )

        graph.add_edge("writer_node", "supervisor_validate_report_node")
        graph.add_edge("supervisor_validate_report_node", END)

        return graph.compile()

    def run(self, initial_state: ResearchState) -> ResearchState:
        """Execute the graph synchronously."""
        logger.info("Starting graph execution for run_id=%s", initial_state["run_id"])
        final_state = self._graph.invoke(initial_state)
        logger.info("Graph execution complete for run_id=%s, status=%s", final_state["run_id"], final_state["status"])
        return final_state
