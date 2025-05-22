from typing import Dict, TypedDict, Annotated, Sequence, Optional, Any
import pandas as pd
from langgraph.graph import Graph, StateGraph, END
from langgraph.prebuilt import ToolNode
from utils.agents.question_generator_agent import QuestionGeneratorAgent, DataQuestion
from utils.agents.chart_designer_agent import ChartDesignerAgent, ChartDesign
from utils.agents.chart_code_generator_agent import ChartCodeGeneratorAgent
from utils.agents.validator_agent import ValidatorAgent


class AnalysisState(TypedDict):
    """State for the analysis workflow."""
    df: pd.DataFrame
    question: DataQuestion | None
    design: ChartDesign | None
    fig: Any | None
    error: str | None
    retry_count: int  # Track number of retries for each step
    max_retries: int  # Maximum number of retries allowed


def create_analysis_graph() -> Graph:
    """Create the analysis workflow graph.

    Returns:
        Graph: The configured analysis workflow graph
    """
    # Initialize agents
    question_agent = QuestionGeneratorAgent(verbose=True)
    designer_agent = ChartDesignerAgent(verbose=True)
    code_agent = ChartCodeGeneratorAgent(verbose=True)
    validator_agent = ValidatorAgent(verbose=True)

    # Define the nodes
    def generate_question(state: AnalysisState) -> AnalysisState:
        """Generate the most important question to analyze."""
        try:
            question = question_agent.get_most_important_question(state["df"])
            if not question:
                return dict(state, error="Could not determine what to analyze in the data")
            return dict(state, question=question, error=None, retry_count=0)
        except Exception as e:
            return dict(state, error=f"Error generating question: {str(e)}")

    def validate_question(state: AnalysisState) -> AnalysisState:
        """Validate the generated question using the validator agent."""

            
        # Get validation result from the validator agent
        validation = validator_agent.validate_question(
            state["question"],
            state["df"].columns.tolist()
        )
        
        if not validation.is_valid:
            error_msg = validation.error_message or "Invalid question generated"
            if validation.suggestions:
                error_msg += f"\nSuggestions: {', '.join(validation.suggestions)}"
            return dict(
                state,
                error=error_msg,
                retry_count=state.get("retry_count", 0) + 1
            )
            
        return dict(state)  # Return complete state on success

    def generate_design(state: AnalysisState) -> AnalysisState:
        """Generate the chart design."""
        try:
            design = designer_agent.get_chart_design(
                state["df"],
                state["question"].question,
                state["question"].data_columns
            )
            if not design:
                return dict(state, error="Could not determine how to visualize the data")
            return dict(state, design=design, error=None, retry_count=0)
        except Exception as e:
            return dict(state, error=f"Error generating design: {str(e)}")

    def validate_design(state: AnalysisState) -> AnalysisState:
        """Validate the generated chart design using the validator agent."""            
        # Get validation result from the validator agent
        validation = validator_agent.validate_design(
            state["design"],
            state["question"],
            state["df"].columns.tolist()
        )
        
        if not validation.is_valid:
            error_msg = validation.error_message or "Invalid chart design generated"
            if validation.suggestions:
                error_msg += f"\nSuggestions: {', '.join(validation.suggestions)}"
            return dict(
                state,
                error=error_msg,
                retry_count=state.get("retry_count", 0) + 1
            )
            
        return dict(state)  # Return complete state on success

    def generate_chart(state: AnalysisState) -> AnalysisState:
        """Generate and execute the chart code."""
        try:
            fig = code_agent.generate_chart(
                state["df"],
                state["question"],
                state["design"]
            )
            if not fig:
                return dict(state, error="Could not generate the visualization")
            return dict(state, fig=fig, error=None)
        except Exception as e:
            return dict(state, error=f"Error generating chart: {str(e)}")

    def should_retry_question(state: AnalysisState) -> str:
        """Determine if we should retry question generation."""
        if state.get("retry_count") < state.get("max_retries"):
            if state.get("error"):
                return "generate_question"
            else:
                return "generate_design"
        return "end"

    def should_retry_design(state: AnalysisState) -> str:
        """Determine if we should retry design generation."""
        if state.get("retry_count") < state.get("max_retries"):
            if state.get("error"):
                return "generate_design"
            else:
                return "generate_chart"
        return "end"

    def should_retry_chart(state: AnalysisState) -> str:
        """Determine if we should retry chart generation."""
        if state.get("retry_count") < state.get("max_retries"):
            if state.get("error"):
                return "generate_chart"
            else:
                return "end"
        return "end"

    # Create the graph
    workflow = StateGraph(AnalysisState)

    # Add nodes
    workflow.add_node("generate_question", generate_question)
    workflow.add_node("validate_question", validate_question)
    workflow.add_node("generate_design", generate_design)
    workflow.add_node("validate_design", validate_design)
    workflow.add_node("generate_chart", generate_chart)

    # Add edges with conditions
    # Question generation and validation
    workflow.add_edge("generate_question", "validate_question")
    
    workflow.add_conditional_edges(
        "validate_question",
        should_retry_question,
        {
            "generate_question": "generate_question",  # Retry question generation
            "generate_design": "generate_design",      # Move to design
            "end": END                                # End workflow if max retries reached
        }
    )

    # Design generation and validation
    workflow.add_edge("generate_design", "validate_design")
    
    workflow.add_conditional_edges(
        "validate_design",
        should_retry_design,
        {
            "generate_design": "generate_design",  # Retry design generation
            "generate_chart": "generate_chart",    # Move to chart generation
            "end": END                            # End workflow if max retries reached
        }
    )

    # Chart generation
    workflow.add_conditional_edges(
        "generate_chart",
        should_retry_chart,
        {
            "generate_chart": "generate_chart",  # Retry chart generation
            "end": END                          # End workflow
        }
    )

    # Set entry and exit points
    workflow.set_entry_point("generate_question")
    workflow.set_finish_point("generate_chart")

    return workflow.compile()