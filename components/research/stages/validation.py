from typing import TypedDict
from research.agents import create_validator_agent
from research.config import WORKFLOW_CONFIG
from research.state import PlanningState

def validate_question(state: PlanningState) -> PlanningState:
    """Validate if the question is related to car accidents."""
    validator = create_validator_agent()
    result = validator.invoke({"input": state["query"]})
    
    if not result.get("is_valid", False):
        if state.get("query_retry_count", 0) < WORKFLOW_CONFIG["max_validation_retries"]:
            state["query_retry_count"] = state.get("query_retry_count", 0) + 1
            state["validation_error"] = result.get("reason", "Unknown validation error")
            print(f"\nInvalid question (Attempt {state['query_retry_count']}/{WORKFLOW_CONFIG['max_validation_retries']}): {result['reason']}")
            print("Please enter a new question related to car accidents:")
            new_question = input("> ").strip()
            state["query"] = new_question
            return state
        else:
            raise ValueError(f"Question validation failed after {WORKFLOW_CONFIG['max_validation_retries']} attempts: {result.get('reason', 'Unknown validation error')}")
    
    state["is_query_valid"] = True
    return state

def should_retry(state: PlanningState) -> str:
    """Determine if we should retry validation or proceed"""
    if state.get("is_query_valid", False):
        return "plan_sections"
    elif state.get("query_retry_count", 0) < WORKFLOW_CONFIG["max_validation_retries"]:
        return "validate_question"
    else:
        print(f"\nMaximum validation attempts ({WORKFLOW_CONFIG['max_validation_retries']}) reached. Please try again with a question related to car accidents.")
        return "end" 