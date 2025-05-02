from agents import create_planner_agent
from state import PlanningState, Section
from research.config import SECTION_CONFIG

def plan_sections(state: PlanningState) -> PlanningState:
    """Plan the sections for the research."""
    print("\n=== Starting Research Planning ===")
    print(f"Planning sections for question: {state['query']}")
    
    # Create planner agent
    planner = create_planner_agent()
    
    # Get section plan
    result = planner.invoke({
        "input": state["query"],
        "max_sections": SECTION_CONFIG["max_sections"]
    })
    
    # Create sections
    state["sections"] = [
        Section(title=section["title"], description=section["description"])
        for section in result["sections"]
    ]
    
    print("\nPlanned Sections:")
    for section in state["sections"]:
        print(f"- {section.title}: {section.description}")
    
    print("\n=== Research Planning Complete ===")
    return state 