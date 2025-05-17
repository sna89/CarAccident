from typing import Optional, Any, List, Dict
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_experimental.agents.agent_toolkits.python.prompt import PREFIX
from pydantic import BaseModel, Field
import pandas as pd
import plotly.express as px


class ChartReasoning(BaseModel):
    """Model for chart generation reasoning and analysis."""
    data_analysis: str = Field(description="Brief analysis of the data structure and patterns")
    visualization_choice: str = Field(description="Explanation of why this visualization type was chosen")
    data_processing: str = Field(description="Description of any data transformations needed")
    edge_cases: List[str] = Field(description="List of edge cases considered and how they're handled")


class ChartResponse(BaseModel):
    """Model for chart generation response."""
    reasoning: ChartReasoning = Field(description="Analysis and reasoning behind the chart generation")
    code: str = Field(description="The complete, runnable Python code")


def run_chart_code(result: str, df: pd.DataFrame) -> Optional[Any]:
    """Execute chart generation code in a safe environment.
    
    Args:
        result: The chart generation response containing code
        df: The DataFrame to use for chart generation
        
    Returns:
        Optional[Any]: The generated Plotly figure or None if generation fails
    """
    if not result:
        print("Failed to generate chart code")
        return None
    
    try:
        # Parse the response using Pydantic model
        chart_response = ChartResponse.model_validate_json(result)
        chart_code = chart_response.code
        
        # Set up execution environment
        local_vars = {
            'px': px,
            'pd': pd,
            'df': df
        }
        
        # Execute the code
        exec(chart_code, globals(), local_vars)
        
        # Get the figure from the executed code
        fig = local_vars.get('fig')
        if fig is None:
            print("Failed to create figure from generated code")
            return None
            
        return fig
        
    except Exception as e:
        print(f"Error executing chart code: {str(e)}")
        return None


class PythonAgent:
    """A Python agent that can execute Python code using a REPL."""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0,
        verbose: bool = True
    ):
        """Initialize the Python agent.
        
        Args:
            model_name: The OpenAI model to use
            temperature: Model temperature (0 for deterministic, higher for more creative)
            verbose: Whether to print agent's reasoning
        """
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.python_repl_tool = PythonREPLTool()
        self.agent_executor = create_python_agent(
            llm=self.llm,
            tool=self.python_repl_tool,
            verbose=verbose,
            prefix=PREFIX
        )
    
    def execute(self, input_data: str) -> Any:
        """Execute the agent with the given input.
        
        Args:
            input_data: Input data or instructions for the agent
            
        Returns:
            Result of the agent's execution
        """
        try:
            result = self.agent_executor.invoke({"input": input_data})
            return result["output"]
        except Exception as e:
            print(f"Error executing task: {str(e)}")
            return None


def main():
    """Main function to demonstrate using the PythonAgent class."""
    # Instantiate the Python agent
    python_agent = PythonAgent(verbose=True)
    
    # Example usage
    result = python_agent.execute("What is 2 + 2?")
    print(f"Result: {result}")


if __name__ == "__main__":
    main()