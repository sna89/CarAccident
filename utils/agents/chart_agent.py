import json
from typing import Optional, List, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain.agents import AgentExecutor
from langgraph.prebuilt import create_react_agent
from langchain.tools import BaseTool
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


class PythonAgent:
    """A general-purpose Python agent that can be configured for different use cases."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None,
        available_imports: Optional[Dict[str, Any]] = None,
        temperature: float = 0
    ):
        """Initialize the Python agent.
        
        Args:
            model_name: The OpenAI model to use
            system_prompt: Custom system prompt for the agent
            available_imports: Dictionary of imports available to the agent
            temperature: Model temperature (0 for deterministic, higher for more creative)
        """
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.system_prompt = system_prompt or self._get_default_prompt()
        self.available_imports = available_imports or {}
        self.tools = self.get_tools()
        self.agent = self._create_agent()
    
    def _get_default_prompt(self) -> str:
        """Get the default system prompt."""
        return """You are a Python expert. Your task is to analyze data and perform operations using Python.
        
        You have access to a Python REPL that can execute code. Use it to accomplish the given task.
        
        Follow these guidelines:
        1. First analyze the input data and requirements
        2. Write clear, well-documented Python code
        3. Handle errors and edge cases appropriately
        4. Return results in the expected format
        
        Always explain your reasoning and approach."""
    
    def get_tools(self) -> List[BaseTool]:
        """Get the tools available to the agent.
        
        Returns:
            List of tools that the agent can use
        """
        return [PythonREPLTool()]
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent with its prompt and tools."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent_executor = create_react_agent(
            self.llm, 
            self.tools,
            prompt=prompt
        )
        return agent_executor
    
    def execute(self, input_data: str) -> Any:
        """Execute the agent with the given input.
        
        Args:
            input_data: Input data or instructions for the agent
            
        Returns:
            Result of the agent's execution
        """
        try:
            result = self.agent.invoke({
                "input": input_data
            })
            return result["output"]
        except Exception as e:
            print(f"Error executing agent: {str(e)}")
            return None

class ChartAgent(PythonAgent):
    """A specialized Python agent for creating Plotly visualizations."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """Initialize the chart generation agent.
        
        Args:
            model_name: The OpenAI model to use for chart generation
        """
        # Define available imports for chart creation
        available_imports = {
            'pd': pd,
            'px': px,
            'go': go,
            'json': json
        }
        
        # Define specialized prompt for chart creation
        system_prompt = """Based on the following data, generate Python code to create the most meaningful visualization using Plotly Express.
        The data is provided as a pandas DataFrame named 'df' with the following columns: {columns}

        Data:
        {data}

        Requirements:
        1. Analyze the data structure and choose the most appropriate visualization:
           - For time-based data: line charts showing trends
           - For categorical data: bar charts showing distributions
           - For hierarchical data: treemaps or sunburst charts
           - For relationships: scatter plots or heatmaps
           - For proportions: pie charts or stacked bar charts
           - For distributions: histograms or box plots

        2. Chart Design Guidelines:
           - Title should clearly describe the main insight
           - Use clear, descriptive axis labels
           - Include a legend if multiple categories are shown
           - Use a color scheme that enhances readability
           - Add hover information for detailed data exploration
           - Consider using subplots if multiple insights need to be shown

        3. Data Processing:
           - Aggregate or transform data if needed to show key patterns
           - Sort categories by value if it helps highlight patterns
           - Calculate percentages or ratios if they provide better insights
           - Handle missing or null values appropriately

        4. Code Requirements:
           - Must be valid, runnable Python code
           - Must use the existing DataFrame 'df'
           - Must create a Plotly Express figure and store it in a variable named 'fig'
           - Must include proper error handling
           - Must be complete and executable

        Your response must be a valid JSON string that follows this exact structure:
        {
            "reasoning": {
                "data_analysis": "Brief analysis of the data structure and patterns",
                "visualization_choice": "Explanation of why this visualization type was chosen",
                "data_processing": "Description of any data transformations needed",
                "edge_cases": ["List", "of", "edge", "cases", "as", "strings"]
            },
            "code": "The complete, runnable Python code without any markdown formatting or explanations"
        }"""
        
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            available_imports=available_imports
        )
    
    def generate_chart(self, data: pd.DataFrame) -> Optional[go.Figure]:
        """Generate an appropriate chart based on the input data.
        
        Args:
            data: Pandas DataFrame containing the data to visualize
            
        Returns:
            Plotly figure object or None if chart generation fails
        """
        try:
            # Format the prompt with the data
            prompt = self.system_prompt.format(
                columns=data.columns.tolist(),
                data=data.to_json(orient='records')
            )
            
            # Execute the agent
            result = self.execute(prompt)
            
            if not result:
                print("Failed to generate chart code")
                return None
            
            # Parse the response using Pydantic model
            try:
                chart_response = ChartResponse.parse_raw(result)
                chart_code = chart_response.code
            except Exception as e:
                print(f"Error parsing chart generation response: {str(e)}")
                return None
            
            # Execute the generated code in a safe environment
            local_vars = {'df': data, 'px': px, 'pd': pd}
            exec(chart_code, globals(), local_vars)
            
            # Get the figure from the executed code
            fig = local_vars.get('fig')
            if fig is None:
                print("Failed to create figure from generated code")
                return None
            
            return fig
            
        except Exception as e:
            print(f"Error generating chart: {str(e)}")
            return None 