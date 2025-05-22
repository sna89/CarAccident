from typing import Optional, Any, Iterator
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import pandas as pd
import plotly.express as px
from utils.agents.chart_designer_agent import ChartDesign
from utils.agents.question_generator_agent import DataQuestion
from langchain.prompts import ChatPromptTemplate
import json


class ChartCode(BaseModel):
    """Model for chart generation code and explanation."""
    code: str = Field(description="The complete, runnable Python code for generating the chart")
    explanation: str = Field(description="Explanation of how the code works and handles the data")


class ChartCodeGeneratorAgent:
    """Agent for generating and executing Python code to create charts based on design specifications."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        verbose: bool = True
    ):
        """Initialize the chart code generator agent.

        Args:
            model_name: The OpenAI model to use
            temperature: Model temperature (0 for deterministic results)
            verbose: Whether to print agent's reasoning
        """
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.verbose = verbose

    def _get_system_message(self) -> str:
        """Get the system message for the LLM."""
        return """You are a Python code generation expert specializing in data visualization.
        Your task is to generate Python code to create a chart based on the given specifications.

        Follow these steps in your reasoning:
        1. Analyze the requirements and data
        2. Plan the data preparation steps
        3. Plan the chart creation steps
        4. Generate the code
        5. Validate the code

        For each step, explain your reasoning before taking action.
        """

    def generate_chart(self, df: pd.DataFrame, question: DataQuestion, design: ChartDesign) -> Optional[Any]:
        """Generate and execute Python code to create the chart based on the design.

        Args:
            df: The DataFrame containing the data
            question: The question to answer with the chart
            design: The chart design specifications

        Returns:
            The generated Plotly figure if successful, None otherwise
        """
        # Create code generation prompt
        code_prompt = f"""Task: Generate Python code to create a chart that follows these specifications.

        Question to Answer: {question.question}
        
        Chart Design:
        - Type: {design.chart_type}
        - Title: {design.title}
        - X-axis: {design.x_axis}
        - Y-axis: {design.y_axis}
        - Data transformation: {design.data_transformation if design.data_transformation else 'None'}
        
        Design Reasoning: {design.reasoning}
        
        Required Columns: {question.data_columns}
        
        Data Types:
        {df[question.data_columns].dtypes}
        
        Sample Data:
        {df[question.data_columns]}
        
        Code Generation Steps:
        1. Data Preparation:
           - Start with the provided DataFrame 'df'
           - Select only the required columns from question.data_columns
           - Handle any missing values if needed
           - Apply any data transformations specified in design.data_transformation

        2. Data Validation:
           - Verify all required columns exist
           - Check data types are appropriate for the chart type
           - Ensure data is not empty
           - Validate any grouping or aggregation columns

        3. Chart Creation:
           - Use plotly.express (px) to create the chart
           - Set the chart type according to design.chart_type
           - Configure x and y axes based on design.x_axis and design.y_axis
           - Add the title from design.title
           - Apply any necessary styling or formatting

        4. Error Handling:
           - Add try-except blocks for data operations
           - Handle potential errors in data transformation
           - Validate the final figure before returning

        Requirements:
        1. Use plotly.express (px) for creating the chart
        2. Use the DataFrame 'df' directly
        3. Follow the chart design specifications exactly
        4. Handle any necessary data transformations
        5. Store the result in a variable named 'fig'
        6. Make sure the code is complete and runnable
        7. Include appropriate error handling
        8. DO NOT include any show() or display() commands
        9. Just create and return the figure object

        Respond with ONLY the JSON object, no markdown formatting, no quotes, no language identifier:
        {{
            "code": string,  // The complete, runnable Python code for generating the chart
            "explanation": string  // Brief explanation of how the code works
        }}
        """
        
        # Get code from the LLM
        messages = [
            ("system", self._get_system_message()),
            ("user", code_prompt)
        ]
        response = self.llm.invoke(messages)
        
        if not response or not response.content:
            if self.verbose:
                print("No response from LLM")
            return None
            
        try:
            # Parse the response using the ChartCode model
            chart_code = ChartCode.model_validate_json(response.content)
            if not chart_code.code:
                if self.verbose:
                    print("No code was generated")
                return None
                
            # Execute the code and get the figure
            local_vars = {'df': df, 'px': px}
            exec(chart_code.code, globals(), local_vars)
            
            # Get the figure from the local namespace
            fig = local_vars.get('fig')
            if fig is None:
                if self.verbose:
                    print("No figure was created by the code")
                return None
                
            return fig
            
        except Exception as e:
            if self.verbose:
                print(f"Error executing chart code: {str(e)}")
            return None 