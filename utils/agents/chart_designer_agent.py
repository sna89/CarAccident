from typing import List, Optional
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import pandas as pd
from langchain.output_parsers import ResponseSchema, StructuredOutputParser


class ChartDesign(BaseModel):
    """Model for chart design specifications."""
    chart_type: str = Field(description="The most appropriate type of chart (e.g., bar, line, scatter, pie)")
    title: str = Field(description="Clear and descriptive title for the chart")
    x_axis: str = Field(description="What should be represented on the x-axis")
    y_axis: str = Field(description="What should be represented on the y-axis")
    data_transformation: Optional[str] = Field(description="Any necessary data transformations (e.g., aggregation, filtering)")
    reasoning: str = Field(description="Explanation of why this chart design is most appropriate for answering the question")


class ChartDesignerAgent:
    """Agent for determining the best chart design based on the question and data."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        verbose: bool = True
    ):
        """Initialize the chart designer agent.

        Args:
            model_name: The OpenAI model to use
            temperature: Model temperature (0 for deterministic results)
            verbose: Whether to print agent's reasoning
        """
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.verbose = verbose
        
        # Chart design response schemas
        self.design_schemas = [
            ResponseSchema(
                name="chart_design",
                description="""The optimal chart design containing:
                - chart_type: The most appropriate type of chart
                - title: Clear and descriptive title
                - x_axis: What to show on x-axis
                - y_axis: What to show on y-axis
                - data_transformation: Any necessary data transformations
                - reasoning: Why this design is most appropriate"""
            )
        ]
        self.design_parser = StructuredOutputParser.from_response_schemas(self.design_schemas)

    def get_chart_design(self, df: pd.DataFrame, question: str, data_columns: List[str]) -> Optional[ChartDesign]:
        """Determine the best chart design for answering the question.

        Args:
            df: The DataFrame containing the data
            question: The question to answer with the chart
            data_columns: The columns needed to answer the question

        Returns:
            ChartDesign object specifying the optimal chart design, or None if design cannot be determined
        """
        # Create design prompt
        design_prompt = f"""Task: Determine the optimal chart design for answering this question about the data.

        Question to Answer: {question}
        
        Required Columns: {data_columns}
        
        Data Types:
        {df[data_columns].dtypes}
        
        Sample Data:
        {df[data_columns].head(30)}
        
        Guidelines:
        1. Choose ONE of these chart types:
           - Line chart: Use for showing trends over time or continuous data
             Example: Monthly accident counts over a year
             Example: Speed vs. distance relationship
             Example: Temperature changes throughout the day
           
           - Bar chart: Use for comparing categories or discrete data
             Example: Number of accidents by vehicle type
             Example: Average speed by road condition
             Example: Accident count by day of week
           
           - Pie chart: Use for showing proportions of a whole
             Example: Distribution of accident severity levels
             Example: Percentage of accidents by weather condition
             Example: Proportion of accidents by time of day (morning/afternoon/evening/night)
        
        2. Consider:
           - What dimensions need to be shown
           - What transformations might be needed
           - How to make the visualization clear and informative
        3. Design should:
           - Have a clear, descriptive title
           - Use appropriate axis labels
           - Make patterns and insights obvious
        4. Consider data characteristics:
           - Number of categories
           - Time series vs categorical
           - Relationships between variables
           - Distribution of values

        {self.design_parser.get_format_instructions()}

        Remember: Your response must be a valid JSON object that matches the format above exactly."""
        
        # Get design from the model
        response = self.llm.invoke(design_prompt)
        
        # Parse the response
        try:
            parsed_result = self.design_parser.parse(response.content)
            design = parsed_result["chart_design"]
            
            return ChartDesign(
                chart_type=design["chart_type"],
                title=design["title"],
                x_axis=design["x_axis"],
                y_axis=design["y_axis"],
                data_transformation=design.get("data_transformation"),
                reasoning=design["reasoning"]
            )
            
        except Exception as e:
            if self.verbose:
                print(f"Error parsing chart design: {str(e)}")
            return None 