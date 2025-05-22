from typing import Optional, Any, Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from utils.agents.question_generator_agent import DataQuestion
from utils.agents.chart_designer_agent import ChartDesign


class ValidationResult(BaseModel):
    """Model for validation results."""
    is_valid: bool = Field(description="Whether the validation passed")
    error_message: Optional[str] = Field(description="Error message if validation failed", default=None)
    suggestions: Optional[list[str]] = Field(description="Suggestions for improvement if any", default=None)


class ValidatorAgent:
    """Agent for validating questions and chart designs."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        verbose: bool = True
    ):
        """Initialize the validator agent.

        Args:
            model_name: The OpenAI model to use
            temperature: Model temperature (0 for deterministic results)
            verbose: Whether to print agent's reasoning
        """
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.verbose = verbose

    def validate_question(self, question: DataQuestion, df_columns: list[str]) -> ValidationResult:
        """Validate if a question makes sense and can be answered with the available data.

        Args:
            question: The question to validate
            df_columns: List of available columns in the DataFrame

        Returns:
            ValidationResult containing validation status and any error messages
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data analysis expert. Your task is to validate if a question can be answered with the available data.
            
            Consider the following:
            1. Is the question clear and well-formed?
            2. Are all required columns available in the data?
            3. Is the question answerable with the given data types?
            4. Are there any potential issues with the question?
            
            Respond with ONLY the JSON object, no markdown formatting, no quotes, no language identifier:
            {{
                "is_valid": boolean,
                "error_message": string or null,
                "suggestions": array of strings or null
            }}
            """),
            ("user", """Question: {question}
            Required columns: {columns}
            Available columns: {available_columns}
            """)
        ])

        # Format the prompt
        formatted_prompt = prompt.format_messages(
            question=question.question,
            columns=question.data_columns,
            available_columns=df_columns
        )

        # Get validation from the model
        response = self.llm.invoke(formatted_prompt)
        
        try:
            # Parse the response into a ValidationResult
            result = ValidationResult.model_validate_json(response.content)
            return result
        except Exception as e:
            if self.verbose:
                print(f"Error parsing validation result: {str(e)}")
            return ValidationResult(is_valid=False, error_message="Failed to validate question")

    def validate_design(
        self,
        design: ChartDesign,
        question: DataQuestion,
        df_columns: list[str]
    ) -> ValidationResult:
        """Validate if a chart design makes sense for the given question and data.

        Args:
            design: The chart design to validate
            question: The question the chart should answer
            df_columns: List of available columns in the DataFrame

        Returns:
            ValidationResult containing validation status and any error messages
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data visualization expert. Your task is to validate if a chart design is appropriate for the given question and data.
            
            Consider the following:
            1. Is the chart type appropriate for the question and data types?
            2. Are all required columns available in the data?
            3. Does the design make sense for the intended visualization?
            4. Are there any potential issues with the design?
            
            Respond with ONLY the JSON object, no markdown formatting, no quotes, no language identifier:
            {{
                "is_valid": boolean,
                "error_message": string or null,
                "suggestions": array of strings or null
            }}
            """),
            ("user", """Question: {question}
            Required columns: {columns}
            Available columns: {available_columns}
            
            Chart Design:
            - Type: {chart_type}
            - Title: {title}
            - X-axis: {x_axis}
            - Y-axis: {y_axis}
            - Data transformation: {data_transformation}
            - Reasoning: {reasoning}
            """)
        ])

        # Format the prompt
        formatted_prompt = prompt.format_messages(
            question=question.question,
            columns=question.data_columns,
            available_columns=df_columns,
            chart_type=design.chart_type,
            title=design.title,
            x_axis=design.x_axis,
            y_axis=design.y_axis,
            data_transformation=design.data_transformation or "None",
            reasoning=design.reasoning
        )

        # Get validation from the model
        response = self.llm.invoke(formatted_prompt)
        
        try:
            # Parse the response into a ValidationResult
            result = ValidationResult.model_validate_json(response.content)
            return result
        except Exception as e:
            if self.verbose:
                print(f"Error parsing validation result: {str(e)}")
            return ValidationResult(is_valid=False, error_message="Failed to validate chart design") 