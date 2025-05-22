from typing import List, Optional
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import pandas as pd
from langchain.output_parsers import ResponseSchema, StructuredOutputParser


class DataQuestion(BaseModel):
    """Model for data-related questions."""
    question: str = Field(description="An interesting question that can be answered by analyzing the data")
    reasoning: str = Field(description="Explanation of why this question is valuable and what insights it might reveal")
    data_columns: List[str] = Field(description="Columns needed to answer this question")
    importance: str = Field(description="Level of importance (High/Medium/Low) based on potential business or analytical value")


class QuestionGeneratorAgent:
    """Agent for generating interesting questions from data."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,  # Deterministic for consistent results
        verbose: bool = True
    ):
        """Initialize the question generator agent.

        Args:
            model_name: The OpenAI model to use
            temperature: Model temperature (0 for deterministic results)
            verbose: Whether to print agent's reasoning
        """
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.verbose = verbose
        
        # Question generation response schemas
        self.question_schemas = [
            ResponseSchema(
                name="questions",
                description="""List of valuable questions about the data, each containing:
                - question: The actual question
                - reasoning: Why this question is valuable and what insights it might reveal
                - data_columns: Columns needed to answer this question
                - importance: Level of importance (High/Medium/Low) based on potential value"""
            )
        ]
        self.question_parser = StructuredOutputParser.from_response_schemas(self.question_schemas)

    def get_most_important_question(self, df: pd.DataFrame) -> Optional[DataQuestion]:
        """Get the most important question for visualization.

        Args:
            df: The DataFrame to analyze

        Returns:
            The most important DataQuestion object, or None if no questions could be generated
        """
        # Create question generation prompt
        question_prompt = f"""Task: Generate exactly 3 of the most valuable questions that can be answered by analyzing this data.
        Focus on questions that would reveal the most meaningful insights or patterns.

        Guidelines:
        1. Questions should be specific and answerable with the available data
        2. Consider different aspects: trends, patterns, relationships, anomalies
        3. Questions should be clear and well-formed
        4. Each question should specify which columns are needed to answer it
        5. Focus on questions that could provide the most valuable business or analytical insights
        6. Consider the potential impact of answering each question
        7. Prioritize questions that could lead to actionable insights
        8. Ensure at least one question is marked as High importance

        DataFrame Info:
        {df.info()}

        Sample Data:
        {df.head(20)}

        {self.question_parser.get_format_instructions()}

        Remember: Your response must be a valid JSON object that matches the format above exactly."""
        
        # Get questions from the model
        response = self.llm.invoke(question_prompt)
        
        # Parse the response
        try:
            parsed_result = self.question_parser.parse(response.content)
            questions = []
            
            for q in parsed_result["questions"]:
                questions.append(DataQuestion(
                    question=q["question"],
                    reasoning=q["reasoning"],
                    data_columns=q["data_columns"],
                    importance=q["importance"]
                ))
            
            # Return the first high importance question, or the first question if none are high importance
            for question in questions:
                if question.importance.lower() == "high":
                    return question
            return questions[0] if questions else None
            
        except Exception as e:
            if self.verbose:
                print(f"Error parsing questions: {str(e)}")
            return None 