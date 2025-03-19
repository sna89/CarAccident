import os
import random
from typing import Tuple, List

import pandas as pd
from langchain_openai import ChatOpenAI

from tqdm import tqdm

from langchain.docstore.document import Document
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage

import config
from rag.document_processor import DocumentProcessor
import openai
from dotenv import load_dotenv

from rag.evaluation.prompts import QA_generation_prompt, question_groundedness_critique_prompt, \
    question_relevance_critique_prompt, question_standalone_critique_prompt, RAG_PROMPT_TEMPLATE, EVALUATION_PROMPT
from rag.rag_config import RAG_CONFIG
from rag.rag_main import RAG
from rag.vector_store_helper import VectorStoreHelper
from utils.sql_db import SqlDb
from config import LLM_RAG_EVAL_MODEL


class RagEvaluator:
    # based on https://huggingface.co/learn/cookbook/rag_evaluation

    def __init__(self,
                 rag: RAG = None,
                 vector_store_helper: VectorStoreHelper = None,
                 path=None,
                 num_questions=10,
                 provider=None,
                 llm_model_name=None,
                 llm_judge_model=None,
                 llm_judge_name="GPT4",
                 sql_db=None):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.rag = rag
        self.vector_store_helper = vector_store_helper

        self.path = path
        self.num_questions = num_questions
        self.document_processor = DocumentProcessor()
        self.llm_model_name = llm_model_name or config.get("llm_model_name", "")
        self.llm_judge_model = llm_judge_model or ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
        self.llm_judge_name = llm_judge_name
        self.provider = provider or "openai"
        self.sql_db = sql_db or SqlDb()

    def _process_data(self):
        docs_processed = self.document_processor.process_documents(self.path)
        return docs_processed

    def call_llm(self, prompt: str):
        if self.provider.lower() == "openai":
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self.llm_model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
            )
            return response.choices[0].message.content
        else:
            return None

    def create_synthetic_evaluation_dataset(self, evaluation_table_name="rag_evaluation_dataset", version=1):
        docs_processed = self._process_data()

        raw_eval_dataset = self._generate_raw_eval_dataset(docs_processed)
        evaluated_raw_eval_dataset = self._evaluate_raw_questions(raw_eval_dataset)

        evaluated_raw_eval_dataset = pd.DataFrame(evaluated_raw_eval_dataset)
        evaluated_eval_dataset = self._filter_dataset(evaluated_raw_eval_dataset)
        evaluated_eval_dataset["version"] = version

        self.sql_db.upload_table_from_pandas_df(evaluation_table_name, evaluated_eval_dataset, "replace")

    def evaluate_dataset(self,
                         evaluation_table_name="rag_evaluation_dataset",
                         version=1,
                         result_table_name="rag_evaluation_result"):
        engine = self.sql_db.engine
        eval_df = pd.read_sql_query(f"select * from {evaluation_table_name} where version={version}", engine)
        rag_tests_output = self._run_rag_tests(eval_df)
        faithfulness_score = self._evaluate_faithfulness(rag_tests_output)
        evaluation_result_df = pd.DataFrame([{"Eval Dataset Version": version,
                                              "Metric": "Faithfulness",
                                              "Eval Score": faithfulness_score}])
        self.sql_db.upload_table_from_pandas_df(result_table_name, evaluation_result_df)

    @staticmethod
    def _filter_dataset(eval_dataset, threshold=3):
        eval_dataset = eval_dataset[
            (eval_dataset["groundedness_score"] >= threshold) &
            (eval_dataset["relevance_score"] >= threshold) &
            (eval_dataset["standalone_score"] >= threshold)
            ]
        return eval_dataset

    def _generate_raw_eval_dataset(self, docs_processed):
        raw_eval_dataset = []
        for sampled_context in tqdm(random.sample(docs_processed, self.num_questions)):

            output_qa_couple = self.call_llm(QA_generation_prompt.format(context=sampled_context.page_content))
            try:
                question = output_qa_couple.split("Factoid question: ")[-1].split("Answer: ")[0]
                answer = output_qa_couple.split("Answer: ")[-1]
                assert len(answer) < 300, "Answer is too long"
                raw_eval_dataset.append(
                    {
                        "context": sampled_context.page_content,
                        "question": question,
                        "answer": answer,
                        "source_doc": sampled_context.metadata["source"],
                    }
                )
            except:
                continue

        return raw_eval_dataset

    def _evaluate_raw_questions(self, raw_eval_dataset):
        for data in tqdm(raw_eval_dataset):
            evaluations = {
                "groundedness": self.call_llm(
                    question_groundedness_critique_prompt.format(context=data["context"],
                                                                 question=data["question"]),
                ),
                "relevance": self.call_llm(
                    question_relevance_critique_prompt.format(question=data["question"]),
                ),
                "standalone": self.call_llm(
                    question_standalone_critique_prompt.format(question=data["question"]),
                ),
            }
            try:
                for criterion, evaluation in evaluations.items():
                    score, eval = (
                        int(evaluation.split("Total rating: ")[-1].strip()),
                        evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
                    )
                    data.update(
                        {
                            f"{criterion}_score": score,
                            f"{criterion}_eval": eval,
                        }
                    )
            except Exception as e:
                continue

        return raw_eval_dataset

    def _run_rag_tests(self, eval_dataset: pd.DataFrame) -> pd.DataFrame:
        outputs = []
        for idx, example in tqdm(eval_dataset.iterrows()):
            question = example["question"]

            answer, relevant_docs = self._answer_question(question)
            result = {
                "question": question,
                "true_answer": example["answer"],
                "source_doc": example["source_doc"],
                "generated_answer": answer,
                "retrieved_docs": [doc for doc in relevant_docs],
            }
            outputs.append(result)
        outputs = pd.DataFrame(outputs)
        return outputs

    def _answer_question(
            self,
            question: str,
            num_retrieved_docs: int = 3,
    ) -> Tuple[str, List[Document]]:

        retriever = self.vector_store_helper.vector_store
        relevant_docs = retriever.similarity_search(query=question, k=num_retrieved_docs)
        relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text

        context = "\nExtracted documents:\n"
        context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

        final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

        answer = self.rag.chat(final_prompt)

        return answer, relevant_docs

    def _evaluate_faithfulness(self, evaluation_answers_df: pd.DataFrame) -> float:

        evaluation_prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="You are a fair evaluator language model."),
                HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT),
            ]
        )

        output_eval_list = []
        total_score = 0

        for idx, experiment in tqdm(evaluation_answers_df.iterrows()):
            eval_prompt = evaluation_prompt_template.format_messages(
                instruction=experiment["question"],
                response=experiment["generated_answer"],
                reference_answer=experiment["true_answer"],
            )
            eval_result = self.llm_judge_model.invoke(eval_prompt)
            feedback, score = [item.strip() for item in eval_result.content.split("[RESULT]")]
            experiment[f"eval_score_{self.llm_judge_name}"] = score
            experiment[f"eval_feedback_{self.llm_judge_name}"] = feedback
            output_eval_list.append(experiment)
            total_score += int(score)

        total_score /= evaluation_answers_df.shape[0]
        return total_score


if __name__ == "__main__":
    path = r"C:\Users\sna89\PycharmProjects\car_accident_app\data\evaluation"
    config = RAG_CONFIG
    config["llm_model_name"] = LLM_RAG_EVAL_MODEL
    rag_eval = RagEvaluator(RAG(config),
                            VectorStoreHelper(),
                            path,
                            250)
    # rag_eval.create_synthetic_evaluation_dataset()
    rag_eval.evaluate_dataset()
