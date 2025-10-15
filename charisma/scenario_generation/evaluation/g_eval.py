from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate
import pandas as pd
from typing import List
from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv
from openai import OpenAI
import os
from logging import getLogger

log = getLogger(__name__)

class OpenRouterLLM(DeepEvalBaseLLM):
    def __init__(self, *, client, model_name: str, provider: str | None = None, max_tokens: int = 1000):
        self._client = client
        self._model_name = model_name
        self._provider = provider
        self._max_tokens = max_tokens

    def load_model(self):
        return self._client

    def get_model_name(self) -> str:
        return f"OpenRouter:{self._model_name}"

    def generate(self, prompt: str) -> str:
        extra_body = {"provider": {"only": [self._provider]}} if self._provider else {}
        resp = self._client.chat.completions.create(
            extra_body=extra_body,
            model=self._model_name,
            # response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._max_tokens,
        )
        return resp.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        return list(map(self.generate, prompts))

def evaluate_scenarios_from_csv(path_to_csv_file: str, save_path: str = None):
    df = pd.read_csv(path_to_csv_file)
    results = []

    for idx, row in df.iterrows():
        scenario = row['scenario']
        scenario_setting = {
            "shared_goal": row['shared_goal'],
            "social_goal_category": row['social_goal_category'],
            "first_agent_goal": row['first_agent_goal'],
            "second_agent_goal": row['second_agent_goal'],
            "first_agent_role": row['agent1_role'],
            "second_agent_role": row['agent2_role']
        }
        test_case = LLMTestCase(
        input=scenario_setting,
        actual_output=scenario
     )
        # result = evaluate([test_case], metrics=[scenario_coherence, scenario_fluency])[0]
        # scenario_coherence.measure(test_case)
        scenario_fluency.measure(test_case)
        score_data = {
        # "coherence_score": scenario_coherence.score,
        # "coherence_reason": scenario_coherence.reason,
        "fluency_score": scenario_fluency.score,
        "fluency_reason": scenario_fluency.reason
        }
        log.info(f"Processed row {idx + 1}/{len(df)}: Fluency: {score_data['fluency_score']}")
        # results.append(score_data)
        output_record = row.to_dict()
        output_record.update(score_data)
        results.append(output_record)
    # Convert results to DataFrame
        pd.DataFrame(results).to_csv(save_path, index=False)

    # Append results to DataFrame
    # results_df = pd.concat([df, pd.DataFrame(results)], axis=1)

    # if save_path:
    #     results_df.to_csv(save_path, index=False)

    # return results_df


if __name__ == "__main__":
    load_dotenv()
    OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=OPEN_ROUTER_API_KEY
    )
    open_router_llm = OpenRouterLLM(client=client, model_name="deepseek/deepseek-chat-v3-0324", provider="deepinfra/fp4")
        
    # coherence_metric = GEval(
    #     model=open_router_llm,
    #     name="Coherence",
    #     criteria="Evaluate the overall coherence and logical flow of the generated output. Score 1‑5.",
    #     # Do not specify expected_output or context
    #     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    # )

    # test = LLMTestCase(
    #     input="Tell me a short story about a space pirate",
    #     actual_output="Captain Zora traveled across the stars. She stole jewels. Then...",
    # )

    # score = coherence_metric.measure(test)
    # print("Score:", score.score)
    # print("Reasoning (CoT):\n", score.reason)
    # scenario_coherence = GEval(
    # name="Scenario-Coherence",
    # evaluation_steps=[
    #     "Check if the 'shared_goal' is clearly instantiated in the scenario.",
    #     "Verify that agent roles and goals match what is described in the input.",
    #     "Assess if the scenario flows logically and consistently with no logical contradictions."
    # ],
    # evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    # model=open_router_llm,
    # threshold=0.5,
    # strict_mode=False
    # )

    scenario_fluency = GEval(
        name="Scenario-Fluency",
        evaluation_steps=[
            "Check if the grammar and sentence structure are correct (e.g., no run-ons, fragments, or tense errors).",
            "Evaluate punctuation and basic formatting (e.g., proper use of commas, periods, line breaks).",
            "Assess whether the text reads smoothly and naturally (i.e., does it flow like native-written English?).",
            "Check if vocabulary is appropriate and not overly repetitive or awkward.",
            "Judge overall clarity: is the scenario easy to understand from start to end?"
        ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=open_router_llm,
        threshold=0.5,
        strict_mode=False
    )
    evaluate_scenarios_from_csv(
        path_to_csv_file="outputs/scenario_evaluation/medium_scenarios_coherence_fluency_scores_final.csv",
        save_path="outputs/scenario_evaluation/medium_scenarios_coherence_fluency_scores_final.csv"
    )