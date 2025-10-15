from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
from charisma.pipeline import run_pipeline
from charisma.scenario_generation.evaluation.emotion_intensity import process_scenarios_csv
from charisma.scenario_generation.evaluation.entailment_scores import calculate_entailment_scores
from charisma.pipeline import evaluation

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    load_dotenv()
    OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=OPEN_ROUTER_API_KEY
    )
    # calculate_entailment_scores("outputs/goals_deepseek__scenario_generation_medium.csv", "outputs/scenario_evaluation/entailment/entailment_scores_medium.csv")
    run_pipeline(client)
    # evaluation(client)
    # process_scenarios_csv(
    #     input_csv_path='outputs/goals_deepseek__scenario_generation_medium.csv',
    #     output_csv_path='outputs/scenario_evaluation/medium_scenarios_2.csv',
    #     lexicon_path='inputs/NRC-Emotion-Intensity-Lexicon-v1.txt'
    # )
    # process_scenarios_csv(
    #     input_csv_path='outputs/goals_deepseek__scenario_generation_Hard.csv',
    #     output_csv_path='outputs/scenario_evaluation/hard_scenarios_2.csv',
    #     lexicon_path='inputs/NRC-Emotion-Intensity-Lexicon-v1.txt'
    # )