from openai import OpenAI
import os
import pandas as pd
from charisma.config import config
from charisma.setup_goals.generation import setup_goals
from charisma.scenario_generation.generation import scenario_generation
from charisma.interaction_generation.generation import interaction_generation
from charisma.util import create_output_path


def run_pipeline(client: OpenAI):
    scen_gen_output_csv_path = create_output_path(
        config.pipeline.input_csv, f"_scenario_generation_{config.pipeline.scenario_generation.difficulty}"
    )
    # setup_goals(
    #     output_csv=config.pipeline.setup_goals.output_csv,
    #     goal_list_csv=config.pipeline.setup_goals.input_csv,
    #     client=client,
    #     model_name=config.pipeline.model,
    #     provider=config.pipeline.provider if config.pipeline.provider else None,
    #     num_records=config.pipeline.setup_goals.num_records,
    # )
    scenario_generation(
        input_csv=config.pipeline.input_csv,
        output_csv=scen_gen_output_csv_path,
        client=client,
        model=config.pipeline.model,
        difficulty=config.pipeline.scenario_generation.difficulty,
        provider=config.pipeline.provider if config.pipeline.provider else None,
        agentic_generation=config.pipeline.scenario_generation.agentic_generation,
    )

def evaluation(client: OpenAI):
    evaluation_output_csv_path = create_output_path(
        config.pipeline.evaluation.input_csv, "_evaluation"
    )
    from charisma.evaluation.generation import evaluation
    evaluation(
        input_csv=config.pipeline.evaluation.input_csv,
        output_csv=evaluation_output_csv_path,
        client=client,
        model=config.pipeline.model,
        provider=config.pipeline.provider if config.pipeline.provider else None,
    )

def run_interaction_pipeline():
    interaction_gen_output_csv_path = create_output_path(
        config.pipeline.interaction_generation.scenarios_filepath, "_interaction_generation"
    )
    interaction_generation(
        charaction_filename=config.pipeline.interaction_generation.characters_filepath,
        scenarios_filename=config.pipeline.interaction_generation.scenarios_filepath,
        output_csv=interaction_gen_output_csv_path,
        model=config.pipeline.model,
        provider=config.pipeline.provider if config.pipeline.provider else None,
        num_samples=config.pipeline.interaction_generation.num_samples if config.pipeline.interaction_generation.num_samples else None,
    )
