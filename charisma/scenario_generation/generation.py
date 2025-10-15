import os
import pandas as pd
from logging import getLogger
from charisma.scenario_generation.prompts import scenario_creation_prompt

log = getLogger(__name__)


def scenario_generation(
    input_csv: str,
    output_csv: str,
    client,
    model,
    difficulty="Medium",
    provider=None,
    agentic_generation: bool = False,
):
    # Load the input data
    data = pd.read_csv(input_csv)
    log.info(
        f"Model: {model}, Provider: {provider}, Difficulty: {difficulty}, Agentic Generation: {agentic_generation}"
    )
    if os.path.exists(output_csv):
        result_df = pd.read_csv(output_csv)
        results = result_df.to_dict("records")
        existing_indices = set(result_df.index)
        log.info(f"Resuming from existing file with {len(results)} records")
    else:
        pd.DataFrame(columns=data.columns.tolist() + ["scenario"]).to_csv(
            output_csv, index=False
        )
        results = []
        existing_indices = set()

    log.info(f"Generating Scenarios for difficulty {difficulty}")

    if not agentic_generation:
        for idx, row in data.iterrows():
            if idx in existing_indices:
                log.info(f"Skipping already processed row {idx}")
                continue

            try:
                scenario_setting = {
                    "shared_goal": row["shared_goal"],
                    "chosen_goal_category": row["social_goal_category"],
                    "first_agent_goal": row["first_agent_goal"],
                    "second_agent_goal": row["second_agent_goal"],
                    "first_agent_role": row["agent1_role"],
                    "second_agent_role": row["agent2_role"],
                }

                result = scenario_creation_prompt(
                    scenario_setting=scenario_setting,
                    difficulty=difficulty,
                    client=client,
                    model_name=model,
                    provider=provider,
                )

                output_record = row.to_dict()
                output_record["scenario"] = result
                results.append(output_record)

                pd.DataFrame(results).to_csv(output_csv, index=False)
                log.info(f"Saved row {idx + 1}/{len(data)} to {output_csv}")
                log.info(f"Scenario: {result}")

            except Exception as e:
                log.error(f"Error processing row {idx}: {str(e)}")
                if results:  # Only save if we have some results
                    pd.DataFrame(results).to_csv(output_csv, index=False)
                    log.info(f"Saved progress up to row {idx} before error")
                raise

        log.info(f"Successfully processed all {len(data)} rows to {output_csv}")
    else:
        # TODO migrate agentic generation
        log.warning("Agentic generation is not yet implemented.")
        pass

def interpolate_scenario(
    scenario_setting: dict,
    scenario_difficulty: str,
    client,
    model_name: str,
    provider: str = None,
):
    result_dict = scenario_creation_prompt(
        scenario_setting=scenario_setting,
        difficulty=scenario_difficulty,
        client=client,
        model_name=model_name,
        provider=provider)
    return result_dict