import os
import pandas as pd
from logging import getLogger
from charisma.evaluation.prompts import evaluation_prompt

log = getLogger(__name__)

def evaluation(
    input_csv: str,
    output_csv: str,
    client,
    model,
    provider=None,
):
    data = pd.read_csv(input_csv)
    log.info(
        f"Model: {model}, Provider: {provider}"
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
            interaction = row["interaction_history"] 

            result = evaluation_prompt(
                scenario_setting=scenario_setting,
                conversation=interaction,
                client=client,
                model_name=model,
                provider=provider,
            )
            log.info(f"Scores for row {idx}: {result}")
            output_record = row.to_dict()
            output_record["shared_goal_completion_score"] = result["shared_goal_completion_score"]
            output_record["agent1_goal_completion_score"] = result["Agent A"]["personal_goal_completion_score"]
            output_record["agent2_goal_completion_score"] = result["Agent B"]["personal_goal_completion_score"]
            # reasonings for both agents
            output_record["agent1_reasoning"] = result["Agent A"]["reasoning"]
            output_record["agent2_reasoning"] = result["Agent B"]["reasoning"]
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
    
def evaluate_conversation_app(scenario_setting, conversation, client, model, provider=None):
    """
    Evaluate a conversation between two agents based on the scenario setting.
    
    Args:
        scenario_setting (dict): Dictionary containing scenario details.
        conversation (str): The conversation text between the agents.
        client: The OpenAI client instance.
        model (str): The model name to use for evaluation.
        provider (str, optional): The provider name if applicable.
        
    Returns:
        dict: A dictionary containing evaluation results including scores and reasonings.
    """
    result = evaluation_prompt(
        scenario_setting=scenario_setting,
        conversation=conversation,
        client=client,
        model_name=model,
        provider=provider,
    )
    log.info(f"Evaluation result: {result}")
    return result