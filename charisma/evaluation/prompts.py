import json
from openai import OpenAI
from charisma.util import extract_json_string


def evaluation_prompt(
    scenario_setting: dict,
    conversation: str,
    client: OpenAI,
    model_name="deepseek/deepseek-chat-v3-0324:free",
    provider=None,
):  # Define the JSON structure for the result
    # User input defining the task
    json_format = {
        "shared_goal_completion_score": "integer (0-10)",
        "reasoning": "string (explanation for the assigned scores)",
        "Agent A": {
            "personal_goal_completion_score": "integer (0-10)",
            "reasoning": "string (explanation for the assigned scores)"
        },
        "Agent B": {
            "personal_goal_completion_score": "integer (0-10)",
            "reasoning": "string (explanation for the assigned scores)"
        }
    }

    user_message = f"""
    ### GOAL: ###
    We aim to explore how personality traits influence behavior in social contexts by simulating psychologically profiled role-playing agents. These agents are placed in interpersonal scenarios involving shared goals. Their behavior should reflect both their individual personality traits and assigned social roles.

    ### TASK: ###
    You should evaluate the following conversation between two agents within a specific scenario. In the following conversation, each agent attempts to achieve:
    - their individual personal goal
    - a shared goal.

    Your task is to assess:
    1. How successfully each agent achieved their personal goal (on a scale from 0 to 10).
    2. How successfully the shared goal was achieved (also on a scale from 0 to 10).

    ### INSTRUCTIONS: ###
    1. Assign a personal goal completion score (0 - 10) to each agent.  
    0 = The agent completely failed to achieve their goal  
    10 = The agent fully achieved their goal

    2. Assign a shared goal completion score (0 - 10) to represent how well the mutual goal was achieved.  
    0 = The shared goal was not achieved at all  
    10 = The shared goal was fully achieved

    3. Provide a brief reasoning for each agent to justify the personal goal score and the shared goal completion score.

    4. All goal completion scores must be integers between 0 and 10.

    ### SCENARIO SETTING: ###
    {scenario_setting}

    ### CONVERSATION: ###
    {conversation}

    ### OUTPUT FORMAT: ###
    Your response must be a JSON object with the following structure:
    json_format = {json_format}
    """
    messages = [
        {"role": "user", "content": user_message},
    ]
    completion = client.chat.completions.create(
        extra_body={"provider": {"only": [provider]}} if provider else {},
        model=model_name,
        response_format={"type": "json_object"},
        messages=messages,
    )
    try:
        return json.loads(completion.choices[0].message.content)
    except json.JSONDecodeError:
        return extract_json_string(completion.choices[0].message.content)