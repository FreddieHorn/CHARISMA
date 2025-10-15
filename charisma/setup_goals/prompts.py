from charisma.util import extract_json_string
from openai import OpenAI
import json
from logging import getLogger
log = getLogger(__name__)
def choose_goal_category_prompt(base_shared_goal: dict, client: OpenAI, model_name, provider):
    json_format = {
        "explanation": "string",
        "chosen_social_goal_category": "string",
        "first_agent_goal": "string",
        "second_agent_goal": "string",
        "agent1_role": "string",
        "agent2_role": "string"
    }

    user_message = f"""
    ### GOAL: ###
    We aim to investigate attribution theory within the field of social psychology through the simulation of psychologically profiled, role-playing agents. These agents are situated in interpersonal scenarios involving shared and individual goals, enabling the study of how dispositional traits and situational factors interact to shape social behaviour.

    ## TASK: ###
    Your task consists of the following steps:

    1. Select one social goal category from the list below for the following shared goal.
    2. Based on the chosen social goal category and the following shared goal, define a distinct personal goal for each agent that reflects their perspective and motivation.
    3. Personal goals should be defined in a way that allows them to be evaluated later based on how well they are achieved in a conversation between two agents.
    4. Assign each agent a social role that shapes how they interact with the other agent in pursuit of their personal and shared goals.

    ### SHARED GOAL: ###
    {base_shared_goal["Full label"]}

    ### SOCIAL GOAL CATEGORIES: ###
    1. Information Acquisition
    2. Information Provision
    3. Relationship Building
    4. Relationship Maintenance
    5. Identity Recognition
    6. Cooperation
    7. Competition
    8. Conflict Resolution


    ### OUTPUT FORMAT: ###
    Output your response in the following JSON format:
    json_format = {json_format}
    """
    messages = [
        {"role": "user", "content": user_message},
    ]

    completion = client.chat.completions.create(
        extra_body = {
            "provider": {"only": [provider]} 
        } if provider else {},
        model=model_name,
        response_format={
            'type': 'json_object'
        },
        messages=messages,
    )
    try:
        return json.loads(completion.choices[0].message.content)
    except json.JSONDecodeError:
        return extract_json_string(completion.choices[0].message.content)

def choose_goal_category_prompt_app(shared_goal: str, agent1_role: str, agent2_role: str, client: OpenAI, model_name, provider):
    json_format = {
        "explanation": "string",
        "chosen_social_goal_category": "string",
        "first_agent_goal": "string",
        "second_agent_goal": "string",
    }

    user_message = f"""
    ### GOAL: ###
    We aim to explore how personality traits influence behavior in social contexts by simulating psychologically profiled role-playing agents. These agents will be placed in interpersonal situations involving shared goals, where their behavior reflects both their personality traits and social roles.

    ## TASK: ###
    Your task consists of the following steps:

    1. Select one social goal category from the list below for the following shared goal.
    2. Based on the chosen social goal category, the following shared goal and provided agent roles define a distinct personal goal for each agent that reflects their perspective and motivation.
    3. Personal goals should be defined in a way that allows them to be evaluated later based on how well they are achieved in a conversation between two agents.

    ### SHARED GOAL: ###
    {shared_goal}

    ### Agent Roles: ###
    Agent 1 role: {agent1_role}
    Agent 2 role: {agent2_role}

    ### SOCIAL GOAL CATEGORIES: ###
    1. Information Acquisition
    2. Information Provision
    3. Relationship Building
    4. Relationship Maintenance
    5. Identity Recognition
    6. Cooperation
    7. Competition
    8. Conflict Resolution


    ### OUTPUT FORMAT ###
    Output your response in the following JSON format:
    json_format = {json_format}
    """
    messages = [
        {"role": "user", "content": user_message},
    ]

    completion = client.chat.completions.create(
        extra_body = {
            "provider": {"only": [provider]} 
        } if provider else {},
        model=model_name,
        response_format={
            'type': 'json_object'
        },
        messages=messages,
    )
    try:
        return json.loads(completion.choices[0].message.content)
    except json.JSONDecodeError:
        return extract_json_string(completion.choices[0].message.content)