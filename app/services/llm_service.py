from logging import getLogger
from openai import OpenAI, RateLimitError

from app.config.settings import OPEN_ROUTER_API_KEY, MODEL_NAME, PROVIDER
from app.util import replace_agent_names
from charisma.setup_goals.generation import interpolate_goals
from charisma.scenario_generation.generation import interpolate_scenario
from charisma.interaction_generation.generation import interaction_generation_app
from charisma.evaluation.generation import evaluate_conversation_app

import time
from typing import Optional

log = getLogger(__name__)

class CharismaService:
    """Handles all LLM-related operations"""
    
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPEN_ROUTER_API_KEY
        )
        self.model_name = MODEL_NAME
        self.provider = PROVIDER
    
    def setup_goals(self, shared_goal: str, agent1_role: str, agent2_role: str, max_retries = 3) -> dict:
        """Setup goals for the scenario"""
        for attempt in range(max_retries):
            try:
                goals = interpolate_goals(
                    shared_goal=shared_goal,
                    agent1_role=agent1_role,
                    agent2_role=agent2_role,
                    client=self.client,
                    model_name=self.model_name,
                    provider=self.provider
                )
                return goals
            except RateLimitError:
                log.warning(f"Rate limit exceeded, retrying... (attempt {attempt + 1})")
                time.sleep(2)

        log.error("Failed to setup goals after 3 attempts")
        return {}

    def generate_scenario(self, goal_setup_data: dict, scenario_difficulty: str, agent1_name: str, agent2_name: str, max_retries = 3) -> tuple:
        """Generate scenario context"""
        for attempt in range(max_retries):
            try:
                scenario_data = interpolate_scenario(
                    scenario_setting=goal_setup_data,
                    scenario_difficulty=scenario_difficulty,
                    client=self.client,
                    model_name=self.model_name,
                    provider=self.provider
                )
                break
            except RateLimitError:
                log.warning(f"Rate limit exceeded, retrying... (attempt {attempt + 1})")
                time.sleep(2)
        else:
            log.error("Failed to generate scenario after 3 attempts")
            return None, None
        
        scenario_content = replace_agent_names(
            scenario_data['scenario_context'],
            agent1_name,
            agent2_name
        )
        
        return scenario_data, scenario_content

    def generate_conversation(self, agent1_name: str, agent2_name: str, scenario_input: dict, scenario_payload: dict, max_retries: int = 3, max_turns: int = 20) -> Optional[str]:
        """Generate conversation between agents"""

        for attempt in range(max_retries):
            try:
                return interaction_generation_app(
                    self.model_name,
                    self.provider,
                    agent1_name,
                    agent2_name,
                    scenario_input,
                    scenario_payload,
                    max_turns=max_turns
                )
            except RateLimitError:
                log.warning(f"Rate limit exceeded, retrying... (attempt {attempt + 1})")
                time.sleep(2)

        log.error("Failed to generate conversation after 3 attempts")
        return None
    
    def evaluate_conversation(self, scenario_setting: dict, conversation: list, max_retries: int = 3) -> Optional[dict]:
        """Evaluate conversation with retry logic for rate limits"""
        for attempt in range(max_retries):
            try:
                return evaluate_conversation_app(
                    scenario_setting=scenario_setting,
                    conversation=conversation,
                    client=self.client,
                    model=self.model_name,
                    provider=self.provider
                )
            except RateLimitError:
                log.warning(f"Rate limit exceeded, retrying... (attempt {attempt + 1})")
                time.sleep(2)
        
        log.error("Failed to evaluate conversation after 3 attempts")
        return None