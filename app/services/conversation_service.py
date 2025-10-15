import time
import streamlit as st
from typing import Generator, Dict, List
from app.services.llm_service import CharismaService
import random

class ConversationService:
    """Handles real-time conversation simulation"""
    
    @staticmethod
    def simulate_typing_effect(text: str, speed: float = 0.02) -> Generator[str, None, None]:
        """
        Simulate typing effect by yielding text character by character
        
        Args:
            text: The text to display
            speed: Delay between characters in seconds
        """
        displayed_text = ""
        for char in text:
            displayed_text += char
            yield displayed_text
            time.sleep(speed)
    
    @staticmethod
    def simulate_thinking_delay(min_delay: float = 1.0, max_delay: float = 3.0) -> None:
        """
        Simulate thinking time before agent responds
        
        Args:
            min_delay: Minimum thinking time in seconds
            max_delay: Maximum thinking time in seconds
        """
        thinking_time = random.uniform(min_delay, max_delay)
        time.sleep(thinking_time)
    
    def generate_real_time_conversation(
        self, 
        llm_service: CharismaService, 
        agent1_name: str, 
        agent2_name: str, 
        scenario_input: dict, 
        scenario_payload: dict,
        typing_speed: float = 0.01,
        thinking_min_delay: float = 0.5,
        thinking_max_delay: float = 2.0
    ) -> Generator[Dict, None, None]:
        """
        Generate conversation with real-time effects
        
        Args:
            ai_service: The AI service instance
            agent1_name: Name of first agent
            agent2_name: Name of second agent  
            scenario_input: Scenario configuration
            scenario_payload: Scenario context
            typing_speed: Speed of typing animation
            thinking_min_delay: Minimum thinking time
            thinking_max_delay: Maximum thinking time
        """
        # Get the conversation generator
        gen = llm_service.generate_conversation(
            agent1_name,
            agent2_name,
            scenario_input,
            scenario_payload,
            max_turns=st.session_state.number_of_turns
        )
        first_message = True
        for msg in gen:
            speaker = msg["agent"]
            text = msg["response"]
            turn = msg.get("turn", 0)
            
            # Simulate thinking before response
            if not first_message:
                self.simulate_thinking_delay(thinking_min_delay, thinking_max_delay)
            first_message = False
            # Yield message with typing effect simulation
            yield {
                "agent": speaker,
                "response": text,
                "behavioral_code": msg["behavioral_code"],
                "explanation": msg["explanation"],
                "turn": turn,
                "typing_effect": True,
                "typing_speed": typing_speed
            }