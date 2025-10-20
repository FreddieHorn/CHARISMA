import pandas as pd
from typing import Dict, List
from collections import Counter
from logging import getLogger
log = getLogger(__name__)

class BehavioralAnalysisService:
    def __init__(self, behavioral_codes_df: pd.DataFrame, config: dict):
        self.behavioral_codes_df = behavioral_codes_df
        self.config = config
    
    def analyze_behavioral_patterns(self, conversation_behavioral_codes: List[Dict]) -> Dict:
        """Analyze behavioral patterns from conversation"""
        # Count frequency of each behavioral code
        code_counts = Counter([code['behavioral_code'] for code in conversation_behavioral_codes])
        
        # Analyze by agent
        agent1_codes = [code for code in conversation_behavioral_codes if code['speaker'] == self.config.get('agent1_name', 'Agent 1')]
        agent2_codes = [code for code in conversation_behavioral_codes if code['speaker'] == self.config.get('agent2_name', 'Agent 2')]

        log.info(f"Agent 1 codes: {agent1_codes}")
        log.info(f"Agent 2 codes: {agent2_codes}")
        
        # Count codes for each agent
        agent1_code_counts = Counter([code['behavioral_code'] for code in agent1_codes])
        agent2_code_counts = Counter([code['behavioral_code'] for code in agent2_codes])

        return {
            'overall_codes': code_counts,
            'agent1_codes': agent1_code_counts,
            'agent2_codes': agent2_code_counts,
            'total_interactions': len(conversation_behavioral_codes),
            'agent1_interactions': len(agent1_codes),
            'agent2_interactions': len(agent2_codes)
        }
    
    def get_personality_traits_for_codes(self, behavioral_codes: List[str]) -> Dict[str, List[str]]:
        """Get personality traits for given behavioral codes"""
        traits_mapping = {}
        for code in behavioral_codes:
            code_data = self.behavioral_codes_df[self.behavioral_codes_df['Behaviour Code'] == code]
            if not code_data.empty:
                traits = code_data['Personality Trait Links'].iloc[0]
                traits_mapping[code] = traits.split(", ") if isinstance(traits, str) else []
        return traits_mapping
    
    def get_personality_traits_by_agent(self, agent_codes: Dict) -> Dict[str, Dict]:
        """Get personality traits breakdown by agent"""
        agent_traits = {}
        
        for agent_name, code_counts in agent_codes.items():
            trait_counter = {}
            for code, count in code_counts.items():
                code_data = self.behavioral_codes_df[self.behavioral_codes_df['Behaviour Code'] == code]
                if not code_data.empty:
                    traits = code_data['Personality Trait Links'].iloc[0]
                    if isinstance(traits, str):
                        for trait in traits.split(", "):
                            trait_counter[trait] = trait_counter.get(trait, 0) + count
            
            # Get top 3 traits for this agent
            top_traits = sorted(trait_counter.items(), key=lambda x: x[1], reverse=True)[:3]
            agent_traits[agent_name] = {
                'all_traits': trait_counter,
                'top_traits': top_traits
            }
        
        return agent_traits
    
    def get_behavior_type_breakdown_by_agent(self, conversation_behavioral_codes: List[Dict]) -> Dict[str, Dict]:
        """Break down behavior by type of act for each agent"""
        agent_type_counts = {
            self.config.get('agent1_name', 'Agent 1'): {},
            self.config.get('agent2_name', 'Agent 2'): {}
        }
        
        for code_data in conversation_behavioral_codes:
            speaker = code_data['speaker']
            code = code_data['behavioral_code']
            behavior_data = self.behavioral_codes_df[self.behavioral_codes_df['Behaviour Code'] == code]
            
            if not behavior_data.empty:
                behavior_type = behavior_data['Type of Act'].iloc[0]
                if speaker in agent_type_counts:
                    agent_type_counts[speaker][behavior_type] = agent_type_counts[speaker].get(behavior_type, 0) + 1
        
        return agent_type_counts