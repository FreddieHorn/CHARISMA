import streamlit as st
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
import os

@dataclass
class Character:
    name: str
    description: str
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float
    source: str = "csv"

class PersonalityService:
    def __init__(self, csv_path: str = "inputs/characters_database.csv"):
        self.csv_path = csv_path
        self.characters = self._load_characters_from_csv()
    
    def _load_characters_from_csv(self) -> List[Character]:
        """Load characters from CSV file with MBTI profiles and personality traits"""
        try:
            if not os.path.exists(self.csv_path):
                st.warning(f"Character CSV file not found at {self.csv_path}. Using default characters.")
                return self._get_default_characters()
            
            df = pd.read_csv(self.csv_path)
            
            characters = []
            for _, row in df.iterrows():
                character = Character(
                    name=str(row.get('mbti_profile', 'Unknown Character')),
                    description=self._generate_description_from_traits(row),
                    openness=float(row.get('Openness', 0.5)),
                    conscientiousness=float(row.get('Conscientiousness', 0.5)),
                    extraversion=float(row.get('Extraversion', 0.5)),
                    agreeableness=float(row.get('Agreeableness', 0.5)),
                    neuroticism=float(row.get('Neuroticism', 0.5)),
                    source="csv"
                )
                characters.append(character)
            
            st.success(f"✅ Loaded {len(characters)} characters from the character database.")
            return characters
            
        except Exception as e:
            st.error(f"❌ Error loading character CSV: {e}")
            return self._get_default_characters()
    
    def _generate_description_from_traits(self, row: pd.Series) -> str:
        """Generate a description based on personality traits"""
        traits = []
        
        openness = float(row.get('Openness', 0.5))
        conscientiousness = float(row.get('Conscientiousness', 0.5))
        extraversion = float(row.get('Extraversion', 0.5))
        agreeableness = float(row.get('Agreeableness', 0.5))
        neuroticism = float(row.get('Neuroticism', 0.5))
        
        # Generate description based on trait combinations
        if openness > 0.7:
            traits.append("creative")
        elif openness < 0.3:
            traits.append("practical")
            
        if conscientiousness > 0.7:
            traits.append("organized")
        elif conscientiousness < 0.3:
            traits.append("spontaneous")
            
        if extraversion > 0.7:
            traits.append("outgoing")
        elif extraversion < 0.3:
            traits.append("reserved")
            
        if agreeableness > 0.7:
            traits.append("cooperative")
        elif agreeableness < 0.3:
            traits.append("assertive")
            
        if neuroticism > 0.7:
            traits.append("sensitive")
        elif neuroticism < 0.3:
            traits.append("calm")
        
        if traits:
            description = f"{', '.join(traits)} individual"
        else:
            description = "balanced personality"
            
        return description.capitalize()
    
    def _get_default_characters(self) -> List[Character]:
        """Provide default characters if CSV loading fails"""
        return [
            Character("Sherlock Holmes", "Analytical and observant detective", 0.9, 0.8, 0.4, 0.3, 0.6),
            Character("Elizabeth Bennet", "Witty and independent-minded", 0.7, 0.6, 0.5, 0.8, 0.4),
            Character("Tony Stark", "Charismatic and innovative inventor", 0.9, 0.7, 0.9, 0.5, 0.6),
            Character("Hermione Granger", "Intelligent and diligent student", 0.8, 0.9, 0.5, 0.7, 0.4),
        ]
    
    def filter_characters_by_personality(
        self, 
        openness_range: Tuple[float, float],
        conscientiousness_range: Tuple[float, float],
        extraversion_range: Tuple[float, float],
        agreeableness_range: Tuple[float, float],
        neuroticism_range: Tuple[float, float]
    ) -> List[Character]:
        """Filter characters based on personality trait ranges"""
        filtered_chars = []
        
        for character in self.characters:
            if (openness_range[0] <= character.openness <= openness_range[1] and
                conscientiousness_range[0] <= character.conscientiousness <= conscientiousness_range[1] and
                extraversion_range[0] <= character.extraversion <= extraversion_range[1] and
                agreeableness_range[0] <= character.agreeableness <= agreeableness_range[1] and
                neuroticism_range[0] <= character.neuroticism <= neuroticism_range[1]):
                filtered_chars.append(character)
        
        return filtered_chars
    
    def get_personality_description(self, trait: str, value: float) -> str:
        """Get descriptive text for personality trait values"""
        descriptions = {
            'openness': {
                (0.0, 0.3): "Practical, conventional",
                (0.3, 0.7): "Balanced curiosity",
                (0.7, 1.0): "Creative, imaginative"
            },
            'conscientiousness': {
                (0.0, 0.3): "Spontaneous, flexible",
                (0.3, 0.7): "Reliable, organized",
                (0.7, 1.0): "Disciplined, careful"
            },
            'extraversion': {
                (0.0, 0.3): "Reserved, thoughtful",
                (0.3, 0.7): "Sociable, balanced",
                (0.7, 1.0): "Energetic, outgoing"
            },
            'agreeableness': {
                (0.0, 0.3): "Competitive, skeptical",
                (0.3, 0.7): "Cooperative, kind",
                (0.7, 1.0): "Compassionate, trusting"
            },
            'neuroticism': {
                (0.0, 0.3): "Resilient, calm",
                (0.3, 0.7): "Sensitive, moody",
                (0.7, 1.0): "Anxious, emotional"
            }
        }
        
        for range_val, desc in descriptions[trait].items():
            if range_val[0] <= value <= range_val[1]:
                return desc
        return "Unknown"