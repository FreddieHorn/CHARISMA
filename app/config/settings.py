import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

OPEN_ROUTER_API_KEY = st.secrets["OPEN_ROUTER_API_KEY"]
MODEL_NAME = "openai/gpt-5" 
PROVIDER = "openai/default"

MODEL_CONFIG = {
    "deepseek/deepseek-chat-v3-0324": {
        "provider": "deepinfra/fp4",
        "display_name": "DeepSeek Chat v3-0324"
    },
    "deepseek/deepseek-chat-v3.1": {
        "provider": "deepinfra/fp4", 
        "display_name": "DeepSeek Chat v3.1"
    },
    "openai/gpt-5": {
        "provider": "openai/default",
        "display_name": "GPT-5"
    }
}

SOCIAL_GOAL_CATEGORIES = [
    "Information Acquisition",
    "Information Provision",
    "Relationship Building",
    "Relationship Maintenance",
    "Identity Recognition",
    "Cooperation",
    "Competition",
    "Conflict Resolution"
]

PAGE_CONFIG = {
    "page_title": "CHARISMA",
    "page_icon": "🤖",
    "layout": "wide"
}