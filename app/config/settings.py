import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

OPEN_ROUTER_API_KEY = st.secrets["OPEN_ROUTER_API_KEY"]
MODEL_NAME = "openai/gpt-5" 
PROVIDER = "openai/default"

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