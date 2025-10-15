import streamlit as st
import pandas as pd

def render_conversation_tab(config: dict):
    """Render the conversation tab with synchronized behavioral codes"""
    if not st.session_state.generate_conversation:
        st.info("Accept the scenario first to view the conversation.")
        return
    
    st.markdown(f'<h2 class="section-header">Conversation between {config["agent1_name"]} and {config["agent2_name"]}</h2>', unsafe_allow_html=True)

    # Display conversation with behavioral codes in synchronized rows
    _display_conversation_with_codes(config)

def _display_conversation_with_codes(config: dict):
    """Display conversation with behavioral codes in synchronized columns"""
    for i, message in enumerate(st.session_state.chat_messages):
        # Create columns for each message row (same as in workflow tab)
        message_chat_col, message_code_col = st.columns([2, 1], vertical_alignment="center")
        
        # Display message in chat column
        with message_chat_col:
            with st.chat_message(message["role"], avatar=message.get("avatar", "🤖")):
                st.markdown(f"**{message['speaker']}**: {message['content']}")
        
        # Display corresponding behavioral code in code column
        with message_code_col:
            if i < len(st.session_state.behavioral_codes):
                code_data = st.session_state.behavioral_codes[i]
                _display_single_behavioral_code_compact(
                    code_data["speaker"], 
                    code_data["behavioral_code"], 
                    config
                )
            else:
                # Fallback if no behavioral code exists
                st.info("No behavioral code")

def _display_single_behavioral_code_compact(speaker: str, behavioral_code: str, config: dict):
    """Display a single behavioral code in compact format"""
    # Determine styling based on agent
    if speaker == config['agent1_name']:
        badge_color = "primary"
        icon = "🤖"
    else:
        badge_color = "secondary" 
        icon = "👨‍💼"
    
    # Simple display in chat message format
    with st.container(vertical_alignment="center"):
        if badge_color == "primary":
            st.markdown(f"**{icon} {config['agent1_name']}**: **{behavioral_code}**")
        else:
            st.markdown(f"**{icon} {config['agent2_name']}**: **:yellow[{behavioral_code}]**")