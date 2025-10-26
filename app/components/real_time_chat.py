import streamlit as st
import time
from app.services.conversation_service import ConversationService
from app.services.llm_service import CharismaService
from logging import getLogger
import time
log = getLogger(__name__)

def render_real_time_chat(llm_service: CharismaService, config: dict, typing_speed: float = 0.01, thinking_delay: float = 1.0):
    """Render real-time chat interface"""
    conversation_service = ConversationService()
    
    # Initialize conversation in session state if not exists
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display existing messages with their behavioral codes in synchronized rows
    _display_existing_messages_with_codes(config)

    # Generate new conversation if not finished
    if not st.session_state.conversation_finished and st.session_state.generate_conversation:
        _generate_conversation_stream(conversation_service, llm_service, config, typing_speed=typing_speed, thinking_delay=thinking_delay)


def _display_existing_messages_with_codes(config: dict):
    """Display existing chat messages with their behavioral codes in synchronized rows"""
    for i, message in enumerate(st.session_state.chat_messages):
        # Create columns for each message row
        message_chat_col, message_code_col = st.columns([2, 1], vertical_alignment="center")
        
        # Display message in chat column
        with message_chat_col:
            with st.chat_message(message["role"], avatar=message.get("avatar", "🤖")):
                st.html(f"<span class='chat-{message["role"]}'></span>")
                st.markdown(message["content"])
        
        # Display corresponding behavioral code in code column
        with message_code_col:
            if i < len(st.session_state.behavioral_codes):
                code_data = st.session_state.behavioral_codes[i]
                _display_single_behavioral_code_compact(
                    code_data["speaker"], 
                    code_data["behavioral_code"], 
                    config, 
                )
                
def _display_single_behavioral_code_compact(speaker: str, behavioral_code: str, config: dict):
    """Display a single behavioral code in compact format"""
    # Determine styling based on agent
    if speaker == config['agent1_name']:
        badge_color = "primary"
        role = "user"
        icon = "🤖"
    else:
        badge_color = "secondary" 
        role = "assistant"
        icon = "👨‍💼"
    
    # Simple display - no text length calculations needed
    with st.container(vertical_alignment="center"):
        if badge_color == "primary":
            st.markdown(f"**{icon} {config['agent1_name']}**: **{behavioral_code}**")
        else:
            st.markdown(f"**{icon} {config['agent2_name']}**: **:yellow[{behavioral_code}]**")

def _generate_conversation_stream(conversation_service: ConversationService, llm_service: CharismaService, config: dict, typing_speed: float = 0.01, thinking_delay: float = 1.0):
    """Generate and stream conversation in real-time with synchronized columns per message"""
    
    scenario_input = { 
        "shared_goal": config['shared_goal'],
        "social_goal_category": st.session_state.goal_setup_data['chosen_social_goal_category'],
        "first_agent_goal": st.session_state.goal_setup_data['first_agent_goal'],
        "second_agent_goal": st.session_state.goal_setup_data['second_agent_goal'],
        "agent1_role": config['agent1_role'],
        "agent2_role": config['agent2_role'],
    }
    scenario_payload = {"scenario_context": st.session_state.scenario_content}

    # Get conversation generator
    conversation_gen = conversation_service.generate_real_time_conversation(
        llm_service=llm_service,
        agent1_name=config['agent1_name'],
        agent2_name=config['agent2_name'], 
        scenario_input=scenario_input,
        scenario_payload=scenario_payload,
        typing_speed=typing_speed,
        thinking_min_delay=thinking_delay-0.3 if thinking_delay > 0.3 else 0.1,
        thinking_max_delay=thinking_delay + 0.3,
    )

    # Process each message
    for message_data in conversation_gen:
        speaker = message_data["agent"]
        full_text = message_data["response"]
        behavioral_code = message_data["behavioral_code"]
        explanation = message_data["explanation"]
        
        # Determine message role and avatar
        if speaker == config['agent1_name']:
            role = "user"
            avatar = "🤖"
        else:
            role = "assistant" 
            avatar = "👨‍💼"
        
        # Create NEW columns for this specific message
        message_chat_col, message_code_col = st.columns([2, 1], vertical_alignment="center")
                # Display behavioral code in the code column (same row)
        with message_code_col:
            # Use your existing compact display function
            _display_single_behavioral_code_compact(speaker, behavioral_code, config)
            time.sleep(0.5) # Small delay to ensure code appears before message
            
        # Display message in the chat column
        with message_chat_col:
            with st.chat_message(role, avatar=avatar):
                st.html(f"<span class='chat-{role}'></span>")
                message_placeholder = st.empty()
                
                # Type out message character by character
                displayed_text = ""
                for char in full_text:
                    displayed_text += char
                    message_placeholder.write(displayed_text + "▊")
                    time.sleep(typing_speed)
                
                # Remove cursor and show final text
                message_placeholder.write(displayed_text)
        
        
        # Store message
        st.session_state.chat_messages.append({
            "role": role,
            "content": full_text,
            "avatar": avatar,
            "speaker": speaker
        })
        
        # Also store in conversation rows for evaluation
        st.session_state.conversation_rows.append({
            "speaker": speaker,
            "message": full_text,
            "turn": message_data.get("turn", 0)
        })
        
        st.session_state.behavioral_codes.append({
            "speaker": speaker,
            "behavioral_code": behavioral_code,
            "turn": message_data.get("turn", 0),
            "explanation": explanation,
        })
    
    st.session_state.conversation_finished = True