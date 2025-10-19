import streamlit as st
from app.util import load_goals
from app.services.char_selection_service import PersonalityService
from app.config.settings import MODEL_CONFIG
def render_sidebar():
    """Render the sidebar configuration"""
    sidebar_disabled = st.session_state.simulation_started
    if st.session_state.simulation_started:
        st.sidebar.warning("Simulation in progress. To modify settings, please start a new simulation.")
    
    with st.sidebar:
        st.header("Configuration")
        
        st.markdown("### 🤖 Model Selection")
        model_display_names = [config["display_name"] for config in MODEL_CONFIG.values()]
        selected_display_name = st.selectbox(
            "AI Model",
            options=model_display_names,
            index=1,  # Default to DeepSeek Chat v3.1
            help="Select the AI model to use for scenario generation and conversation",
            disabled=sidebar_disabled,
            key="model_selection"
        )
        
        # Get the actual model name and provider from the selected display name
        selected_model = None
        selected_provider = None
        for model_name, config in MODEL_CONFIG.items():
            if config["display_name"] == selected_display_name:
                selected_model = model_name
                selected_provider = config["provider"]
                break
            
        # Character Selection Method
        st.markdown("### 👥 Agent Selection Method")
        selection_method = st.radio(
            "Choose selection method:",
            ["🎯 Personality-Based", "✏️ Custom Names"],
            help="Select agents by personality traits or enter custom names",
            disabled=sidebar_disabled,
            horizontal=True,
            key="selection_method"
        )
        
        # Get agent names based on selection method
        if selection_method == "🎯 Personality-Based":
            agent1_name, agent2_name = _render_personality_based_selection(sidebar_disabled)
        else:
            agent1_name, agent2_name = _render_custom_name_selection(sidebar_disabled)
        
        # Goals and Roles (common to both methods)
        goals_list = load_goals()
        st.markdown("### 🎯 Shared Goal Selection")
        shared_goal = st.selectbox(
            "Shared Goal", 
            goals_list,
            help="Type to search through the available goals",
            placeholder="Select a shared goal",
            disabled=sidebar_disabled,
            key="shared_goal"
        )
        
        st.markdown("### 🎭 Agent Role Selection")
        agent1_role = st.text_input(
            f"{agent1_name} Role", 
            st.session_state.get('agent1_role', ''),  # Preserve role on new simulation
            placeholder="e.g. Project Manager",
            help="Define the role of Agent 1 in the scenario. It's best to keep the role related to the shared goal",
            disabled=sidebar_disabled,
            key="agent1_role"
        )
        agent2_role = st.text_input(
            f"{agent2_name} Role", 
            st.session_state.get('agent2_role', ''),  # Preserve role on new simulation
            placeholder="e.g. Client Representative",
            help="Define the role of Agent 2 in the scenario. It's best to keep the role related to the shared goal",
            disabled=sidebar_disabled,
            key="agent2_role"
        )

        scenario_difficulty = st.selectbox(
            "Scenario Difficulty", 
            ["Easy", "Hard"], 
            index=0,
            help="Select the difficulty level for the scenario generation",
            disabled=sidebar_disabled,
            key="scenario_difficulty"
        )
        
        # Start simulation button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Start Simulation", 
                                        type="primary", 
                                        use_container_width=True,
                                        disabled=sidebar_disabled):
                # Store current values in session state before starting
                st.session_state.agent1_role_persist = agent1_role
                st.session_state.agent2_role_persist = agent2_role
                st.session_state.agent1_name_persist = agent1_name
                st.session_state.agent2_name_persist = agent2_name
                
                st.session_state.simulation_started = True
                st.session_state.initial_run_flag = True
                st.rerun()
        with col2:
            if st.button("🆕 New Simulation", 
                                        type="secondary", 
                                        use_container_width=True):
                # Clear simulation state but preserve configuration
                simulation_state_keys = [
                    'simulation_started', 'initial_run_flag', 'scenario_data', 
                    'scenario_content', 'goal_setup_data', 'generate_conversation',
                    'evaluation_data', 'conversation_finished', 'conversation_rows',
                    'conversation_started', 'chat_messages', 'behavioral_codes'
                ]
                for key in simulation_state_keys:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        return {
            'agent1_name': agent1_name,
            'agent2_name': agent2_name,
            'shared_goal': shared_goal,
            'agent1_role': agent1_role,
            'agent2_role': agent2_role,
            'scenario_difficulty': scenario_difficulty,
            'model_name': selected_model,
            'provider': selected_provider
        }

def _render_personality_based_selection(sidebar_disabled: bool):
    """Render personality-based character selection interface"""
    # Initialize personality service
    personality_service = PersonalityService()
    
    # Personality trait sliders with dual handles
    with st.expander("🎛️ Adjust Personality Ranges", expanded=True):
        st.markdown("**Adjust min/max values for each personality trait:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            openness_range = st.slider(
                "**Openness**",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.1,
                help="Imagination, creativity, curiosity",
                disabled=sidebar_disabled,
                key="openness_range"
            )
            
            conscientiousness_range = st.slider(
                "**Conscientiousness**",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.1,
                help="Organization, diligence, reliability",
                disabled=sidebar_disabled,
                key="conscientiousness_range"
            )
            
            extraversion_range = st.slider(
                "**Extraversion**",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.1,
                help="Sociability, assertiveness, energy",
                disabled=sidebar_disabled,
                key="extraversion_range"
            )
        
        with col2:
            agreeableness_range = st.slider(
                "**Agreeableness**",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.1,
                help="Cooperation, compassion, trust",
                disabled=sidebar_disabled,
                key="agreeableness_range"
            )
            
            neuroticism_range = st.slider(
                "**Neuroticism**",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.1,
                help="Emotional stability, anxiety, sensitivity",
                disabled=sidebar_disabled,
                key="neuroticism_range"
            )
    
    # Filter characters based on personality ranges
    filtered_characters = personality_service.filter_characters_by_personality(
        openness_range, conscientiousness_range, extraversion_range, 
        agreeableness_range, neuroticism_range
    )
    
    # Display matching characters
    st.markdown(f"### 🎭 Matching Characters ({len(filtered_characters)})")
    
    if not filtered_characters:
        st.warning("No characters match your selected personality ranges. Try broadening your criteria.")
        return "Character 1", "Character 2"
    
    # Display character selection
    character_names = [char.name for char in filtered_characters]
    
    col1, col2 = st.columns(2)
    with col1:
        agent1_name = st.selectbox(
            "Select Agent 1",
            options=character_names,
            key="agent1_personality",
            disabled=sidebar_disabled
        )
    with col2:
        agent2_name = st.selectbox(
            "Select Agent 2", 
            options=character_names,
            index=min(1, len(character_names)-1),
            key="agent2_personality",
            disabled=sidebar_disabled
        )
    
    return agent1_name, agent2_name

def _render_custom_name_selection(sidebar_disabled: bool):
    """Render traditional custom name selection"""
    col1, col2 = st.columns(2)
    with col1:
        agent1_name = st.text_input(
            "Agent 1 Name", 
            st.session_state.get('agent1_name_persist', ''),  # Preserve name on new simulation
            placeholder="e.g. Indiana Jones", 
            disabled=sidebar_disabled,
            key="agent1_custom"
        )
    with col2:
        agent2_name = st.text_input(
            "Agent 2 Name", 
            st.session_state.get('agent2_name_persist', ''),  # Preserve name on new simulation
            placeholder="e.g. Walter White", 
            disabled=sidebar_disabled,
            key="agent2_custom"
        )
    
    return agent1_name, agent2_name