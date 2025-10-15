import streamlit as st
from app.util import load_goals

def render_sidebar():
    """Render the sidebar configuration"""
    sidebar_disabled = st.session_state.simulation_started
    if st.session_state.simulation_started:
        st.sidebar.warning("Simulation in progress. To modify settings, please start a new simulation.")
    
    with st.sidebar:
        st.header("Configuration")
        # Agent names
        st.markdown("### Agent Name Selection")
        agent1_name = st.text_input("Agent 1 Name", "", placeholder="e.g. Indiana Jones", disabled=sidebar_disabled)
        agent2_name = st.text_input("Agent 2 Name", "", placeholder="e.g. Walter White", disabled=sidebar_disabled)

        # Goals
        goals_list = load_goals()
        st.markdown("### Shared Goal Selection")
        shared_goal = st.selectbox(
            "Shared Goal", 
            goals_list,
            help="Type to search through the available goals",
            placeholder="Select a shared goal",
            disabled=sidebar_disabled
        )
        
        # Roles
        st.markdown("### Agent Role Selection")
        agent1_role = st.text_input(
            "Agent 1 Role", 
            "", 
            placeholder="e.g. Project Manager",
            help="Define the role of Agent 1 in the scenario. It's best to keep the role related to the shared goal",
            disabled=sidebar_disabled
        )
        agent2_role = st.text_input(
            "Agent 2 Role", 
            "", 
            placeholder="e.g. Client Representative",
            help="Define the role of Agent 2 in the scenario. It's best to keep the role related to the shared goal",
            disabled=sidebar_disabled
        )

        scenario_difficulty = st.selectbox(
            "Scenario Difficulty", 
            ["Easy", "Hard"], 
            index=0,
            help="Select the difficulty level for the scenario generation",
            disabled=sidebar_disabled
        )
        
        # Start simulation button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Start Simulation", 
                                        type="primary", 
                                        use_container_width=True,
                                        disabled=sidebar_disabled):
                st.session_state.simulation_started = True # state to keep track 
                st.session_state.initial_run_flag = True # impulse to start the simulation
                st.rerun()
        with col2:
            if st.button("🆕 New Simulation", 
                                        type="secondary", 
                                        use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        return {
            'agent1_name': agent1_name,
            'agent2_name': agent2_name,
            'shared_goal': shared_goal,
            'agent1_role': agent1_role,
            'agent2_role': agent2_role,
            'scenario_difficulty': scenario_difficulty,
        }