import streamlit as st

class SessionState:
    """Manages session state for the application"""
    
    @staticmethod
    def initialize():
        """Initialize all session state variables"""
        state_vars = {
            'scenario_accepted': False,
            'scenario_rejected': False,
            'scenario_editing': False,
            'scenario_data': None,
            'scenario_content': None,
            'goal_setup_data': None,
            'regenerate_context': False,
            'generate_conversation': False,
            'evaluation_data': None,
            'conversation_finished': False,
            'conversation_rows': [],
            'behavioral_codes': [],
            'chat_messages': [],
            'sentiment_results': {},
            'simulation_started': False, # may cause bugs with scenario-regeneration - if so, remove this
            'initial_run_flag' : False,
            'conversation_started': False,
            'number_of_turns': 20
        }
        
        for key, default_value in state_vars.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def reset_simulation(): # this is used exclusively when the user rejects a scenario
        """Reset simulation state"""
        st.session_state.scenario_rejected = False
        st.session_state.scenario_accepted = False
        st.session_state.scenario_editing = False
        st.session_state.evaluation_data = None
        st.session_state.conversation_finished = False
        st.session_state.conversation_rows = []
        st.session_state.chat_messages = []
        st.session_state.sentiment_results = {}
        st.session_state.behavioral_codes = []
        st.session_state.scenario_data = None
        st.session_state.scenario_content = None
        st.session_state.goal_setup_data = None
        st.session_state.number_of_turns = 20
        st.session_state.generate_conversation = False