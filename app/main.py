import streamlit as st
import pandas as pd

from app.config.settings import PAGE_CONFIG
from app.config.styles import CUSTOM_CSS
from app.models.session_state import SessionState
from app.services.llm_service import CharismaService
from app.components.sidebar import render_sidebar
from app.components.workflow_tab import render_workflow_tab
from app.components.scenario_tab import render_scenario_tab
from app.components.conversation_tab import render_conversation_tab
from app.components.evaluation_tab import render_evaluation_tab
from logging import getLogger
log = getLogger(__name__)
class AIAgentScenarioPlayer:
    """Main application class for CHARISMA"""
    
    def __init__(self):
        self.setup_page()
        SessionState.initialize()
        self.llm_service = None
        self.behavioral_codes_df = self.load_behavioral_codes()
    
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(**PAGE_CONFIG)
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        st.markdown('<h1 class="main-header"><span style="font-size:1.3em">🤖 CHARISMA</span>:</br>Character-based Interaction Simulation </br>with Multi LLM-based Agents</br>for Computational Social Psychology</h1>', unsafe_allow_html=True)
    
    def handle_simulation_start(self, sidebar_config: dict):
        """Handle simulation start"""
        if st.session_state.initial_run_flag or st.session_state.scenario_rejected:
            if st.session_state.scenario_rejected:
                SessionState.reset_simulation()
            self.run_simulation(sidebar_config)
    
    def load_behavioral_codes(self) -> pd.DataFrame:
        """Load behavioral codes from CSV"""
        try:
            # Update this path to where your CSV file is located
            return pd.read_csv('inputs/behavioral_coding.csv')
        except FileNotFoundError:
            st.warning("Behavioral codes CSV file not found. Behavioral analysis will be disabled.")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading behavioral codes: {e}")
            return pd.DataFrame()
    def run_simulation(self, config: dict):
        """Run the main simulation pipeline"""
        with st.spinner("Running simulation... This may take a few moments."):
            log.info("Starting simulation with configuration: " + str(config))
            # Setup goals
            goal_setup_data = self.llm_service.setup_goals(
                shared_goal=config['shared_goal'],
                agent1_role=config['agent1_role'],
                agent2_role=config['agent2_role']
            )
            
            # Generate scenario
            scenario_data, scenario_content = self.llm_service.generate_scenario(
                goal_setup_data=goal_setup_data,
                scenario_difficulty=config['scenario_difficulty'],
                agent1_name=config['agent1_name'],
                agent2_name=config['agent2_name']
            )
            
            # Store in session state
            st.session_state.scenario_data = scenario_data
            st.session_state.scenario_content = scenario_content
            st.session_state.goal_setup_data = goal_setup_data
            st.session_state.initial_run_flag = False
    
    def render_tabs(self, sidebar_config: dict):
        """Render application tabs"""
        tab1, tab2, tab3, tab4 = st.tabs([
            "🚀 Simulation Workflow", 
            "📋 Scenario Analysis", 
            "💬 Conversation Analysis", 
            "📊 Agent Analysis"
        ])
        
        with tab1:
            render_workflow_tab(self.llm_service, sidebar_config, self.behavioral_codes_df)
        
        with tab2:
            render_scenario_tab(sidebar_config)
        
        with tab3:
            render_conversation_tab(sidebar_config)
        
        with tab4:
            render_evaluation_tab(sidebar_config, self.behavioral_codes_df)
    
    def render_footer(self):
        """Render application footer"""
        st.markdown("---")
        st.caption("CHARISMA © 2025")
    
    def run(self):
        """Main application entry point"""
        sidebar_config = render_sidebar()
        self.llm_service = CharismaService(model_name=sidebar_config['model_name'], provider=sidebar_config['provider'])
        self.handle_simulation_start(sidebar_config)
        self.render_tabs(sidebar_config)
        self.render_footer()

if __name__ == "__main__":
    app = AIAgentScenarioPlayer()
    app.run()