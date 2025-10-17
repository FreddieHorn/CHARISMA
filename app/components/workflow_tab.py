import streamlit as st
import pandas as pd
from app.config.settings import SOCIAL_GOAL_CATEGORIES
from app.services.llm_service import CharismaService
from app.services.behavioral_analisis_service import BehavioralAnalysisService
from app.services.sentiment_service import SentimentAnalysisService
from typing import List, Dict
from app.util import goal_achievment_int_to_str
from app.components.real_time_chat import render_real_time_chat
import plotly.express as px
import pandas as pd

def render_workflow_tab(llm_service: CharismaService, sidebar_config: dict, behavioral_codes_df: pd.DataFrame):
    """Render the main simulation workflow tab"""
    st.markdown('<h2 class="section-header">Simulation Workflow</h2>', unsafe_allow_html=True)
    if st.session_state.scenario_data is None:
        _render_welcome_screen()
        return
    
    _render_scenario_section(llm_service, sidebar_config)
    
    if st.session_state.scenario_accepted and st.session_state.generate_conversation:
        _render_conversation_section(llm_service, sidebar_config, behavioral_codes_df)
    
    if st.session_state.evaluation_data:
        _render_evaluation_section(sidebar_config, behavioral_codes_df)

def _render_welcome_screen():
    """Render welcome screen when no simulation has run"""
    st.info("👈 Configure the agents and scenario in the sidebar, then click 'Start Simulation' to begin!")
    st.info("🚨 Work in progress 🚨")
    
    st.markdown("""
    ### How to use this application:
    1. Enter the names of your AI agents in the sidebar
    2. Select a shared goal from the dropdown menu
    3. Define the roles for each agent
    4. Click the 'Start Simulation' button to run the scenario
    5. View the generated scenario, conversation, and evaluation results
    6. Download the results as CSV files for further analysis
    """)

def _render_scenario_section(llm_service: CharismaService, config: dict):
    """Render scenario section"""
    st.markdown('<h3 class="section-header">📋 Scenario Overview</h3>', unsafe_allow_html=True)
    
    _render_scenario_actions()
    _render_scenario_details(llm_service, config)
    _render_scenario_context(llm_service, config)

def _render_scenario_actions():
    """Render scenario action buttons"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Accept Scenario", use_container_width=True, key="accept_main", disabled=st.session_state.conversation_started):
            st.session_state.scenario_accepted = True
            st.session_state.scenario_editing = False
            st.session_state.generate_conversation = True
            st.rerun()
    
    with col2:
        if st.button("🔄 Reject & Regenerate", use_container_width=True, key="reject_main", disabled=st.session_state.conversation_started):
            st.session_state.scenario_rejected = True
            st.rerun()
    
    with col3:
        if st.button("✏️ Edit Scenario", use_container_width=True, key="edit_main", disabled=st.session_state.conversation_started):
            st.session_state.scenario_editing = not st.session_state.scenario_editing
            st.rerun()

def _render_scenario_details(llm_service: CharismaService, config: dict):
    """Render scenario details"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Selected goal category**")
        if st.session_state.scenario_editing:
            original_category = st.session_state.goal_setup_data['chosen_social_goal_category']
            edited_category = st.selectbox(
                "Goal Category", 
                options=SOCIAL_GOAL_CATEGORIES,
                index=SOCIAL_GOAL_CATEGORIES.index(original_category) if original_category in SOCIAL_GOAL_CATEGORIES else 0,
                label_visibility="collapsed",
                key="category_main"
            )
            
            if edited_category != original_category:
                st.session_state.goal_setup_data['chosen_social_goal_category'] = edited_category
                st.session_state.goal_setup_data['explanation'] = ""
        else:
            st.info(st.session_state.goal_setup_data['chosen_social_goal_category'])
        
        st.markdown("**Explanation for the choice**")
        st.info(st.session_state.goal_setup_data['explanation'])

    with col2:
        st.markdown("**Agent 1 personal goal**")
        if st.session_state.scenario_editing:
            edited_agent1_goal = st.text_area(
                "Agent 1 Goal", 
                value=st.session_state.goal_setup_data['first_agent_goal'],
                height=100,
                label_visibility="collapsed",
                key="agent1_main"
            )
            if edited_agent1_goal != st.session_state.goal_setup_data['first_agent_goal']:
                st.session_state.goal_setup_data['first_agent_goal'] = edited_agent1_goal
        else:
            st.text_area(
                "", 
                st.session_state.goal_setup_data['first_agent_goal'], 
                height=100, 
                disabled=True, 
                key="agent1_display_main"
            )

        st.markdown("**Agent 2 personal goal**")
        if st.session_state.scenario_editing:
            edited_agent2_goal = st.text_area(
                "Agent 2 Goal", 
                value=st.session_state.goal_setup_data['second_agent_goal'],
                height=100,
                label_visibility="collapsed",
                key="agent2_main"
            )
            if edited_agent2_goal != st.session_state.goal_setup_data['second_agent_goal']:
                st.session_state.goal_setup_data['second_agent_goal'] = edited_agent2_goal
        else:
            st.text_area(
                "", 
                st.session_state.goal_setup_data['second_agent_goal'], 
                height=100, 
                disabled=True, 
                key="agent2_display_main"
            )
    
    return col1, col2

def _render_scenario_context(llm_service: CharismaService, config: dict):
    """Render scenario context section"""
    st.markdown("""
        <h3 style='text-align: center; font-weight: bold;'>
            🎭 Scenario Context
        </h3>
    """, unsafe_allow_html=True)
    
    if st.session_state.scenario_editing:
        if st.session_state.regenerate_context:
            _regenerate_scenario_context(llm_service, config)
            st.session_state.regenerate_context = False
            st.rerun()
        
        edited_scenario_context = st.text_area(
            "Scenario Context", 
            value=st.session_state.scenario_content,
            height=150,
            label_visibility="collapsed",
            key="scenario_main"
        )
        if edited_scenario_context != st.session_state.scenario_content:
            st.session_state.scenario_content = edited_scenario_context
        
        if st.button("🔄 Regenerate Context", use_container_width=True, key="regenerate_main"):
            st.session_state.regenerate_context = True
            st.rerun()
    else:
        st.text_area(
            "", 
            st.session_state.scenario_content, 
            height=150, 
            disabled=True, 
            key="scenario_display_main"
        )
    
    if st.session_state.scenario_editing:
        _render_save_changes_button()

def _regenerate_scenario_context(llm_service: CharismaService, config: dict):
    """Regenerate scenario context"""
    with st.spinner("Regenerating scenario context..."):
        scenario_data = llm_service.generate_scenario(
            goal_setup_data=st.session_state.goal_setup_data,
            scenario_difficulty=config['scenario_difficulty'],
            agent1_name=config['agent1_name'],
            agent2_name=config['agent2_name']
        )
        st.session_state.scenario_content = scenario_data[1]

def _render_save_changes_button():
    """Render save changes button"""
    if st.button("💾 Save Changes", type="primary", key="save_main"):
        st.session_state.scenario_editing = False
        st.rerun()

def _render_conversation_section(llm_service: CharismaService, config: dict, behavioral_codes_df: pd.DataFrame):
    """Render conversation section"""
    st.markdown(f'<h3 class="section-header">💬 Conversation between {config["agent1_name"]} and {config["agent2_name"]}</h3>', unsafe_allow_html=True)
    
    run_col, code_col = st.columns([2, 1])

    with run_col:
        with st.expander("⚙️ Conversation Settings"):
            col1, col2 = st.columns(2)
            with col1:
                typing_speed = st.slider("Typing Speed", 0.001, 0.05, 0.01, 
                                    help="Speed of the typing animation", disabled=st.session_state.conversation_started)
            with col2:
                thinking_delay = st.slider("Thinking Delay", 0.1, 3.0, 1.0,
                                        help="Maximum delay between responses", disabled=st.session_state.conversation_started)
            number_of_turns = st.slider("Number of Turns", 2, 20, 20, step=2, disabled=st.session_state.conversation_started)
            st.session_state.number_of_turns = number_of_turns    
        if st.button("▶️ Start Conversation", use_container_width=True, 
                    key="run_interaction_main", disabled=st.session_state.conversation_started):
            st.session_state.conversation_started = True
            st.rerun()
    with code_col:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacer
        st.markdown("### 🏷️ Behavioral Codes")

    if st.session_state.conversation_started:
        _run_real_time_interaction(llm_service, config, typing_speed, thinking_delay, behavioral_codes_df)

def _run_real_time_interaction(llm_service: CharismaService, config: dict, typing_speed: float, thinking_delay: float, behavioral_codes_df: pd.DataFrame):
    """Run interaction with real-time effects"""
    
    # Initialize chat state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "behavioral_codes" not in st.session_state:
        st.session_state.behavioral_codes = []
    # Render the real-time chat interface
    render_real_time_chat(llm_service, config=config, typing_speed=typing_speed, thinking_delay=thinking_delay)

    # Trigger evaluation when conversation finishes
    if st.session_state.conversation_finished and not st.session_state.evaluation_data:
        _trigger_evaluation(llm_service, config, behavioral_codes_df=behavioral_codes_df)

def _trigger_evaluation(llm_service: CharismaService, config: dict, behavioral_codes_df: pd.DataFrame):
    """Trigger conversation evaluation"""
    if not st.session_state.evaluation_data:
        scenario_setting_payload = {
            "shared_goal": config['shared_goal'],
            "chosen_goal_category": st.session_state.goal_setup_data['chosen_social_goal_category'],
            "first_agent_goal": st.session_state.goal_setup_data['first_agent_goal'],
            "second_agent_goal": st.session_state.goal_setup_data['second_agent_goal'],
            "first_agent_role": config['agent1_role'],
            "second_agent_role": config['agent2_role'],
        }
        st.session_state.evaluation_data = llm_service.evaluate_conversation(
            scenario_setting=scenario_setting_payload,
            scenario=st.session_state.scenario_content,
            conversation=st.session_state.conversation_rows
        )
        
        # Add behavioral analysis to evaluation data
        if st.session_state.behavioral_codes and not behavioral_codes_df.empty:
            analysis_service = BehavioralAnalysisService(behavioral_codes_df, config=config)
            behavioral_analysis = analysis_service.analyze_behavioral_patterns(st.session_state.behavioral_codes)
            st.session_state.evaluation_data['behavioral_analysis'] = behavioral_analysis


def _render_evaluation_section(config: dict, behavioral_codes_df: pd.DataFrame):
    """Render evaluation section with behavioral analysis"""
    st.markdown('<h3 class="section-header">📊 Evaluation Results</h3>', unsafe_allow_html=True)
    
    # Basic evaluation metrics
    _render_basic_evaluation_metrics(config)
    
    # Behavioral analysis (if available)
    if (st.session_state.behavioral_codes and 
        not behavioral_codes_df.empty and 
        'behavioral_analysis' in st.session_state.evaluation_data):
        _render_behavioral_analysis_workflow(config, behavioral_codes_df)
    
    # Sentiment analysis section
    _render_sentiment_analysis_eval(config)
    _render_download_buttons()

def _render_basic_evaluation_metrics(config: dict):
    """Render basic evaluation metrics"""
    eval_col1, eval_col2 = st.columns(2)
    
    with eval_col1:
        st.metric(
            label="Shared Goal Completion Score",
            value=f"{st.session_state.evaluation_data['shared_goal_achievement_score']}/10"
        )
        
        st.metric(
            label=f"{config['agent1_name']} Score",
            value=f"{st.session_state.evaluation_data['Agent A']['personal_goal_completion_score']}/10",
            delta="Goal Achievement: " + goal_achievment_int_to_str(int(st.session_state.evaluation_data['Agent A']['personal_goal_completion_score']))
        )
        
        st.metric(
            label=f"{config['agent2_name']} Score",
            value=f"{st.session_state.evaluation_data['Agent B']['personal_goal_completion_score']}/10",
            delta="Goal Achievement: " + goal_achievment_int_to_str(int(st.session_state.evaluation_data['Agent B']['personal_goal_completion_score']))
        )
                
    with eval_col2:
        st.markdown("**Detailed Explanations:**")
        
        with st.expander(f"Explanation for Shared Goal"):
            st.write(st.session_state.evaluation_data['reasoning'])
        
        with st.expander(f"Explanation for {config['agent1_name']}"):
            st.write(st.session_state.evaluation_data['Agent A']['reasoning'])
        
        with st.expander(f"Explanation for {config['agent2_name']}"):
            st.write(st.session_state.evaluation_data['Agent B']['reasoning'])

def _render_behavioral_analysis_workflow(config: dict, behavioral_codes_df: pd.DataFrame):
    """Render behavioral analysis in workflow tab"""
    st.markdown("---")
    st.markdown('<h4 class="section-header">🧠 Behavioral Pattern Analysis</h4>', unsafe_allow_html=True)
    
    if behavioral_codes_df.empty:
        st.info("📊 Behavioral analysis unavailable - behavioral codes CSV not found")
        return
    
    if not st.session_state.behavioral_codes:
        st.info("📊 Run a conversation to see behavioral analysis")
        return
    
    analysis_service = BehavioralAnalysisService(behavioral_codes_df, config)
    analysis_results = st.session_state.evaluation_data['behavioral_analysis']
    # Top 3 overall behavioral codes
    _render_top_overall_codes(analysis_results)
    # Behavioral codes bar charts
    st.markdown("#### 📊 Behavioral Codes Usage")
    _render_behavioral_codes_charts(analysis_results, config)
    
    # Create two columns for personality insights and behavior types
    col1, col2 = st.columns(2)
    
    with col1:
        _render_personality_insights_workflow(analysis_service, analysis_results, config)
    
    with col2:
        st.markdown("#### 📋 Behavior Type Distribution")
        _render_behavior_type_breakdown_workflow(analysis_service, st.session_state.behavioral_codes, config)

def _render_top_overall_codes(analysis_results: Dict):
    """Render top 3 overall behavioral codes with counts"""
    st.markdown("#### 🏆 Top 3 Most Used Behavioral Codes")
    
    overall_top = analysis_results['overall_codes'].most_common(3)
    
    if overall_top:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if len(overall_top) >= 1:
                code, count = overall_top[0]
                st.metric(
                    label="🥇 Most Used",
                    value=code,
                    delta=f"{count} times"
                )
        
        with col2:
            if len(overall_top) >= 2:
                code, count = overall_top[1]
                st.metric(
                    label="🥈 Second Most",
                    value=code,
                    delta=f"{count} times"
                )
        
        with col3:
            if len(overall_top) >= 3:
                code, count = overall_top[2]
                st.metric(
                    label="🥉 Third Most",
                    value=code,
                    delta=f"{count} times"
                )
    else:
        st.info("No behavioral codes recorded")
        
def _render_behavioral_codes_charts(analysis_results: Dict, config: dict):
    """Render bar charts showing all behavioral codes used by each agent"""
    import plotly.express as px
    import pandas as pd
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Agent 1 behavioral codes chart
        agent1_codes = analysis_results['agent1_codes']
        
        if agent1_codes:
            # Create dataframe for agent 1
            agent1_data = []
            for code, count in agent1_codes.most_common():  # Show all codes, sorted by frequency
                agent1_data.append({'Behavioral Code': code, 'Count': count, 'Agent': config['agent1_name']})
            
            agent1_df = pd.DataFrame(agent1_data)
            
            if not agent1_df.empty:
                fig1 = px.bar(
                    agent1_df, 
                    x='Count', 
                    y='Behavioral Code',
                    orientation='h',
                    title=f"{config['agent1_name']} - Behavioral Codes",
                    color='Count',
                    color_continuous_scale='blues'
                )
                fig1.update_layout(
                    showlegend=False,
                    height=400,
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig1, use_container_width=True, key="agent1_behavioral_chart")
            else:
                st.info(f"No behavioral codes recorded for {config['agent1_name']}")
        else:
            st.info(f"No behavioral codes recorded for {config['agent1_name']}")
    
    with col2:
        # Agent 2 behavioral codes chart
        agent2_codes = analysis_results['agent2_codes']
        
        if agent2_codes:
            # Create dataframe for agent 2
            agent2_data = []
            for code, count in agent2_codes.most_common():  # Show all codes, sorted by frequency
                agent2_data.append({'Behavioral Code': code, 'Count': count, 'Agent': config['agent2_name']})
            
            agent2_df = pd.DataFrame(agent2_data)
            
            if not agent2_df.empty:
                fig2 = px.bar(
                    agent2_df, 
                    x='Count', 
                    y='Behavioral Code',
                    orientation='h',
                    title=f"{config['agent2_name']} - Behavioral Codes",
                    color='Count',
                    color_continuous_scale='oranges'
                )
                fig2.update_layout(
                    showlegend=False,
                    height=400,
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig2, use_container_width=True, key="agent2_behavioral_chart")
            else:
                st.info(f"No behavioral codes recorded for {config['agent2_name']}")
        else:
            st.info(f"No behavioral codes recorded for {config['agent2_name']}")

def _render_personality_insights_workflow(analysis_service: BehavioralAnalysisService, analysis_results: Dict, config: dict):
    """Render personality insights in workflow tab"""
    st.markdown("#### 🎭 Personality Traits by Agent")
    
    # Get personality traits by agent
    agent_codes = {
        config['agent1_name']: analysis_results['agent1_codes'],
        config['agent2_name']: analysis_results['agent2_codes']
    }
    
    agent_traits = analysis_service.get_personality_traits_by_agent(agent_codes)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{config['agent1_name']}:**")
        traits_data = agent_traits.get(config['agent1_name'], {})
        if traits_data.get('top_traits'):
            for trait, count in traits_data['top_traits']:
                st.write(f"• **{trait}** (in {count} behaviors)")
        else:
            st.write("• No personality trait data available")
    
    with col2:
        st.markdown(f"**{config['agent2_name']}:**")
        traits_data = agent_traits.get(config['agent2_name'], {})
        if traits_data.get('top_traits'):
            for trait, count in traits_data['top_traits']:
                st.write(f"• **{trait}** (in {count} behaviors)")
        else:
            st.write("• No personality trait data available")

def _render_behavior_type_breakdown_workflow(analysis_service: BehavioralAnalysisService, behavioral_codes: List[Dict], config: dict):
    """Render behavior type breakdown by agent in workflow tab"""
    agent_type_breakdown = analysis_service.get_behavior_type_breakdown_by_agent(behavioral_codes)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{config['agent1_name']}:**")
        agent1_types = agent_type_breakdown.get(config['agent1_name'], {})
        if agent1_types:
            total_agent1 = sum(agent1_types.values())
            for behavior_type, count in agent1_types.items():
                percentage = (count / total_agent1) * 100
                st.write(f"• **{behavior_type}**: {count} ({percentage:.1f}%)")
        else:
            st.write("• No behavior type data")
    
    with col2:
        st.markdown(f"**{config['agent2_name']}:**")
        agent2_types = agent_type_breakdown.get(config['agent2_name'], {})
        if agent2_types:
            total_agent2 = sum(agent2_types.values())
            for behavior_type, count in agent2_types.items():
                percentage = (count / total_agent2) * 100
                st.write(f"• **{behavior_type}**: {count} ({percentage:.1f}%)")
        else:
            st.write("• No behavior type data")

def _render_sentiment_analysis_eval(config: dict):
    """Render sentiment analysis section"""
    st.markdown("---")
    st.markdown('<h3 class="section-header">😊 Sentiment Analysis</h3>', unsafe_allow_html=True)
    
    if not st.session_state.conversation_rows:
        st.info("No conversation data available for sentiment analysis.")
        return
    
    # Initialize sentiment service
    sentiment_service = SentimentAnalysisService()
    
    with st.spinner("Analyzing conversation sentiment..."):
        sentiment_results = sentiment_service.analyze_conversation_sentiment(st.session_state.conversation_rows)
    
    if not sentiment_results:
        st.error("Failed to analyze sentiment.")
        return
    
    # Store sentiment results in session state for download
    st.session_state.sentiment_results = sentiment_results
    
    # Display sentiment summary
    _render_sentiment_summary(sentiment_results, config)
    
    # Display detailed sentiment breakdown
    _render_sentiment_breakdown(sentiment_results, config)

def _render_sentiment_summary(sentiment_results: Dict, config: dict):
    """Render sentiment summary"""
    summary = sentiment_results['conversation_summary']
    
    st.markdown("### 📈 Sentiment Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Overall Conversation Sentiment",
            value=summary['average_sentiment_label'],
            delta=f"Score: {summary['average_sentiment_score']:.2f}/4.0"
        )
    
    with col2:
        agent1_avg = summary.get(f"{config['agent1_name']}_average_score", 0)
        agent1_label = summary.get(f"{config['agent1_name']}_average_label", "Unknown")
        st.metric(
            label=f"{config['agent1_name']} Average Sentiment",
            value=agent1_label,
            delta=f"Score: {agent1_avg:.2f}/4.0"
        )
    
    with col3:
        agent2_avg = summary.get(f"{config['agent2_name']}_average_score", 0)
        agent2_label = summary.get(f"{config['agent2_name']}_average_label", "Unknown")
        st.metric(
            label=f"{config['agent2_name']} Average Sentiment",
            value=agent2_label,
            delta=f"Score: {agent2_avg:.2f}/4.0"
        )
    
    # Sentiment distribution
    st.markdown("#### Sentiment Distribution")
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        st.markdown(f"**{config['agent1_name']}:**")
        agent1_data = sentiment_results['speaker_sentiments'].get(config['agent1_name'], {})
        if agent1_data:
            for sentiment, count in agent1_data['sentiment_distribution'].items():
                if count > 0:
                    percentage = (count / agent1_data['messages_analyzed']) * 100
                    st.write(f"• {sentiment}: {count} messages ({percentage:.1f}%)")
    
    with dist_col2:
        st.markdown(f"**{config['agent2_name']}:**")
        agent2_data = sentiment_results['speaker_sentiments'].get(config['agent2_name'], {})
        if agent2_data:
            for sentiment, count in agent2_data['sentiment_distribution'].items():
                if count > 0:
                    percentage = (count / agent2_data['messages_analyzed']) * 100
                    st.write(f"• {sentiment}: {count} messages ({percentage:.1f}%)")

def _render_sentiment_breakdown(sentiment_results: Dict, config: dict):
    """Render detailed sentiment breakdown"""
    st.markdown("#### 📋 Detailed Sentiment Analysis")
    
    with st.expander("View Message-by-Message Sentiment", expanded=False):
        for result in sentiment_results['detailed_results']:
            sentiment_color = {
                "Very Negative": "red",
                "Negative": "orange", 
                "Neutral": "gray",
                "Positive": "lightgreen",
                "Very Positive": "green"
            }.get(result['sentiment_label'], "gray")
            
            st.markdown(
                f"""
                <div style='border-left: 4px solid {sentiment_color}; padding-left: 10px; margin: 5px 0;'>
                    <strong>{result['speaker']} (Turn {result['turn']}):</strong> {result['sentiment_label']} 
                    <br>
                    <em>{result['message'][:100]}{'...' if len(result['message']) > 100 else ''}</em>
                </div>
                """,
                unsafe_allow_html=True
            )
            
def _render_download_buttons():
    """Render download buttons"""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    scenario_df = pd.DataFrame({k: [v] for k, v in st.session_state.goal_setup_data.items()})
    conversation_df = pd.DataFrame(st.session_state.conversation_rows)
    evaluation_df = pd.DataFrame(st.session_state.evaluation_data)
    
    with col1:
        st.download_button(
            label="📥 Download Scenario",
            data=scenario_df.to_csv(index=False),
            file_name="scenario.csv",
            mime="text/csv",
            key="dl_scenario_main"
        )
    
    with col2:
        st.download_button(
            label="📥 Download Conversation",
            data=conversation_df.to_csv(index=False),
            file_name="conversation.csv",
            mime="text/csv",
            key="dl_conversation_main"
        )
    
    with col3:
        st.download_button(
            label="📥 Download Evaluation",
            data=evaluation_df.to_csv(index=False),
            file_name="evaluation.csv",
            mime="text/csv",
            key="dl_evaluation_main"
        )