import streamlit as st
from app.util import goal_achievment_int_to_str
from app.services.behavioral_analisis_service import BehavioralAnalysisService
from app.services.sentiment_service import SentimentAnalysisService
from app.components.workflow_tab import _render_compact_semantic_arc
from typing import Dict, List
import pandas as pd

def render_evaluation_tab(config: dict, behavioral_codes_df: pd.DataFrame):
    """Render the evaluation tab"""
    if not st.session_state.evaluation_data:
        st.info("Conversation must run first in order to view evaluation results.")
        return
    
    st.markdown('<h2 class="section-header">Evaluation Results</h2>', unsafe_allow_html=True)
    
    eval_col1, eval_col2 = st.columns(2)
    
    with eval_col1:
        st.metric(
            label="Shared Goal Completion Score",
            value=f"{st.session_state.evaluation_data['shared_goal_achievement_score']}/10"
        )
        
        st.metric(
            label=f"{config['agent1_name']} Score",
            value=f"{st.session_state.evaluation_data['Agent A']['personal_goal_achievement_score']}/10",
            delta="Goal Achievement: " + goal_achievment_int_to_str(int(st.session_state.evaluation_data['Agent A']['personal_goal_achievement_score']))
        )
        
        st.metric(
            label=f"{config['agent2_name']} Score",
            value=f"{st.session_state.evaluation_data['Agent B']['personal_goal_achievement_score']}/10",
            delta="Goal Achievement: " + goal_achievment_int_to_str(int(st.session_state.evaluation_data['Agent B']['personal_goal_achievement_score']))
        )
                
    with eval_col2:
        st.markdown("**Detailed Explanations:**")
        
        with st.expander(f"Explanation for Shared Goal"):
            st.write(st.session_state.evaluation_data['reasoning'])
        
        with st.expander(f"Explanation for {config['agent1_name']}"):
            st.write(st.session_state.evaluation_data['Agent A']['reasoning'])
        
        with st.expander(f"Explanation for {config['agent2_name']}"):
            st.write(st.session_state.evaluation_data['Agent B']['reasoning'])
            
    _render_behavioral_analysis_workflow(config, behavioral_codes_df)
    _render_sentiment_analysis_eval(config)
            
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
                st.plotly_chart(fig1, use_container_width=True)
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
                st.plotly_chart(fig2, use_container_width=True)
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
    
    sentiment_service = SentimentAnalysisService()
    
    if not st.session_state.sentiment_results:
        sentiment_results = sentiment_service.analyze_conversation_sentiment(st.session_state.conversation_rows)
        st.session_state.sentiment_results = sentiment_results

    if not st.session_state.sentiment_results:
        st.error("Failed to analyze sentiment.")
        return


    # Display sentiment summary
    _render_sentiment_summary(st.session_state.sentiment_results, config)
    
    # Display detailed sentiment breakdown
    _render_sentiment_breakdown(st.session_state.sentiment_results, config)

    # Display Semantic Arc
    _render_compact_semantic_arc(st.session_state.sentiment_results, config, key="eval_sentiment_compact_arc_eval")

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
            