import streamlit as st
import plotly.express as px
import pandas as pd
from app.services.scenario_eval_services import EmotionIntensityService, EntailmentService, G_EvalService
from openai import OpenAI
from app.config.settings import OPEN_ROUTER_API_KEY
import logging
log = logging.getLogger(__name__)
def render_scenario_tab(config: dict):
    """Render the scenario details tab"""
    if st.session_state.scenario_data is None:
        st.info("Run the simulation first to view scenario details.")
        return
    
    st.markdown('<h2 class="section-header">Scenario Details</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Selected goal category**")
        st.info(st.session_state.goal_setup_data['chosen_social_goal_category'])
        
        st.markdown("**Explanation for the choice**")
        st.info(st.session_state.goal_setup_data['explanation'])

    with col2:
        st.markdown("**Agent 1 personal goal**")
        st.text_area("", st.session_state.goal_setup_data['first_agent_goal'], height=100, disabled=True)
        
        st.markdown("**Agent 2 personal goal**")
        st.text_area("", st.session_state.goal_setup_data['second_agent_goal'], height=100, disabled=True)
    
    st.markdown("**Scenario Context**")
    st.text_area("", st.session_state.scenario_content, height=150, disabled=True)
    
    # Emotion intensity analysis
    if st.session_state.scenario_content and st.session_state.scenario_accepted:
        _render_emotion_analysis()
        
        # Entailment analysis
        _render_entailment_analysis(config=config)

        # Cohesiveness and Fluency (placeholder for G-Eval)
        _render_geval_analysis(config=config, model_name=config['model_name'], provider=config['provider'])

def _render_emotion_analysis():
    """Render emotion intensity analysis"""
    st.markdown("#### 😊 Emotion Intensity Analysis")
    
    if not st.session_state.scenario_content:
        st.info("No scenario content available for emotion analysis.")
        return
    
    # Initialize emotion service
    emotion_service = EmotionIntensityService()
    
    with st.spinner("Analyzing emotional content..."):
        emotion_results = emotion_service.analyze_scenario_emotions(st.session_state.scenario_content)
    
    if not emotion_results:
        st.error("Emotion analysis unavailable - lexicon not loaded")
        return
    
    # Create emotion intensity bar chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        _render_emotion_chart(emotion_results)
    
    with col2:
        _render_emotion_metrics(emotion_results)

def _render_emotion_chart(emotion_results: dict):
    """Render emotion intensity bar chart"""
    # Extract mean intensity for each emotion
    emotions = ['anger', 'fear', 'sadness', 'surprise', 'joy', 'disgust', 'trust', 'anticipation']
    emotion_data = []
    
    for emotion in emotions:
        mean_intensity = emotion_results.get(f'{emotion}_mean', 0.0)
        word_count = emotion_results.get(f'{emotion}_count', 0)
        emotion_data.append({
            'Emotion': emotion.title(),
            'Mean Intensity': mean_intensity,
            'Word Count': word_count
        })
    
    emotion_df = pd.DataFrame(emotion_data)
    
    if not emotion_df.empty:
        fig = px.bar(
            emotion_df,
            x='Emotion',
            y='Mean Intensity',
            color='Mean Intensity',
            color_continuous_scale='reds',
            title="Emotion Intensity in Scenario",
            hover_data=['Word Count']
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title="Emotion",
            yaxis_title="Mean Intensity",
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No emotion data to display")

def _render_emotion_metrics(emotion_results: dict):
    """Render emotion analysis metrics"""
    st.markdown("**Emotion Metrics**")
    
    total_emotion_words = emotion_results.get('total_emotion_words', 0)
    unique_emotion_words = emotion_results.get('unique_emotion_words', 0)
    
    st.metric("Total Emotion Words", total_emotion_words)
    st.metric("Unique Emotion Words", unique_emotion_words)
    
    # Top emotions
    emotions = ['anger', 'fear', 'sadness', 'surprise', 'joy', 'disgust', 'trust', 'anticipation']
    emotion_scores = [(emotion, emotion_results.get(f'{emotion}_mean', 0.0)) for emotion in emotions]
    emotion_scores.sort(key=lambda x: x[1], reverse=True)
    
    st.markdown("**Top Emotions:**")
    for emotion, score in emotion_scores[:3]:
        if score > 0:
            st.write(f"• {emotion.title()}: {score:.3f}")

def _render_entailment_analysis(config: dict):
    """Render entailment analysis"""
    st.markdown("#### 🔗 Schema Entailment Analysis")
    
    if not st.session_state.scenario_content or not st.session_state.goal_setup_data:
        st.info("Scenario content or setup data not available for entailment analysis.")
        return
    
    # Initialize entailment service
    entailment_service = EntailmentService()
    
    # Prepare schema from goal setup data
    schema = {
        "shared_goal": st.session_state.goal_setup_data.get('shared_goal', ''),
        "social_goal_category": st.session_state.goal_setup_data.get('chosen_social_goal_category', ''),
        "first_agent_goal": st.session_state.goal_setup_data.get('first_agent_goal', ''),
        "second_agent_goal": st.session_state.goal_setup_data.get('second_agent_goal', ''),
        "first_agent_role": config.get('agent1_role', ''),
        "second_agent_role": config.get('agent2_role', '')
    }
    
    if st.session_state.entailment_score is None:
        if st.button("🔗 Calculate Entailment Score", type="secondary"):
            with st.spinner("Calculating entailment scores..."):
                slot_scores, average_entailment = entailment_service.calculate_scenario_entailment(
                    st.session_state.scenario_content, 
                    schema
                )
                st.session_state.entailment_score = average_entailment
            st.rerun()
    
    
    if st.session_state.entailment_score is not None:
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            _render_entailment_score(st.session_state.entailment_score)
        
        with col2:
            _render_entailment_explanation()

def _render_entailment_score(average_entailment: float):
    """Render the main entailment score"""
    st.metric(
        label="Overall Entailment Score",
        value=f"{average_entailment:.3f}",
        delta="Schema Alignment"
    )
    
    # Color code based on score
    if average_entailment >= 0.7:
        st.success("✅ High alignment with schema")
    elif average_entailment >= 0.4:
        st.warning("⚠️ Moderate alignment with schema")
    else:
        st.error("❌ Low alignment with schema")

def _render_entailment_explanation():
    """Render explanation of entailment scores"""
    with st.expander("What does entailment score mean?"):
        st.markdown("""
        **Natural Language Inference (NLI) Entailment Score**
        
        This score measures how well the generated scenario aligns with the intended schema:
        
        - **0.7-1.0**: High alignment - Scenario strongly follows the intended goals and roles
        - **0.4-0.7**: Moderate alignment - Scenario generally follows but may have some deviations  
        - **0.0-0.4**: Low alignment - Scenario may not properly reflect the intended setup
        
        The score is calculated using a transformer model that evaluates whether the scenario
        text *entails* (logically follows from) each element of your scenario setup.
        """)

def _render_geval_analysis(config: dict, model_name: str, provider: str):
    """Render G-Eval text quality analysis"""
    st.markdown("#### 📝 Text Quality Analysis (G-Eval)")
    
    if not st.session_state.scenario_content or not st.session_state.goal_setup_data:
        st.info("Scenario content not available for quality analysis.")
        return
    client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPEN_ROUTER_API_KEY
        )
    # Initialize G-Eval service
    geval_service = G_EvalService(client=client, model_name=model_name, provider=provider)
    
    # Prepare scenario setting for evaluation
    scenario_setting = {
        "shared_goal": st.session_state.goal_setup_data.get('shared_goal', ''),
        "social_goal_category": st.session_state.goal_setup_data.get('chosen_social_goal_category', ''),
        "first_agent_goal": st.session_state.goal_setup_data.get('first_agent_goal', ''),
        "second_agent_goal": st.session_state.goal_setup_data.get('second_agent_goal', ''),
        "first_agent_role": config.get('agent1_role', ''),
        "second_agent_role": config.get('agent2_role', '')
    }
    
    if not st.session_state.geval_results:
    # Run G-Eval analysis
        if st.button("🔍 Run Text Quality Analysis", type="secondary"):
            coherence_results, fluency_results = geval_service.evaluate_scenario_quality(
                st.session_state.scenario_content,
                scenario_setting
            )
            
            # Store results in session state
            st.session_state.geval_results = {
                'coherence': coherence_results,
                'fluency': fluency_results
            }
    
    # Display results if available
    if st.session_state.geval_results != {}:
        _render_geval_results(st.session_state.geval_results)
    else:
        st.info("Click the button above to run text quality analysis")

def _render_geval_results(geval_results: dict):
    """Render G-Eval results"""
    coherence = geval_results.get('coherence', {})
    fluency = geval_results.get('fluency', {})
    coherence_reason = coherence.get('reason', 'No feedback available')
    fluency_reason = fluency.get('reason', 'No feedback available')
    log.info(f"G-Eval Results - Coherence: {coherence}, Fluency: {fluency}")
    col1, col2 = st.columns(2)
    
    with col1:
        _render_coherence_score(coherence)
    
    with col2:
        _render_fluency_score(fluency)
    
    # Detailed reasoning
    _render_geval_detailed_feedback(coherence_reason, fluency_reason)

def _render_coherence_score(coherence: dict):
    """Render coherence score with explanation"""
    score = coherence.get('score', 0.0)
    
    st.metric(
        label="Coherence Score",
        value=f"{score:.2f}",
        delta=_get_coherence_feedback(score)
    )
    
    with st.expander("About Coherence"):
        st.markdown("""
        **Coherence** measures the logical flow and consistency of the scenario:
        
        - **0.8-1.0**: Excellent - Clear logical progression, consistent characters
        - **0.6-0.8**: Good - Generally coherent with minor inconsistencies  
        - **0.4-0.6**: Fair - Some logical gaps or contradictions
        - **0.0-0.4**: Poor - Significant coherence issues
        
        Evaluates: Goal alignment, role consistency, logical flow, character consistency
        """)

def _render_fluency_score(fluency: dict):
    """Render fluency score with explanation"""
    score = fluency.get('score', 0.0)
    
    st.metric(
        label="Fluency Score", 
        value=f"{score:.2f}",
        delta=_get_fluency_feedback(score)
    )
    
    with st.expander("About Fluency"):
        st.markdown("""
        **Fluency** measures the naturalness and grammatical quality:
        
        - **0.8-1.0**: Excellent - Very natural, well-written English
        - **0.6-0.8**: Good - Generally fluent with minor issues  
        - **0.4-0.6**: Fair - Noticeable grammar or flow issues
        - **0.0-0.4**: Poor - Significant language problems
        
        Evaluates: Grammar, punctuation, natural flow, vocabulary, clarity
        """)

def _render_geval_detailed_feedback(coherence_reason: str, fluency_reason: str):
    """Render detailed feedback from G-Eval"""
    st.markdown("#### 📋 Detailed Feedback")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Coherence Analysis:**")
        log.info(f"Coherence Reason v3: {coherence_reason}")
        st.write(coherence_reason)
    
    with col2:
        st.markdown("**Fluency Analysis:**")
        log.info(f"Fluency Reason v3: {fluency_reason}")
        st.write(fluency_reason)

def _get_coherence_feedback(score: float) -> str:
    """Get feedback text for coherence score"""
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    else:
        return "Needs Improvement"

def _get_fluency_feedback(score: float) -> str:
    """Get feedback text for fluency score"""
    if score >= 0.8:
        return "Very Natural"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    else:
        return "Needs Work"