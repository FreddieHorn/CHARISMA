import streamlit as st
import pandas as pd
import plotly.express as px
from app.services.scenario_eval_services import EmotionIntensityService
from typing import List, Dict, Tuple

def render_conversation_tab(config: dict):
    """Render the conversation tab with synchronized behavioral codes and emotion analysis"""
    if not st.session_state.generate_conversation:
        st.info("Accept the scenario first to view the conversation.")
        return
    
    message_chat_col, message_code_col, message_emotion_col = st.columns([2, 1, 1], vertical_alignment="center")
    with message_chat_col:
        st.markdown(f'<h2 class="section-header">Conversation between {config["agent1_name"]} and {config["agent2_name"]}</h2>', unsafe_allow_html=True)
    with message_code_col:
        st.markdown(f'<h2 class="section-header">Behavioral Codes</h2>', unsafe_allow_html=True)
    with message_emotion_col:
        st.markdown(f'<h2 class="section-header">Top emotions by message</h2>', unsafe_allow_html=True)
    # Automatically analyze emotions if not already done
    if not hasattr(st.session_state, 'emotion_results') or not st.session_state.emotion_results:
        _perform_emotion_analysis(config)
    
    # Display conversation with behavioral codes and emotions
    _display_conversation_with_codes_and_emotions(config)
    
    # Show emotion summary by default
    _render_emotion_summary_chart(config)

def _perform_emotion_analysis(config: dict):
    """Perform emotion analysis for the conversation"""
    emotion_service = EmotionIntensityService()
    
    with st.spinner("Analyzing emotional content..."):
        emotion_results = []
        
        for i, message in enumerate(st.session_state.chat_messages):
            analysis = emotion_service.analyze_scenario_emotions(message['content'])
            top_emotions = _get_top_emotions(analysis, 3)
            
            emotion_data = {
                'turn': i + 1,
                'speaker': message['speaker'],
                'message': message['content'],
                'top_emotions': top_emotions,
                'analysis': analysis
            }
            emotion_results.append(emotion_data)
        
        # Store results in session state
        st.session_state.emotion_results = emotion_results

def _get_top_emotions(analysis: Dict, top_n: int = 3) -> List[Tuple[str, float]]:
    """Get top N emotions by intensity from analysis"""
    emotions = ['anger', 'fear', 'sadness', 'surprise', 'joy', 'disgust', 'trust', 'anticipation']
    emotion_scores = []
    
    for emotion in emotions:
        score = analysis.get(f'{emotion}_mean', 0.0)
        if score > 0:  # Only include emotions with non-zero intensity
            emotion_scores.append((emotion, score))
    
    # Sort by intensity and return top N
    emotion_scores.sort(key=lambda x: x[1], reverse=True)
    return emotion_scores[:top_n]

def _display_conversation_with_codes_and_emotions(config: dict):
    """Display conversation with behavioral codes and emotions in synchronized columns"""
    # Always use three columns: chat, behavioral codes, emotions
    for i, message in enumerate(st.session_state.chat_messages):
        message_chat_col, message_code_col, message_emotion_col = st.columns([2, 1, 1], vertical_alignment="center")
        
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
                st.info("No code")
        
        # Display emotion analysis in emotion column
        with message_emotion_col:
            if i < len(st.session_state.emotion_results):
                emotion_data = st.session_state.emotion_results[i]
                _display_single_message_emotions(emotion_data)
            else:
                st.info("Analyzing...")

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
    with st.container():
        if badge_color == "primary":
            st.markdown(f"**{icon} {config['agent1_name']}**: **{behavioral_code}**")
        else:
            st.markdown(f"**{icon} {config['agent2_name']}**: **:yellow[{behavioral_code}]**")

def _display_single_message_emotions(emotion_data: Dict):
    """Display top emotions for a single message"""
    
    for emotion, intensity in emotion_data['top_emotions']:
        color = _get_emotion_color(emotion)
        # Show emotion with intensity and color coding
        st.markdown(
            f"<span style='color: {color}; font-size: 0.9em;'>• {emotion.title()}: {intensity:.3f}</span>",
            unsafe_allow_html=True
        )

def _get_emotion_color(emotion: str) -> str:
    """Get color for emotion type"""
    color_map = {
        'anger': '#e74c3c',
        'fear': '#9b59b6', 
        'sadness': '#3498db',
        'surprise': '#f39c12',
        'joy': '#2ecc71',
        'disgust': '#16a085',
        'trust': '#2980b9',
        'anticipation': '#e67e22'
    }
    return color_map.get(emotion, '#666666')

def _render_emotion_summary_chart(config: dict):
    """Render overall emotion summary chart per agent"""
    if not hasattr(st.session_state, 'emotion_results') or not st.session_state.emotion_results:
        return
    
    st.markdown("---")
    st.markdown("#### 📊 Emotion Profile Summary")
    
    # Aggregate emotions by agent
    agent_emotions = {config['agent1_name']: {}, config['agent2_name']: {}}
    
    for result in st.session_state.emotion_results:
        speaker = result['speaker']
        analysis = result['analysis']
        
        for emotion in ['anger', 'fear', 'sadness', 'surprise', 'joy', 'disgust', 'trust', 'anticipation']:
            intensity = analysis.get(f'{emotion}_mean', 0.0)
            if emotion not in agent_emotions[speaker]:
                agent_emotions[speaker][emotion] = []
            agent_emotions[speaker][emotion].append(intensity)
    
    # Calculate average intensities
    summary_data = []
    for agent, emotions in agent_emotions.items():
        for emotion, intensities in emotions.items():
            if intensities:  # Only include if we have data
                avg_intensity = sum(intensities) / len(intensities)
                summary_data.append({
                    'Agent': agent,
                    'Emotion': emotion.title(),
                    'Average Intensity': avg_intensity
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Create grouped bar chart
        fig = px.bar(
            summary_df,
            x='Emotion',
            y='Average Intensity',
            color='Agent',
            barmode='group',
            color_discrete_map={
                config['agent1_name']: '#1f77b4',
                config['agent2_name']: '#ff7f0e'
            }
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Emotion",
            yaxis_title="Average Intensity",
            showlegend=True,
            title=None  # Remove title for cleaner look
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Quick insights
        _render_quick_emotion_insights(agent_emotions, config)

def _render_quick_emotion_insights(agent_emotions: Dict, config: dict):
    """Render quick emotion insights"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{config['agent1_name']}:**")
        insights_1 = _get_quick_insights(agent_emotions[config['agent1_name']])
        for insight in insights_1:
            st.write(f"• {insight}")
    
    with col2:
        st.markdown(f"**{config['agent2_name']}:**")
        insights_2 = _get_quick_insights(agent_emotions[config['agent2_name']])
        for insight in insights_2:
            st.write(f"• {insight}")

def _get_quick_insights(emotion_data: Dict) -> List[str]:
    """Generate quick insights for emotion data"""
    if not emotion_data:
        return ["No emotion data"]
    
    insights = []
    avg_intensities = {emotion: sum(intensities)/len(intensities) for emotion, intensities in emotion_data.items()}
    
    # Dominant emotion
    if avg_intensities:
        dominant = max(avg_intensities.items(), key=lambda x: x[1])
        if dominant[1] > 0.1:
            insights.append(f"Most {dominant[0]}")
    
    # Emotional tone
    positive_emotions = ['joy', 'trust', 'surprise', 'anticipation']
    negative_emotions = ['anger', 'fear', 'sadness', 'disgust']
    
    pos_score = sum(avg_intensities.get(e, 0) for e in positive_emotions)
    neg_score = sum(avg_intensities.get(e, 0) for e in negative_emotions)
    
    if pos_score > neg_score + 0.1:
        insights.append("Positive tone")
    elif neg_score > pos_score + 0.1:
        insights.append("Negative tone")
    else:
        insights.append("Balanced tone")
    
    # Emotional intensity level
    max_intensity = max(avg_intensities.values()) if avg_intensities else 0
    if max_intensity > 0.3:
        insights.append("High emotional intensity")
    elif max_intensity < 0.1:
        insights.append("Low emotional intensity")
    else:
        insights.append("Moderate emotional intensity")
    
    return insights if insights else ["Neutral emotional expression"]