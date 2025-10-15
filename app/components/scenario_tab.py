import streamlit as st

def render_scenario_tab():
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