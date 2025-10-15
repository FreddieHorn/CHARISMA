import streamlit as st
import pandas as pd
# Function to load goals from CSV
@st.cache_data
def load_goals():
    try:
        # Load the CSV file with human goals
        goals_df = pd.read_csv("inputs/Human_Goals_List_Clean_Updated.csv")
        # Extract the full labels for the dropdown
        goals_list = goals_df['Full label'].tolist()
        return goals_list
    except FileNotFoundError:
        st.error("inputs/Human_Goals_List_Clean_Updated.csv file not found. Using default goals.")
        # Fallback to default goals if file not found
        return [
            "Achieving salvation",
            "Appreciating the arts",
            "Achieving my aspirations",
            "Being able to attract, please, sexually excite a sexual partner",
            "Avoiding failure",
            "Avoiding feelings of guilt",
            "Avoiding rejection by others",
            "Avoiding stress",
            "Being able to fantasize, imagine"
        ]

# Custom component for searchable dropdown
def searchable_dropdown(options, default_label="Select a goal", key="goal_selector"):
    # Create a unique key for the session state
    search_key = f"{key}_search"
    selected_key = f"{key}_selected"
    
    # Initialize session state
    if search_key not in st.session_state:
        st.session_state[search_key] = ""
    if selected_key not in st.session_state:
        st.session_state[selected_key] = None
    
    # Search input
    search_text = st.text_input("🔍 Search for a goal", 
                               value=st.session_state[search_key],
                               key=f"{key}_input")
    
    # Update search text in session state
    st.session_state[search_key] = search_text
    
    # Filter options based on search
    if search_text:
        filtered_options = [opt for opt in options if search_text.lower() in opt.lower()]
    else:
        filtered_options = options
    
    # Display the selectbox with filtered options
    if filtered_options:
        # Add a default option at the top
        select_options = [default_label] + filtered_options
        
        # Get the current selection
        current_selection = st.session_state[selected_key]
        
        # If current selection is not in filtered options, reset it
        if current_selection not in filtered_options and current_selection != default_label:
            current_selection = default_label
        
        # Create the selectbox
        selected = st.selectbox("Shared Goal", 
                               select_options, 
                               index=select_options.index(current_selection) if current_selection in select_options else 0,
                               key=f"{key}_select")
        
        # Update session state
        st.session_state[selected_key] = selected
        
        # Return the selected value (if not the default label)
        return selected if selected != default_label else None
    else:
        st.warning("No goals match your search.")
        return None
    
def replace_agent_names(text, agent1_name, agent2_name):
    """
    Replace [Agent 1] and [Agent 2] placeholders with actual agent names.
    
    Args:
        text (str): The input text containing placeholders
        agent1_name (str): Name to replace [Agent 1]
        agent2_name (str): Name to replace [Agent 2]
    
    Returns:
        str: Text with placeholders replaced by actual names
    """
    # Replace both placeholders with their respective names
    text = text.replace("[Agent 1]", agent1_name)
    text = text.replace("[Agent 2]", agent2_name)
    
    return text

def goal_achievment_int_to_str(score):
    """
    Convert an integer goal achievement score to a descriptive string.
    
    Args:
        score (int): The goal achievement score (0-10)
    
    Returns:
        str: Descriptive string corresponding to the score
    """
    if score == 10:
        return "Fully achieved"
    elif 7 <= score < 10:
        return "Mostly achieved"
    elif 4 <= score < 7:
        return "Partially achieved"
    elif 1 <= score < 4:
        return "Minimally achieved"
    elif score == 0:
        return "Not achieved at all"
    else:
        return "Invalid score"