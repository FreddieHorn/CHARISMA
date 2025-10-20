CUSTOM_CSS = """
<style>
    section[data-testid="stSidebar"] {
        min-width: 400px; max-width: 800px; width: 650px;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .agent-bubble {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border: 1px solid #bbdefb;
    }
    .scenario-box {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
    .evaluation-box {
        background-color: #fff3e0;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: #2c3e50;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #2c3e50;
        border-radius: 4px 4px 0px 0px;
        padding: 15px 20px;
        font-size: 18px;
        font-weight: 600;
        transition: all 0.3s ease;
        flex: 1;
        text-align: center;
        border-bottom: 3px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e6eef9;
        color: #1f77b4;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white !important;
        border-bottom: 3px solid #ff9800;
    }
        .stSlider [data-baseweb="slider"] {
        margin: 10px 0;
    }
    
    /* Character card styles */
    .character-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        background-color: #f9f9f9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    .character-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .personality-badge {
        display: inline-block;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 10px;
        margin: 1px;
    }
    
    /* Range display styles */
    .range-display {
        background-color: #e3f2fd;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 12px;
    }
</style>
"""