CUSTOM_CSS = """
<style>
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
</style>
"""