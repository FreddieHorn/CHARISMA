import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _extract_model_name(dir_path: str) -> str:
    """
    Extract model name from directory name.

    Example:
    'deepseekv3.1_evaluation_visualisations_deepseekv3.1-GPT_SCENARIOS'
    -> 'deepseekv3.1'
    """
    base = os.path.basename(os.path.normpath(dir_path))
    if "_evaluation_visualisations_" in base:
        return base.split("_evaluation_visualisations_")[0]
    return base

def _extract_eval_model_name(dir_path: str) -> str:
    """
    Extract evaluation model name from directory name.

    Example:
    'gemini-2.5-flash_evaluation_visualisations_deepseekv3.1-GPT_SCENARIOS'
    -> 'deepseekv3.1'
    """
    base = os.path.basename(os.path.normpath(dir_path))
    if "_evaluation_visualisations_" not in base:
        return "UNKNOWN"

    tail = base.split("_evaluation_visualisations_", 1)[1]
    # tail looks like 'deepseekv3.1-GPT_SCENARIOS' or 'gpt5-mini-DEEPSEEK_SCENARIOS'
    eval_model = tail.rsplit("-", 1)[0]
    return eval_model



def plot_agent_personal_shared_by_model(
    directories,
    agent_name,
    personal_col="Personal_Score",
    shared_col="Shared_Score",
    csv_name="agent_performance_results.csv",
    save_dir="agent_plots",
):
    """
    For a given agent, compute average Personal & Shared scores per MODEL
    (aggregating across all its directories) and save a grouped bar plot.

    There will be 4 models on the x-axis, with 2 bars per model:
    - personal score
    - shared score
    """

    os.makedirs(save_dir, exist_ok=True)

    # model -> list of scores
    personal_by_model = {}
    shared_by_model = {}

    for d in directories:
        csv_path = os.path.join(d, csv_name)
        if not os.path.isfile(csv_path):
            print(f"Warning: {csv_path} not found, skipping.")
            continue

        df = pd.read_csv(csv_path)
        if "Agent" not in df.columns:
            print(f"Warning: 'Agent' column missing in {csv_path}, skipping.")
            continue

        missing_cols = [c for c in [personal_col, shared_col] if c not in df.columns]
        if missing_cols:
            print(f"Warning: {missing_cols} missing in {csv_path}, skipping.")
            continue

        agent_rows = df[df["Agent"] == agent_name]
        if agent_rows.empty:
            print(f"Warning: agent {agent_name!r} not found in {csv_path}, skipping.")
            continue

        model_name = _extract_model_name(d)

        p_scores = agent_rows[personal_col].dropna().tolist()
        s_scores = agent_rows[shared_col].dropna().tolist()

        if p_scores:
            personal_by_model.setdefault(model_name, []).extend(p_scores)
        if s_scores:
            shared_by_model.setdefault(model_name, []).extend(s_scores)

    # models that have both scores
    models = sorted(set(personal_by_model.keys()) | set(shared_by_model.keys()))
    if not models:
        print("No data collected. Check paths, agent name, and column names.")
        return

    personal_avgs = [
        np.mean(personal_by_model[m]) if m in personal_by_model else np.nan
        for m in models
    ]
    shared_avgs = [
        np.mean(shared_by_model[m]) if m in shared_by_model else np.nan
        for m in models
    ]

    # ---- Plot: 2 bars per model (Personal vs Shared) ----
    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, personal_avgs, width=width, label="Personal_Score")
    plt.bar(x + width/2, shared_avgs,  width=width, label="Shared_Score")

    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel("Average score")
    plt.title(
        f'Personal & Shared scores for "{agent_name}"\n'
        f"by interaction generation model"
    )
    plt.legend()
    plt.tight_layout()

    safe_agent = agent_name.replace(" ", "_")
    out_path = os.path.join(save_dir, f"{safe_agent}_personal_shared_by_model.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved plot → {out_path}")

def plot_agent_personal_shared_by_model_for_eval(
    directories,
    agent_name,
    eval_model_filter,  # 'gpt5-mini' or 'deepseekv3.1'
    personal_col="Personal_Score",
    shared_col="Shared_Score",
    csv_name="agent_performance_results.csv",
    save_dir="agent_plots",
):
    """
    For a given agent and a chosen EVALUATION MODEL, compute average
    Personal & Shared scores per GENERATOR MODEL and save a grouped bar plot.

    - Filters directories by eval_model_filter (e.g. 'gpt5-mini' or 'deepseekv3.1')
    - Aggregates across all dirs that match that evaluation model
    - X-axis: generator models (deepseekv3.1, gpt-5-mini, mistral-medium-3.1, gemini-2.5-flash)
    - Bars:   2 per model (Personal_Score, Shared_Score)
    """

    os.makedirs(save_dir, exist_ok=True)

    personal_by_model = {}
    shared_by_model = {}

    for d in directories:
        # filter by evaluation model in directory name
        eval_name = _extract_eval_model_name(d)
        if eval_name != eval_model_filter:
            continue

        csv_path = os.path.join(d, csv_name)
        if not os.path.isfile(csv_path):
            print(f"Warning: {csv_path} not found, skipping.")
            continue

        df = pd.read_csv(csv_path)
        if "Agent" not in df.columns:
            print(f"Warning: 'Agent' column missing in {csv_path}, skipping.")
            continue

        missing_cols = [c for c in [personal_col, shared_col] if c not in df.columns]
        if missing_cols:
            print(f"Warning: {missing_cols} missing in {csv_path}, skipping.")
            continue

        agent_rows = df[df["Agent"] == agent_name]
        if agent_rows.empty:
            print(f"Warning: agent {agent_name!r} not found in {csv_path}, skipping.")
            continue

        model_name = _extract_model_name(d)

        p_scores = agent_rows[personal_col].dropna().tolist()
        s_scores = agent_rows[shared_col].dropna().tolist()

        if p_scores:
            personal_by_model.setdefault(model_name, []).extend(p_scores)
        if s_scores:
            shared_by_model.setdefault(model_name, []).extend(s_scores)

    models = sorted(set(personal_by_model.keys()) | set(shared_by_model.keys()))
    if not models:
        print(
            f"No data collected for eval_model_filter={eval_model_filter!r}. "
            "Check paths, agent name, and directory names."
        )
        return

    personal_avgs = [
        np.mean(personal_by_model[m]) if m in personal_by_model else np.nan
        for m in models
    ]
    shared_avgs = [
        np.mean(shared_by_model[m]) if m in shared_by_model else np.nan
        for m in models
    ]

    # ---- Plot: 2 bars per generator model (Personal & Shared) ----
    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, personal_avgs, width=width, label=personal_col)
    plt.bar(x + width / 2, shared_avgs,  width=width, label=shared_col)

    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel("Average score")
    plt.title(
        f'Personal & Shared scores for "{agent_name}"\n'
        f"by interaction generation model (eval: {eval_model_filter})"
    )
    plt.legend()
    plt.tight_layout()

    safe_agent = agent_name.replace(" ", "_")
    safe_eval = eval_model_filter.replace(" ", "_")
    out_path = os.path.join(
        save_dir, f"{safe_agent}_personal_shared_by_model_eval_{safe_eval}.png"
    )
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved plot → {out_path}")

    
dirs = [
    # --- deepseekv3.1 model ---
    "deepseekv3.1_evaluation_visualisations_deepseekv3.1-DEEPSEEK_SCENARIOS",
    "deepseekv3.1_evaluation_visualisations_deepseekv3.1-GPT_SCENARIOS",
    "deepseekv3.1_evaluation_visualisations_gpt5-mini-DEEPSEEK_SCENARIOS",
    "deepseekv3.1_evaluation_visualisations_gpt5-mini-GPT_SCENARIOS",

    # --- gemini-2.5-flash model ---
    "gemini-2.5-flash_evaluation_visualisations_deepseekv3.1-DEEPSEEK_SCENARIOS",
    "gemini-2.5-flash_evaluation_visualisations_deepseekv3.1-GPT_SCENARIOS",
    "gemini-2.5-flash_evaluation_visualisations_gpt5-mini-DEEPSEEK_SCENARIOS",
    "gemini-2.5-flash_evaluation_visualisations_gpt5-mini-GPT_SCENARIOS",

    # --- gpt-5-mini model ---
    "gpt-5-mini_evaluation_visualisations_deepseekv3.1-DEEPSEEK_SCENARIOS",
    "gpt-5-mini_evaluation_visualisations_deepseekv3.1-GPT_SCENARIOS",
    "gpt-5-mini_evaluation_visualisations_gpt5-mini-DEEPSEEK_SCENARIOS",
    "gpt-5-mini_evaluation_visualisations_gpt5-mini-GPT_SCENARIOS",

    # --- NEW: mistral-medium-3.1 model ---
    "mistral-medium-3.1_evaluation_visualisations_deepseekv3.1-DEEPSEEK_SCENARIOS",
    "mistral-medium-3.1_evaluation_visualisations_deepseekv3.1-GPT_SCENARIOS",
    "mistral-medium-3.1_evaluation_visualisations_gpt5-mini-DEEPSEEK_SCENARIOS",
    "mistral-medium-3.1_evaluation_visualisations_gpt5-mini-GPT_SCENARIOS",
]

root = "./"

dirs = [
    os.path.join(root, d)
    for d in os.listdir(root)
    if os.path.isdir(os.path.join(root, d)) 
    and d.endswith((
        "DEEPSEEK_SCENARIOS",
        "GPT_SCENARIOS",
    ))
]

agent_name_list = [
    'Joe Biden',
    'Christine "Lady Bird" McPherson',
    'Love Quinn',
    'Fallon Carrington',
    'Michael Scott',
    'Tyler Durden',
    'Hannibal Lecter',
    'Dwight K. Schrute',
    'Skyler White',
    'Walter White',
]
for agent_name in agent_name_list:
    plot_agent_personal_shared_by_model(
        directories=dirs,
        agent_name=agent_name
    )
    plot_agent_personal_shared_by_model_for_eval(
        directories=dirs,
        agent_name=agent_name,
        eval_model_filter='gpt5-mini'
    )
    