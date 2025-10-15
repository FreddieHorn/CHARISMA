import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

INPUT_FILES = {
    "Easy": "outputs/scenario_evaluation/entailment/entailment_scores_easy.csv",
    "Medium": "outputs/scenario_evaluation/entailment/entailment_scores_medium.csv", 
    "Hard": "outputs/scenario_evaluation/entailment/entailment_scores_hard.csv"
}
OUTPUT_PLOT = "charisma/visualisations/plots/entailment_scores/"

# 1. Load and preprocess data
def load_and_label_data(file_path, difficulty):
    df = pd.read_csv(file_path)
    df['difficulty'] = difficulty  # Add difficulty label
    return df  # Keep only needed columns

# Combine all data
all_data = pd.concat([
    load_and_label_data(file, difficulty) 
    for difficulty, file in INPUT_FILES.items()
])

# Create one plot per emotion
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='difficulty',
    y=f'average_entailment_score',
    data=all_data,
    order=['Easy', 'Medium', 'Hard'],
    palette='Set2',
    width=0.5
)
plt.title(f'Average entailment score by difficulty')
plt.ylabel('Mean Entailment Score')
plt.savefig(Path(OUTPUT_PLOT) / f'average_entailment_score_comparison.png', dpi=300)
plt.close()