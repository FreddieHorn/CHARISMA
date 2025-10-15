import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
INPUT_FILES = {
    "Easy": "outputs/scenario_evaluation/easy_scenarios_2.csv",
    "Medium": "outputs/scenario_evaluation/medium_scenarios_2.csv", 
    "Hard": "outputs/scenario_evaluation/hard_scenarios_2.csv"
}
OUTPUT_PLOT = "charisma/visualisations/plots/"

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
for emotion in ['anger', 'fear', 'sadness', 'surprise', 'joy', 'disgust', 'trust', 'anticipation']:
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x='difficulty',
        y=f'{emotion}_mean',
        data=all_data,
        order=['Easy', 'Medium', 'Hard'],
        palette='Set2',
        width=0.5
    )
    plt.title(f'{emotion.capitalize()} Intensity by Difficulty')
    plt.ylabel('Mean Intensity Score')
    plt.savefig(Path(OUTPUT_PLOT) / f'{emotion}_comparison.png', dpi=300)
    plt.close()

# # 3. Add annotations
# plt.title('Emotion Score Distribution by Scenario Difficulty', pad=20)
# plt.xlabel('Difficulty Level', labelpad=10)
# plt.ylabel('Mean Emotion Intensity (0-1 scale)', labelpad=10)
# plt.ylim(0, 1)  # Adjust if your scores use different range

# # 4. Save and show
# plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
# plt.show()