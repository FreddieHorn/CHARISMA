import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import ast

OUTPUT_PLOT = "charisma/visualisations/plots/data_analisis/"
# Load your CSV file
df = pd.read_csv("outputs/goals_deepseek__scenario_generation_Easy.csv")  # Replace wcith your file name
# Step 1: Parse base_shared_goal column (stringified dict)
if isinstance(df['base_shared_goal'].iloc[0], str):
    df['base_shared_goal'] = df['base_shared_goal'].apply(ast.literal_eval)

# Step 2: Extract Abbreviation + category
df['abbreviation'] = df['base_shared_goal'].apply(lambda x: x.get('Abbreviation') if isinstance(x, dict) else None)
df = df.dropna(subset=['abbreviation'])

# Keep one social goal per unique abbreviation (if duplicate, pick first)
df = df.drop_duplicates(subset=['abbreviation']).reset_index(drop=True)

# Step 3: Sentence Embedding
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['abbreviation'])

# Step 4: t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
tsne_result = tsne.fit_transform(embeddings)

# Step 5: Prepare plot data
tsne_df = pd.DataFrame(tsne_result, columns=['x', 'y'])
tsne_df['abbreviation'] = df['abbreviation']
tsne_df['category'] = df['social_goal_category']

# Step 6: Plot
plt.figure(figsize=(12, 9))
sns.scatterplot(data=tsne_df, x='x', y='y', hue='category', s=100, palette='tab10')

# Annotate every point
for _, row in tsne_df.iterrows():
    plt.text(row['x'] + 0.2, row['y'], row['abbreviation'], fontsize=13)

plt.title("Semantic Embedding of Shared Goal Colored by Social Goal Category", fontsize=22)
plt.legend(title='Social Goal Category')
plt.tight_layout()
plt.savefig(Path(OUTPUT_PLOT) / f'shared_goals_2.png', dpi=300)