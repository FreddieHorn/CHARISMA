import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer


OUTPUT_PLOT = "charisma/visualisations/plots/data_analisis/"
# Load your CSV file
df = pd.read_csv("outputs/goals_deepseek__scenario_generation_Easy.csv")  # Replace with your file name
# Step 1: Stack agent1 and agent2 goals into a long format
agent1_df = df[['agent1_role', 'social_goal_category']].rename(columns={'agent1_role': 'role'})
agent2_df = df[['agent2_role', 'social_goal_category']].rename(columns={'agent2_role': 'role'})
all_roles_df = pd.concat([agent1_df, agent2_df])

# Drop duplicates to avoid embedding the same role multiple times
all_roles_df = all_roles_df.drop_duplicates(subset=['role']).reset_index(drop=True)

# Step 2: Embed role names using SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
role_embeddings = model.encode(all_roles_df['role'])

# Step 3: Dimensionality reduction
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
tsne_result = tsne.fit_transform(role_embeddings)

# Step 4: Prepare DataFrame for plotting
tsne_df = pd.DataFrame(tsne_result, columns=['x', 'y'])
tsne_df['role'] = all_roles_df['role']
tsne_df['category'] = all_roles_df['social_goal_category']

# Step 5: Plot with color-coded social goal categories
plt.figure(figsize=(12, 9))
sns.scatterplot(data=tsne_df, x='x', y='y', hue='category', s=100, palette='tab10')

# Annotate role names
for _, row in tsne_df.iterrows():
    plt.text(row['x'] + 0.2, row['y'], row['role'], fontsize=8)

plt.title("Semantic Embedding of Agent Roles Colored by Social Goal Category")
plt.legend(title='Social Goal')
plt.tight_layout()
# Step 1: Extract unique role names from both columns
# all_roles = pd.unique(df[['agent1_role', 'agent2_role']].values.ravel())

# # Step 2: Embed roles using a pretrained model
# model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast, and accurate
# role_embeddings = model.encode(all_roles)

# # Step 3: Optional - Cluster roles
# k = 8  # You can tune this
# kmeans = KMeans(n_clusters=k, random_state=42)
# clusters = kmeans.fit_predict(role_embeddings)

# # Step 4: Reduce dimensions for visualization
# tsne = TSNE(n_components=2, random_state=42, perplexity=5)
# tsne_result = tsne.fit_transform(role_embeddings)

# # Step 5: Plot
# tsne_df = pd.DataFrame(tsne_result, columns=['x', 'y'])
# tsne_df['role'] = all_roles
# tsne_df['cluster'] = clusters

# plt.figure(figsize=(10, 7))
# sns.scatterplot(data=tsne_df, x='x', y='y', hue='cluster', s=100, palette='tab10')

# # Annotate points
# for _, row in tsne_df.iterrows():
#     plt.text(row['x'] + 0.2, row['y'], row['role'], fontsize=9)

# plt.title("Semantic Embedding of Role Names")
# plt.legend(title='Cluster')
# plt.tight_layout()

# # Step 1: Create a co-occurrence matrix
# pairs = [tuple(sorted([r1, r2])) for r1, r2 in zip(df['agent1_role'], df['agent2_role'])]
# all_roles = sorted(set(df['agent1_role']).union(set(df['agent2_role'])))

# # Initialize co-occurrence matrix
# co_matrix = pd.DataFrame(0, index=all_roles, columns=all_roles)

# for r1, r2 in pairs:
#     co_matrix.loc[r1, r2] += 1
#     if r1 != r2:
#         co_matrix.loc[r2, r1] += 1  # Symmetric

# # Optional: Normalize rows (can help)
# co_matrix_normalized = co_matrix.div(co_matrix.sum(axis=1) + 1e-9, axis=0)

# # Step 2: Dimensionality Reduction (PCA → t-SNE)
# pca = PCA(n_components=10)
# pca_result = pca.fit_transform(co_matrix_normalized)

# tsne = TSNE(n_components=2, perplexity=5, random_state=42)
# tsne_result = tsne.fit_transform(pca_result)

# # Step 3: KMeans Clustering
# k = 5  # Choose number of clusters (adjust as needed)
# kmeans = KMeans(n_clusters=k, random_state=42)
# clusters = kmeans.fit_predict(pca_result)

# # Step 4: Visualization
# tsne_df = pd.DataFrame(tsne_result, columns=['x', 'y'])
# tsne_df['role'] = co_matrix.index
# tsne_df['cluster'] = clusters

# plt.figure(figsize=(10, 7))
# sns.scatterplot(data=tsne_df, x='x', y='y', hue='cluster', palette='tab10', s=100)

# # Annotate roles
# for _, row in tsne_df.iterrows():
#     plt.text(row['x'] + 0.5, row['y'], row['role'], fontsize=9)

# plt.title('Clustering of Roles Based on Co-occurrence')
# plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig(Path(OUTPUT_PLOT) / f'roles_3.png', dpi=300)