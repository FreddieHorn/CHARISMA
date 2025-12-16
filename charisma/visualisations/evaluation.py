# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# # Read the CSV file
# df = pd.read_csv('copies/goals_deepseek_masters__scenarios_Easy__evaluation.csv')

# # Create lists to store agent data
# agent_data = {}

# # Process each row
# for index, row in df.iterrows():
#     agents = eval(row['agents'])  # Convert string list to actual list
    
#     # Extract agent names
#     agent1 = agents[0]
#     agent2 = agents[1]
    
#     # Extract scores
#     shared_score = row['shared_goal_completion_score']
#     agent1_score = row['agent1_goal_completion_score']
#     agent2_score = row['agent2_goal_completion_score']
    
#     # Initialize agent data if not exists
#     if agent1 not in agent_data:
#         agent_data[agent1] = {'personal_scores': [], 'shared_scores': [], 'count': 0}
#     if agent2 not in agent_data:
#         agent_data[agent2] = {'personal_scores': [], 'shared_scores': [], 'count': 0}
    
#     # Add scores to respective agents
#     agent_data[agent1]['personal_scores'].append(agent1_score)
#     agent_data[agent1]['shared_scores'].append(shared_score)
#     agent_data[agent1]['count'] += 1
    
#     agent_data[agent2]['personal_scores'].append(agent2_score)
#     agent_data[agent2]['shared_scores'].append(shared_score)
#     agent_data[agent2]['count'] += 1

# # Calculate averages for each agent
# results = []
# for agent, data in agent_data.items():
#     avg_personal = np.mean(data['personal_scores'])
#     avg_shared = np.mean(data['shared_scores'])
#     count = data['count']
    
#     results.append({
#         'Agent': agent,
#         'Average Personal Goal Completion': round(avg_personal, 2),
#         'Average Shared Goal Completion': round(avg_shared, 2),
#         'Number of Interactions': count
#     })

# # Create results dataframe
# results_df = pd.DataFrame(results)

# # Sort by average personal goal completion (descending)
# results_df = results_df.sort_values('Average Personal Goal Completion', ascending=False)

# # Display results
# print("Agent Performance Summary:")
# print("=" * 70)
# for _, row in results_df.iterrows():
#     print(f"{row['Agent']:25} | Personal: {row['Average Personal Goal Completion']:4.2f} | Shared: {row['Average Shared Goal Completion']:4.2f} | Interactions: {row['Number of Interactions']}")

# # Calculate overall averages
# overall_avg_personal = results_df['Average Personal Goal Completion'].mean()
# overall_avg_shared = results_df['Average Shared Goal Completion'].mean()

# print("\n" + "=" * 70)
# print(f"{'Overall Averages':25} | Personal: {overall_avg_personal:4.2f} | Shared: {overall_avg_shared:4.2f}")

# # Additional analysis: Show agents with highest and lowest scores
# print("\n" + "=" * 70)
# print("Top Performers (Personal Goals):")
# top_personal = results_df.nlargest(3, 'Average Personal Goal Completion')
# for _, row in top_personal.iterrows():
#     print(f"  {row['Agent']}: {row['Average Personal Goal Completion']:.2f}")

# print("\nLowest Performers (Personal Goals):")
# bottom_personal = results_df.nsmallest(3, 'Average Personal Goal Completion')
# for _, row in bottom_personal.iterrows():
#     print(f"  {row['Agent']}: {row['Average Personal Goal Completion']:.2f}")
# def create_stacked_bar_chart():
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
#     # Personal vs Shared scores
#     bars1 = ax1.bar(results_df['Agent'], results_df['Personal_Score'], 
#                    label='Personal Goals', alpha=0.8)
#     bars2 = ax1.bar(results_df['Agent'], results_df['Shared_Score'], 
#                    bottom=results_df['Personal_Score'], 
#                    label='Shared Goals', alpha=0.8)
    
#     ax1.set_ylabel('Scores')
#     ax1.set_title('Agent Performance: Personal vs Shared Goals')
#     ax1.legend()
#     ax1.tick_params(axis='x', rotation=45)
    
#     # Overall performance with interaction count
#     colors = plt.cm.YlOrRd(results_df['Interactions'] / max(results_df['Interactions']))
#     bars = ax2.bar(results_df['Agent'], results_df['Overall_Score'], color=colors)
#     ax2.set_ylabel('Overall Score')
#     ax2.set_title('Overall Performance (Color intensity = Number of Interactions)')
#     ax2.tick_params(axis='x', rotation=45)
    
#     # Add value labels on bars
#     for bar in bars:
#         height = bar.get_height()
#         ax2.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{height:.2f}',
#                 ha='center', va='bottom')
    
#     plt.tight_layout()
#     plt.show()
    
# # Save results to CSV
# results_df.to_csv('outputs/agent_performance_analysis.csv', index=False)
# print(f"\nDetailed results saved to 'outputs/agent_performance_analysis.csv'")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

model = "deepseekv3.1"
DIFFICULTY = "Easy"
output_dir = f"exp1-grok-4-fast-interaction_generation-{DIFFICULTY}"
# os.makedirs(output_dir, exist_ok=True)

# Set up styling
plt.style.use('default')
sns.set_palette("husl")

# Read the CSV file
df = pd.read_csv(f"experiment_1/goals_deepseek_masters__scenarios_Hard__mistral-medium-3.1-interaction_generation__evaluation_deepseekv3.1.csv")
# Create lists to store agent data
CONF_THRESHOLD = 0.8

agent_data = {}
confidence_data = {}
shared_confidence_data = {}

# Track lowest confidence seen in the raw file (all rows)
lowest_personal_conf_raw = float("inf")
lowest_shared_conf_raw = float("inf")

# Process each row
for index, row in df.iterrows():
    agents = eval(row['agents'])
    agent1 = agents[0]
    agent2 = agents[1]

    shared_score = row['shared_goal_completion_score']
    agent1_score = row['agent1_goal_completion_score']
    agent2_score = row['agent2_goal_completion_score']
    agent1_confidence = row['agent1_confidence']
    agent2_confidence = row['agent2_confidence']
    shared_confidence = row['shared_goal_confidence']

    # Update raw minima (regardless of threshold)
    lowest_personal_conf_raw = min(lowest_personal_conf_raw, float(agent1_confidence), float(agent2_confidence))
    lowest_shared_conf_raw = min(lowest_shared_conf_raw, float(shared_confidence))

    # Initialize agent data if not exists
    if agent1 not in agent_data:
        agent_data[agent1] = {'personal_scores': [], 'shared_scores': []}
        confidence_data[agent1] = []
        shared_confidence_data[agent1] = []
    if agent2 not in agent_data:
        agent_data[agent2] = {'personal_scores': [], 'shared_scores': []}
        confidence_data[agent2] = []
        shared_confidence_data[agent2] = []

    # -------------------------
    # Add scores ONLY if confidence passes threshold
    # -------------------------

    # Agent 1 personal
    if agent1_confidence >= CONF_THRESHOLD:
        agent_data[agent1]['personal_scores'].append(agent1_score)
        confidence_data[agent1].append(agent1_confidence)

    # Agent 2 personal
    if agent2_confidence >= CONF_THRESHOLD:
        agent_data[agent2]['personal_scores'].append(agent2_score)
        confidence_data[agent2].append(agent2_confidence)

    # Shared (applies to BOTH agents for this row)
    if shared_confidence >= CONF_THRESHOLD:
        agent_data[agent1]['shared_scores'].append(shared_score)
        shared_confidence_data[agent1].append(shared_confidence)

        agent_data[agent2]['shared_scores'].append(shared_score)
        shared_confidence_data[agent2].append(shared_confidence)

# Calculate averages for each agent
results = []
all_personal_scores = []
all_shared_scores = []
all_confidences = []
all_shared_confidences = []

for agent, data in agent_data.items():
    personal_scores = data['personal_scores']
    shared_scores = data['shared_scores']
    personal_confs = confidence_data[agent]
    shared_confs = shared_confidence_data[agent]

    avg_personal = np.mean(personal_scores) if personal_scores else np.nan
    avg_shared = np.mean(shared_scores) if shared_scores else np.nan
    avg_confidence = np.mean(personal_confs) if personal_confs else np.nan
    avg_shared_confidence = np.mean(shared_confs) if shared_confs else np.nan

    # Collect all kept scores for overall averages (only those passing threshold)
    all_personal_scores.extend(personal_scores)
    all_shared_scores.extend(shared_scores)
    all_confidences.extend(personal_confs)
    all_shared_confidences.extend(shared_confs)

    results.append({
        'Agent': agent,
        'Personal_Score': avg_personal,
        'Shared_Score': avg_shared,
        'Personal_Confidence': avg_confidence,
        'Shared_Confidence': avg_shared_confidence,
        'Overall_Score': np.nanmean([avg_personal, avg_shared]),
        'Overall_Confidence': np.nanmean([avg_confidence, avg_shared_confidence])
    })

results_df = pd.DataFrame(results)
# keep agents that have at least one kept score (personal or shared)
results_df = results_df.dropna(subset=['Personal_Score', 'Shared_Score'], how='all')
results_df = results_df.sort_values('Personal_Score', ascending=False, na_position='last')

# Calculate overall averages (over kept rows only)
overall_avg_personal = np.mean(all_personal_scores) if all_personal_scores else np.nan
overall_avg_shared = np.mean(all_shared_scores) if all_shared_scores else np.nan
overall_avg_personal_confidence = np.mean(all_confidences) if all_confidences else np.nan
overall_avg_shared_confidence = np.mean(all_shared_confidences) if all_shared_confidences else np.nan
overall_avg_total = np.nanmean([overall_avg_personal, overall_avg_shared])
overall_avg_total_confidence = np.nanmean([overall_avg_personal_confidence, overall_avg_shared_confidence])


# print("\nAgent Performance Summary (with Confidence):")
# print("=" * 110)
# for _, row in results_df.iterrows():
#     print(f"{row['Agent']:20} | Personal: {row['Personal_Score']:4.2f} | Shared: {row['Shared_Score']:4.2f} | Pers_Conf: {row['Personal_Confidence']:4.2f} | Shrd_Conf: {row['Shared_Confidence']:4.2f} | Overall: {row['Overall_Score']:4.2f}")

print(f"\nAgent Performance Summary (Personal_Conf > {CONF_THRESHOLD} AND Shared_Conf > {CONF_THRESHOLD}):")
print("=" * 110)

if results_df.empty:
    print("No agents passed both confidence thresholds.")
else:
    for _, row in results_df.iterrows():
        print(
            f"{row['Agent']:20} | Personal: {row['Personal_Score']:4.2f} | Shared: {row['Shared_Score']:4.2f} "
        )
def create_scatter_plot():
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot colored by overall performance
    scatter = plt.scatter(results_df['Personal_Score'], 
                         results_df['Shared_Score'],
                         s=100,  # Fixed size
                         alpha=0.7,
                         c=results_df['Overall_Score'],
                         cmap='viridis')
    
    # Add agent names as annotations
    for i, row in results_df.iterrows():
        plt.annotate(row['Agent'], 
                    (row['Personal_Score'], row['Shared_Score']),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=9)
    
    plt.colorbar(scatter, label='Overall Score')
    plt.xlabel('Personal Goal Completion Score')
    plt.ylabel('Shared Goal Completion Score')
    plt.title('Agent Performance: Personal vs Shared Goals\n(Color = Overall Performance)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{output_dir}/scatter_plot_personal_vs_shared_{DIFFICULTY}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved scatter plot")

# 2. PERFORMANCE QUADRANT ANALYSIS
def create_quadrant_analysis():
    plt.figure(figsize=(12, 10))
    
    # Calculate quadrant boundaries
    personal_avg = results_df['Personal_Score'].mean()
    shared_avg = results_df['Shared_Score'].mean()
    
    # Create scatter plot with colors based on quadrant
    colors = []
    for i, row in results_df.iterrows():
        if row['Personal_Score'] >= personal_avg and row['Shared_Score'] >= shared_avg:
            colors.append('green')  # High Personal, High Shared
        elif row['Personal_Score'] >= personal_avg and row['Shared_Score'] < shared_avg:
            colors.append('orange')  # High Personal, Low Shared
        elif row['Personal_Score'] < personal_avg and row['Shared_Score'] >= shared_avg:
            colors.append('blue')    # Low Personal, High Shared
        else:
            colors.append('red')     # Low Personal, Low Shared
    
    plt.scatter(results_df['Personal_Score'], results_df['Shared_Score'], 
               s=100, c=colors, alpha=0.7)
    
    # Add quadrant lines
    plt.axvline(x=personal_avg, color='black', linestyle='--', alpha=0.7, label=f'Personal Avg: {personal_avg:.2f}')
    plt.axhline(y=shared_avg, color='black', linestyle='--', alpha=0.7, label=f'Shared Avg: {shared_avg:.2f}')
    
    # Add agent labels
    for i, row in results_df.iterrows():
        plt.annotate(row['Agent'], 
                    (row['Personal_Score'], row['Shared_Score']),
                    xytext=(10, 10), 
                    textcoords='offset points',
                    fontsize=9)
    
    # Add quadrant labels
    plt.text(personal_avg/2, shared_avg*1.8, 'Low Personal\nHigh Shared', 
             fontsize=12, ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", alpha=0.7))
    plt.text(personal_avg*1.8, shared_avg*1.8, 'High Personal\nHigh Shared', 
             fontsize=12, ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", alpha=0.7))
    plt.text(personal_avg/2, shared_avg/2, 'Low Personal\nLow Shared', 
             fontsize=12, ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.5", fc="lightcoral", alpha=0.7))
    plt.text(personal_avg*1.8, shared_avg/2, 'High Personal\nLow Shared', 
             fontsize=12, ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.7))
    
    plt.xlabel('Personal Goal Completion Score')
    plt.ylabel('Shared Goal Completion Score')
    plt.title('Agent Performance Quadrant Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{output_dir}/quadrant_analysis_{DIFFICULTY}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved quadrant analysis")


# 5. SIDE-BY-SIDE COMPARISON
def create_side_by_side():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Personal scores
    ax1.bar(results_df['Agent'], results_df['Personal_Score'], color='skyblue', alpha=0.7)
    ax1.set_ylabel('Score')
    ax1.set_title('Personal Goal Completion Scores')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Shared scores
    ax2.bar(results_df['Agent'], results_df['Shared_Score'], color='lightcoral', alpha=0.7)
    ax2.set_ylabel('Score')
    ax2.set_title('Shared Goal Completion Scores')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for ax, scores in [(ax1, results_df['Personal_Score']), (ax2, results_df['Shared_Score'])]:
        for i, v in enumerate(scores):
            ax.text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{output_dir}/side_by_side_comparison_{DIFFICULTY}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved side-by-side comparison")

def create_grouped_bar_chart():
    plt.figure(figsize=(14, 8))
    
    # Set the width of the bars and the positions
    bar_width = 0.35
    x_pos = np.arange(len(results_df))
    
    # Create bars for personal and shared scores
    bars_personal = plt.bar(x_pos - bar_width/2, results_df['Personal_Score'], 
                           bar_width, label='Personal Goals', 
                           color='skyblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    bars_shared = plt.bar(x_pos + bar_width/2, results_df['Shared_Score'], 
                         bar_width, label='Shared Goals', 
                         color='lightcoral', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize the chart
    plt.xlabel('Agents')
    plt.ylabel('Performance Score')
    plt.title('Agent Performance: Personal vs Shared Goals')
    plt.xticks(x_pos, results_df['Agent'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars_personal)
    add_value_labels(bars_shared)
    
    # Adjust y-axis to accommodate labels
    plt.ylim(0, max(max(results_df['Personal_Score']), max(results_df['Shared_Score'])) + 1)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{output_dir}/grouped_bar_chart_{DIFFICULTY}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved grouped bar chart")
# Run all visualizations
print(f"\nGenerating and saving visualizations to '{output_dir}' directory...")

# create_scatter_plot()
# create_quadrant_analysis()
# create_ranking_chart()
# create_grouped_bar_chart()
# create_side_by_side()

print("\nAll visualizations completed and saved!")

# Save detailed results
results_df.to_csv(f'{output_dir}/agent_performance_results.csv', index=False)
print(f"Detailed results saved to '{output_dir}/agent_performance_results.csv'")

# Print key insights
print("\n" + "="*50)
print("KEY INSIGHTS:")
print(f"Most Consistent Performer: {results_df.loc[results_df['Overall_Score'].idxmax(), 'Agent']}")
print(f"Best at Personal Goals: {results_df.loc[results_df['Personal_Score'].idxmax(), 'Agent']}")
print(f"Best at Shared Goals: {results_df.loc[results_df['Shared_Score'].idxmax(), 'Agent']}")
print(f"Performance Range: {results_df['Overall_Score'].min():.2f} - {results_df['Overall_Score'].max():.2f}")

print(f"\nAll plots saved in: {os.path.abspath(output_dir)}")