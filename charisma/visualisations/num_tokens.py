import pandas as pd
import matplotlib.pyplot as plt

def analyze_and_plot_with_pandas(file_path, output_file=None):
    df = pd.read_csv(file_path)
    
    # Calculate token counts and group averages
    df['length'] = df['scenario'].str.split().str.len()
    averages = df.groupby('social_goal_category')['length'].mean().sort_values(ascending=False)
    
    # Create plot
    ax = averages.plot.bar(figsize=(12, 6), color='teal', alpha=0.7)
    
    # Customize plot
    ax.set_title('Average Scenario Length by Social Goal Category', pad=20)
    ax.set_xlabel('Social Goal Category')
    ax.set_ylabel('Average Length (tokens)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 10),
                   textcoords='offset points')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    analyze_and_plot_with_pandas("outputs/goals_deepseek__scenario_generation_medium.csv", 
                                  "charisma/visualisations/plots/average_scenario_length_by_category.png")