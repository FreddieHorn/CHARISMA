import pandas as pd
import ast
from collections import Counter

# Load the CSV
df = pd.read_csv("experiment_1/goals_deepseek_masters__scenarios_Easy__mistral-medium-3.1-interaction_generation__evaluation_deepseekv3.1.csv")

# Dictionary to store counts of behavioral codes for each agent
agent_code_counts = {}

# Iterate through each row's interaction history
for _, row in df.iterrows():
    history = row.get("interaction_history")
    
    if pd.isna(history):
        continue
    
    # Convert the string representation of list-of-dicts into actual Python objects
    try:
        interactions = ast.literal_eval(history)
    except Exception:
        continue
    
    # Count each behavioral code for each agent
    for entry in interactions:
        agent = entry.get("agent")
        code = entry.get("behavioral_code")
        
        if agent and code:
            if agent not in agent_code_counts:
                agent_code_counts[agent] = Counter()
            agent_code_counts[agent][code] += 1

# Compute the top 3 behavioral codes per agent
result = {}
for agent, counter in agent_code_counts.items():
    result[agent] = counter.most_common(3)

# Print results
for agent, codes in result.items():
    print(f"\nAgent: {agent}")
    for code, count in codes:
        print(f"  {code}: {count}")