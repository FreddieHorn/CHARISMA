import json
import re
from pathlib import Path
from datetime import datetime
import csv
import random


def extract_json_string(raw_response: str) -> str:
    """
    Extracts a JSON block from a Markdown-style raw response and returns it as a JSON-formatted string.

    Args:
        raw_response (str): The raw response containing a JSON block (e.g., inside ```json ... ```).

    Returns:
        str: A valid JSON-formatted string.

    Raises:
        ValueError: If no JSON block is found or if JSON is invalid.
    """
    # match = re.search(r"```json\s*(\{.*?\})\s*```", raw_response, re.DOTALL)
    match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    json_block = match.group()
    if not match:
        raise ValueError("No valid JSON block found in the response.")

    data = json.loads(json_block)
    return data


def create_output_path(input_path, output_suffix: str = "_output") -> Path:
    input_path = Path(input_path)

    new_filename = f"{input_path.stem}_{output_suffix}{input_path.suffix}"

    output_path = input_path.parent.parent / "outputs" / new_filename

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return output_path


def sample_shared_goal(goals_csv: str, num_samples: int = 1) -> dict:
    """
    Sample a specified number of goals from a CSV file containing goals.
    
    Args:
        goals_csv (str): Path to the CSV file containing goals.
        num_samples (int): Number of goals to sample. Default is 2.

    Returns:
        dict: A dictionary containing sampled goals.
    """
    with open(goals_csv, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        records = list(csv_reader)  # Convert to list to work with it

    # Randomly select 1 unique record
    selected_records = random.sample(records, num_samples)
    return selected_records

def sample_unique_shared_goals_excluding_existing(goal_csv_path, existing_goals, num_samples):
    all_goals = sample_shared_goal(goal_csv_path, num_samples=135)  # Load all goals
    filtered_goals = [g for g in all_goals if g["Full label"] not in existing_goals]

    if len(filtered_goals) < num_samples:
        raise ValueError(f"Only {len(filtered_goals)} unique goals available; {num_samples} requested.")

    return random.sample(filtered_goals, num_samples)

def extract_words_from_text(text):
    """Remove punctuation and convert to lowercase words."""
    if not isinstance(text, str):
        return []
    words = re.findall(r'\b[\w-]+\b', text.lower())
    return words
