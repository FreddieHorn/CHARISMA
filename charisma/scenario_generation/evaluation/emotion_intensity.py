import csv
import pandas as pd
from collections import defaultdict
from statistics import mean
import re
def extract_words_from_text(text):
    """Remove punctuation and convert to lowercase words."""
    if not isinstance(text, str):
        return []
    words = re.findall(r'\b[\w-]+\b', text.lower())
    return words
def load_lexicon(
    lexicon_path: str
)  -> dict:
    """
    Ingests the emotion intensity lexicon from a CSV file.

    Args:
        lexicon_path (str): Path to the emotion intensity lexicon CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the emotion intensity lexicon.
    """
    lexicon = defaultdict(dict)
    with open(lexicon_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                word = parts[0].lower()
                emotion = parts[1]
                intensity = float(parts[2])
                lexicon[word][emotion] = intensity
    return lexicon

def analyze_scenario(scenario_text, lexicon):
    """
    Analyze a single scenario text using the lexicon.
    Returns a dictionary with key metrics for difficulty comparison.
    """
    words = extract_words_from_text(scenario_text)
    emotion_results = defaultdict(list)
    matched_words = defaultdict(list)
    
    # Collect scores for each emotion
    for word in words:
        if word in lexicon:
            for emotion, score in lexicon[word].items():
                emotion_results[emotion].append(score)
                matched_words[emotion].append(word)
    
    # Calculate statistics for each emotion
    results = {}
    for emotion in ['anger', 'fear', 'sadness', 'surprise', 'joy', 'disgust', 'trust', 'anticipation']:  # All target emotions
        scores = emotion_results.get(emotion, [])
        matched = matched_words.get(emotion, [])
        
        results.update({
            f'{emotion}_mean': mean(scores) if scores else 0.0,
            f'{emotion}_median': pd.Series(scores).median() if scores else 0.0,
            f'{emotion}_iqr': pd.Series(scores).quantile(0.75) - pd.Series(scores).quantile(0.25) if scores else 0.0,
            f'{emotion}_count': len(scores),
            f'{emotion}_words': '|'.join(matched)
        })
    
    return results

def process_scenarios_csv(input_csv_path, output_csv_path, lexicon_path):
    """Process all scenario files and save results."""
    lexicon = load_lexicon(lexicon_path)

    with open(input_csv_path, 'r', encoding='utf-8') as infile, \
         open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        
        # Prepare output columns for all emotions
        emotion_metrics = ['mean', 'median', 'iqr', 'count', 'words']
        emotions = ['anger', 'fear', 'sadness', 'surprise', 'joy', 'disgust', 'trust', 'anticipation']
        new_columns = [f'{emotion}_{metric}' for emotion in emotions for metric in emotion_metrics]
        
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames + new_columns)
        writer.writeheader()
        
        for row in reader:
            scenario_text = row.get('scenario', '')
            words = extract_words_from_text(scenario_text)
            
            # Initialize emotion trackers
            emotion_scores = {emotion: [] for emotion in emotions}
            emotion_words = {emotion: [] for emotion in emotions}
            
            # Analyze each word
            for word in words:
                if word in lexicon:
                    for emotion, score in lexicon[word].items():
                        if emotion in emotions:  # Only track our target emotions
                            emotion_scores[emotion].append(score)
                            emotion_words[emotion].append(word)
            
            # Calculate statistics for each emotion
            analysis = {}
            for emotion in emotions:
                scores = emotion_scores[emotion]
                series = pd.Series(scores)
                
                analysis.update({
                    f'{emotion}_mean': series.mean() if not series.empty else 0.0,
                    f'{emotion}_median': series.median() if not series.empty else 0.0,
                    f'{emotion}_iqr': (series.quantile(0.75) - series.quantile(0.25)) if not series.empty else 0.0,
                    f'{emotion}_count': len(scores),
                    f'{emotion}_words': '|'.join(emotion_words[emotion])
                })
            
            # Write combined row
            writer.writerow({**row, **analysis})


if __name__ == "__main__":
    process_scenarios_csv(
        input_csv_path='outputs/goals_deepseek_masters__scenarios_Hard.csv',
        output_csv_path='outputs/scenario_evaluation/hard_deepseek_scenarios.csv',
        lexicon_path='inputs/NRC-Emotion-Intensity-Lexicon-v1.txt'
    )
    # print(ingest_emotion_intensity_lexicon('inputs/NRC-Emotion-Intensity-Lexicon-v1.txt'))