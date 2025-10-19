import streamlit as st
from transformers import pipeline
from typing import Dict, List, Tuple
from logging import getLogger

log = getLogger(__name__)

class SentimentAnalysisService:
    def __init__(self):
        self.sentiment_map = {
            "Very Negative": 0,
            "Negative": 1, 
            "Neutral": 2,
            "Positive": 3,
            "Very Positive": 4
        }
        self.pipe = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentiment analysis model"""
        try:
            with st.spinner("🔄 Loading sentiment analysis model..."):
                self.pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
            log.info("Sentiment analysis model loaded successfully")
        except Exception as e:
            log.error(f"Failed to load sentiment analysis model: {e}")
            st.error("❌ Failed to load sentiment analysis model. Sentiment analysis will be disabled.")
            self.pipe = None
            
    def analyze_sentiment(self, text: str) -> Tuple[int, str, float]:
        """Analyze sentiment of a text and return (sentiment_id, sentiment_label, confidence)"""
        if self.pipe is None:
            return 2, "Neutral", 1.0  # Default to neutral if model not loaded
        
        try:
            result = self.pipe(text)[0]
            sentiment_score = self.sentiment_map[result['label']]  # This might be the numeric label or string
            return sentiment_score, result['label']

        except Exception as e:
            log.error(f"Error in sentiment analysis: {e}")
            return 2, "Neutral"

    def analyze_conversation_sentiment(self, conversation_rows: List[Dict]) -> Dict:
        """Analyze sentiment for entire conversation"""
        if not conversation_rows:
            return {}
        
        sentiment_results = []
        speaker_sentiments = {}
        progress_bar = st.progress(0.0, text="Analyzing sentiment...")
        for i, row in enumerate(conversation_rows):
            progress_bar.progress((i + 1) / len(conversation_rows), text=f"Analyzing sentiment... {i + 1}/{len(conversation_rows)}")
            log.info(f"{len(conversation_rows)} messages to analyze, processing message {i + 1}")
            speaker = row['speaker']
            message = row['message']
            turn = i + 1
            
            sentiment_score, sentiment_label = self.analyze_sentiment(message)
            
            sentiment_data = {
                'turn': turn,
                'speaker': speaker,
                'message': message,
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label
            }
            sentiment_results.append(sentiment_data)
            
            # Initialize speaker data if not exists
            if speaker not in speaker_sentiments:
                speaker_sentiments[speaker] = {
                    'sentiment_scores': [],
                    'messages_analyzed': 0,
                    'sentiment_distribution': {label: 0 for label in self.sentiment_map.keys()}
                }
            
            # Update speaker statistics
            speaker_sentiments[speaker]['sentiment_scores'].append(sentiment_score)
            speaker_sentiments[speaker]['messages_analyzed'] += 1
            speaker_sentiments[speaker]['sentiment_distribution'][sentiment_label] += 1
        
        # Calculate averages and summaries
        conversation_summary = self._calculate_sentiment_summary(sentiment_results, speaker_sentiments)
        
        return {
            'detailed_results': sentiment_results,
            'speaker_sentiments': speaker_sentiments,
            'conversation_summary': conversation_summary,
        }
    
    def _calculate_sentiment_summary(self, sentiment_results: List[Dict], speaker_sentiments: Dict) -> Dict:
        """Calculate summary statistics for the conversation"""
        overall_sentiments = [result['sentiment_score'] for result in sentiment_results]
        log.debug(f"Overall sentiment scores: {overall_sentiments}")
        if not overall_sentiments:
            return {}
        
        avg_sentiment = sum(overall_sentiments) / len(overall_sentiments)
        overall_label = self._get_sentiment_label_from_score(avg_sentiment)
        
        summary = {
            'average_sentiment_score': avg_sentiment,
            'average_sentiment_label': overall_label,
            'total_messages_analyzed': len(sentiment_results),
            'sentiment_distribution': {label: 0 for label in self.sentiment_map.keys()}
        }
        
        
        # Calculate overall distribution
        for result in sentiment_results:
            summary['sentiment_distribution'][result['sentiment_label']] += 1
        
        # Add speaker-specific summaries
        for speaker, data in speaker_sentiments.items():
            if data['sentiment_scores']:
                speaker_avg = sum(data['sentiment_scores']) / len(data['sentiment_scores'])
                summary[f'{speaker}_average_score'] = speaker_avg
                summary[f'{speaker}_average_label'] = self._get_sentiment_label_from_score(speaker_avg)
                summary[f'{speaker}_messages_analyzed'] = data['messages_analyzed']
        
        return summary
    
    def _get_sentiment_label_from_score(self, score: float) -> str:
        """Convert numeric sentiment score to label"""
        if score < 0.5:
            return "Very Negative"
        elif score < 1.5:
            return "Negative"
        elif score < 2.5:
            return "Neutral"
        elif score < 3.5:
            return "Positive"
        else:
            return "Very Positive"