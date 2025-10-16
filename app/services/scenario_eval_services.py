import streamlit as st
from collections import defaultdict
from statistics import mean
import pandas as pd
from charisma.util import extract_words_from_text
from logging import getLogger
from typing import Dict, Tuple, List
from dataclasses import dataclass
from transformers import pipeline
from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI
log = getLogger(__name__)

@dataclass
class AlignmentScore:
    label: str
    score: float

class EmotionIntensityService:
    def __init__(self, lexicon_path: str = "inputs/NRC-Emotion-Intensity-Lexicon-v1.txt"):
        self.lexicon_path = lexicon_path
        self.lexicon = None
        self._load_lexicon()
    
    def _load_lexicon(self):
        """Load the emotion intensity lexicon"""
        try:
            self.lexicon = defaultdict(dict)
            with open(self.lexicon_path, 'r') as file:
                for line in file:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        word = parts[0].lower()
                        emotion = parts[1]
                        intensity = float(parts[2])
                        self.lexicon[word][emotion] = intensity
            log.info(f"Loaded emotion lexicon with {len(self.lexicon)} words")
        except FileNotFoundError:
            st.error(f"❌ Emotion lexicon file not found at {self.lexicon_path}")
            self.lexicon = defaultdict(dict)
        except Exception as e:
            log.error(f"Error loading emotion lexicon: {e}")
            st.error("❌ Failed to load emotion lexicon")
            self.lexicon = defaultdict(dict)
    
    def analyze_scenario_emotions(self, scenario_text: str) -> dict:
        """
        Analyze emotion intensity in scenario text
        
        Returns:
            dict: Emotion intensity scores and statistics
        """
        if not self.lexicon:
            return {}
        
        words = extract_words_from_text(scenario_text)
        emotion_results = defaultdict(list)
        matched_words = defaultdict(list)
        
        # Collect scores for each emotion
        for word in words:
            if word in self.lexicon:
                for emotion, score in self.lexicon[word].items():
                    emotion_results[emotion].append(score)
                    matched_words[emotion].append(word)
        
        # Calculate statistics for each emotion
        results = {}
        target_emotions = ['anger', 'fear', 'sadness', 'surprise', 'joy', 'disgust', 'trust', 'anticipation']
        
        for emotion in target_emotions:
            scores = emotion_results.get(emotion, [])
            matched = matched_words.get(emotion, [])
            
            results.update({
                f'{emotion}_mean': mean(scores) if scores else 0.0,
                f'{emotion}_median': pd.Series(scores).median() if scores else 0.0,
                f'{emotion}_count': len(scores),
                f'{emotion}_words': matched
            })
        
        # Add overall metrics
        results['total_emotion_words'] = sum(len(words) for words in matched_words.values())
        results['unique_emotion_words'] = len(set(word for words in matched_words.values() for word in words))
        
        return results
    
class EntailmentService:
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.model_name = model_name
        self.nli_model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the NLI model"""
        try:
            with st.spinner("🔄 Loading entailment model...it may take a while..."):
                self.nli_model = pipeline(
                    task="zero-shot-classification",
                    model=self.model_name,
                    multi_label=False
                )
            log.info("Entailment model loaded successfully")
        except Exception as e:
            log.error(f"Failed to load entailment model: {e}")
            st.error("❌ Failed to load entailment model. Entailment analysis will be disabled.")
            self.nli_model = None
    
    def calculate_scenario_entailment(self, scenario: str, schema: Dict[str, str]) -> Tuple[Dict[str, AlignmentScore], float]:
        """
        Calculate NLI entailment scores between scenario and schema
        
        Args:
            scenario: The generated scenario text
            schema: Dictionary with schema elements (goal_setup_data)
            
        Returns:
            Tuple[Dict[str, AlignmentScore], float]: Per-slot scores and overall average
        """
        if self.nli_model is None:
            return {}, 0.0
        
        slot_scores = {}
        total_score = 0.0
        count = 0
        
        for slot_key, hypothesis in schema.items():
            try:
                result = self.nli_model(scenario, candidate_labels=[hypothesis], hypothesis_template="{}")
                label = result["labels"][0]
                score = result["scores"][0]
                slot_scores[slot_key] = AlignmentScore(label=label, score=score)
                total_score += score
                count += 1
            except Exception as e:
                log.error(f"Error calculating entailment for {slot_key}: {e}")
                slot_scores[slot_key] = AlignmentScore(label="ERROR", score=0.0)
        
        average_entailment = total_score / count if count > 0 else 0.0
        return slot_scores, average_entailment
    
    
class OpenRouterLLM(DeepEvalBaseLLM):
    def __init__(self, *, client, model_name: str, provider: str | None = None, max_tokens: int = 1000):
        self._client = client
        self._model_name = model_name
        self._provider = provider
        self._max_tokens = max_tokens

    def load_model(self):
        return self._client

    def get_model_name(self) -> str:
        return f"OpenRouter:{self._model_name}"

    def generate(self, prompt: str) -> str:
        extra_body = {"provider": {"only": [self._provider]}} if self._provider else {}
        resp = self._client.chat.completions.create(
            extra_body=extra_body,
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._max_tokens,
        )
        return resp.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        return list(map(self.generate, prompts))

class G_EvalService:
    def __init__(self, client: OpenAI, model_name: str = "deepseek/deepseek-chat-v3.1", provider: str = "deepinfra/fp4"):
        self.client = client
        self.model_name = model_name
        self.provider = provider
        self.open_router_llm = None
        self.coherence_metric = None
        self.fluency_metric = None
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize the G-Eval metrics"""
        try:
            with st.spinner("🔄 Initializing G-Eval metrics..."):
                self.open_router_llm = OpenRouterLLM(
                    client=self.client, 
                    model_name=self.model_name, 
                    provider=self.provider
                )
                
                # Initialize coherence metric
                self.coherence_metric = GEval(
                    name="Scenario-Coherence",
                    evaluation_steps=[
                        "Check if the 'shared_goal' is clearly instantiated in the scenario.",
                        "Verify that agent roles and goals match what is described in the input.",
                        "Assess if the scenario flows logically and consistently with no logical contradictions.",
                        "Evaluate if the scenario maintains consistent character behaviors and motivations.",
                        "Check if the scenario progression follows a natural and believable sequence."
                    ],
                    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                    model=self.open_router_llm,
                    threshold=0.5,
                    strict_mode=False
                )
                
                # Initialize fluency metric
                self.fluency_metric = GEval(
                    name="Scenario-Fluency",
                    evaluation_steps=[
                        "Check if the grammar and sentence structure are correct (e.g., no run-ons, fragments, or tense errors).",
                        "Evaluate punctuation and basic formatting (e.g., proper use of commas, periods, line breaks).",
                        "Assess whether the text reads smoothly and naturally (i.e., does it flow like native-written English?).",
                        "Check if vocabulary is appropriate and not overly repetitive or awkward.",
                        "Judge overall clarity: is the scenario easy to understand from start to end?"
                    ],
                    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
                    model=self.open_router_llm,
                    threshold=0.5,
                    strict_mode=False
                )
                
            log.info("G-Eval metrics initialized successfully")
        except Exception as e:
            log.error(f"Failed to initialize G-Eval metrics: {e}")
            st.error("❌ Failed to initialize G-Eval metrics. Text quality analysis will be disabled.")
    
    def evaluate_scenario_quality(self, scenario: str, scenario_setting: Dict) -> Tuple[Dict, Dict]:
        """
        Evaluate scenario quality using G-Eval
        
        Returns:
            Tuple[Dict, Dict]: (coherence_results, fluency_results)
        """
        if not self.coherence_metric or not self.fluency_metric:
            return {}, {}
        
        try:
        #     Create test case
            test_case = LLMTestCase(
                input=str(scenario_setting),
                actual_output=scenario
            )
            
            # Measure coherence
            with st.spinner("Evaluating scenario coherence..."):
                coherence_score = self.coherence_metric.measure(test_case)
                log.info(f"Coherence score: {coherence_score}")
                coherence_results = {
                    'score': coherence_score,
                    'reason': self.coherence_metric.reason,
                    'success': self.coherence_metric.success
                }
            
            # Measure fluency
            with st.spinner("Evaluating scenario fluency..."):
                fluency_score = self.fluency_metric.measure(test_case)
                log.info(f"Fluency score: {fluency_score}")
                log.info(f"Fluency reason: {self.fluency_metric.reason}")
                fluency_results = {
                    'score': fluency_score,
                    'reason': self.fluency_metric.reason,
                    'success': self.fluency_metric.success
                }
            
            log.info(f"G-Eval results - Coherence: {coherence_results['score']}, Fluency: {fluency_results['score']}")
            
            return coherence_results, fluency_results
                
        except Exception as e:
            log.error(f"Error in G-Eval evaluation: {e}")
            st.error("❌ Error during scenario quality evaluation")
            return {}, {}