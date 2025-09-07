# # -*- coding: utf-8 -*-
# Import libraries
import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

@dataclass
class PatientProfile:
    age_band: str
    gender: str
    setting: str
    region: str
    season: str

class NigerianDiseaseGenerator:
    """
    Synthetic data generator for simulating disease occurrence in Nigeria.
    Includes disease-specific characteristic symptoms, realistic probabilities,
    correlations, comorbidities, and seasonal effects.
    """
    def __init__(self, random_seed: int = 42, healthy_share: float = 0.18):
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Demographic distributions
        self.demo_distributions = {
            'age_band': {'0-4': 0.18, '5-14': 0.28, '15-24': 0.17, '25-44': 0.22, '45-64': 0.11, '65+': 0.04},
            'gender': {'male': 0.51, 'female': 0.49},
            'setting': {'urban': 0.52, 'rural': 0.48},
            'region': {'north': 0.40, 'middle_belt': 0.25, 'south': 0.35},
            'season': {'dry': 0.33, 'rainy': 0.50, 'transition': 0.17}
        }

        # Disease base prevalence (before demographic nudges)
        self.diseases = {
            'malaria': 0.22,
            'typhoid': 0.18,
            'peptic_ulcer': 0.10,
            'tuberculosis': 0.12,
            'measles': 0.08,
            'gastroenteritis': 0.19,
            'pneumonia_ari': 0.20,
            'diabetes': 0.10,
            'hypertension': 0.15,
            'hiv': 0.06
        }

        # Symptom list including characteristic symptoms
        self.symptoms = [
            # General symptoms
            'fever', 'headache', 'cough', 'chronic_cough', 'productive_cough', 'fatigue', 'body_ache',
            'chills', 'sweats', 'night_sweats', 'weight_loss', 'loss_of_appetite',
            # GI symptoms
            'nausea', 'vomiting', 'diarrhea', 'constipation', 'abdominal_pain', 'epigastric_pain',
            'heartburn', 'hunger_pain',
            # Respiratory symptoms
            'sore_throat', 'runny_nose', 'chest_pain', 'shortness_of_breath', 'rapid_breathing',
            'hemoptysis',
            # Genitourinary
            'dysuria', 'polyuria', 'oliguria',
            # Metabolic
            'polydipsia', 'polyphagia', 'blurred_vision',
            # Neurological
            'dizziness', 'confusion',
            # Dermatological/Physical signs
            'rash', 'maculopapular_rash', 'rose_spots', 'conjunctivitis', 'lymph_nodes',
            # Infection-related
            'recurrent_infections', 'oral_thrush'
        ]

        # Target share of healthy (or undiagnosed) cases in the population
        self.healthy_share = float(np.clip(healthy_share, 0.05, 0.4))

        # Disease configs with revised demographic nudges and symptom probabilities
        self.malaria_config = {
            'demographic_nudges': {
                'age_band': {'0-4': +0.20, '5-14': +0.08, '15-24': +0.06, '25-44': +0.03, '45-64': -0.05, '65+': -0.10},
                'gender': {'male': +0.02, 'female': -0.02}, 
                'setting': {'rural': +0.12, 'urban': -0.12},
                'region': {'south': +0.06, 'middle_belt': +0.06, 'north': +0.04},
                'season': {'rainy': +0.15, 'dry': -0.15, 'transition': +0.05}
            },
            'symptom_probabilities': {
                'fever': 0.946,  
                'headache': 0.854,  
                'body_ache': 0.718,  
                'chills': 0.762,  
                'sweats': 0.815, 
                # Common symptoms
                'fatigue': 0.85, 'nausea': 0.65, 'vomiting': 0.45, 'loss_of_appetite': 0.60,
                'diarrhea': 0.25, 'abdominal_pain': 0.20,
                # Less common in uncomplicated malaria
                'cough': 0.15, 'sore_throat': 0.08, 'runny_nose': 0.05, 'constipation': 0.10,
                'rash': 0.05, 'confusion': 0.08, 'dysuria': 0.03,
                # Rare/absent symptoms
                'chronic_cough': 0.02, 'hemoptysis': 0.01, 'chest_pain': 0.05,
                'shortness_of_breath': 0.05, 'night_sweats': 0.15
            }
        }

        self.typhoid_config = {
            'demographic_nudges': {
                'age_band': {'0-4': -0.02, '5-14': +0.05, '15-24': +0.08, '25-44': +0.06, '45-64': +0.04, '65+': -0.05},
                'gender': {'female': +0.05, 'male': -0.05},
                'setting': {'rural': +0.15, 'urban': -0.15},
                'region': {'north': +0.05, 'middle_belt': +0.02, 'south': -0.07},
                'season': {'rainy': +0.12, 'dry': -0.12, 'transition': 0}
            },
            'symptom_probabilities': {
                'fever': 0.915, 
                'headache': 0.821,  
                'constipation': 0.653,  
                # Other typhoid symptoms
                'fatigue': 0.80, 'loss_of_appetite': 0.75, 'abdominal_pain': 0.55,
                'body_ache': 0.50, 'rose_spots': 0.25,
                # Other GI symptoms
                'nausea': 0.50, 'vomiting': 0.35, 'diarrhea': 0.20,  
                # General symptoms
                'chills': 0.35, 'sweats': 0.30, 'confusion': 0.15,
                # Less relevant symptoms
                'cough': 0.20, 'sore_throat': 0.08, 'runny_nose': 0.05, 'dysuria': 0.05,
                'rash': 0.08, 'chest_pain': 0.05
            }
        }

        self.peptic_ulcer_config = {
            'demographic_nudges': {
                'age_band': {'0-4': -0.55, '5-14': -0.50, '15-24': +0.05, '25-44': +0.35, '45-64': +0.50, '65+': +0.35},
                'gender': {'male': +0.08, 'female': -0.08},
                'setting': {'urban': +0.08, 'rural': -0.08},
                'region': {'north': +0.03, 'middle_belt': +0.01, 'south': -0.04},
                'season': {'dry': +0.02, 'rainy': -0.02, 'transition': 0}
            },
            'symptom_probabilities': {
                # Characteristic peptic ulcer symptoms
                'epigastric_pain': 0.90, 'hunger_pain': 0.75, 'heartburn': 0.70,
                'abdominal_pain': 0.85, 'loss_of_appetite': 0.50, 'nausea': 0.55,
                'vomiting': 0.30,
                # General symptoms
                'fatigue': 0.40, 'headache': 0.20, 'body_ache': 0.15, 'fever': 0.05,
                # GI symptoms
                'constipation': 0.25, 'diarrhea': 0.10,
                # Rare symptoms
                'cough': 0.03, 'sore_throat': 0.03, 'runny_nose': 0.03, 'dysuria': 0.03,
                'chills': 0.05, 'sweats': 0.05, 'rash': 0.02
            }
        }

        self.tuberculosis_config = {
            'demographic_nudges': {
                'age_band': {'0-4': -0.02, '5-14': -0.08, '15-24': +0.15, '25-44': +0.18, '45-64': +0.08, '65+': +0.05},
                'gender': {'male': +0.10, 'female': -0.10},
                'setting': {'urban': +0.05, 'rural': -0.05},
                'region': {'north': +0.08, 'middle_belt': +0.02, 'south': -0.10},
                'season': {'dry': +0.02, 'rainy': -0.02, 'transition': 0}
            },
            'symptom_probabilities': {
                # Classic TB symptoms
                'chronic_cough': 0.85, 'productive_cough': 0.75, 'hemoptysis': 0.40,
                'night_sweats': 0.70, 'weight_loss': 0.80, 'fatigue': 0.75,
                'fever': 0.65, 'loss_of_appetite': 0.65,
                # Other symptoms
                'chest_pain': 0.50, 'shortness_of_breath': 0.45, 'body_ache': 0.45,
                'chills': 0.40, 'headache': 0.30,
                # GI symptoms (less common)
                'nausea': 0.20, 'vomiting': 0.10, 'abdominal_pain': 0.15, 'diarrhea': 0.08,
                # Rare symptoms
                'sore_throat': 0.10, 'runny_nose': 0.08, 'dysuria': 0.05, 'rash': 0.03,
                # Regular cough much lower than chronic
                'cough': 0.20
            }
        }

        self.measles_config = {
            'demographic_nudges': {
                'age_band': {'0-4': +0.30, '5-14': +0.15, '15-24': -0.10, '25-44': -0.15, '45-64': -0.15, '65+': -0.05},
                'gender': {'male': +0.02, 'female': -0.02},
                'setting': {'urban': +0.08, 'rural': -0.08},
                'region': {'north': +0.15, 'middle_belt': +0.05, 'south': -0.20},
                'season': {'dry': +0.20, 'rainy': -0.20, 'transition': 0}
            },
            'symptom_probabilities': {
                'cough': 0.942,  
                'runny_nose': 0.920,  
                'maculopapular_rash': 0.965,  
                'conjunctivitis': 0.918, 
                # Other classic symptoms
                'fever': 0.95,
                # General symptoms
                'fatigue': 0.75, 'headache': 0.60, 'body_ache': 0.50, 'sore_throat': 0.45,
                'loss_of_appetite': 0.60, 'chills': 0.50, 'sweats': 0.45,
                # GI symptoms (common in children)
                'nausea': 0.40, 'vomiting': 0.30, 'diarrhea': 0.25, 'abdominal_pain': 0.20,
                # Less common
                'dysuria': 0.03, 'chest_pain': 0.10, 'lymph_nodes': 0.30,
                'rash': 0.05  # Non-specific rash (vs maculopapular)
            }
        }

        self.gastroenteritis_config = {
            'demographic_nudges': {
                'age_band': {'0-4': +0.25, '5-14': +0.08, '15-24': -0.05, '25-44': -0.08, '45-64': -0.08, '65+': +0.02},
                'gender': {'male': +0.06, 'female': -0.06},
                'setting': {'rural': +0.12, 'urban': -0.12},
                'region': {'north': +0.10, 'middle_belt': +0.03, 'south': -0.13},
                'season': {'rainy': +0.15, 'dry': -0.15, 'transition': 0}
            },
            'symptom_probabilities': {
                # Classic gastroenteritis symptoms
                'diarrhea': 0.95, 'abdominal_pain': 0.75, 'vomiting': 0.70,
                'nausea': 0.70, 'fatigue': 0.65, 'loss_of_appetite': 0.50,
                'fever': 0.45, 'headache': 0.30, 'body_ache': 0.25,
                'chills': 0.25, 'sweats': 0.20,
                # Rare respiratory symptoms
                'cough': 0.08, 'sore_throat': 0.08, 'runny_nose': 0.08,
                'dysuria': 0.05, 'constipation': 0.05, 'rash': 0.03,
                'chest_pain': 0.05, 'confusion': 0.05
            }
        }

        self.pneumonia_ari_config = {
            'demographic_nudges': {
                'age_band': {'0-4': +0.20, '5-14': +0.05, '15-24': -0.08, '25-44': -0.08, '45-64': +0.02, '65+': +0.15},
                'gender': {'male': +0.08, 'female': -0.08},
                'setting': {'rural': +0.08, 'urban': -0.08},
                'region': {'north': +0.05, 'middle_belt': +0.02, 'south': -0.07},
                'season': {'dry': +0.15, 'rainy': -0.15, 'transition': 0}
            },
            'symptom_probabilities': {
                'cough': 0.905,  
                'productive_cough': 0.682, 
                'chest_pain': 0.720, 
                'shortness_of_breath': 0.783, 
                'rapid_breathing': 0.751,  
                # Other pneumonia symptoms
                'fever': 0.80,
                # General symptoms
                'fatigue': 0.70, 'chills': 0.55, 'sweats': 0.50, 'headache': 0.45,
                'body_ache': 0.50, 'loss_of_appetite': 0.45,
                # Less common
                'nausea': 0.25, 'vomiting': 0.20, 'sore_throat': 0.20, 'runny_nose': 0.15,
                'abdominal_pain': 0.10, 'diarrhea': 0.08, 'dysuria': 0.05,
                'confusion': 0.15, 'hemoptysis': 0.10
            }
        }

        self.diabetes_config = {
            'demographic_nudges': {
                'age_band': {'0-4': -0.65, '5-14': -0.60, '15-24': -0.35, '25-44': -0.05, '45-64': +0.75, '65+': +0.65},
                'gender': {'female': +0.08, 'male': -0.08},
                'setting': {'urban': +0.20, 'rural': -0.20},
                'region': {'south': +0.15, 'middle_belt': -0.05, 'north': -0.10},
                'season': {'rainy': 0, 'dry': 0, 'transition': 0}
            },
            'symptom_probabilities': {
                # Classic diabetes triad
                'polyuria': 0.70, 'polydipsia': 0.65, 'polyphagia': 0.60,
                # Common symptoms
                'fatigue': 0.80, 'blurred_vision': 0.50, 'recurrent_infections': 0.45,
                'weight_loss': 0.40, 'dysuria': 0.35,
                # General symptoms
                'headache': 0.30, 'body_ache': 0.25, 'dizziness': 0.25,
                'nausea': 0.20, 'loss_of_appetite': 0.15,
                # Complications
                'confusion': 0.10, 'shortness_of_breath': 0.08,
                # Rare/absent acute symptoms
                'fever': 0.08, 'chills': 0.05, 'sweats': 0.15, 'vomiting': 0.10,
                'abdominal_pain': 0.15, 'diarrhea': 0.08, 'cough': 0.08,
                'sore_throat': 0.05, 'runny_nose': 0.05
            }
        }

        self.hypertension_config = {
            'demographic_nudges': {
                'age_band': {'0-4': -0.70, '5-14': -0.65, '15-24': -0.40, '25-44': -0.10, '45-64': +0.80, '65+': +0.95},
                'gender': {'male': +0.03, 'female': -0.03},
                'setting': {'urban': +0.08, 'rural': -0.08},
                'region': {'south': +0.02, 'middle_belt': +0.01, 'north': -0.03},
                'season': {'rainy': 0, 'dry': 0, 'transition': 0}
            },
            'symptom_probabilities': {
                # Most hypertension is asymptomatic, but when symptomatic:
                'headache': 0.40, 'dizziness': 0.35, 'fatigue': 0.30,
                'blurred_vision': 0.20, 'chest_pain': 0.15, 'shortness_of_breath': 0.15,
                # Less common
                'body_ache': 0.15, 'nausea': 0.12, 'confusion': 0.08,
                'loss_of_appetite': 0.10, 'sweats': 0.10,
                # Rare symptoms
                'fever': 0.03, 'chills': 0.03, 'vomiting': 0.05, 'abdominal_pain': 0.05,
                'cough': 0.08, 'sore_throat': 0.03, 'runny_nose': 0.03,
                'dysuria': 0.05, 'diarrhea': 0.03
            }
        }

        self.hiv_config = {
            'demographic_nudges': {
                'age_band': {'0-4': -0.58, '5-14': -0.53, '15-24': +0.05, '25-44': +0.25, '45-64': +0.08, '65+': -0.48},
                'gender': {'female': +0.15, 'male': -0.15},
                'setting': {'urban': +0.10, 'rural': -0.10},
                'region': {'south': +0.08, 'middle_belt': +0.05, 'north': -0.13},
                'season': {'rainy': 0, 'dry': 0, 'transition': 0}
            },
            'symptom_probabilities': {
                # Advanced HIV/AIDS symptoms (chronic stage)
                'fatigue': 0.75, 'weight_loss': 0.70, 'recurrent_infections': 0.80,
                'chronic_cough': 0.50, 'diarrhea': 0.55, 'fever': 0.45,
                'night_sweats': 0.60, 'loss_of_appetite': 0.55,
                # Opportunistic infections signs
                'oral_thrush': 0.40, 'lymph_nodes': 0.65,
                # General symptoms
                'headache': 0.40, 'body_ache': 0.45, 'nausea': 0.25,
                'abdominal_pain': 0.25, 'sore_throat': 0.25,
                # Less common
                'vomiting': 0.15, 'chills': 0.25, 'dysuria': 0.15,
                'runny_nose': 0.12, 'confusion': 0.20, 'blurred_vision': 0.15,
                'shortness_of_breath': 0.25, 'chest_pain': 0.20
            }
        }

    def _sample_demographics(self) -> PatientProfile:
        """Randomly sample patient demographics based on Nigerian population distributions"""
        def pick(d: Dict[str, float]) -> str:
            return np.random.choice(list(d.keys()), p=list(d.values()))
        return PatientProfile(
            age_band=pick(self.demo_distributions['age_band']),
            gender=pick(self.demo_distributions['gender']),
            setting=pick(self.demo_distributions['setting']),
            region=pick(self.demo_distributions['region']),
            season=pick(self.demo_distributions['season']),
        )

    def _calculate_disease_probability(self, disease: str, profile: PatientProfile) -> float:
        """Calculate disease probability for a patient profile with modified clipping for realistic distributions"""
        config = getattr(self, f"{disease}_config")
        base_prob = float(self.diseases[disease])
        for demo_type, demo_value in [
            ('age_band', profile.age_band),
            ('gender', profile.gender),
            ('setting', profile.setting),
            ('region', profile.region),
            ('season', profile.season),
        ]:
            nudge = config['demographic_nudges'].get(demo_type, {}).get(demo_value, 0.0)
            base_prob += nudge

        # Modified clipping for realistic HIV distribution in children
        if disease == 'hiv' and profile.age_band in ['0-4', '5-14', '65+']:
            return float(np.clip(base_prob, 0.0001, 0.999)) # Allow very low probabilities
        else:
            return float(np.clip(base_prob, 0.001, 0.999))

    def _select_disease(self, profile: PatientProfile) -> str:
        """Select disease for patient based on adjusted probabilities"""
        raw = {d: self._calculate_disease_probability(d, profile) for d in self.diseases.keys()}
        total = sum(raw.values())
        norm = {d: raw[d] / total for d in raw}
        scaled = {d: norm[d] * (1.0 - self.healthy_share) for d in norm}
        scaled['healthy'] = self.healthy_share
        z = sum(scaled.values())
        final_probs = {k: v / z for k, v in scaled.items()}
        return np.random.choice(list(final_probs.keys()), p=list(final_probs.values()))

    def _apply_correlations(self, probs: Dict[str, float], disease: str, profile: PatientProfile):
        """Apply improved symptom correlations based on disease-specific patterns"""
        if disease == 'malaria':
            if probs['fever'] > 0.8 and probs['chills'] > 0.7:
                probs['sweats'] = min(0.95, probs['sweats'] + 0.15)
                probs['headache'] = min(0.90, probs['headache'] + 0.10)
            # Cerebral malaria complications
            if random.random() < 0.05:  # 5% severe malaria
                probs['confusion'] = min(0.60, probs['confusion'] + 0.50)
                probs['fever'] = min(0.98, probs['fever'] + 0.03)

        elif disease == 'typhoid':
            if probs['fever'] > 0.8 and probs['headache'] > 0.7:
                probs['fatigue'] = min(0.90, probs['fatigue'] + 0.10)
                probs['rose_spots'] = min(0.40, probs['rose_spots'] + 0.15)
            # Constipation is more common than diarrhea in adults
            if profile.age_band not in ['0-4', '5-14']:
                probs['constipation'] = min(0.80, probs['constipation'] + 0.20)
                probs['diarrhea'] = max(0.10, probs['diarrhea'] - 0.10)

        elif disease == 'tuberculosis':
            if probs['chronic_cough'] > 0.7:
                probs['night_sweats'] = min(0.85, probs['night_sweats'] + 0.15)
                probs['weight_loss'] = min(0.90, probs['weight_loss'] + 0.10)
                probs['hemoptysis'] = min(0.60, probs['hemoptysis'] + 0.20)
            if probs['weight_loss'] > 0.6:
                probs['fatigue'] = min(0.90, probs['fatigue'] + 0.15)
                probs['loss_of_appetite'] = min(0.80, probs['loss_of_appetite'] + 0.15)

        elif disease == 'measles':
            # Characteristic triad
            if probs['fever'] > 0.8:
                probs['maculopapular_rash'] = min(0.98, probs['maculopapular_rash'] + 0.03)
                probs['conjunctivitis'] = min(0.95, probs['conjunctivitis'] + 0.10)
                probs['cough'] = min(0.95, probs['cough'] + 0.10)
                probs['runny_nose'] = min(0.95, probs['runny_nose'] + 0.10)

        elif disease == 'pneumonia_ari':
            if probs['cough'] > 0.8 and probs['fever'] > 0.6:
                probs['chest_pain'] = min(0.80, probs['chest_pain'] + 0.15)
                probs['shortness_of_breath'] = min(0.85, probs['shortness_of_breath'] + 0.15)
                probs['rapid_breathing'] = min(0.80, probs['rapid_breathing'] + 0.20)

        elif disease == 'diabetes':
            # Classic triad correlation
            if probs['polyuria'] > 0.6:
                probs['polydipsia'] = min(0.85, probs['polydipsia'] + 0.20)
                probs['polyphagia'] = min(0.75, probs['polyphagia'] + 0.15)
                probs['recurrent_infections'] = min(0.65, probs.get('recurrent_infections', 0.2) + 0.20)

        elif disease == 'peptic_ulcer':
            if probs['epigastric_pain'] > 0.8:
                probs['hunger_pain'] = min(0.90, probs['hunger_pain'] + 0.15)
                probs['heartburn'] = min(0.85, probs['heartburn'] + 0.15)

    def _apply_age_modifiers(self, probs: Dict[str, float], disease: str, age_band: str):
        """Apply age-specific symptom modifications with better clinical accuracy"""
        if age_band == '0-4':
            # Children have different symptom presentations
            if disease in ['malaria', 'typhoid', 'gastroenteritis', 'measles', 'pneumonia_ari']:
                probs['vomiting'] = min(0.80, probs['vomiting'] + 0.15)
                probs['fatigue'] = min(0.90, probs['fatigue'] + 0.10)
                probs['rapid_breathing'] = min(0.85, probs.get('rapid_breathing', 0.2) + 0.20)
            if disease == 'malaria':
                probs['fever'] = min(0.98, probs['fever'] + 0.03)
                probs['confusion'] = min(0.25, probs['confusion'] + 0.15)  # Febrile seizures
            if disease in ['typhoid', 'gastroenteritis']:
                probs['diarrhea'] = min(0.70, probs['diarrhea'] + 0.30)  # More diarrhea in children
                probs['constipation'] = max(0.10, probs.get('constipation', 0.3) - 0.20)

        elif age_band in ['45-64', '65+']:
            # Older adults have different presentations
            if disease in ['diabetes', 'hypertension']:
                probs['fatigue'] = min(0.70, probs['fatigue'] + 0.10)
                probs['dizziness'] = min(0.50, probs.get('dizziness', 0.2) + 0.15)
                probs['confusion'] = min(0.25, probs.get('confusion', 0.08) + 0.10)
            if disease == 'tuberculosis':
                probs['fatigue'] = min(0.85, probs['fatigue'] + 0.10)
                probs['weight_loss'] = min(0.90, probs['weight_loss'] + 0.10)
                probs['fever'] = max(0.50, probs['fever'] - 0.15)  # Less fever in elderly
            if disease == 'pneumonia_ari':
                probs['confusion'] = min(0.40, probs.get('confusion', 0.15) + 0.25)
                probs['fever'] = max(0.60, probs['fever'] - 0.20)  # Less fever in elderly

    def _apply_comorbidities(self, probs: Dict[str, float], disease: str, profile: PatientProfile):
        """Apply realistic comorbidity patterns common in Nigeria"""

        # Malaria-Typhoid co-infection
        if disease == 'typhoid' and random.random() < 0.25:
            probs['fever'] = min(0.98, probs['fever'] + 0.03)
            probs['headache'] = min(0.95, probs['headache'] + 0.10)
            probs['chills'] = min(0.70, probs['chills'] + 0.35)
            probs['sweats'] = min(0.60, probs['sweats'] + 0.30)
            probs['body_ache'] = min(0.80, probs['body_ache'] + 0.30)

        # TB-HIV co-infection
        if disease == 'tuberculosis' and random.random() < 0.30:
            probs['weight_loss'] = min(0.95, probs['weight_loss'] + 0.15)
            probs['recurrent_infections'] = min(0.90, probs.get('recurrent_infections', 0.2) + 0.70)
            probs['oral_thrush'] = min(0.60, probs.get('oral_thrush', 0.1) + 0.50)
            probs['chronic_cough'] = min(0.95, probs['chronic_cough'] + 0.10)
            probs['night_sweats'] = min(0.85, probs['night_sweats'] + 0.15)

        # Diabetes-Hypertension comorbidity
        if disease == 'diabetes' and random.random() < 0.45:
            probs['headache'] = min(0.55, probs['headache'] + 0.25)
            probs['dizziness'] = min(0.45, probs.get('dizziness', 0.25) + 0.20)
            probs['blurred_vision'] = min(0.70, probs['blurred_vision'] + 0.20)
            probs['chest_pain'] = min(0.25, probs.get('chest_pain', 0.05) + 0.20)

        # Gastroenteritis with dehydration
        if disease == 'gastroenteritis' and random.random() < 0.20:
            probs['dizziness'] = min(0.50, probs.get('dizziness', 0.1) + 0.40)
            probs['fatigue'] = min(0.85, probs['fatigue'] + 0.20)
            probs['confusion'] = min(0.30, probs.get('confusion', 0.05) + 0.25)
            probs['oliguria'] = min(0.40, probs.get('oliguria', 0.05) + 0.35)

        # Malaria with severe complications
        if disease == 'malaria' and random.random() < 0.08:
            probs['confusion'] = min(0.70, probs['confusion'] + 0.60)
            probs['oliguria'] = min(0.40, probs.get('oliguria', 0.02) + 0.38)
            probs['shortness_of_breath'] = min(0.50, probs.get('shortness_of_breath', 0.05) + 0.45)

    def _generate_symptoms(self, disease: str, profile: PatientProfile) -> Dict[str, int]:
        """Generate symptoms for a patient with given disease using improved clinical logic"""
        if disease == 'healthy':
            # Very low background rates for healthy individuals
            return {s: int(random.random() < 0.01) for s in self.symptoms}

        config = getattr(self, f"{disease}_config")
        probs = config['symptom_probabilities'].copy()

        # Ensure all symptoms have baseline probability
        for s in self.symptoms:
            if s not in probs:
                probs[s] = 0.02  # Very low baseline

        # Apply modifiers
        self._apply_age_modifiers(probs, disease, profile.age_band)

        # Seasonal effects
        if profile.season == 'rainy' and disease in ['malaria', 'typhoid', 'gastroenteritis']:
            for s in ['fever', 'diarrhea', 'vomiting']:
                if s in probs:
                    probs[s] = min(0.98, probs[s] + 0.05)
        elif profile.season == 'dry' and disease in ['tuberculosis', 'measles', 'pneumonia_ari']:
            for s in ['cough', 'chronic_cough', 'fever']:
                if s in probs:
                    probs[s] = min(0.98, probs[s] + 0.05)

        # Apply correlations and comorbidities
        self._apply_correlations(probs, disease, profile)
        self._apply_comorbidities(probs, disease, profile)

        # Ensure mutually exclusive symptoms are handled properly
        if probs.get('chronic_cough', 0) > 0.5:
            probs['cough'] = min(probs.get('cough', 0.2), 0.3)  # Regular cough less likely if chronic

        if probs.get('maculopapular_rash', 0) > 0.7:
            probs['rash'] = min(probs.get('rash', 0.1), 0.2)  # Non-specific rash less likely

        if probs.get('epigastric_pain', 0) > 0.7:
            probs['abdominal_pain'] = min(probs.get('abdominal_pain', 0.3), 0.5)  # More specific pain

        # Sample binary symptoms
        symptoms = {}
        for s in self.symptoms:
            symptoms[s] = int(random.random() < probs.get(s, 0.02))

        return symptoms

    def generate_patient(self) -> dict:
        """Generate a single patient with realistic disease selection and symptoms"""
        profile = self._sample_demographics()
        disease = self._select_disease(profile)
        symptoms = self._generate_symptoms(disease, profile)
        return {
            'age_band': profile.age_band,
            'gender': profile.gender,
            'setting': profile.setting,
            'region': profile.region,
            'season': profile.season,
            'diagnosis': disease,
            **symptoms
        }

    def generate_dataset(self, n_patients: int = 10000,
                        balance_diseases: bool = False,
                        stratify_demographics: bool = False,
                        include_metadata: bool = True) -> pd.DataFrame:
        """
        Generate a comprehensive dataset with enhanced options
        """
        patients = []

        if balance_diseases:
            diseases_list = list(self.diseases.keys()) + ['healthy']
            per_disease = n_patients // len(diseases_list)
            remainder = n_patients % len(diseases_list)

            # Generate balanced diseases
            for i, disease in enumerate(diseases_list):
                # Add one extra for remainder
                count = per_disease + (1 if i < remainder else 0)
                for _ in range(count):
                    if stratify_demographics:
                        profile = self._sample_stratified_demographics()
                    else:
                        profile = self._sample_demographics()
                    symptoms = self._generate_symptoms(disease, profile)
                    patients.append({
                        'age_band': profile.age_band, 'gender': profile.gender,
                        'setting': profile.setting, 'region': profile.region,
                        'season': profile.season, 'diagnosis': disease,
                        **symptoms
                    })
        else:
            # Natural distribution
            for _ in range(n_patients):
                patients.append(self.generate_patient())

        df = pd.DataFrame(patients)

        if include_metadata:
            df.insert(0, 'patient_id', [f'PT_{i+1:06d}' for i in range(len(df))])
            df['generation_timestamp'] = pd.Timestamp.now()

        # Shuffle to avoid systematic ordering
        df = df.sample(frac=1).reset_index(drop=True)

        return df

    def _sample_stratified_demographics(self) -> PatientProfile:
        """Sample demographics ensuring better representation across all groups"""
        age_weights = np.array(list(self.demo_distributions['age_band'].values()))
        age_weights = np.sqrt(age_weights)  # Reduce extreme weights
        age_weights = age_weights / age_weights.sum()

        age_band = np.random.choice(
            list(self.demo_distributions['age_band'].keys()),
            p=age_weights
        )

        # Standard sampling for others
        def pick(d): return np.random.choice(list(d.keys()), p=list(d.values()))

        return PatientProfile(
            age_band=age_band,
            gender=pick(self.demo_distributions['gender']),
            setting=pick(self.demo_distributions['setting']),
            region=pick(self.demo_distributions['region']),
            season=pick(self.demo_distributions['season']),
        )

    def get_disease_statistics(self, df: pd.DataFrame,
                             pretty: bool = False,
                             include_symptom_correlations: bool = False) -> Dict:
        """
        Get comprehensive statistics with enhanced options
        """
        stats = {
            'total_patients': len(df),
            'disease_distribution': df['diagnosis'].value_counts().to_dict(),
            'disease_percentages': (df['diagnosis'].value_counts(normalize=True) * 100).round(2).to_dict(),
            'demographic_stats': {
                'age_distribution': df['age_band'].value_counts().to_dict(),
                'gender_distribution': df['gender'].value_counts().to_dict(),
                'setting_distribution': df['setting'].value_counts().to_dict(),
                'region_distribution': df['region'].value_counts().to_dict(),
                'season_distribution': df['season'].value_counts().to_dict()
            },
            'symptom_prevalence': {},
            'pathognomonic_symptoms': {},
            'age_disease_patterns': {},
            'seasonal_patterns': {},
            'gender_disease_patterns': {},
            'setting_disease_patterns': {},
            'region_disease_patterns': {}
        }

        # Enhanced symptom analysis
        for disease in df['diagnosis'].unique():
            disease_df = df[df['diagnosis'] == disease]
            symptom_prev = {}

            for s in self.symptoms:
                if s in df.columns:
                    prevalence = (disease_df[s].mean() * 100).round(1)
                    symptom_prev[s] = prevalence

            stats['symptom_prevalence'][disease] = symptom_prev

            # Identify pathognomonic symptoms (>70% prevalence and disease-specific)
            pathognomonic = {}
            for symptom, prevalence in symptom_prev.items():
                if prevalence > 70:
                    # Check if it's specific to this disease
                    other_diseases = df[df['diagnosis'] != disease]
                    if len(other_diseases) > 0:
                        other_prevalence = (other_diseases[symptom].mean() * 100) if symptom in other_diseases.columns else 0
                        if prevalence > (other_prevalence * 2):  # At least 2x more common
                            pathognomonic[symptom] = {
                                'prevalence_in_disease': prevalence,
                                'prevalence_in_others': round(other_prevalence, 1),
                                'specificity_ratio': round(prevalence / max(other_prevalence, 0.1), 1)
                            }
            stats['pathognomonic_symptoms'][disease] = pathognomonic

        # Age-disease patterns
        for disease in df['diagnosis'].unique():
            age_dist = df[df['diagnosis'] == disease]['age_band'].value_counts(normalize=True)
            stats['age_disease_patterns'][disease] = (age_dist * 100).round(1).to_dict()

        # Gender-disease patterns
        for disease in df['diagnosis'].unique():
            if disease != 'healthy':
                gender_dist = df[df['diagnosis'] == disease]['gender'].value_counts(normalize=True)
                stats['gender_disease_patterns'][disease] = (gender_dist * 100).round(1).to_dict()

        # Setting-disease patterns
        for disease in df['diagnosis'].unique():
            if disease != 'healthy':
                setting_dist = df[df['diagnosis'] == disease]['setting'].value_counts(normalize=True)
                stats['setting_disease_patterns'][disease] = (setting_dist * 100).round(1).to_dict()

        # Region-disease patterns (Added this section)
        for disease in df['diagnosis'].unique():
            if disease != 'healthy':
                region_dist = df[df['diagnosis'] == disease]['region'].value_counts(normalize=True)
                stats['region_disease_patterns'][disease] = (region_dist * 100).round(1).to_dict()

        # Seasonal patterns
        for disease in df['diagnosis'].unique():
            if disease != 'healthy':
                seasonal_dist = df[df['diagnosis'] == disease]['season'].value_counts(normalize=True)
                stats['seasonal_patterns'][disease] = (seasonal_dist * 100).round(1).to_dict()

        if include_symptom_correlations:
            stats['symptom_correlations'] = self._calculate_symptom_correlations(df)

        if pretty:
            self._print_enhanced_statistics(stats)

        return stats

    def _calculate_symptom_correlations(self, df: pd.DataFrame) -> Dict:
        """Calculate meaningful symptom correlations for insight"""
        correlations = {}

        # Key symptom pairs to examine
        key_pairs = [
            ('fever', 'chills'), ('fever', 'sweats'), ('cough', 'chest_pain'),
            ('diarrhea', 'vomiting'), ('polyuria', 'polydipsia'),
            ('chronic_cough', 'night_sweats'), ('maculopapular_rash', 'conjunctivitis')
        ]

        for disease in df['diagnosis'].unique():
            if disease == 'healthy':
                continue
            disease_df = df[df['diagnosis'] == disease]
            disease_corrs = {}

            for sym1, sym2 in key_pairs:
                if sym1 in disease_df.columns and sym2 in disease_df.columns:
                    if disease_df[sym1].var() > 0 and disease_df[sym2].var() > 0:
                        corr = disease_df[sym1].corr(disease_df[sym2])
                        if abs(corr) > 0.3:
                            disease_corrs[f"{sym1}_{sym2}"] = round(corr, 3)

            if disease_corrs:
                correlations[disease] = disease_corrs

        return correlations

    def _print_enhanced_statistics(self, stats: Dict):
        """Pretty printing with insights"""
        print("\n" + "="*60)
        print("         NIGERIAN DISEASE DATASET SUMMARY")
        print("="*60)
        print(f"Total patients: {stats['total_patients']:,}")

        print(f"\n{'='*20} DISEASE DISTRIBUTION {'='*20}")
        for disease, pct in stats['disease_percentages'].items():
            count = stats['disease_distribution'][disease]
            print(f"{disease:<15}: {count:>6} patients ({pct:>5}%)")

        print(f"\n{'='*20} CHARACTERISTIC SYMPTOMS {'='*20}\n")
        for disease, symptoms in stats['pathognomonic_symptoms'].items():
            if symptoms:
                print(f"{disease.upper()}:")
                for symptom, details in symptoms.items():
                    print(f"  {symptom:<20}: {details['prevalence_in_disease']:>5}% (vs {details['prevalence_in_others']:>5}% others, {details['specificity_ratio']:>4}x specific)")
                print("-" * 20)

        print(f"\n{'='*20} AGE DISTRIBUTION BY DISEASE {'='*20}\n")
        for disease, age_dist in stats['age_disease_patterns'].items():
            print(f"{disease}:")
            for age_band, pct in age_dist.items():
                print(f"  {age_band:<8}: {pct:>5}%")
            print("-" * 10)

        print(f"\n{'='*20} GENDER DISTRIBUTION BY DISEASE {'='*20}\n")
        for disease, gender_dist in stats['gender_disease_patterns'].items():
            print(f"{disease}:")
            for gender, pct in gender_dist.items():
                print(f"  {gender:<8}: {pct:>5}%")
            print("-" * 10)

        print(f"\n{'='*20} SETTING DISTRIBUTION BY DISEASE {'='*20}\n")
        for disease, setting_dist in stats['setting_disease_patterns'].items():
            print(f"{disease}:")
            for setting, pct in setting_dist.items():
                print(f"  {setting:<8}: {pct:>5}%")
            print("-" * 10)

        print(f"\n{'='*20} REGION DISTRIBUTION BY DISEASE {'='*20}\n")
        for disease, region_dist in stats['region_disease_patterns'].items():
            print(f"{disease}:")
            for region, pct in region_dist.items():
                print(f"  {region:<15}: {pct:>5}%")
            print("-" * 10)

        print(f"\n{'='*20} SEASONAL PATTERNS {'='*20}")
        for disease, seasonal_dist in stats['seasonal_patterns'].items():
            peak_season = max(seasonal_dist, key=seasonal_dist.get)
            peak_pct = seasonal_dist[peak_season]
            print(f"{disease:<15}: Peak in {peak_season} ({peak_pct:>5}%)")

        if 'symptom_correlations' in stats and stats['symptom_correlations']:
            print(f"\n{'='*20} SYMPTOM CORRELATIONS {'='*20}\n")
            for disease, corrs in stats['symptom_correlations'].items():
                print(f"{disease.upper()}:")
                for pair, corr_value in corrs.items():
                    print(f"  {pair:<20}: {corr_value:>5}")
                print("-" * 20)

        print("\n" + "="*60)
        print("         END OF SUMMARY")
        print("="*60)
