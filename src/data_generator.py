# -*- coding: utf-8 -*-
# Import libraries
import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Define patient profile (basic demographics)
@dataclass
class PatientProfile:
    """Patient demographic profile"""
    age_band: str
    gender: str
    setting: str  # urban/rural
    region: str   # north/middle_belt/south
    season: str   # dry/rainy/transition

# Define disease generator
class NigerianDiseaseGenerator:
    """
    Synthetic data generator for simulating disease occurrence in Nigeria.
    Includes disease-specific symptom probabilities, demographic nudges,
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
            'pneumonia': 0.20,
            'diabetes': 0.10,
            'hypertension': 0.15,
            'hiv': 0.06
        }

        # Unified symptom list for all rows
        self.symptoms = [
            'fever', 'headache', 'cough', 'fatigue', 'body_ache', 'chills', 'sweats',
            'nausea', 'vomiting', 'diarrhea', 'abdominal_pain', 'loss_of_appetite',
            'sore_throat', 'runny_nose', 'dysuria'
        ]

        # Target share of healthy (or undiagnosed) cases in the population
        self.healthy_share = float(np.clip(healthy_share, 0.05, 0.4))

        # Disease configs
        self.malaria_config = {
            'demographic_nudges': {
                'age_band': {'0-4': +0.15, '5-14': +0.12, '15-24': +0.08, '25-44': +0.05, '45-64': -0.05, '65+': -0.10},
                'gender': {'male': +0.03, 'female': -0.03},
                'setting': {'rural': +0.12, 'urban': -0.12},
                'region': {'south': +0.08, 'middle_belt': +0.10, 'north': +0.05},
                'season': {'rainy': +0.15, 'dry': -0.15, 'transition': +0.05}
            },
            'symptom_probabilities': {
                'fever': 0.95, 'chills': 0.85, 'headache': 0.80, 'body_ache': 0.75,
                'sweats': 0.70, 'fatigue': 0.85, 'nausea': 0.65, 'vomiting': 0.45,
                'loss_of_appetite': 0.60, 'diarrhea': 0.25, 'abdominal_pain': 0.20,
                'cough': 0.15, 'sore_throat': 0.08, 'runny_nose': 0.05, 'dysuria': 0.03
            }
        }
        self.typhoid_config = {
            'demographic_nudges': {
                'age_band': {'0-4': -0.02, '5-14': +0.08, '15-24': +0.06, '25-44': +0.04, '45-64': +0.02, '65+': -0.05},
                'gender': {'female': +0.05, 'male': -0.05},
                'setting': {'rural': +0.15, 'urban': -0.15},
                'region': {'north': +0.05, 'middle_belt': +0.02, 'south': -0.07},
                'season': {'rainy': +0.12, 'dry': -0.12, 'transition': 0}
            },
            'symptom_probabilities': {
                'fever': 0.95, 'headache': 0.80, 'fatigue': 0.75, 'loss_of_appetite': 0.70,
                'abdominal_pain': 0.55, 'nausea': 0.50, 'diarrhea': 0.45, 'body_ache': 0.45,
                'vomiting': 0.40, 'chills': 0.35, 'sweats': 0.30, 'cough': 0.25,
                'sore_throat': 0.10, 'runny_nose': 0.08, 'dysuria': 0.05
            }
        }
        self.peptic_ulcer_config = {
            'demographic_nudges': {
                'age_band': {'0-4': -0.15, '5-14': -0.10, '15-24': -0.05, '25-44': +0.05, '45-64': +0.15, '65+': +0.10},
                'gender': {'male': +0.08, 'female': -0.08},
                'setting': {'urban': +0.08, 'rural': -0.08},
                'region': {'north': +0.03, 'middle_belt': +0.01, 'south': -0.04},
                'season': {'dry': +0.02, 'rainy': -0.02, 'transition': 0}
            },
            'symptom_probabilities': {
                'abdominal_pain': 0.85, 'loss_of_appetite': 0.60, 'nausea': 0.55,
                'vomiting': 0.35, 'fatigue': 0.40, 'headache': 0.25, 'body_ache': 0.20,
                'fever': 0.15, 'diarrhea': 0.10, 'cough': 0.05, 'sore_throat': 0.05,
                'runny_nose': 0.05, 'dysuria': 0.05, 'chills': 0.10, 'sweats': 0.10
            }
        }
        self.tuberculosis_config = {
            'demographic_nudges': {
                'age_band': {'0-4': +0.05, '5-14': -0.02, '15-24': +0.08, '25-44': +0.10, '45-64': +0.05, '65+': +0.08},
                'gender': {'male': +0.10, 'female': -0.10},
                'setting': {'urban': +0.05, 'rural': -0.05},
                'region': {'north': +0.08, 'middle_belt': +0.02, 'south': -0.10},
                'season': {'dry': +0.15, 'rainy': -0.15, 'transition': 0}
            },
            'symptom_probabilities': {
                'cough': 0.75, 'fever': 0.70, 'fatigue': 0.75, 'sweats': 0.65,
                'loss_of_appetite': 0.60, 'body_ache': 0.45, 'chills': 0.40,
                'headache': 0.35, 'nausea': 0.25, 'vomiting': 0.15, 'abdominal_pain': 0.20,
                'diarrhea': 0.10, 'sore_throat': 0.15, 'runny_nose': 0.10, 'dysuria': 0.05
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
                'fever': 0.95, 'cough': 0.85, 'runny_nose': 0.85, 'fatigue': 0.75,
                'chills': 0.60, 'loss_of_appetite': 0.60, 'sweats': 0.55, 'headache': 0.50,
                'body_ache': 0.45, 'sore_throat': 0.40, 'nausea': 0.35, 'diarrhea': 0.30,
                'vomiting': 0.25, 'abdominal_pain': 0.20, 'dysuria': 0.05
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
                'diarrhea': 0.95, 'abdominal_pain': 0.75, 'vomiting': 0.70, 'fatigue': 0.70,
                'nausea': 0.65, 'loss_of_appetite': 0.50, 'fever': 0.45, 'headache': 0.35,
                'body_ache': 0.30, 'chills': 0.25, 'sweats': 0.20, 'cough': 0.10,
                'sore_throat': 0.10, 'runny_nose': 0.10, 'dysuria': 0.05
            }
        }
        self.pneumonia_config = {
            'demographic_nudges': {
                'age_band': {'0-4': +0.20, '5-14': +0.05, '15-24': -0.08, '25-44': -0.08, '45-64': +0.02, '65+': +0.15},
                'gender': {'male': +0.08, 'female': -0.08},
                'setting': {'rural': +0.08, 'urban': -0.08},
                'region': {'north': +0.05, 'middle_belt': +0.02, 'south': -0.07},
                'season': {'dry': +0.15, 'rainy': -0.15, 'transition': 0}
            },
            'symptom_probabilities': {
                'cough': 0.94, 'fever': 0.82, 'fatigue': 0.70, 'body_ache': 0.50,
                'chills': 0.45, 'loss_of_appetite': 0.45, 'headache': 0.45, 'sweats': 0.40,
                'nausea': 0.30, 'vomiting': 0.25, 'sore_throat': 0.25, 'runny_nose': 0.20,
                'abdominal_pain': 0.15, 'diarrhea': 0.10, 'dysuria': 0.05
            }
        }
        self.diabetes_config = {
            'demographic_nudges': {
                'age_band': {'0-4': -0.20, '5-14': -0.15, '15-24': -0.08, '25-44': +0.05, '45-64': +0.25, '65+': +0.15},
                'gender': {'female': +0.08, 'male': -0.08},
                'setting': {'urban': +0.20, 'rural': -0.20},
                'region': {'south': +0.15, 'middle_belt': -0.05, 'north': -0.10},
                'season': {'rainy': +0.05, 'dry': -0.05, 'transition': 0}
            },
            'symptom_probabilities': {
                'fatigue': 0.85, 'dysuria': 0.45, 'loss_of_appetite': 0.25, 'headache': 0.35,
                'body_ache': 0.30, 'nausea': 0.25, 'sweats': 0.20, 'vomiting': 0.15,
                'fever': 0.15, 'abdominal_pain': 0.20, 'chills': 0.10, 'diarrhea': 0.10,
                'cough': 0.10, 'sore_throat': 0.10, 'runny_nose': 0.05
            }
        }
        self.hypertension_config = {
            'demographic_nudges': {
                'age_band': {'0-4': -0.18, '5-14': -0.12, '15-24': -0.08, '25-44': +0.05, '45-64': +0.20, '65+': +0.25},
                'gender': {'male': +0.08, 'female': -0.08},
                'setting': {'urban': +0.08, 'rural': -0.08},
                'region': {'south': +0.02, 'middle_belt': +0.01, 'north': -0.03},
                'season': {'rainy': +0.05, 'dry': -0.05, 'transition': 0}
            },
            'symptom_probabilities': {
                'headache': 0.35, 'fatigue': 0.40, 'body_ache': 0.20, 'loss_of_appetite': 0.20,
                'nausea': 0.15, 'sweats': 0.15, 'abdominal_pain': 0.10, 'chills': 0.08,
                'vomiting': 0.08, 'fever': 0.05, 'cough': 0.10, 'sore_throat': 0.05,
                'runny_nose': 0.05, 'dysuria': 0.05, 'diarrhea': 0.05
            }
        }
        self.hiv_config = {
            'demographic_nudges': {
                'age_band': {'0-4': -0.05, '5-14': +0.02, '15-24': +0.15, '25-44': +0.20, '45-64': +0.05, '65+': -0.08},
                'gender': {'female': +0.15, 'male': -0.15},
                'setting': {'rural': +0.08, 'urban': -0.08},
                'region': {'south': +0.08, 'middle_belt': +0.05, 'north': -0.13},
                'season': {'rainy': +0.02, 'dry': -0.02, 'transition': 0}
            },
            'symptom_probabilities': {
                'fatigue': 0.70, 'diarrhea': 0.55, 'cough': 0.50, 'loss_of_appetite': 0.50,
                'sweats': 0.50, 'fever': 0.45, 'body_ache': 0.45, 'headache': 0.40,
                'chills': 0.30, 'sore_throat': 0.25, 'nausea': 0.25, 'abdominal_pain': 0.25,
                'vomiting': 0.15, 'dysuria': 0.15, 'runny_nose': 0.15
            }
        }

    # --- Sample demographics
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

    # --- Calculate disease probability after demographic nudges
    def _calculate_disease_probability(self, disease: str, profile: PatientProfile) -> float:
        """Calculate disease probability for a patient profile"""
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
        return float(np.clip(base_prob, 0.001, 0.999))

    # --- Select disease (with explicit healthy share)
    def _select_disease(self, profile: PatientProfile) -> str:
        """Select disease for patient based on adjusted probabilities"""
        # Get unnormalized adjusted probs
        raw = {d: self._calculate_disease_probability(d, profile) for d in self.diseases.keys()}
        total = sum(raw.values())

        # Normalize to sum=1
        norm = {d: raw[d] / total for d in raw}

        # Allocate healthy share, scale others
        scaled = {d: norm[d] * (1.0 - self.healthy_share) for d in norm}

        # Add healthy
        scaled['healthy'] = self.healthy_share

        # Renormalize
        z = sum(scaled.values())
        final_probs = {k: v / z for k, v in scaled.items()}

        # Sample
        return np.random.choice(list(final_probs.keys()), p=list(final_probs.values()))

    # --- Symptom correlations
    def _apply_correlations(self, probs: Dict[str, float], disease: str, profile: PatientProfile):
        """Apply symptom correlations based on disease-specific patterns"""
        if disease == 'malaria':
            if probs['fever'] > 0.5 and probs['chills'] > 0.5:
                probs['sweats'] = min(0.95, probs['sweats'] + 0.25)
                probs['headache'] = min(0.95, probs['headache'] + 0.15)
            if probs['fever'] > 0.5:
                probs['runny_nose'] = max(0.05, probs['runny_nose'] - 0.10)
                probs['sore_throat'] = max(0.05, probs['sore_throat'] - 0.08)

        elif disease == 'typhoid':
            if probs['fever'] > 0.5 and probs['headache'] > 0.5:
                probs['fatigue'] = min(0.95, probs['fatigue'] + 0.20)
            if probs['diarrhea'] > 0.3:
                probs['abdominal_pain'] = min(0.85, probs['abdominal_pain'] + 0.25)
                probs['nausea'] = min(0.75, probs['nausea'] + 0.15)

        elif disease == 'gastroenteritis':
            if probs['diarrhea'] > 0.7 and probs['vomiting'] > 0.5:
                probs['abdominal_pain'] = min(0.90, probs['abdominal_pain'] + 0.25)
                probs['fatigue'] = min(0.90, probs['fatigue'] + 0.25)
                probs['loss_of_appetite'] = min(0.85, probs['loss_of_appetite'] + 0.20)
            if probs['diarrhea'] > 0.7:
                probs['sore_throat'] = max(0.05, probs['sore_throat'] - 0.12)
                probs['runny_nose'] = max(0.05, probs['runny_nose'] - 0.12)
                probs['cough'] = max(0.05, probs['cough'] - 0.08)

        elif disease == 'pneumonia_ari':
            if probs['cough'] > 0.8 and probs['fever'] > 0.6:
                probs['chills'] = min(0.70, probs['chills'] + 0.20)
                probs['sweats'] = min(0.65, probs['sweats'] + 0.15)
                probs['fatigue'] = min(0.85, probs['fatigue'] + 0.15)
            if probs['cough'] > 0.8:
                probs['runny_nose'] = max(0.10, probs['runny_nose'] - 0.15)
                probs['abdominal_pain'] = max(0.05, probs['abdominal_pain'] - 0.10)
                probs['diarrhea'] = max(0.05, probs['diarrhea'] - 0.05)

        elif disease == 'tuberculosis':
            if probs['cough'] > 0.6 and probs['fever'] > 0.5:
                probs['sweats'] = min(0.85, probs['sweats'] + 0.20)
                probs['fatigue'] = min(0.90, probs['fatigue'] + 0.15)
            if probs['sweats'] > 0.5 and probs['fatigue'] > 0.6:
                probs['runny_nose'] = max(0.05, probs['runny_nose'] - 0.08)
                probs['sore_throat'] = max(0.10, probs['sore_throat'] - 0.08)
                probs['diarrhea'] = max(0.05, probs['diarrhea'] - 0.05)

        elif disease == 'measles':
            if probs['fever'] > 0.8 and probs['cough'] > 0.7:
                probs['runny_nose'] = min(0.95, probs['runny_nose'] + 0.25)
                probs['fatigue'] = min(0.90, probs['fatigue'] + 0.15)
            if probs['fever'] > 0.8:
                probs['chills'] = min(0.80, probs['chills'] + 0.20)
                probs['sweats'] = min(0.75, probs['sweats'] + 0.15)
                probs['loss_of_appetite'] = min(0.80, probs['loss_of_appetite'] + 0.15)

    # --- Age modifiers
    def _apply_age_modifiers(self, probs: Dict[str, float], disease: str, age_band: str):
        """Apply age-specific symptom modifications"""
        if age_band == '0-4':
            if disease in ['malaria', 'typhoid', 'gastroenteritis', 'measles', 'pneumonia_ari']:
                probs['vomiting'] = min(0.80, probs['vomiting'] + 0.15)
                probs['fatigue'] = min(0.90, probs['fatigue'] + 0.10)
            if disease == 'malaria':
                probs['fever'] = min(0.98, probs['fever'] + 0.05)
                probs['headache'] = max(0.35, probs['headache'] - 0.15)

        elif age_band in ['45-64', '65+']:
            if disease in ['diabetes', 'hypertension']:
                probs['fatigue'] = min(0.70, probs['fatigue'] + 0.10)
                probs['headache'] = min(0.65, probs['headache'] + 0.08)
            if disease == 'tuberculosis':
                probs['fatigue'] = min(0.85, probs['fatigue'] + 0.12)
                probs['fever'] = max(0.60, probs['fever'] - 0.05)

    # --- Comorbidities
    def _apply_comorbidities(self, probs: Dict[str, float], disease: str, profile: PatientProfile):
        """Apply common disease comorbidities and overlaps"""
        if disease == 'typhoid' and random.random() < 0.15:  # malaria-typhoid overlap
            probs['fever'] = min(0.98, probs['fever'] + 0.03)
            probs['headache'] = min(0.95, probs['headache'] + 0.10)
            probs['fatigue'] = min(0.90, probs['fatigue'] + 0.08)
            probs['body_ache'] = min(0.85, probs['body_ache'] + 0.15)

        if disease == 'gastroenteritis' and random.random() < 0.17:  # gastro-malaria overlap
            probs['fever'] = min(0.80, probs['fever'] + 0.25)
            probs['headache'] = min(0.70, probs['headache'] + 0.20)
            probs['body_ache'] = min(0.60, probs['body_ache'] + 0.15)

        if disease == 'tuberculosis' and random.random() < 0.22:  # TB-HIV
            probs['fever'] = min(0.85, probs['fever'] + 0.05)
            probs['fatigue'] = min(0.95, probs['fatigue'] + 0.15)
            probs['sweats'] = min(0.85, probs['sweats'] + 0.15)

        if disease == 'diabetes' and random.random() < 0.40:  # DM-HTN
            probs['headache'] = min(0.65, probs['headache'] + 0.20)
            probs['fatigue'] = min(0.95, probs['fatigue'] + 0.05)

    # --- Generate symptoms
    def _generate_symptoms(self, disease: str, profile: PatientProfile) -> Dict[str, int]:
        """Generate symptoms for a patient with given disease"""
        if disease == 'healthy':
            return {s: int(random.random() < 0.03) for s in self.symptoms}  # very low background

        config = getattr(self, f"{disease}_config")
        probs = config['symptom_probabilities'].copy()

        # Ensure all symptoms have a (low) baseline if unspecified
        for s in self.symptoms:
            probs.setdefault(s, 0.05)

        # Age & season
        self._apply_age_modifiers(probs, disease, profile.age_band)
        if profile.season == 'rainy' and disease in ['malaria', 'typhoid', 'gastroenteritis']:
            for s in ['fever', 'diarrhea', 'vomiting']:
                probs[s] = min(0.98, probs[s] + 0.05)
        elif profile.season == 'dry' and disease in ['tuberculosis', 'measles', 'pneumonia_ari']:
            for s in ['cough', 'fever']:
                probs[s] = min(0.98, probs[s] + 0.05)

        # Correlations & comorbidities
        self._apply_correlations(probs, disease, profile)
        self._apply_comorbidities(probs, disease, profile)

        # Sample binary symptoms
        return {s: int(random.random() < probs[s]) for s in self.symptoms}

    # --- Public API
    def generate_patient(self) -> Dict[str, Any]:
        """Generate a single patient record with demographics, disease, and symptoms"""
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

    def generate_dataset(self, n_patients: int = 10000, balance_diseases: bool = False) -> pd.DataFrame:
        """ Generate a complete dataset of n_patients"""
        patients = []
        if balance_diseases:
            diseases_list = list(self.diseases.keys()) + ['healthy']
            per = n_patients // len(diseases_list)
            for disease in diseases_list:
                for _ in range(per):
                    profile = self._sample_demographics()
                    symptoms = self._generate_symptoms(disease, profile)
                    patients.append({
                        'age_band': profile.age_band, 'gender': profile.gender,
                        'setting': profile.setting, 'region': profile.region,
                        'season': profile.season, 'diagnosis': disease, **symptoms
                    })
            # Fill remainder naturally
            for _ in range(n_patients - len(patients)):
                patients.append(self.generate_patient())
        else:
            for _ in range(n_patients):
                patients.append(self.generate_patient())
        return pd.DataFrame(patients)

    def get_disease_statistics(self, df: pd.DataFrame, pretty: bool = False) -> Dict:
        """Get comprehensive statistics about the generated dataset"""
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
            'symptom_prevalence': {}
        }
        for disease in df['diagnosis'].unique():
            disease_df = df[df['diagnosis'] == disease]
            symptom_prev = {s: (disease_df[s].mean() * 100).round(1) for s in self.symptoms}
            stats['symptom_prevalence'][disease] = symptom_prev

        if pretty:
            print("\n=== DATASET SUMMARY ===")
            print(f"Total patients: {stats['total_patients']}")
            print("\n-- Disease Distribution --")
            for d, pct in stats['disease_percentages'].items():
                print(f"{d:<15}: {stats['disease_distribution'][d]} patients ({pct}%)")
            print("\n-- Demographics --")
            for key, dist in stats['demographic_stats'].items():
                print(f"\n{key.replace('_', ' ').title()}:")
                for val, count in dist.items():
                    pct = round(count / stats['total_patients'] * 100, 1)
                    print(f"  {val:<12}: {count} ({pct}%)")
            print("\n-- Symptom Prevalence by Disease (top 5) --")
            for disease, symp in stats['symptom_prevalence'].items():
                top5 = sorted(symp.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"\n{disease}:")
                for s, p in top5:
                    print(f"  {s:<14}: {p}%")
        return stats

    def validate_realism(self, df: pd.DataFrame) -> Dict:
        """Validate the realism of generated data against expected patterns"""
        validation = {'warnings': [], 'passes': [], 'disease_specific_checks': {}, 'realism_score': None}
        total_checks = 0
        total_passes = 0

        # Healthy share check
        total_checks += 1
        healthy_pct = (df['diagnosis'].eq('healthy').mean())
        if 0.05 <= healthy_pct <= 0.40:
            validation['passes'].append("Reasonable healthy/diseased ratio")
            total_passes += 1
        else:
            validation['warnings'].append(f"Healthy share out of range: {healthy_pct:.1%}")

        # Disease-specific
        diseases_to_check = ['malaria', 'typhoid', 'tuberculosis', 'measles', 'gastroenteritis']
        for disease in diseases_to_check:
            if disease in df['diagnosis'].values:
                disease_df = df[df['diagnosis'] == disease]
                checks = []

                # Each block adds 1–2 checks with pass/fail
                if disease == 'measles':
                    total_checks += 1
                    child_pct = disease_df['age_band'].isin(['0-4', '5-14']).mean()
                    if child_pct >= 0.70:
                        checks.append("Pass: Measles age distribution correct")
                        total_passes += 1
                    else:
                        checks.append(f"⚠️ Measles should be 70%+ in children, got {child_pct:.1%}")

                if disease == 'malaria':
                    total_checks += 2
                    rainy_pct = (disease_df['season'] == 'rainy').mean()
                    if rainy_pct >= 0.45:
                        checks.append("Pass: Malaria rainy-season predominance")
                        total_passes += 1
                    else:
                        checks.append(f"⚠️ Malaria should peak in rainy season, got {rainy_pct:.1%}")
                    fever_rate = disease_df['fever'].mean()
                    if fever_rate >= 0.85:
                        checks.append("Pass: Malaria fever rate realistic")
                        total_passes += 1
                    else:
                        checks.append(f"⚠️ Malaria fever rate should be >85%, got {fever_rate:.1%}")

                if disease == 'tuberculosis':
                    total_checks += 2
                    dry_pct = (disease_df['season'] == 'dry').mean()
                    if dry_pct >= 0.40:
                        checks.append("Pass: TB dry-season pattern plausible")
                        total_passes += 1
                    else:
                        checks.append(f"⚠️ TB should peak in dry season, got {dry_pct:.1%}")
                    male_pct = (disease_df['gender'] == 'male').mean()
                    if male_pct >= 0.52:
                        checks.append("Pass: TB male predominance")
                        total_passes += 1
                    else:
                        checks.append(f"⚠️ TB should show male predominance, got {male_pct:.1%}")

                if disease == 'gastroenteritis':
                    total_checks += 2
                    child_pct = disease_df['age_band'].isin(['0-4', '5-14']).mean()
                    if child_pct >= 0.55:
                        checks.append("Pass Gastroenteritis pediatric focus")
                        total_passes += 1
                    else:
                        checks.append(f"⚠️ Gastroenteritis should be 55%+ in children, got {child_pct:.1%}")
                    diarrhea_rate = disease_df['diarrhea'].mean()
                    if diarrhea_rate >= 0.85:
                        checks.append("Pass: Gastroenteritis diarrhea rate realistic")
                        total_passes += 1
                    else:
                        checks.append(f"⚠️ Gastroenteritis diarrhea rate should be >85%, got {diarrhea_rate:.1%}")

                if disease == 'typhoid':
                    total_checks += 1
                    rural_pct = (disease_df['setting'] == 'rural').mean()
                    if rural_pct >= 0.55:
                        checks.append("Pass: Typhoid rural predominance")
                        total_passes += 1
                    else:
                        checks.append(f"⚠️ Typhoid should have rural predominance, got {rural_pct:.1%}")

                validation['disease_specific_checks'][disease] = checks

        validation['realism_score'] = round(total_passes / total_checks, 3) if total_checks else None
        return validation

