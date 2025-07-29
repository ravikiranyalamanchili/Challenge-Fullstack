#!/usr/bin/env python3
"""
Test script for the patient-trial matching system with synthetic evaluation data.
This script creates synthetic patients and trials to test the matching functionality.
"""

import asyncio
import json
import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.match import PatientTrialMatcher, Patient, TrialMatch


# Synthetic patient data for testing
SYNTHETIC_PATIENTS = [
    {
        'patient_id': 'SYNTH_001',
        'age': 45,
        'gender': 'Female',
        'cancer_type': 'Breast Cancer',
        'stage': 'Stage II',
        'biomarkers': 'HER2+;ER+',
        'location': 'Boston MA',
        'ecog_status': 0,
        'prior_treatments': 'Surgery, Chemotherapy'
    },
    {
        'patient_id': 'SYNTH_002',
        'age': 62,
        'gender': 'Male',
        'cancer_type': 'Lung Cancer',
        'stage': 'Stage IIIA',
        'biomarkers': 'EGFR mutation',
        'location': 'Seattle WA',
        'ecog_status': 1,
        'prior_treatments': 'Radiation, Targeted therapy'
    },
    {
        'patient_id': 'SYNTH_003',
        'age': 38,
        'gender': 'Female',
        'cancer_type': 'Ovarian Cancer',
        'stage': 'Stage IV',
        'biomarkers': 'BRCA2',
        'location': 'New York NY',
        'ecog_status': 1,
        'prior_treatments': 'Surgery, Chemotherapy, Immunotherapy'
    },
    {
        'patient_id': 'SYNTH_004',
        'age': 55,
        'gender': 'Female',
        'cancer_type': 'Breast Cancer',
        'stage': 'Stage III',
        'biomarkers': 'Triple Negative',
        'location': 'Chicago IL',
        'ecog_status': 0,
        'prior_treatments': 'Chemotherapy, Radiation'
    }
]

# Synthetic trial data for testing
SYNTHETIC_TRIALS = [
    {
        'NCT Number': 'NCT12345678',
        'Study Title': 'HER2-Positive Breast Cancer Treatment Study',
        'Study URL': 'https://clinicaltrials.gov/study/NCT12345678',
        'Study Status': 'RECRUITING',
        'Brief Summary': 'This study evaluates the effectiveness of trastuzumab and pertuzumab in patients with HER2-positive breast cancer. Eligible patients must have Stage I-III HER2+ breast cancer, be 18-75 years old, and have ECOG performance status of 0-1. Prior chemotherapy and surgery are allowed.',
        'Study Results': 'NO',
        'Conditions': 'Breast Cancer|HER2 Positive Breast Cancer',
        'Interventions': 'DRUG: Trastuzumab|DRUG: Pertuzumab',
        'Phases': 'Phase 3',
        'Enrollment': '500',
        'Study Type': 'INTERVENTIONAL',
        'Study Design': 'Allocation: RANDOMIZED|Intervention Model: PARALLEL|Masking: DOUBLE|Primary Purpose: TREATMENT',
        'Start Date': '2024-01-15',
        'Completion Date': '2026-12'
    },
    {
        'NCT Number': 'NCT87654321',
        'Study Title': 'Early Stage Breast Cancer Immunotherapy',
        'Study URL': 'https://clinicaltrials.gov/study/NCT87654321',
        'Study Status': 'RECRUITING',
        'Brief Summary': 'This phase 2 study investigates pembrolizumab immunotherapy in early-stage breast cancer patients. Eligible patients must have Stage I-II breast cancer of any subtype, be 18-70 years old, ECOG 0-1, and have completed surgery and chemotherapy.',
        'Study Results': 'NO',
        'Conditions': 'Breast Cancer|Early Stage Breast Cancer',
        'Interventions': 'DRUG: Pembrolizumab',
        'Phases': 'Phase 2',
        'Enrollment': '200',
        'Study Type': 'INTERVENTIONAL',
        'Study Design': 'Allocation: RANDOMIZED|Intervention Model: PARALLEL|Masking: NONE|Primary Purpose: TREATMENT',
        'Start Date': '2024-03-01',
        'Completion Date': '2025-06'
    },
    {
        'NCT Number': 'NCT11111111',
        'Study Title': 'Triple Negative Breast Cancer Study',
        'Study URL': 'https://clinicaltrials.gov/study/NCT11111111',
        'Study Status': 'RECRUITING',
        'Brief Summary': 'Phase 1 study of olaparib in triple negative breast cancer patients. Eligible patients must have Stage II-IV triple negative breast cancer, be 18-65 years old, ECOG 0-2, and have received at least one prior chemotherapy regimen.',
        'Study Results': 'NO',
        'Conditions': 'Breast Cancer|Triple Negative Breast Cancer',
        'Interventions': 'DRUG: Olaparib',
        'Phases': 'Phase 1',
        'Enrollment': '50',
        'Study Type': 'INTERVENTIONAL',
        'Study Design': 'Allocation: NON_RANDOMIZED|Intervention Model: SINGLE_GROUP|Masking: NONE|Primary Purpose: TREATMENT',
        'Start Date': '2024-02-01',
        'Completion Date': '2025-12'
    },
    {
        'NCT Number': 'NCT22222222',
        'Study Title': 'EGFR-Mutated Lung Cancer Treatment',
        'Study URL': 'https://clinicaltrials.gov/study/NCT22222222',
        'Study Status': 'RECRUITING',
        'Brief Summary': 'This study evaluates osimertinib in patients with EGFR-mutated non-small cell lung cancer. Eligible patients must have Stage III-IV EGFR+ lung cancer, be 18-80 years old, ECOG 0-2, and have received prior EGFR-TKI therapy.',
        'Study Results': 'NO',
        'Conditions': 'Lung Cancer|EGFR Mutation|Non-Small Cell Lung Cancer',
        'Interventions': 'DRUG: Osimertinib',
        'Phases': 'Phase 2',
        'Enrollment': '150',
        'Study Type': 'INTERVENTIONAL',
        'Study Design': 'Allocation: NON_RANDOMIZED|Intervention Model: SINGLE_GROUP|Masking: NONE|Primary Purpose: TREATMENT',
        'Start Date': '2024-04-01',
        'Completion Date': '2026-03'
    },
    {
        'NCT Number': 'NCT33333333',
        'Study Title': 'BRCA-Mutated Ovarian Cancer PARP Inhibitor Study',
        'Study URL': 'https://clinicaltrials.gov/study/NCT33333333',
        'Study Status': 'RECRUITING',
        'Brief Summary': 'This study evaluates niraparib maintenance therapy in patients with BRCA-mutated ovarian cancer. Eligible patients must have Stage III-IV BRCA+ ovarian cancer, be 18-75 years old, ECOG 0-1, and have completed platinum-based chemotherapy.',
        'Study Results': 'NO',
        'Conditions': 'Ovarian Cancer|BRCA Mutation',
        'Interventions': 'DRUG: Niraparib',
        'Phases': 'Phase 3',
        'Enrollment': '300',
        'Study Type': 'INTERVENTIONAL',
        'Study Design': 'Allocation: RANDOMIZED|Intervention Model: PARALLEL|Masking: DOUBLE|Primary Purpose: TREATMENT',
        'Start Date': '2024-05-01',
        'Completion Date': '2027-04'
    }
]

class SyntheticTestMatcher(PatientTrialMatcher):
    """Test version of PatientTrialMatcher using synthetic data."""
    
    def __init__(self, openai_api_key: str = None):
        """Initialize with synthetic data instead of real CSV."""
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            import openai
            openai.api_key = self.openai_api_key
        
        # Create synthetic patient dataframe
        self.patients_df = pd.DataFrame(SYNTHETIC_PATIENTS)
        

    
    async def get_trials_for_cancer_type(self, cancer_type: str):
        """Return synthetic trials filtered by cancer type."""
        filtered_trials = []
        for trial in SYNTHETIC_TRIALS:
            if cancer_type.lower() in trial['Conditions'].lower():
                filtered_trials.append(trial)
        
        print(f"Synthetic: Found {len(filtered_trials)} trials for {cancer_type}")
        return filtered_trials
    
    async def rank_trials_with_llm(self, patient: Patient, trials):
        """Simple ranking """
        ranked_trials = []
        
        for i, trial in enumerate(trials):
            # Simple scoring based on patient characteristics
            base_score = 50.0
            
            # Boost score for biomarker matches
            if patient.biomarkers:
                for biomarker in patient.biomarkers.split(';'):
                    if biomarker.strip().lower() in trial['Study Title'].lower():
                        base_score += 20
                        break
            
            # Boost for stage appropriateness
            if patient.stage.lower() in trial['Brief Summary'].lower():
                base_score += 15
            
            # Boost for age compatibility
            age_mentions = ['18', '21', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80']
            for age in age_mentions:
                if age in trial['Brief Summary'] and int(age) - 10 <= patient.age <= int(age) + 10:
                    base_score += 10
                    break
            
            # Add some variation
            base_score += (i * 2)
            
            reasoning = f"Simple analysis: {patient.cancer_type} patient with {patient.biomarkers or 'no biomarkers'} matched to {trial['Study Title']}"
            
            trial_match = TrialMatch(
                nct_id=trial['NCT Number'],
                title=trial['Study Title'],
                score=base_score,
                reasoning=reasoning,
                eligibility_criteria=trial['Brief Summary'],
                phase=trial['Phases'],
                status=trial['Study Status']
            )
            ranked_trials.append(trial_match)
        
        # Sort by score (highest first)
        ranked_trials.sort(key=lambda x: x.score, reverse=True)
        return ranked_trials

def evaluate_matching_quality(patient: Patient, ranked_trials: list) -> dict:
    """Evaluate the quality of matching results."""
    evaluation = {
        'patient_id': patient.patient_id,
        'cancer_type': patient.cancer_type,
        'total_trials': len(ranked_trials),
        'top_match_score': ranked_trials[0].score if ranked_trials else 0,
        'biomarker_matches': 0,
        'stage_matches': 0,
        'age_appropriate': 0,
        'relevant_trials': 0
    }
    
    for trial in ranked_trials:
        # Check biomarker matches
        if patient.biomarkers:
            for biomarker in patient.biomarkers.split(';'):
                if biomarker.strip().lower() in trial.title.lower():
                    evaluation['biomarker_matches'] += 1
                    break
        
        # Check stage appropriateness
        if patient.stage.lower() in trial.eligibility_criteria.lower():
            evaluation['stage_matches'] += 1
        
        # Check age appropriateness
        age_mentions = ['18', '21', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80']
        for age in age_mentions:
            if age in trial.eligibility_criteria and int(age) - 10 <= patient.age <= int(age) + 10:
                evaluation['age_appropriate'] += 1
                break
        
        # Check if trial is relevant (score > 50)
        if trial.score > 50:
            evaluation['relevant_trials'] += 1
    
    return evaluation

def print_evaluation_results(evaluations: list):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*80)
    print("MATCHING QUALITY EVALUATION")
    print("="*80)
    
    for eval_result in evaluations:
        print(f"\nPatient {eval_result['patient_id']} ({eval_result['cancer_type']}):")
        print(f"  Total trials found: {eval_result['total_trials']}")
        print(f"  Top match score: {eval_result['top_match_score']:.1f}/100")
        print(f"  Trials with biomarker matches: {eval_result['biomarker_matches']}")
        print(f"  Trials with stage matches: {eval_result['stage_matches']}")
        print(f"  Age-appropriate trials: {eval_result['age_appropriate']}")
        print(f"  Relevant trials (score >50): {eval_result['relevant_trials']}")
        
        # Calculate quality metrics
        if eval_result['total_trials'] > 0:
            relevance_rate = eval_result['relevant_trials'] / eval_result['total_trials'] * 100
            print(f"  Relevance rate: {relevance_rate:.1f}%")
        
        print("-" * 40)

def print_results(ranked_trials: list, patient_id: str):
    """Print the ranked trial results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"RANKED TRIALS FOR PATIENT {patient_id}")
    print(f"{'='*60}")
    
    if not ranked_trials:
        print("No suitable trials found.")
        return
    
    for i, trial in enumerate(ranked_trials, 1):
        print(f"\n{i}. {trial.nct_id}")
        print(f"   Title: {trial.title}")
        print(f"   Score: {trial.score:.1f}/100")
        print(f"   Phase: {trial.phase}")
        print(f"   Status: {trial.status}")
        print(f"   Reasoning: {trial.reasoning}")
        print("-" * 60)

async def run_synthetic_tests():
    """Run comprehensive tests with synthetic data."""
    print("Synthetic Clinical Trial Matching Test Suite")
    print("Testing with synthetic patients and trials...")
    
    matcher = SyntheticTestMatcher()
    evaluations = []
    
    # Test each synthetic patient
    for patient_data in SYNTHETIC_PATIENTS:
        patient_id = patient_data['patient_id']
        print(f"\n{'='*60}")
        print(f"TESTING PATIENT {patient_id}")
        print(f"{'='*60}")
        
        try:
            # Match patient to trials
            ranked_trials = await matcher.match_patient_to_trials(patient_id)
            
            # Print results
            print_results(ranked_trials, patient_id)
            
            # Evaluate matching quality
            patient = matcher.get_patient_by_id(patient_id)
            evaluation = evaluate_matching_quality(patient, ranked_trials)
            evaluations.append(evaluation)
            
        except Exception as e:
            print(f"Error testing patient {patient_id}: {e}")
    
    # Print evaluation summary
    print_evaluation_results(evaluations)
    
    # Calculate overall metrics
    if evaluations:
        avg_top_score = sum(e['top_match_score'] for e in evaluations) / len(evaluations)
        avg_relevance_rate = sum(e['relevant_trials'] / e['total_trials'] * 100 for e in evaluations if e['total_trials'] > 0) / len(evaluations)
        
        print(f"\n{'='*60}")
        print("OVERALL PERFORMANCE METRICS")
        print(f"{'='*60}")
        print(f"Average top match score: {avg_top_score:.1f}/100")
        print(f"Average relevance rate: {avg_relevance_rate:.1f}%")
        print(f"Total patients tested: {len(evaluations)}")

def main():
    """Run the synthetic test suite."""
    print("Clinical Trial Matching System - Synthetic Test Suite")
    print("This test uses synthetic data to evaluate matching quality.")
    print("No actual API calls are made.")
    
    asyncio.run(run_synthetic_tests())
    
    print("\n" + "="*60)
    print("SYNTHETIC TEST COMPLETE")
    print("="*60)
    print("\nTo run with real data:")
    print("1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
    print("2. Run: python match.py --patient_id P001")

if __name__ == "__main__":
    main() 