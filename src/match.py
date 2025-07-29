import pandas as pd
import json
import asyncio
import argparse
import os
from typing import List, Dict, Any
from dataclasses import dataclass
import openai
from biomcp.trials.search import (
    RecruitingStatus,
    TrialPhase,
    TrialQuery,
    search_trials,
)

@dataclass
class Patient:
    patient_id: str
    age: int
    gender: str
    cancer_type: str
    stage: str
    biomarkers: str
    location: str
    ecog_status: int
    prior_treatments: str

@dataclass
class TrialMatch:
    nct_id: str
    title: str
    score: float
    reasoning: str
    eligibility_criteria: str
    phase: str
    status: str

class PatientTrialMatcher:
    def __init__(self, openai_api_key: str = None):
        """Initialize the matcher with OpenAI API key."""
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        else:
            raise ValueError("No OpenAI API key provided. Please set the OPENAI_API_KEY environment variable or provide an API key when initializing the matcher.")
        
        # Load patient data
        try:
            self.patients_df = pd.read_csv('patients.csv')
        except Exception as e:
            raise ValueError(f"Error loading patient data: {e}")
        
    def get_patient_by_id(self, patient_id: str) -> Patient:
        """Retrieve patient data by ID."""
        patient_data = self.patients_df[self.patients_df['patient_id'] == patient_id]
        
        if patient_data.empty:
            raise ValueError(f"Patient ID {patient_id} not found in database")
        
        row = patient_data.iloc[0]
        return Patient(
            patient_id=row['patient_id'],
            age=row['age'],
            gender=row['gender'],
            cancer_type=row['cancer_type'],
            stage=row['stage'],
            biomarkers=row['biomarkers'] if pd.notna(row['biomarkers']) else "",
            location=row['location'],
            ecog_status=row['ecog_status'],
            prior_treatments=row['prior_treatments']
        )
    
    async def get_trials_for_cancer_type(self, cancer_type: str) -> List[Dict[str, Any]]:
        """Retrieve active trials for a specific cancer type using BioMCP."""
        try:
            query = TrialQuery(
                conditions=[cancer_type],
                recruiting_status=RecruitingStatus.OPEN,
                # Don't limit to specific phase to get more options
            )
            
            json_output_str = await search_trials(query, output_json=True)
            trials_data = json.loads(json_output_str)
            
            # Extract relevant trial information
            trials = []
            # Convert the JSON structure to match expected format
            for trial in trials_data:
                trial_info = {
                    'nct_id': trial.get('NCT Number', ''),
                    'title': trial.get('Study Title', ''),
                    'eligibility_criteria': trial.get('Brief Summary', ''),
                    'phase': trial.get('Phases', ''),
                    'status': trial.get('Study Status', ''),
                    'conditions': trial.get('Conditions', '').split('|') if trial.get('Conditions') else [],
                    'interventions': [{'name': i} for i in trial.get('Interventions', '').split('|')] if trial.get('Interventions') else [],
                    'locations': []  # Location data not available in this format
                }
                trials.append(trial_info)
         
            return trials
            
        except Exception as e:
            print(f"Error retrieving trials: {e}")
            return []
    
    def create_patient_summary(self, patient: Patient) -> str:
        """Create a comprehensive patient summary for LLM matching."""
        summary = f"""
Patient Profile:
- ID: {patient.patient_id}
- Age: {patient.age} years old
- Gender: {patient.gender}
- Cancer Type: {patient.cancer_type}
- Stage: {patient.stage}
- Biomarkers: {patient.biomarkers if patient.biomarkers else 'None specified'}
- Location: {patient.location}
- ECOG Status: {patient.ecog_status} (0=fully active, 1=restricted, 2=ambulatory)
- Prior Treatments: {patient.prior_treatments}
"""
        return summary
    
    def create_trial_summary(self, trial: Dict[str, Any]) -> str:
        """Create a summary of trial information for LLM matching."""
        summary = f"""
Trial: {trial['nct_id']}
Title: {trial['title']}
Phase: {trial['phase']}
Status: {trial['status']}
Conditions: {', '.join(trial['conditions']) if trial['conditions'] else 'Not specified'}
Interventions: {', '.join([i.get('name', '') for i in trial['interventions']]) if trial['interventions'] else 'Not specified'}
Eligibility Criteria: {trial['eligibility_criteria'][:500] + '...' if len(trial['eligibility_criteria']) > 500 else trial['eligibility_criteria']}
"""
        return summary
    
    async def rank_trials_with_llm(self, patient: Patient, trials: List[Dict[str, Any]]) -> List[TrialMatch]:
        # TODO: Refactor this to use cosine similarity
        """Use ChatGPT to rank trials based on patient compatibility."""
        if not self.openai_api_key:
            # Fallback to simple ranking if no API key
            # TODO: Implement simple ranking
            print("No OpenAI API key provided. LLM ranking will be disabled.")
            return
        
        patient_summary = self.create_patient_summary(patient)
        
        ranked_trials = []
        
        for trial in trials:
            trial_summary = self.create_trial_summary(trial)
            
            prompt = f"""
You are a clinical trial matching expert. Given the following patient profile and clinical trial, rate how well this trial matches the patient on a scale of 0-100, where:
- 0-20: Poor match (major contraindications, wrong cancer type, etc.)
- 21-40: Fair match (some relevant factors but significant issues)
- 41-60: Good match (relevant cancer type, appropriate stage, minor issues)
- 61-80: Very good match (excellent fit, appropriate biomarkers, stage)
- 81-100: Excellent match (perfect fit, ideal patient profile)

{patient_summary}

{trial_summary}

Please respond with a JSON object containing:
{{
    "score": <0-100>,
    "reasoning": "<detailed explanation of why this score was given>"
}}

Consider factors like:
- Cancer type match
- Disease stage appropriateness
- Biomarker compatibility
- Age and gender eligibility
- Prior treatment history
- Geographic location
- ECOG status compatibility
"""
            
            try:
                response = await openai.ChatCompletion.acreate(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
                
                result = json.loads(response.choices[0].message.content)
                
                trial_match = TrialMatch(
                    nct_id=trial['nct_id'],
                    title=trial['title'],
                    score=result['score'],
                    reasoning=result['reasoning'],
                    eligibility_criteria=trial['eligibility_criteria'],
                    phase=trial['phase'],
                    status=trial['status']
                )
                ranked_trials.append(trial_match)
                
            except Exception as e:
                print(f"Error ranking trial {trial['nct_id']}: {e}")
                # Fallback score
                trial_match = TrialMatch(
                    nct_id=trial['nct_id'],
                    title=trial['title'],
                    score=50.0,
                    reasoning="Error in LLM ranking - assigned neutral score",
                    eligibility_criteria=trial['eligibility_criteria'],
                    phase=trial['phase'],
                    status=trial['status']
                )
                ranked_trials.append(trial_match)
        
        # Sort by score (highest first)
        # TODO: take into account the study status and phase to give more weight to open trials
        ranked_trials.sort(key=lambda x: x.score, reverse=True)
        return ranked_trials
    
    async def match_patient_to_trials(self, patient_id: str) -> List[TrialMatch]:
        """Main function to match a patient to clinical trials."""
        print(f"Finding clinical trials for patient {patient_id}...")
        
        # Get patient data
        patient = self.get_patient_by_id(patient_id)
        print(f"Patient: {patient.cancer_type}, Stage: {patient.stage}")
        
        # Get trials for cancer type
        trials = await self.get_trials_for_cancer_type(patient.cancer_type)
        print(f"Found {len(trials)} active trials for {patient.cancer_type}")
        
        if not trials:
            print("No trials found for this cancer type.")
            return []
        
        # Rank trials using LLM
        ranked_trials = await self.rank_trials_with_llm(patient, trials)
        
        return ranked_trials

def print_results(ranked_trials: List[TrialMatch]):
    """Print the ranked trial results in a formatted way."""
    print("\n" + "="*80)
    print("RANKED CLINICAL TRIAL MATCHES")
    print("="*80)
    
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
        print("-" * 80)

async def main():
    parser = argparse.ArgumentParser(description='Match patients to clinical trials')
    parser.add_argument('--patient_id', required=True, help='Patient ID to match')
    parser.add_argument('--openai_key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Initialize matcher
    matcher = PatientTrialMatcher(openai_api_key=args.openai_key)
    
    try:
        # Match patient to trials
        ranked_trials = await matcher.match_patient_to_trials(args.patient_id)
        
        # Print results
        print_results(ranked_trials)
        
        # Return NCT IDs in order (for programmatic use)
        nct_ids = [trial.nct_id for trial in ranked_trials]
        return nct_ids
        
    except ValueError as e:
        print(f"Error: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

if __name__ == "__main__":
    asyncio.run(main())










