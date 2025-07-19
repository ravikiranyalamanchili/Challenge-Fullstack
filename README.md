#  Full-Stack Engineer Take‑Home (Clinical‑Trial Matching)

## 0  Overview
This is a coding challenge designed to test your ability to turn raw oncology data into actionable options for patients and clinicians. In this takehome, you will build a lean service that turns a **patient snapshot** (`data/patients.csv`) into an ordered list of *actively-recruiting* cancer trials and a clear measure of how well that ranking works.  

**Heads-up:** the `biomcp` SDK only streams live trial/biomedical data. **It is _not_ a matcher.** Designing the matching logic (rules, ML or LLM) **and** a sound evaluation harness is the real assignment.



## 1  Objectives

1. **Matching workflow**: Query trials with BioMCP, decide eligibility & relevance, output a ranked list of NCT IDs.  
2. **Evaluation & test design (primary deliverable)**: E.g., craft metrics, gold labels or synthetic patients, automated tests, false-negative analysis, etc. 
3. **Interface**: Expose one simple endpoint or CLI command such that  
     ```bash
     python match.py --patient_id P002
     ```  
     returns a ranked list of NCT IDs.



## 2  Repo skeleton
```bash
patients.csv   # add/augment as you like
src/           # your code (language agnostic)
tests/         # unit + integration tests & eval scripts
README.md      # you are here
```


## 3  Optional, but valuable, LLM boosts
1. Eligibility → features, parsing free-text criteria into JSON with LLMs
2. Ranking: Embed patient + trial summaries and score via cosine similarity.
3. Synthetic eval data: Few-shot an LLM to generate labelled fake patients/trials.
4. Explainability: Have the LLM narrate why each trial was chosen.
5. Feel free to ignore or modify these; creativity highly encouraged.


## 4  Submission
1. Push your work to a private fork, add bryan as a collaborator.
2. Email the link to bryan@radicalhealth.ai & simone@radicalhealth.ai and book a 45‑minute walkthrough call.
3. Rough time target ≈ 3 h. It’s OK to leave TODOs, just mark them explicitly. 
Hint: It's okay to start by focussing on just one cancer type, breast cancer is a great start.

Good luck! We’re excited to see how you think!

