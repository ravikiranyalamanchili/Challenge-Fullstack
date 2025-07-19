#  Clinical‑Trial Matching Take‑Home

Welcome — your mission is to build a small, end‑to‑end service that matches cancer patients to the most relevant **open clinical trials**, then shows how well the matching works.

---

## 1  Objectives

1. **Trial‑matching workflow**  
   - Use the open‑source **[biomcp](https://github.com/genomoncology/biomcp)** repo to interpret patient features and query clinical‑trial registries.  
   - Rank trials by relevance **and** eligibility.

2. **Evaluation & test design** *most important*  
   - Propose *and implement* a minimal yet rigorous evaluation strategy (e.g., synthetic gold labels, precision/recall, false‑negative analysis).  
   - Write automated tests that fail clearly when the matcher degrades.

3. **Presentation**  
   - Expose one simple endpoint or CLI command such that  
     ```bash
     python match.py --patient_id P002
     ```  
     returns a ranked list of NCT IDs.

---

## 2  Repository layout

| Path | Purpose |
|------|---------|
| `data/patients.csv` | Sample dataset (15 records) — please augment or mock more. |
| `src/` | Your code (Python, TypeScript, any language is fine). |
| `tests/` | Unit / integration tests & evaluation scripts. |
| `README.md` | _you are here_ |

---

## 3  Quick‑start

Create a fork.

## 4  What we expect

| Area                  | Minimum bar                                                          | Stretch ideas (optional)                                                 |
| --------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Correctness**       | Basic match list returned without crashing.                          | Heuristics for trial inclusion/exclusion, genomic filters, geo‑distance. |
| **Evaluation design** | At least one reproducible metric and ≥3 test cases.                  | Want to see what you come up with here.          |                 |
| **Communication**     | Inline comments & a short `DECISIONS.md` if you made big trade‑offs. |                         |

## 5  Submission
1. Push your work to a public GitHub fork (private is fine if you add us as collaborators).

2. Email the link to simone@radicalhealth.ai & bryan@radicalhealth.ai and book a 30‑minute walkthrough call.

3. Total suggested time: ~3 h. It’s OK to leave TODOs; just be explicit. 
Hint: It's okay to start by focussing on just one cancer type, breast cancer is a great start.

Good luck! We’re excited to see how you think!

