# Tech Stack Recommender

A content-based recommendation system built for DecodeLabs AI Project 3.
It takes 3 skills from the user and recommends the Top 3 matching job roles.

## How it works
1. **Ingestion** - user enters 3 skills
2. **Scoring** - skills and job roles are converted to TF-IDF vectors; cosine similarity is calculated
3. **Sorting** - roles are ranked by score (highest first)
4. **Filtering** - only the Top 3 are displayed

## Tech used
Python, pandas, scikit-learn (TfidfVectorizer, cosine_similarity)

## Run it
```
pip install pandas scikit-learn
python main.py
```

## Sample output
```
Enter skill 1: python
Enter skill 2: sql
Enter skill 3: statistics

Top 3 Recommended Roles:
1. Data Analyst - 54% match
2. Product Analyst - 54% match
3. Data Scientist - 44% match
```

## Extra features
- "Why this match?" shows which of your skills matched each role

## Handles edge cases
- Empty input is rejected
- No match (cold start) shows a helpful message

## Live demo
https://github.com/emanshafiq1402-en/DecodeLabs-Internship-Project-3

## Files
- `main.py` - the recommender
- `raw_skills.csv` - dataset of roles and skills
- `index.html` - web version (for GitHub Pages)
- `requirements.txt` - Python libraries needed
