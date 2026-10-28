"""
Project 3: Tech Stack Recommender (DecodeLabs - AI Recommendation Logic)

Pipeline:  Ingestion -> Scoring -> Sorting -> Filtering
Method:    Content-Based Filtering using TF-IDF + Cosine Similarity
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOP_N = 3          # how many recommendations to show
NUM_SKILLS = 3     # minimum skills required from the user


def load_dataset(path="raw_skills.csv"):
    """Load job roles (the 'items') and their skills from the CSV file."""
    df = pd.read_csv(path)
    df["skills"] = df["skills"].str.lower()   # same vocabulary = lowercase everywhere
    return df


def get_user_skills():
    """STEP 1 - INGESTION: ask the user for skills (min 3, no empty answers)."""
    print("=== Tech Stack Recommender ===")
    print(f"Enter {NUM_SKILLS} skills (example: Python, Cloud, Automation)\n")
    skills = []
    while len(skills) < NUM_SKILLS:
        skill = input(f"Enter skill {len(skills) + 1}: ").strip().lower()
        if skill == "":
            print("  Skill cannot be empty. Try again.")
            continue
        skills.append(skill)
    return skills


def recommend(user_skills, df, top_n=TOP_N):
    """STEPS 2-4: Scoring, Sorting, Filtering."""
    # Turn text into TF-IDF vectors. Fit on the job roles only,
    # so the vocabulary is the unique skills in our dataset.
    vectorizer = TfidfVectorizer()
    role_vectors = vectorizer.fit_transform(df["skills"])

    # Convert the user's skills into a vector in the SAME vocabulary space
    user_text = " ".join(user_skills)
    user_vector = vectorizer.transform([user_text])

    # STEP 2 - SCORING: cosine similarity between user and every role
    scores = cosine_similarity(user_vector, role_vectors).flatten()

    # STEP 3 - SORTING: highest score first
    ranked_indexes = scores.argsort()[::-1]

    # STEP 4 - FILTERING: keep only the Top-N
    # Words the user typed (same tokenizer style as TF-IDF: 2+ letters)
    user_words = set(re.findall(r"\b\w\w+\b", user_text))

    results = []
    for i in ranked_indexes[:top_n]:
        if scores[i] > 0:               # skip roles with zero match
            role_words = re.findall(r"\b\w\w+\b", df.loc[i, "skills"])
            # "Why this match?" -> skills the user and the role have in common
            matched = [w for w in dict.fromkeys(role_words) if w in user_words]
            results.append((df.loc[i, "role"], scores[i], matched))
    return results


def main():
    df = load_dataset()
    user_skills = get_user_skills()
    results = recommend(user_skills, df)

    print("\nYour skills:", ", ".join(user_skills))

    # Cold-start / no-match handling: all scores are 0
    if len(results) == 0:
        print("\nNo matching roles found. Try different or more common skills,")
        print("for example: python, sql, aws, docker, linux, javascript.")
        return

    print(f"\nTop {len(results)} Recommended Roles:")
    for rank, (role, score, matched) in enumerate(results, start=1):
        print(f"{rank}. {role} - {score * 100:.0f}% match")
        print(f"   Why: matched on -> {', '.join(matched)}")


if __name__ == "__main__":
    main()
