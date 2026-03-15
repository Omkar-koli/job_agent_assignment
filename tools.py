import pandas as pd


def load_jobs(csv_path="data/jobs.csv"):
    df = pd.read_csv(csv_path)
    return df


def normalize_skills(skill_text):
    if pd.isna(skill_text):
        return set()
    return {s.strip().lower() for s in str(skill_text).split(";") if s.strip()}


def filter_jobs(df, preferred_location, years_experience, exclude_companies=None, remote_only=False):
    """
    Filtering Tool
    Rules:
    - location preference
    - experience limit
    - company exclusion
    - remote-only optional
    """
    if exclude_companies is None:
        exclude_companies = []

    filtered_df = df.copy()

    # Exclude companies
    filtered_df = filtered_df[~filtered_df["company"].isin(exclude_companies)]

    # Experience filter: keep jobs requiring <= candidate experience
    filtered_df = filtered_df[filtered_df["years_experience"] <= years_experience]

    # Location filter
    if remote_only:
        filtered_df = filtered_df[
            filtered_df["location"].astype(str).str.lower().str.contains("remote", na=False)
        ]
    else:
        filtered_df = filtered_df[
            filtered_df["location"].astype(str).str.lower().str.contains(preferred_location.lower(), na=False)
            | filtered_df["location"].astype(str).str.lower().str.contains("remote", na=False)
        ]

    return filtered_df.reset_index(drop=True)


def rank_jobs(df, candidate_skills, candidate_experience, preferred_location):
    """
    Ranking Tool
    Score based on:
    - skill match
    - experience alignment
    - location match
    """
    ranked_rows = []

    candidate_skill_set = {skill.strip().lower() for skill in candidate_skills}

    for _, row in df.iterrows():
        job_skills = normalize_skills(row["required_skills"])
        matched_skills = candidate_skill_set.intersection(job_skills)

        skill_score = len(matched_skills) * 10

        exp_required = row["years_experience"]
        exp_gap = abs(candidate_experience - exp_required)
        exp_score = max(0, 20 - (exp_gap * 5))

        location_text = str(row["location"]).lower()
        if preferred_location.lower() in location_text:
            location_score = 10
        elif "remote" in location_text:
            location_score = 8
        else:
            location_score = 0

        total_score = skill_score + exp_score + location_score

        ranked_rows.append({
            "job_title": row["job_title"],
            "company": row["company"],
            "location": row["location"],
            "required_skills": row["required_skills"],
            "years_experience": row["years_experience"],
            "job_description": row["job_description"],
            "url": row["url"],
            "matched_skills": ", ".join(sorted(matched_skills)),
            "skill_score": skill_score,
            "experience_score": exp_score,
            "location_score": location_score,
            "total_score": total_score
        })

    ranked_df = pd.DataFrame(ranked_rows)
    ranked_df = ranked_df.sort_values(by="total_score", ascending=False).reset_index(drop=True)
    return ranked_df


def tailor_resume(base_resume, top_job):
    """
    Resume Tailoring Tool
    - rewrite professional summary
    - modify exactly 2 experience bullet points
    - highlight aligned skills
    """
    required_skills = [s.strip() for s in str(top_job["required_skills"]).split(";")]
    aligned_skills = [s for s in base_resume["skills"] if s.lower() in [x.lower() for x in required_skills]]

    tailored_summary = (
        f"Engineering Data Science graduate student with hands-on experience in "
        f"{', '.join(aligned_skills[:5])}. Strong interest in the {top_job['job_title']} role at "
        f"{top_job['company']}, with experience in data analysis, machine learning, and solving "
        f"real-world business problems using data-driven methods."
    )

    modified_bullets = [
        f"Built data-driven projects using {', '.join(aligned_skills[:4])} to analyze datasets and support machine learning workflows.",
        f"Applied analytical and technical skills relevant to the {top_job['job_title']} role, including preprocessing data, modeling, and communicating insights effectively."
    ]

    return {
        "professional_summary": tailored_summary,
        "modified_experience_bullets": modified_bullets,
        "aligned_skills": aligned_skills
    }