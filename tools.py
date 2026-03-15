import os
import re
import pandas as pd


REQUIRED_COLUMNS = [
    "job_title",
    "company",
    "location",
    "required_skills",
    "years_experience",
    "job_description",
    "url"
]

SENIOR_TITLE_KEYWORDS = [
    "director", "head", "vp", "vice president",
    "manager", "principal", "staff", "lead"
]

TITLE_RELEVANCE_KEYWORDS = {
    "machine learning engineer": 12,
    "ml engineer": 12,
    "ai engineer": 12,
    "data scientist": 11,
    "applied scientist": 11,
    "research scientist": 10,
    "data analyst": 9,
    "analytics engineer": 9,
    "ai product manager": 6,
    "product manager": 4
}


def validate_jobs_csv(csv_path="data/jobs.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    if df.empty:
        raise ValueError("The jobs CSV is empty.")

    return df


def load_jobs(csv_path="data/jobs.csv"):
    df = validate_jobs_csv(csv_path)

    # Basic cleanup
    for col in ["job_title", "company", "location", "required_skills", "job_description", "url"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["years_experience"] = pd.to_numeric(df["years_experience"], errors="coerce").fillna(0).astype(int)

    return df


def normalize_skills(skill_text):
    if pd.isna(skill_text) or not str(skill_text).strip():
        return set()
    return {s.strip().lower() for s in str(skill_text).split(";") if s.strip()}


def normalize_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip().lower()


def is_senior_title(job_title):
    title = normalize_text(job_title)
    return any(keyword in title for keyword in SENIOR_TITLE_KEYWORDS)


def count_description_keywords(job_description, candidate_skills):
    text = normalize_text(job_description)

    default_keywords = {
        "python", "sql", "machine learning", "deep learning", "analytics",
        "data analysis", "statistics", "model", "deployment", "dashboard",
        "experimentation", "scikit-learn", "pandas", "numpy"
    }

    candidate_keywords = {skill.strip().lower() for skill in candidate_skills}
    all_keywords = default_keywords.union(candidate_keywords)

    matches = [kw for kw in all_keywords if kw in text]
    return len(matches), sorted(matches)


def score_title_relevance(job_title):
    title = normalize_text(job_title)
    score = 0

    for keyword, value in TITLE_RELEVANCE_KEYWORDS.items():
        if keyword in title:
            score = max(score, value)

    return score


def seniority_penalty(job_title, candidate_experience):
    title = normalize_text(job_title)

    if not is_senior_title(title):
        return 0

    if candidate_experience <= 2:
        return 15
    if candidate_experience <= 4:
        return 8
    return 0


def filter_jobs(df, preferred_location, years_experience, exclude_companies=None, remote_only=False):
    """
    Filtering Tool
    Rules:
    - company exclusion
    - years of experience filter
    - location filter
    - title seniority filter
    """
    if exclude_companies is None:
        exclude_companies = []

    exclude_companies_normalized = {str(c).strip().lower() for c in exclude_companies}

    filtered_rows = []
    stats = {
        "total_input_jobs": len(df),
        "removed_company_exclusion": 0,
        "removed_experience": 0,
        "removed_location": 0,
        "removed_seniority_title": 0,
        "remaining_jobs": 0
    }

    preferred_location_lower = preferred_location.strip().lower()

    for _, row in df.iterrows():
        company = normalize_text(row["company"])
        location = normalize_text(row["location"])
        job_title = normalize_text(row["job_title"])
        job_exp = int(row["years_experience"])

        # Company exclusion
        if company in exclude_companies_normalized:
            stats["removed_company_exclusion"] += 1
            continue

        # Experience filter
        if job_exp > years_experience:
            stats["removed_experience"] += 1
            continue

        # Senior title filter for junior candidate
        if years_experience <= 2 and is_senior_title(job_title):
            stats["removed_seniority_title"] += 1
            continue

        # Location filter
        is_remote = "remote" in location
        location_match = preferred_location_lower in location

        if remote_only:
            if not is_remote:
                stats["removed_location"] += 1
                continue
        else:
            if not (location_match or is_remote):
                stats["removed_location"] += 1
                continue

        filtered_rows.append(row.to_dict())

    filtered_df = pd.DataFrame(filtered_rows)
    stats["remaining_jobs"] = len(filtered_df)

    if filtered_df.empty:
        return filtered_df, stats

    return filtered_df.reset_index(drop=True), stats


def rank_jobs(df, candidate_skills, candidate_experience, preferred_location):
    """
    Ranking Tool
    Score dimensions:
    - skill match
    - experience alignment
    - location match
    - title relevance
    - description keyword match
    - seniority penalty
    """
    ranked_rows = []
    candidate_skill_set = {skill.strip().lower() for skill in candidate_skills}
    preferred_location_lower = preferred_location.strip().lower()

    for _, row in df.iterrows():
        job_skills = normalize_skills(row["required_skills"])
        matched_skills = sorted(candidate_skill_set.intersection(job_skills))
        matched_skill_count = len(matched_skills)

        # 1. Skill score
        skill_score = matched_skill_count * 12

        # 2. Experience score
        exp_required = int(row["years_experience"])
        exp_gap = abs(candidate_experience - exp_required)
        experience_score = max(0, 20 - (exp_gap * 4))

        # 3. Location score
        location_text = normalize_text(row["location"])
        if preferred_location_lower in location_text:
            location_score = 10
        elif "remote" in location_text:
            location_score = 8
        else:
            location_score = 0

        # 4. Title relevance
        title_score = score_title_relevance(row["job_title"])

        # 5. Description keyword score
        description_keyword_count, description_keywords = count_description_keywords(
            row["job_description"],
            candidate_skills
        )
        description_score = min(description_keyword_count * 2, 12)

        # 6. Seniority penalty
        penalty = seniority_penalty(row["job_title"], candidate_experience)

        total_score = (
            skill_score +
            experience_score +
            location_score +
            title_score +
            description_score -
            penalty
        )

        ranked_rows.append({
            "job_title": row["job_title"],
            "company": row["company"],
            "location": row["location"],
            "required_skills": row["required_skills"],
            "years_experience": exp_required,
            "job_description": row["job_description"],
            "url": row["url"],
            "matched_skills": ", ".join(matched_skills),
            "matched_skill_count": matched_skill_count,
            "description_keywords": ", ".join(description_keywords[:8]),
            "skill_score": skill_score,
            "experience_score": experience_score,
            "location_score": location_score,
            "title_score": title_score,
            "description_score": description_score,
            "seniority_penalty": penalty,
            "total_score": total_score,
            "experience_gap": exp_gap
        })

    ranked_df = pd.DataFrame(ranked_rows)

    if ranked_df.empty:
        return ranked_df

    # Tie-breakers
    ranked_df = ranked_df.sort_values(
        by=["total_score", "matched_skill_count", "title_score", "experience_gap", "location_score"],
        ascending=[False, False, False, True, False]
    ).reset_index(drop=True)

    return ranked_df


def extract_top_keywords(top_job):
    skills = [s.strip() for s in str(top_job["required_skills"]).split(";") if s.strip()]
    desc_keywords = [s.strip() for s in str(top_job.get("description_keywords", "")).split(",") if s.strip()]

    ordered = []
    seen = set()

    for item in skills + desc_keywords:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(item)

    return ordered[:8]


def tailor_resume(base_resume, top_job):
    """
    Resume Tailoring Tool
    - rewrite professional summary
    - modify exactly 2 experience bullet points
    - highlight aligned skills
    """
    top_keywords = extract_top_keywords(top_job)
    aligned_skills = [s for s in base_resume["skills"] if s.lower() in [k.lower() for k in top_keywords]]

    if len(aligned_skills) < 3:
        aligned_skills = base_resume["skills"][:5]

    role_title = top_job["job_title"]
    company = top_job["company"]

    tailored_summary = (
        f"Engineering Data Science graduate student with hands-on experience in "
        f"{', '.join(aligned_skills[:5])}. Strong alignment with the {role_title} role at {company}, "
        f"with experience building data and machine learning workflows, analyzing structured datasets, "
        f"and translating technical results into actionable insights for real-world decision-making."
    )

    modified_bullets = [
        (
            f"Built machine learning and analytics workflows using {', '.join(aligned_skills[:4])} "
            f"to clean structured datasets, support predictive modeling, and deliver reproducible insights "
            f"aligned with business and engineering needs."
        ),
        (
            f"Applied data analysis, preprocessing, and model-oriented problem solving in academic and project work, "
            f"developing experience relevant to the {role_title} position, including structured experimentation, "
            f"technical communication, and solution design."
        )
    ]

    return {
        "professional_summary": tailored_summary,
        "modified_experience_bullets": modified_bullets,
        "aligned_skills": aligned_skills,
        "top_keywords_used": top_keywords
    }