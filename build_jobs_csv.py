import csv
import requests
from typing import List, Dict

# -------------------------------------------------
# CONFIG: add more Greenhouse / Lever company boards
# -------------------------------------------------

GREENHOUSE_BOARDS = [
    "openai",
    "huggingface",
    "scaleai",
    "databricks",
    "stripe"
]

LEVER_BOARDS = [
    "netflix",
    "airbnb",
    "discord",
    "figma",
    "roblox"
]

AI_KEYWORDS = [
    "ai", "ml", "machine learning", "data scientist", "data science",
    "applied scientist", "research scientist", "nlp", "llm",
    "computer vision", "deep learning", "analytics", "data engineer",
    "business intelligence", "decision scientist", "prompt engineer"
]


def is_ai_related(title: str, description: str) -> bool:
    text = f"{title} {description}".lower()
    return any(keyword in text for keyword in AI_KEYWORDS)


def clean_text(text: str) -> str:
    if not text:
        return ""
    return (
        text.replace("\n", " ")
            .replace("\r", " ")
            .replace("\t", " ")
            .replace(",", " ")
            .strip()
    )


def shorten_description(text: str, max_chars: int = 600) -> str:
    text = clean_text(text)
    return text[:max_chars]


def infer_skills(text: str) -> str:
    skill_bank = [
        "Python", "SQL", "Machine Learning", "Deep Learning", "NLP",
        "LLM", "PyTorch", "TensorFlow", "Scikit-learn", "Pandas",
        "NumPy", "Spark", "AWS", "Azure", "GCP", "Statistics",
        "Data Analysis", "Data Engineering", "MLOps", "Computer Vision"
    ]
    text_lower = text.lower()
    found = [skill for skill in skill_bank if skill.lower() in text_lower]
    return "; ".join(found[:8]) if found else "Python; SQL; Machine Learning"


def infer_years_experience(text: str) -> int:
    text_lower = text.lower()

    for n in range(0, 11):
        patterns = [
            f"{n}+ years",
            f"{n} years",
            f"{n} year",
            f"{n}+ year"
        ]
        if any(p in text_lower for p in patterns):
            return n

    if "senior" in text_lower or "staff" in text_lower:
        return 5
    if "lead" in text_lower or "principal" in text_lower:
        return 7
    if "intern" in text_lower:
        return 0

    return 2


def fetch_greenhouse_jobs(board_token: str) -> List[Dict]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs"
    jobs = []

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()

        for job in data.get("jobs", []):
            title = clean_text(job.get("title", ""))
            location = clean_text(job.get("location", {}).get("name", ""))
            absolute_url = job.get("absolute_url", "")

            # Greenhouse list endpoint may not include full description
            desc = title

            if is_ai_related(title, desc):
                combined_text = f"{title} {location}"
                jobs.append({
                    "job_title": title,
                    "company": board_token,
                    "location": location if location else "Not specified",
                    "required_skills": infer_skills(combined_text),
                    "years_experience": infer_years_experience(combined_text),
                    "job_description": shorten_description(
                        f"{title} role at {board_token}. Public job posting collected from Greenhouse board."
                    ),
                    "url": absolute_url
                })
    except Exception as e:
        print(f"[Greenhouse] Failed for {board_token}: {e}")

    return jobs


def fetch_lever_jobs(company_slug: str) -> List[Dict]:
    url = f"https://api.lever.co/v0/postings/{company_slug}?mode=json"
    jobs = []

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()

        for job in data:
            title = clean_text(job.get("text", ""))
            categories = job.get("categories", {}) or {}
            location = clean_text(categories.get("location", ""))
            description_plain = clean_text(job.get("descriptionPlain", "") or "")
            hosted_url = job.get("hostedUrl", "")

            if is_ai_related(title, description_plain):
                combined_text = f"{title} {description_plain}"

                jobs.append({
                    "job_title": title,
                    "company": company_slug,
                    "location": location if location else "Not specified",
                    "required_skills": infer_skills(combined_text),
                    "years_experience": infer_years_experience(combined_text),
                    "job_description": shorten_description(description_plain if description_plain else title),
                    "url": hosted_url
                })
    except Exception as e:
        print(f"[Lever] Failed for {company_slug}: {e}")

    return jobs


def deduplicate_jobs(jobs: List[Dict]) -> List[Dict]:
    seen = set()
    unique_jobs = []

    for job in jobs:
        key = (
            job["job_title"].strip().lower(),
            job["company"].strip().lower(),
            job["location"].strip().lower()
        )
        if key not in seen:
            seen.add(key)
            unique_jobs.append(job)

    return unique_jobs


def save_to_csv(jobs: List[Dict], path: str = "data/jobs.csv"):
    fieldnames = [
        "job_title",
        "company",
        "location",
        "required_skills",
        "years_experience",
        "job_description",
        "url"
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(jobs)


def main():
    all_jobs = []

    print("Fetching Greenhouse jobs...")
    for board in GREENHOUSE_BOARDS:
        all_jobs.extend(fetch_greenhouse_jobs(board))

    print("Fetching Lever jobs...")
    for board in LEVER_BOARDS:
        all_jobs.extend(fetch_lever_jobs(board))

    all_jobs = deduplicate_jobs(all_jobs)

    # Keep 20-30 jobs as assignment needs
    final_jobs = all_jobs[:30]

    save_to_csv(final_jobs, "data/jobs.csv")

    print(f"\nSaved {len(final_jobs)} jobs to data/jobs.csv")
    if final_jobs:
        print("Sample row:")
        print(final_jobs[0])


if __name__ == "__main__":
    main()