import os
import json
import textwrap
import requests
from datetime import datetime
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from candidate_profile import CANDIDATE_PROFILE, BASE_RESUME
from tools import load_jobs, filter_jobs, rank_jobs, tailor_resume
from prompts import SYSTEM_PROMPT, build_user_prompt
from build_jobs_csv import main as build_jobs_main


def local_reasoning(candidate_profile, jobs_df):
    return """
Agent Decision Trace:
1. The candidate profile was analyzed first.
2. Filtering is needed because the dataset contains jobs from different locations, companies, and experience levels.
3. The Filtering Tool should be called first to remove jobs that do not match preferred location, experience, and excluded companies.
4. After filtering, the Ranking Tool should be called to score jobs based on:
   - skill matching
   - experience alignment
   - location match
5. The highest scoring job should be selected as the best job.
6. The Resume Tailoring Tool should then rewrite the professional summary and modify exactly two experience bullet points for the selected top job.

Why this order?
Because filtering reduces irrelevant jobs first, then ranking works only on the best-matching subset.
""".strip()


def call_llm_for_reasoning(candidate_profile, dataset_preview, jobs_df):
    prompt = build_user_prompt(candidate_profile, dataset_preview)

    payload = {
        "model": "llama3",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        text = data["message"]["content"]
        return text, "ollama"
    except Exception as e:
        print("\n[WARNING] Ollama call failed.")
        print(f"Reason: {e}")
        print("Switching to local fallback reasoning mode...\n")
        return local_reasoning(candidate_profile, jobs_df), "fallback"


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_output_paths():
    os.makedirs("output", exist_ok=True)
    ts = get_timestamp()
    txt_path = f"output/agent_output_{ts}.txt"
    pdf_path = f"output/agent_output_{ts}.pdf"
    return txt_path, pdf_path


def save_text_file(content, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def save_pdf(content, path):
    c = canvas.Canvas(path, pagesize=LETTER)
    width, height = LETTER

    left_margin = 50
    top_margin = 50
    bottom_margin = 50
    line_height = 14
    max_width_chars = 95

    y = height - top_margin
    c.setFont("Helvetica", 10)

    for raw_line in content.split("\n"):
        wrapped_lines = textwrap.wrap(raw_line, width=max_width_chars) if raw_line.strip() else [""]

        for line in wrapped_lines:
            if y <= bottom_margin:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - top_margin

            c.drawString(left_margin, y, line)
            y -= line_height

    c.save()


def save_all_outputs(content):
    txt_path, pdf_path = get_output_paths()
    save_text_file(content, txt_path)
    save_pdf(content, pdf_path)
    return txt_path, pdf_path


def append_line(output_lines, text=""):
    print(text)
    output_lines.append(text)


def format_job_block(job, index=None):
    prefix = f"[{index}] " if index is not None else ""
    return (
        f"{prefix}{job['job_title']}\n"
        f"   Company: {job['company']}\n"
        f"   Location: {job['location']}\n"
        f"   Experience Required: {job['years_experience']}\n"
        f"   Skills: {job['required_skills']}\n"
        f"   URL: {job['url']}\n"
    )


def format_compact_job(job, index=None):
    prefix = f"[{index}] " if index is not None else ""
    return (
        f"{prefix}{job['job_title']} | {job['company']} | "
        f"{job['location']} | Exp: {job['years_experience']}"
    )


def format_compact_rank(job, index=None):
    prefix = f"[Rank {index}] " if index is not None else ""
    return (
        f"{prefix}{job['job_title']} | {job['company']} | "
        f"Score: {job['total_score']} | Skills: {job['matched_skills']}"
    )


def format_ranked_job_block(job, index=None):
    prefix = f"[Rank {index}] " if index is not None else ""
    return (
        f"{prefix}{job['job_title']}\n"
        f"   Company: {job['company']}\n"
        f"   Location: {job['location']}\n"
        f"   Matched Skills: {job['matched_skills']}\n"
        f"   Skill Score: {job['skill_score']}\n"
        f"   Experience Score: {job['experience_score']}\n"
        f"   Location Score: {job['location_score']}\n"
        f"   Total Score: {job['total_score']}\n"
        f"   URL: {job['url']}\n"
    )


def main():
    output_lines = []

    append_line(output_lines, "\n========== AI AGENT FOR JOB SEARCH & RESUME OPTIMIZATION ==========\n")

    # STEP 0: Rebuild jobs.csv every run
    append_line(output_lines, "STEP 0: Refreshing jobs.csv")
    build_jobs_main()
    append_line(output_lines, "jobs.csv refreshed successfully.\n")

    # STEP 1: Load dataset
    jobs_df = load_jobs("data/jobs.csv")
    append_line(output_lines, "STEP 1: Dataset Preview")
    append_line(output_lines, f"Total jobs in dataset: {len(jobs_df)}")
    append_line(output_lines, "\nShowing first 5 jobs:")
    for i, job in enumerate(jobs_df.head(5).to_dict(orient="records"), start=1):
        append_line(output_lines, format_compact_job(job, i))

    # STEP 2: Agent reasoning
    append_line(output_lines, "\nSTEP 2: Agent Reasoning")
    dataset_preview = jobs_df.head(5).to_string(index=False)
    reasoning_text, reasoning_mode = call_llm_for_reasoning(
        CANDIDATE_PROFILE,
        dataset_preview,
        jobs_df
    )
    append_line(output_lines, reasoning_text)
    append_line(output_lines, f"\nReasoning mode used: {reasoning_mode}")

    # STEP 3: Filtering tool
    append_line(output_lines, "\nSTEP 3: Filtering Tool Output")
    filtered_jobs = filter_jobs(
        jobs_df,
        preferred_location=CANDIDATE_PROFILE["preferred_location"],
        years_experience=CANDIDATE_PROFILE["years_experience"],
        exclude_companies=CANDIDATE_PROFILE["exclude_companies"],
        remote_only=CANDIDATE_PROFILE["remote_only"]
    )

    if filtered_jobs.empty:
        append_line(output_lines, "No jobs matched filtering criteria.")
        final_output = "\n".join(output_lines)
        txt_path, pdf_path = save_all_outputs(final_output)
        print(f"Saved text output to: {txt_path}")
        print(f"Saved PDF output to: {pdf_path}")
        return

    append_line(output_lines, f"Filtered jobs count: {len(filtered_jobs)}")
    append_line(output_lines, "Compact filtered list:")
    for i, job in enumerate(filtered_jobs.to_dict(orient="records"), start=1):
        append_line(output_lines, format_compact_job(job, i))

    # STEP 4: Ranking tool
    append_line(output_lines, "\nSTEP 4: Ranking Tool Output")
    ranked_jobs = rank_jobs(
        filtered_jobs,
        candidate_skills=CANDIDATE_PROFILE["skills"],
        candidate_experience=CANDIDATE_PROFILE["years_experience"],
        preferred_location=CANDIDATE_PROFILE["preferred_location"]
    )

    append_line(output_lines, "Compact ranking list:")
    for i, job in enumerate(ranked_jobs.head(10).to_dict(orient="records"), start=1):
        append_line(output_lines, format_compact_rank(job, i))

    # STEP 5: Top 3 jobs
    append_line(output_lines, "\nSTEP 5: Top 3 Jobs")
    top_3 = ranked_jobs.head(3)

    for i, job in enumerate(top_3.to_dict(orient="records"), start=1):
        append_line(output_lines, format_ranked_job_block(job, i))

    # STEP 6: Best job selected
    top_job = ranked_jobs.iloc[0]
    append_line(output_lines, "\nSTEP 6: Best Job Selected")
    append_line(output_lines, format_ranked_job_block(top_job, 1))

    # STEP 7: Resume tailoring
    append_line(output_lines, "\nSTEP 7: Resume Tailoring Tool Output")
    tailored = tailor_resume(BASE_RESUME, top_job)

    append_line(output_lines, "\nTailored Professional Summary:")
    append_line(output_lines, tailored["professional_summary"])

    append_line(output_lines, "\nModified Experience Bullet 1:")
    append_line(output_lines, f"- {tailored['modified_experience_bullets'][0]}")

    append_line(output_lines, "\nModified Experience Bullet 2:")
    append_line(output_lines, f"- {tailored['modified_experience_bullets'][1]}")

    append_line(output_lines, "\nAligned Skills:")
    append_line(output_lines, ", ".join(tailored["aligned_skills"]))

    # STEP 8: Final reasoning trace
    append_line(output_lines, "\nSTEP 8: Final Reasoning Trace")
    append_line(
        output_lines,
        "The agent analyzed the candidate profile, filtered jobs based on location, "
        "experience, and excluded companies, ranked the remaining jobs using skill match, "
        "experience alignment, and location match, selected the highest scoring job, "
        "and tailored the resume only for that top-ranked role."
    )

    append_line(output_lines, "\n========== END ==========\n")

    final_output = "\n".join(output_lines)

    # timestamp created here, at actual save time
    txt_path, pdf_path = save_all_outputs(final_output)

    print(f"Saved text output to: {txt_path}")
    print(f"Saved PDF output to: {pdf_path}")


if __name__ == "__main__":
    main()