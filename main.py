import os
import json
import textwrap
import requests
from datetime import datetime
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from candidate_profile import CANDIDATE_PROFILE, BASE_RESUME
from tools import load_jobs, filter_jobs, rank_jobs, tailor_resume
from prompts import (
    SYSTEM_PROMPT,
    TAILORING_SYSTEM_PROMPT,
    build_filtering_decision_prompt,
    build_ranking_decision_prompt,
    build_top_job_justification_prompt,
    build_resume_tailoring_prompt
)


OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3"


def safe_json_load(text):
    try:
        return json.loads(text)
    except Exception:
        return None


def local_filtering_decision():
    return {
        "next_tool": "filtering",
        "reason": "Filtering is required because the dataset contains jobs with different locations, experience levels, and company constraints.",
        "important_rules": ["location", "experience", "company exclusion", "seniority title filtering"]
    }


def local_ranking_decision():
    return {
        "next_tool": "ranking",
        "reason": "The filtered job set is now relevant and should be ranked based on skills, experience, location, and title fit.",
        "ranking_focus": ["skills", "experience", "location", "title relevance"]
    }


def local_top_job_justification(top_job):
    return {
        "next_tool": "resume_tailoring",
        "reason": f"The top-ranked job is {top_job.get('job_title', 'the selected role')} because it has the strongest overall score.",
        "why_top_job_wins": [
            "It achieved the highest total ranking score.",
            "It shows strong skill alignment with the candidate profile.",
            "It is a better overall fit than the remaining top jobs."
        ]
    }


def call_ollama_json(system_prompt, user_prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=120
    )
    response.raise_for_status()

    data = response.json()
    text = data["message"]["content"]

    parsed = safe_json_load(text)
    if parsed is not None:
        return parsed

    # try extracting JSON block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        maybe_json = text[start:end + 1]
        parsed = safe_json_load(maybe_json)
        if parsed is not None:
            return parsed

    raise ValueError("Model did not return valid JSON.")


def decide_filtering(candidate_profile, jobs_df):
    dataset_summary = jobs_df.head(8)[["job_title", "company", "location", "years_experience"]].to_dict(orient="records")
    prompt = build_filtering_decision_prompt(candidate_profile, json.dumps(dataset_summary, indent=2))

    try:
        return call_ollama_json(SYSTEM_PROMPT, prompt), "ollama"
    except Exception as e:
        return local_filtering_decision(), f"fallback ({e})"


def decide_ranking(candidate_profile, filtering_stats, filtered_jobs_df):
    filtered_summary = filtered_jobs_df.head(10)[["job_title", "company", "location", "years_experience"]].to_dict(orient="records")
    prompt = build_ranking_decision_prompt(
        candidate_profile,
        filtering_stats,
        json.dumps(filtered_summary, indent=2)
    )

    try:
        return call_ollama_json(SYSTEM_PROMPT, prompt), "ollama"
    except Exception as e:
        return local_ranking_decision(), f"fallback ({e})"


def justify_top_job(candidate_profile, top_3_df):
    top_3_summary = top_3_df[
        ["job_title", "company", "location", "total_score", "matched_skills", "title_score", "description_score"]
    ].to_dict(orient="records")

    prompt = build_top_job_justification_prompt(
        candidate_profile,
        json.dumps(top_3_summary, indent=2)
    )

    try:
        return call_ollama_json(SYSTEM_PROMPT, prompt), "ollama"
    except Exception as e:
        top_job = top_3_df.iloc[0].to_dict() if not top_3_df.empty else {}
        return local_top_job_justification(top_job), f"fallback ({e})"


def llm_tailor_resume(candidate_profile, base_resume, top_job):
    prompt = build_resume_tailoring_prompt(
        candidate_profile,
        base_resume,
        top_job
    )

    try:
        result = call_ollama_json(TAILORING_SYSTEM_PROMPT, prompt)
        if (
            "professional_summary" in result and
            "modified_experience_bullets" in result and
            len(result["modified_experience_bullets"]) >= 2 and
            "aligned_skills" in result
        ):
            return result, "ollama"
    except Exception as e:
        return tailor_resume(base_resume, top_job), f"fallback ({e})"

    return tailor_resume(base_resume, top_job), "fallback"


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_output_paths():
    os.makedirs("output", exist_ok=True)
    ts = get_timestamp()
    return (
        f"output/agent_output_{ts}.txt",
        f"output/agent_output_{ts}.pdf"
    )


def save_text_file(content, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def wrap_line_by_width(c, text, max_width, font_name="Helvetica", font_size=10):
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]

    for word in words[1:]:
        test = current + " " + word
        if c.stringWidth(test, font_name, font_size) <= max_width:
            current = test
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines


def save_pdf(content, path):
    c = canvas.Canvas(path, pagesize=LETTER)
    width, height = LETTER

    left_margin = 50
    right_margin = 50
    top_margin = 50
    bottom_margin = 50
    line_height = 14
    max_width = width - left_margin - right_margin

    y = height - top_margin
    c.setFont("Helvetica", 10)
    page_num = 1

    def new_page():
        nonlocal y, page_num
        c.setFont("Helvetica", 9)
        c.drawRightString(width - 50, 30, f"Page {page_num}")
        c.showPage()
        page_num += 1
        c.setFont("Helvetica", 10)
        y = height - top_margin

    for raw_line in content.split("\n"):
        if raw_line.startswith("STEP ") or "========== " in raw_line or raw_line.endswith(":"):
            c.setFont("Helvetica-Bold", 10)
        else:
            c.setFont("Helvetica", 10)

        wrapped_lines = wrap_line_by_width(c, raw_line, max_width, font_name=c._fontname, font_size=10)

        for line in wrapped_lines:
            if y <= bottom_margin:
                new_page()
            c.drawString(left_margin, y, line)
            y -= line_height

        y -= 4  # extra spacing between blocks

    c.setFont("Helvetica", 9)
    c.drawRightString(width - 50, 30, f"Page {page_num}")
    c.save()


def save_all_outputs(content):
    txt_path, pdf_path = get_output_paths()
    save_text_file(content, txt_path)
    save_pdf(content, pdf_path)
    return txt_path, pdf_path


def append_line(output_lines, text=""):
    print(text)
    output_lines.append(text)


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
        f"Score: {job['total_score']} | Matched Skills: {job['matched_skills']}"
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
        f"   Title Score: {job['title_score']}\n"
        f"   Description Score: {job['description_score']}\n"
        f"   Seniority Penalty: {job['seniority_penalty']}\n"
        f"   Total Score: {job['total_score']}\n"
        f"   URL: {job['url']}\n"
    )


def validate_resume(base_resume):
    if "skills" not in base_resume or not isinstance(base_resume["skills"], list):
        raise ValueError("Base resume must contain a skills list.")
    if "experience_bullets" not in base_resume or len(base_resume["experience_bullets"]) < 2:
        raise ValueError("Base resume must contain at least 2 experience bullet points.")


def main():
    output_lines = []

    try:
        validate_resume(BASE_RESUME)

        append_line(output_lines, "\n========== AI AGENT FOR JOB SEARCH & RESUME OPTIMIZATION ==========\n")

        # STEP 1
        jobs_df = load_jobs("data/jobs.csv")
        append_line(output_lines, "STEP 1: Dataset Preview")
        append_line(output_lines, f"Total jobs loaded: {len(jobs_df)}")
        append_line(output_lines, "\nShowing first 5 jobs:")
        for i, job in enumerate(jobs_df.head(5).to_dict(orient="records"), start=1):
            append_line(output_lines, format_compact_job(job, i))

        # STEP 2 - LLM decides filtering
        append_line(output_lines, "\nSTEP 2: LLM Decision for Filtering")
        filtering_decision, filtering_mode = decide_filtering(CANDIDATE_PROFILE, jobs_df)
        append_line(output_lines, json.dumps(filtering_decision, indent=2))
        append_line(output_lines, f"Decision mode used: {filtering_mode}")

        if filtering_decision.get("next_tool") != "filtering":
            raise ValueError("LLM did not select filtering as the next tool.")

        # STEP 3 - run filtering
        append_line(output_lines, "\nSTEP 3: Filtering Tool Output")
        filtered_jobs, filtering_stats = filter_jobs(
            jobs_df,
            preferred_location=CANDIDATE_PROFILE["preferred_location"],
            years_experience=CANDIDATE_PROFILE["years_experience"],
            exclude_companies=CANDIDATE_PROFILE["exclude_companies"],
            remote_only=CANDIDATE_PROFILE["remote_only"]
        )

        append_line(output_lines, "Filtering Stats:")
        append_line(output_lines, json.dumps(filtering_stats, indent=2))

        if filtered_jobs.empty:
            append_line(output_lines, "No jobs matched filtering criteria.")
            final_output = "\n".join(output_lines)
            txt_path, pdf_path = save_all_outputs(final_output)
            print(f"Saved text output to: {txt_path}")
            print(f"Saved PDF output to: {pdf_path}")
            return

        append_line(output_lines, "\nFiltered jobs:")
        for i, job in enumerate(filtered_jobs.to_dict(orient="records"), start=1):
            append_line(output_lines, format_compact_job(job, i))

        # STEP 4 - LLM decides ranking
        append_line(output_lines, "\nSTEP 4: LLM Decision for Ranking")
        ranking_decision, ranking_mode = decide_ranking(CANDIDATE_PROFILE, filtering_stats, filtered_jobs)
        append_line(output_lines, json.dumps(ranking_decision, indent=2))
        append_line(output_lines, f"Decision mode used: {ranking_mode}")

        if ranking_decision.get("next_tool") != "ranking":
            raise ValueError("LLM did not select ranking as the next tool.")

        # STEP 5 - run ranking
        append_line(output_lines, "\nSTEP 5: Ranking Tool Output")
        ranked_jobs = rank_jobs(
            filtered_jobs,
            candidate_skills=CANDIDATE_PROFILE["skills"],
            candidate_experience=CANDIDATE_PROFILE["years_experience"],
            preferred_location=CANDIDATE_PROFILE["preferred_location"]
        )

        if ranked_jobs.empty:
            raise ValueError("Ranking produced no rows.")

        append_line(output_lines, "Compact ranking list:")
        for i, job in enumerate(ranked_jobs.head(10).to_dict(orient="records"), start=1):
            append_line(output_lines, format_compact_rank(job, i))

        # STEP 6 - top 3
        append_line(output_lines, "\nSTEP 6: Top 3 Jobs")
        top_3 = ranked_jobs.head(3)

        for i, job in enumerate(top_3.to_dict(orient="records"), start=1):
            append_line(output_lines, format_ranked_job_block(job, i))

        # STEP 7 - justify best job
        append_line(output_lines, "\nSTEP 7: LLM Justification for Top Job")
        top_job_justification, justification_mode = justify_top_job(CANDIDATE_PROFILE, top_3)
        append_line(output_lines, json.dumps(top_job_justification, indent=2))
        append_line(output_lines, f"Decision mode used: {justification_mode}")

        if top_job_justification.get("next_tool") != "resume_tailoring":
            raise ValueError("LLM did not select resume_tailoring as the next tool.")

        # STEP 8 - best job selected
        top_job = ranked_jobs.iloc[0].to_dict()
        append_line(output_lines, "\nSTEP 8: Best Job Selected")
        append_line(output_lines, format_ranked_job_block(top_job, 1))

        # STEP 9 - resume tailoring
        append_line(output_lines, "\nSTEP 9: Resume Tailoring Tool Output")
        tailored, tailoring_mode = llm_tailor_resume(CANDIDATE_PROFILE, BASE_RESUME, top_job)

        append_line(output_lines, f"Tailoring mode used: {tailoring_mode}")
        append_line(output_lines, "\nTailored Professional Summary:")
        append_line(output_lines, tailored["professional_summary"])

        append_line(output_lines, "\nModified Experience Bullet 1:")
        append_line(output_lines, f"- {tailored['modified_experience_bullets'][0]}")

        append_line(output_lines, "\nModified Experience Bullet 2:")
        append_line(output_lines, f"- {tailored['modified_experience_bullets'][1]}")

        append_line(output_lines, "\nAligned Skills:")
        append_line(output_lines, ", ".join(tailored["aligned_skills"]))

        if "top_keywords_used" in tailored:
            append_line(output_lines, "\nTop Keywords Used:")
            append_line(output_lines, ", ".join(tailored["top_keywords_used"]))

        # STEP 10 - final reasoning trace
        append_line(output_lines, "\nSTEP 10: Final Reasoning Trace")
        append_line(
            output_lines,
            f"The agent loaded {len(jobs_df)} jobs, filtered them using location, experience, company exclusion, "
            f"and seniority-title rules, ranked the remaining roles using skill match, experience alignment, "
            f"location fit, title relevance, and description keyword relevance, selected the highest-scoring job, "
            f"and tailored the resume only for that top-ranked role."
        )

        append_line(output_lines, "\n========== END ==========\n")

        final_output = "\n".join(output_lines)
        txt_path, pdf_path = save_all_outputs(final_output)

        print(f"Saved text output to: {txt_path}")
        print(f"Saved PDF output to: {pdf_path}")

    except Exception as e:
        append_line(output_lines, "\n[ERROR] Pipeline failed.")
        append_line(output_lines, f"Reason: {e}")

        final_output = "\n".join(output_lines)
        txt_path, pdf_path = save_all_outputs(final_output)

        print(f"Saved text output to: {txt_path}")
        print(f"Saved PDF output to: {pdf_path}")


if __name__ == "__main__":
    main()