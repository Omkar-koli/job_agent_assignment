import json


SYSTEM_PROMPT = """
You are a single LLM-based AI job-search agent.

You must reason step by step and decide which tool should be used next.
You must not invent data.
You must only use the information provided to you.
Return concise, structured outputs.
"""

TAILORING_SYSTEM_PROMPT = """
You are an AI resume tailoring assistant.
Rewrite the resume only for the selected top-ranked job.
You must:
1. Rewrite the professional summary
2. Modify exactly 2 experience bullet points
3. Highlight aligned skills
Do not regenerate the full resume.
Do not invent fake experience or false claims.
Return valid JSON only.
"""


def build_filtering_decision_prompt(candidate_profile, dataset_summary):
    return f"""
Candidate Profile:
{json.dumps(candidate_profile, indent=2)}

Dataset Summary:
{dataset_summary}

Decide whether the Filtering Tool should be used first.

Return valid JSON only in this format:
{{
  "next_tool": "filtering",
  "reason": "short explanation",
  "important_rules": ["location", "experience", "company exclusion", "seniority title filtering"]
}}
"""


def build_ranking_decision_prompt(candidate_profile, filtering_stats, filtered_jobs_summary):
    return f"""
Candidate Profile:
{json.dumps(candidate_profile, indent=2)}

Filtering Stats:
{json.dumps(filtering_stats, indent=2)}

Filtered Jobs Summary:
{filtered_jobs_summary}

Decide whether the Ranking Tool should be used next.

Return valid JSON only in this format:
{{
  "next_tool": "ranking",
  "reason": "short explanation",
  "ranking_focus": ["skills", "experience", "location", "title relevance"]
}}
"""


def build_top_job_justification_prompt(candidate_profile, top_3_jobs_summary):
    return f"""
Candidate Profile:
{json.dumps(candidate_profile, indent=2)}

Top 3 Ranked Jobs:
{top_3_jobs_summary}

Decide whether Resume Tailoring Tool should be called for the top-ranked job and justify why rank 1 is better than rank 2 and rank 3.

Return valid JSON only in this format:
{{
  "next_tool": "resume_tailoring",
  "reason": "short explanation",
  "why_top_job_wins": [
    "reason 1",
    "reason 2",
    "reason 3"
  ]
}}
"""


def build_resume_tailoring_prompt(candidate_profile, base_resume, top_job):
    return f"""
Candidate Profile:
{json.dumps(candidate_profile, indent=2)}

Base Resume:
{json.dumps(base_resume, indent=2)}

Selected Top Job:
{json.dumps(top_job, indent=2)}

Rewrite the professional summary and modify exactly 2 experience bullet points to better match the selected job.
Also return aligned skills.

Return valid JSON only in this format:
{{
  "professional_summary": "rewritten summary",
  "modified_experience_bullets": [
    "bullet 1",
    "bullet 2"
  ],
  "aligned_skills": ["skill1", "skill2", "skill3"]
}}
"""