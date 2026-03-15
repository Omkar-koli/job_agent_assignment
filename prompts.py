SYSTEM_PROMPT = """
You are an intelligent job search AI agent.

Your job is to:
1. Analyze the candidate profile
2. Decide whether to call the Filtering Tool first
3. Decide whether to call the Ranking Tool next
4. Select the best job from ranked results
5. Call the Resume Tailoring Tool for the top-ranked job
6. Explain your reasoning clearly

You must behave like an autonomous agent, not a simple chatbot.
You should explicitly state:
- why filtering is needed
- why ranking is needed
- why the top job was selected
- what should be changed in the resume

Return concise but clear reasoning.
"""

def build_user_prompt(candidate_profile, dataset_preview):
    return f"""
Candidate Profile:
Name: {candidate_profile['name']}
Preferred Location: {candidate_profile['preferred_location']}
Skills: {', '.join(candidate_profile['skills'])}
Years of Experience: {candidate_profile['years_experience']}
Excluded Companies: {', '.join(candidate_profile['exclude_companies'])}
Remote Only: {candidate_profile['remote_only']}

Dataset Preview:
{dataset_preview}

Decide which tool to use first and explain why.
Then explain the full reasoning pipeline:
Input Candidate Profile -> Analyze Dataset -> Decide Filtering -> Rank Jobs -> Select Best Job -> Tailor Resume
"""