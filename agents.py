import logging
import re
import json
from dotenv import load_dotenv
import os
import time
import google.generativeai as genai
from typing import Dict, List, Tuple
from collections import deque
from fuzzywuzzy import fuzz

# Setup logging
logging.basicConfig(level=logging.DEBUG, filename='logs.txt', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
try:
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    llm = genai.GenerativeModel("gemini-2.5-flash")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    raise

# Rate limiter for LLM calls (10 RPM = 6 seconds per request)
class RateLimiter:
    def __init__(self, requests_per_minute: int = 10, period: int = 60):
        self.requests_per_minute = requests_per_minute
        self.period = period
        self.requests = deque(maxlen=requests_per_minute)
        logger.info(f"Initialized RateLimiter with {requests_per_minute} requests per {period} seconds")

    def acquire(self):
        current_time = time.time()
        while self.requests and current_time - self.requests[0] > self.period:
            self.requests.popleft()
        if len(self.requests) >= self.requests_per_minute:
            sleep_time = self.period - (current_time - self.requests[0])
            logger.debug(f"Rate limit reached, waiting {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            current_time = time.time()
            while self.requests and current_time - self.requests[0] > self.period:
                self.requests.popleft()
        self.requests.append(current_time)
        logger.debug(f"LLM call allowed, {len(self.requests)}/{self.requests_per_minute} requests in queue")

# Initialize rate limiter
rate_limiter = RateLimiter(requests_per_minute=10, period=60)

# Define skill aliases for matching
SKILL_ALIASES = {
    'react': ['react', 'react.js', 'reactjs'],
    'python': ['python', 'python3', 'python 3'],
    'javascript': ['javascript', 'js', 'ecmascript'],
    'r': ['r', 'r programming'],
    'sql': ['sql', 'structured query language'],
    'tensorflow': ['tensorflow', 'tf'],
    'pytorch': ['pytorch', 'torch'],
    'scikit-learn': ['scikit-learn', 'sklearn'],
    'pandas': ['pandas'],
    'numpy': ['numpy'],
    'matplotlib': ['matplotlib'],
    'nlp': ['nlp', 'natural language processing'],
    'computer vision': ['computer vision', 'cv'],
    'aws': ['aws', 'amazon web services'],
    'azure': ['azure', 'microsoft azure'],
    'gcp': ['gcp', 'google cloud platform', 'google cloud'],
    'docker': ['docker'],
    'kubernetes': ['kubernetes', 'k8s'],
    'git': ['git'],
    'deep learning': ['deep learning', 'dl', 'neural networks', 'ann', 'cnn', 'rnn', 'lstm'],
    'mlops': ['mlops', 'machine learning operations'],
    'apache airflow': ['apache airflow', 'airflow'],
    'apache kafka': ['apache kafka', 'kafka'],
    'reinforcement learning': ['reinforcement learning', 'rl'],
    'model interpretability': ['model interpretability', 'interpretability'],
    'model fairness': ['model fairness', 'fairness'],
    'gpt': ['gpt', 'generative pre-trained transformer'],
    'bert': ['bert'],
    'dall-e': ['dall-e', 'dalle']
}

def normalize_skill(skill: str) -> str:
    skill = skill.lower().strip()
    for canonical, aliases in SKILL_ALIASES.items():
        if skill in aliases:
            return canonical
    return skill

def extract_projects_from_resume(resume_text: str) -> List[Dict]:
    try:
        rate_limiter.acquire()
        prompt = f"""
        **TASK:** Extract ALL projects from the resume text with 100% accuracy.
        
        **RESUME TEXT:**
        {resume_text}
        
        **EXTRACTION REQUIREMENTS:**
        1. Find ALL project sections (may be labeled as: Projects, Academic Projects, Personal Projects, Work Projects, Portfolio, etc.)
        2. For each project, extract:
           - Project Name (if not explicitly named, create a descriptive name)
           - Description (complete project description)
           - Skills/Technologies used (extract ALL technical skills, tools, frameworks, languages, etc.)
        
        3. **SKILL EXTRACTION RULES:**
           - Extract programming languages (e.g., Python, Java, JavaScript, R)
           - Extract frameworks (e.g., React, Django, Flask, TensorFlow, PyTorch)
           - Extract databases (e.g., MySQL, MongoDB, PostgreSQL)
           - Extract cloud platforms (e.g., AWS, Azure, GCP)
           - Extract tools (e.g., Git, Docker, Kubernetes)
           - Extract libraries (e.g., NumPy, Pandas, Scikit-learn)
           - Extract methodologies (e.g., MLOps, Agile)
           - Include version numbers if mentioned (e.g., "Python 3.8")
           - For single-character skills like "R", ensure they refer to the programming language
           - Exclude non-technical terms (e.g., "teamwork", "research")
           - Include AI-specific terms (e.g., NLP, computer vision, deep learning, MLOps)
        
        **OUTPUT FORMAT - Return ONLY valid JSON:**
        {{
            "projects": [
                {{
                    "name": "<project name>",
                    "description": "<full project description>",
                    "skills": ["<skill1>", "<skill2>", "<skill3>", ...]
                }},
                ...
            ]
        }}
        
        **IMPORTANT:**
        - If no projects found, return empty projects array
        - Extract skills exactly as mentioned
        - Include contextual AI skills (e.g., deep learning from "CNN", "LSTM")
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = llm.generate_content(prompt)
                output = response.text.strip()
                if output.startswith("```json"):
                    output = output[len("```json"):].rstrip("```").strip()
                elif output.startswith("```"):
                    output = output[len("```"):].rstrip("```").strip()
                result = json.loads(output)
                projects = result.get("projects", [])
                for project in projects:
                    project['skills'] = [normalize_skill(skill) for skill in project.get('skills', [])]
                logger.info(f"Extracted {len(projects)} projects from resume")
                return projects
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Rate limit error on attempt {attempt + 1}, retrying...")
                    time.sleep(6 * (attempt + 1))
                    continue
                logger.error(f"Failed to extract projects using LLM: {e}")
                raise
    except Exception as e:
        logger.error(f"Failed to extract projects after retries: {e}")
        return extract_projects_fallback(resume_text)

def extract_projects_fallback(resume_text: str) -> List[Dict]:
    projects = []
    project_patterns = [
        r'(?i)(?:^|\n)\s*(?:projects?|academic\s+projects?|personal\s+projects?|work\s+projects?|portfolio)\s*[:\-]?\s*\n',
        r'(?i)(?:^|\n)\s*(?:key\s+projects?|major\s+projects?|relevant\s+projects?)\s*[:\-]?\s*\n'
    ]
    for pattern in project_patterns:
        matches = list(re.finditer(pattern, resume_text))
        if matches:
            for match in matches:
                start_pos = match.end()
                next_section = re.search(r'(?i)\n\s*(?:experience|education|skills|certifications?|achievements?|awards?)\s*[:\-]?\s*\n', resume_text[start_pos:])
                end_pos = start_pos + next_section.start() if next_section else len(resume_text)
                projects_text = resume_text[start_pos:end_pos]
                projects.extend(parse_projects_text(projects_text))
    for project in projects:
        project['skills'] = [normalize_skill(skill) for skill in project.get('skills', [])]
    if not projects:
        logger.warning("No projects found using fallback extraction")
    return projects

def parse_projects_text(projects_text: str) -> List[Dict]:
    projects = []
    project_blocks = re.split(r'(?:\n\s*[•▪▫\-\*]\s*|\n\s*\d+\.\s*)', projects_text)
    for block in project_blocks:
        if len(block.strip()) < 20:
            continue
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if not lines:
            continue
        project_name = lines[0]
        description = ' '.join(lines[1:]) if len(lines) > 1 else lines[0]
        skills = extract_skills_from_text(description)
        if skills:
            projects.append({
                "name": project_name,
                "description": description,
                "skills": skills
            })
    return projects

def extract_skills_from_text(text: str) -> List[str]:
    skills = []
    skill_patterns = {
        'languages': r'\b(?:Python|Java|JavaScript|TypeScript|C\+\+|C#|PHP|Ruby|Go|Rust|Swift|Kotlin|Scala|R(?!\w)|MATLAB|SQL)\b',
        'frameworks': r'\b(?:React|Angular|Vue|Django|Flask|Spring|Express|Laravel|Rails|ASP\.NET|Bootstrap|Tailwind)\b',
        'databases': r'\b(?:MySQL|PostgreSQL|MongoDB|Redis|Cassandra|DynamoDB|SQLite|Oracle|SQL Server)\b',
        'cloud': r'\b(?:AWS|Azure|GCP|Google Cloud|Heroku|DigitalOcean|Docker|Kubernetes)\b',
        'tools': r'\b(?:Git|Jenkins|Travis|CircleCI|Jira|Slack|Postman|Swagger|Webpack|Babel)\b',
        'libraries': r'\b(?:TensorFlow|PyTorch|NumPy|Pandas|Scikit-learn|OpenCV|React Native|jQuery|Lodash)\b',
        'ai': r'\b(?:NLP|Natural Language Processing|Computer Vision|Deep Learning|MLOps|Reinforcement Learning|Model Interpretability|Model Fairness|GPT|BERT|DALL-E)\b'
    }
    for category, pattern in skill_patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        skills.extend(matches)
    skills = list(dict.fromkeys(skills))
    known_technical_skills = set(SKILL_ALIASES.keys())
    validated_skills = [normalize_skill(skill) for skill in skills if normalize_skill(skill) in known_technical_skills]
    logger.debug(f"Extracted skills: {validated_skills}")
    return validated_skills

def extract_nice_to_have_skills_from_jd(job_description: str) -> List[str]:
    try:
        rate_limiter.acquire()
        prompt = f"""
        **TASK:** Extract ONLY nice-to-have technical skills from the job description.
        
        **JOB DESCRIPTION:**
        {job_description}
        
        **EXTRACTION RULES:**
        1. Look for sections with keywords: "nice to have", "preferred", "desirable", "optional", "bonus", "good to have"
        2. Extract ONLY technical skills (programming languages, frameworks, tools, databases, AI terms, etc.)
        3. Include contextual AI skills (e.g., deep learning, MLOps, reinforcement learning)
        4. Ignore mandatory/required skills, soft skills, experience years, or degrees
        5. Extract skills from parentheses (e.g., "Familiarity with (TensorFlow, PyTorch)" → "TensorFlow", "PyTorch")
        6. Split compound requirements (e.g., "Python or R" → "Python", "R")
        7. For single-character skills like "R", ensure they refer to the programming language
        
        **OUTPUT FORMAT - Return ONLY valid JSON:**
        {{
            "nice_to_have_skills": ["<skill1>", "<skill2>", "<skill3>", ...]
        }}
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = llm.generate_content(prompt)
                output = response.text.strip()
                if output.startswith("```json"):
                    output = output[len("```json"):].rstrip("```").strip()
                elif output.startswith("```"):
                    output = output[len("```"):].rstrip("```").strip()
                result = json.loads(output)
                skills = result.get("nice_to_have_skills", [])
                normalized_skills = [normalize_skill(skill) for skill in skills]
                logger.info(f"Extracted {len(normalized_skills)} nice-to-have skills from JD: {normalized_skills}")
                return normalized_skills
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Rate limit error on attempt {attempt + 1}, retrying...")
                    time.sleep(6 * (attempt + 1))
                    continue
                logger.error(f"Failed to extract nice-to-have skills using LLM: {e}")
                raise
    except Exception as e:
        logger.error(f"Failed to extract nice-to-have skills after retries: {e}")
        return extract_nice_to_have_skills_fallback(job_description)

def extract_nice_to_have_skills_fallback(job_description: str) -> List[str]:
    nice_to_have_skills = []
    patterns = [
        r'(?:nice to have|preferred|desirable|optional|bonus|good to have)\s*[:\s]*(.*?)(?=\n[A-Z][a-zA-Z\s]+:|\n\n|\Z)',
        r'(?:preferred qualifications?|additional skills?)\s*[:\s]*(.*?)(?=\n[A-Z][a-zA-Z\s]+:|\n\n|\Z)'
    ]
    for pattern in patterns:
        matches = re.findall(pattern, job_description, re.IGNORECASE | re.DOTALL)
        for match in matches:
            lines = [line.strip() for line in match.split('\n') if line.strip()]
            for line in lines:
                line = re.sub(r'^(?:Familiarity with|Knowledge of|Experience with)\s+', '', line, flags=re.IGNORECASE)
                paren_skills = re.findall(r'\((.*?)\)', line)
                for skill_group in paren_skills:
                    skills = [s.strip() for s in skill_group.split(',') if s.strip()]
                    nice_to_have_skills.extend(skills)
                line = re.sub(r'\s*\(.*?\)', '', line)
                skills = re.split(r',\s*|\s+or\s+', line, flags=re.IGNORECASE)
                skills = [skill.strip() for skill in skills if skill.strip() and len(skill) > 1]
                nice_to_have_skills.extend(skills)
    known_technical_skills = set(SKILL_ALIASES.keys())
    technical_skills = [normalize_skill(skill) for skill in nice_to_have_skills if normalize_skill(skill) in known_technical_skills]
    return list(set(technical_skills))

def extract_soft_skills(resume_text: str) -> List[str]:
    try:
        rate_limiter.acquire()
        prompt = f"""
        **TASK:** Extract ONLY soft skills from the resume text.
        
        **RESUME TEXT:**
        {resume_text[:2000]}
        
        **EXTRACTION RULES:**
        1. Extract ONLY soft skills (e.g., communication, teamwork, problem-solving, analytical thinking)
        2. Ignore technical skills, experience, projects, or certifications
        3. Look for explicit mentions in sections like "Skills", "Profile", or project descriptions
        
        **OUTPUT FORMAT - Return ONLY valid JSON:**
        {{
            "soft_skills": ["<skill1>", "<skill2>", "<skill3>", ...]
        }}
        """
        response = llm.generate_content(prompt)
        output = response.text.strip()
        if output.startswith("```json"):
            output = output[len("```json"):].rstrip("```").strip()
        elif output.startswith("```"):
            output = output[len("```"):].rstrip("```").strip()
        result = json.loads(output)
        soft_skills = result.get("soft_skills", [])
        logger.info(f"Extracted {len(soft_skills)} soft skills: {soft_skills}")
        return soft_skills
    except Exception as e:
        logger.error(f"Failed to extract soft skills: {e}")
        return []

def extract_certifications(resume_text: str) -> List[str]:
    try:
        rate_limiter.acquire()
        prompt = f"""
        **TASK:** Extract ALL certifications from the resume text.
        
        **RESUME TEXT:**
        {resume_text[:2000]}
        
        **EXTRACTION RULES:**
        1. Look for sections labeled "Certifications", "Courses", or similar
        2. Extract certification names exactly as mentioned
        3. Include provider if mentioned (e.g., "AWS Certified Solutions Architect - AWS")
        
        **OUTPUT FORMAT - Return ONLY valid JSON:**
        {{
            "certifications": ["<cert1>", "<cert2>", "<cert3>", ...]
        }}
        """
        response = llm.generate_content(prompt)
        output = response.text.strip()
        if output.startswith("```json"):
            output = output[len("```json"):].rstrip("```").strip()
        elif output.startswith("```"):
            output = output[len("```"):].rstrip("```").strip()
        result = json.loads(output)
        certifications = result.get("certifications", [])
        logger.info(f"Extracted {len(certifications)} certifications: {certifications}")
        return certifications
    except Exception as e:
        logger.error(f"Failed to extract certifications: {e}")
        return []

def calculate_proficiency_score(resume_text: str, job_description: str, projects: List[Dict]) -> int:
    try:
        # Nice-to-have skills (8 points)
        nice_to_have_skills = extract_nice_to_have_skills_from_jd(job_description)
        matched_nice_skills = []
        for project in projects:
            for skill in project.get("skills", []):
                for nice_skill in nice_to_have_skills:
                    if fuzz.ratio(normalize_skill(nice_skill), normalize_skill(skill)) > 80:
                        matched_nice_skills.append(nice_skill)
        matched_nice_skills = list(set(matched_nice_skills))
        nice_skill_score = min(8, int((len(matched_nice_skills) / max(len(nice_to_have_skills), 1)) * 8)) if nice_to_have_skills else 0
        logger.debug(f"Nice-to-have skills score: {nice_skill_score}, Matched: {matched_nice_skills}")

        # Resume quality (8 points)
        rate_limiter.acquire()
        resume_quality_prompt = f"""
        **TASK:** Evaluate the quality of the resume's project section based on clarity, relevance, and complexity.
        
        **RESUME PROJECTS:**
        {json.dumps(projects, indent=2)[:2000]}
        
        **JOB DESCRIPTION:**
        {job_description[:2000]}
        
        **EVALUATION CRITERIA:**
        1. Clarity: Are project descriptions clear and detailed? (0-3 points)
        2. Relevance: Do projects align with JD's technical requirements? (0-3 points)
        3. Complexity: Do projects demonstrate advanced technical skills? (0-2 points)
        
        **OUTPUT FORMAT - Return ONLY valid JSON:**
        {{
            "clarity": <score>,
            "relevance": <score>,
            "complexity": <score>,
            "total": <sum of scores>
        }}
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = llm.generate_content(resume_quality_prompt)
                output = response.text.strip()
                if output.startswith("```json"):
                    output = output[len("```json"):].rstrip("```").strip()
                elif output.startswith("```"):
                    output = output[len("```"):].rstrip("```").strip()
                quality_result = json.loads(output)
                resume_quality_score = min(8, quality_result.get("total", 0))
                logger.debug(f"Resume quality score: {resume_quality_score}, Details: {quality_result}")
                break
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Rate limit error on attempt {attempt + 1}, retrying...")
                    time.sleep(6 * (attempt + 1))
                    continue
                logger.error(f"Failed to calculate resume quality score: {e}")
                raise
        else:
            resume_quality_score = 0

        # Experience alignment (8 points)
        rate_limiter.acquire()
        experience_prompt = f"""
        **TASK:** Evaluate how well the candidate's experience aligns with the job description.
        
        **RESUME TEXT:**
        {resume_text[:2000]}
        
        **JOB DESCRIPTION:**
        {job_description[:2000]}
        
        **EVALUATION CRITERIA:**
        1. Years of experience: Match years to JD requirements (0-3 points)
        2. Role relevance: Do past roles align with JD's responsibilities? (0-3 points)
        3. Industry fit: Does experience match JD's industry/domain? (0-2 points)
        
        **OUTPUT FORMAT - Return ONLY valid JSON:**
        {{
            "years": <score>,
            "role_relevance": <score>,
            "industry_fit": <score>,
            "total": <sum of scores>
        }}
        """
        for attempt in range(max_retries):
            try:
                response = llm.generate_content(experience_prompt)
                output = response.text.strip()
                if output.startswith("```json"):
                    output = output[len("```json"):].rstrip("```").strip()
                elif output.startswith("```"):
                    output = output[len("```"):].rstrip("```").strip()
                experience_result = json.loads(output)
                experience_score = min(8, experience_result.get("total", 0))
                logger.debug(f"Experience alignment score: {experience_score}, Details: {experience_result}")
                break
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Rate limit error on attempt {attempt + 1}, retrying...")
                    time.sleep(6 * (attempt + 1))
                    continue
                logger.error(f"Failed to calculate experience alignment score: {e}")
                raise
        else:
            experience_score = 0

        # Soft skills (3 points)
        soft_skills = extract_soft_skills(resume_text)
        jd_soft_skills = ['problem-solving', 'communication', 'collaboration', 'analytical thinking']
        matched_soft_skills = [skill for skill in soft_skills if skill.lower() in [s.lower() for s in jd_soft_skills]]
        soft_skill_score = min(3, int((len(matched_soft_skills) / max(len(jd_soft_skills), 1)) * 3))
        logger.debug(f"Soft skills score: {soft_skill_score}, Matched: {matched_soft_skills}")

        # Certifications (3 points)
        certifications = extract_certifications(resume_text)
        relevant_certs = [cert for cert in certifications if any(keyword in cert.lower() for keyword in ['ai', 'data science', 'machine learning', 'cloud', 'aws', 'azure', 'gcp'])]
        cert_score = min(3, int((len(relevant_certs) / 2) * 3))  # Up to 2 relevant certs for full points
        logger.debug(f"Certifications score: {cert_score}, Relevant: {relevant_certs}")

        total_proficiency_score = nice_skill_score + resume_quality_score + experience_score + soft_skill_score + cert_score
        logger.info(f"Proficiency score: {total_proficiency_score}/30 (Nice-to-have: {nice_skill_score}, Quality: {resume_quality_score}, Experience: {experience_score}, Soft Skills: {soft_skill_score}, Certifications: {cert_score})")
        return total_proficiency_score
    except Exception as e:
        logger.error(f"Failed to calculate proficiency score: {e}")
        return 0

def extract_mandatory_skills_from_jd(job_description: str) -> List[str]:
    try:
        rate_limiter.acquire()
        prompt = f"""
        **TASK:** Extract ONLY mandatory/required technical skills from the job description.
        
        **JOB DESCRIPTION:**
        {job_description}
        
        **EXTRACTION RULES:**
        1. Look for sections with keywords: "required", "must have", "essential", "mandatory", "must-have"
        2. Extract ONLY technical skills (programming languages, frameworks, tools, databases, AI terms, etc.)
        3. Ignore soft skills, experience years, or degrees
        4. Extract skills from parentheses (e.g., "AI frameworks (TensorFlow, PyTorch)" → "TensorFlow", "PyTorch")
        5. Split compound requirements (e.g., "Python and R" → "Python", "R")
        6. For single-character skills like "R", ensure they refer to the programming language
        
        **OUTPUT FORMAT - Return ONLY valid JSON:**
        {{
            "mandatory_skills": ["<skill1>", "<skill2>", "<skill3>", ...]
        }}
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = llm.generate_content(prompt)
                output = response.text.strip()
                if output.startswith("```json"):
                    output = output[len("```json"):].rstrip("```").strip()
                elif output.startswith("```"):
                    output = output[len("```"):].rstrip("```").strip()
                result = json.loads(output)
                skills = result.get("mandatory_skills", [])
                normalized_skills = [normalize_skill(skill) for skill in skills]
                logger.info(f"Extracted {len(normalized_skills)} mandatory skills from JD: {normalized_skills}")
                return normalized_skills
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Rate limit error on attempt {attempt + 1}, retrying...")
                    time.sleep(6 * (attempt + 1))
                    continue
                logger.error(f"Failed to extract mandatory skills using LLM: {e}")
                raise
    except Exception as e:
        logger.error(f"Failed to extract mandatory skills after retries: {e}")
        return extract_mandatory_skills_fallback(job_description)

def extract_mandatory_skills_fallback(job_description: str) -> List[str]:
    mandatory_skills = []
    mandatory_patterns = [
        r'(?:required|must have|essential|mandatory|must-have)\s*[:\s]*(.*?)(?=\n[A-Z][a-zA-Z\s]+:|\n\n|\Z)',
        r'(?:requirements?|qualifications?)\s*[:\s]*(.*?)(?=\n[A-Z][a-zA-Z\s]+:|\n\n|\Z)'
    ]
    for pattern in mandatory_patterns:
        matches = re.findall(pattern, job_description, re.IGNORECASE | re.DOTALL)
        for match in matches:
            lines = [line.strip() for line in match.split('\n') if line.strip()]
            for line in lines:
                line = re.sub(r'^(?:Proficiency in|Strong|Experience with|Knowledge of)\s+', '', line, flags=re.IGNORECASE)
                paren_skills = re.findall(r'\((.*?)\)', line)
                for skill_group in paren_skills:
                    skills = [s.strip() for s in skill_group.split(',') if s.strip()]
                    mandatory_skills.extend(skills)
                line = re.sub(r'\s*\(.*?\)', '', line)
                skills = re.split(r',\s*|\s+and\s+', line, flags=re.IGNORECASE)
                skills = [skill.strip() for skill in skills if skill.strip() and len(skill) > 1]
                mandatory_skills.extend(skills)
    known_technical_skills = set(SKILL_ALIASES.keys())
    technical_skills = [normalize_skill(skill) for skill in mandatory_skills if normalize_skill(skill) in known_technical_skills]
    logger.debug(f"Fallback extracted mandatory skills: {technical_skills}")
    return list(set(technical_skills))

def match_project_skills_to_mandatory(projects: List[Dict], mandatory_skills: List[str]) -> Tuple[List[Dict], List[str], int]:
    matched_projects = []
    all_matched_skills = set()
    skill_weights = {
        'python': 3, 'r': 3, 'tensorflow': 3, 'pytorch': 3, 'scikit-learn': 3, 'pandas': 3, 'numpy': 3,
        'matplotlib': 3, 'nlp': 3, 'computer vision': 3, 'gpt': 3, 'bert': 3, 'dall-e': 3,
        'aws': 2, 'azure': 2, 'gcp': 2, 'sql': 2
    }
    total_possible_weight = sum(skill_weights.get(normalize_skill(skill), 1) for skill in mandatory_skills)
    current_weight = 0
    for project in projects:
        project_skills = project.get("skills", [])
        matched_skills_for_project = []
        for mandatory_skill in mandatory_skills:
            normalized_mandatory = normalize_skill(mandatory_skill)
            for project_skill in project_skills:
                if fuzz.ratio(normalized_mandatory, normalize_skill(project_skill)) > 80:
                    matched_skills_for_project.append(mandatory_skill)
                    all_matched_skills.add(mandatory_skill)
                    current_weight += skill_weights.get(normalized_mandatory, 1)
                    break
        if matched_skills_for_project:
            matched_projects.append({
                "name": project.get("name", "Unnamed Project"),
                "description": project.get("description", "No description provided"),
                "skills": project_skills,
                "relevance": f"Matches mandatory skills: {', '.join(matched_skills_for_project)}"
            })
            logger.debug(f"Project '{project.get('name', 'Unnamed')}' matched skills: {matched_skills_for_project}")
    score = int((current_weight / max(total_possible_weight, 1)) * 100) if mandatory_skills else 0
    score = min(100, max(0, score))
    logger.debug(f"Matched skills: {all_matched_skills}, Total weight: {current_weight}/{total_possible_weight}, Score: {score}")
    return matched_projects, list(all_matched_skills), score

def enhanced_technical_analysis(resume_text: str, job_description: str) -> Dict:
    logger.info("Starting enhanced technical analysis...")
    projects = extract_projects_from_resume(resume_text)
    logger.info(f"Extracted {len(projects)} projects from resume")
    mandatory_skills = extract_mandatory_skills_from_jd(job_description)
    logger.info(f"Extracted {len(mandatory_skills)} mandatory skills from JD")
    matched_projects, matched_skills, technical_score = match_project_skills_to_mandatory(projects, mandatory_skills)
    proficiency_score = calculate_proficiency_score(resume_text, job_description, projects)
    missing_skills = [skill for skill in mandatory_skills if skill not in matched_skills]
    pain_points = {
        "critical": [],
        "major": [],
        "minor": []
    }
    if missing_skills:
        pain_points["critical"].append(f"Missing mandatory technical skills: {', '.join(missing_skills[:5])}")
        if len(missing_skills) > 5:
            pain_points["critical"].append(f"And {len(missing_skills) - 5} more missing skills")
    if not matched_projects:
        pain_points["critical"].append("No projects found with mandatory technical skills")
    elif len(matched_projects) < len(projects) // 2:
        pain_points["major"].append("Limited project alignment with mandatory requirements")
    if technical_score >= 70:
        pain_points["minor"].append("Strong technical alignment - recommend advanced technical interview")
    elif technical_score >= 50:
        pain_points["minor"].append("Moderate alignment - focus interview on missing skills")
    else:
        pain_points["minor"].append("Significant gaps - consider extensive technical assessment")
    if proficiency_score < 10:
        pain_points["major"].append("Low proficiency score indicates gaps in experience or resume quality")
    status = "Shortlisted" if technical_score >= 70 and proficiency_score >= 20 else "Under Consideration" if technical_score >= 50 else "Rejected"
    summary = generate_enhanced_technical_summary(technical_score, len(matched_projects), len(projects), len(matched_skills), len(mandatory_skills), missing_skills)
    result = {
        "technical_score": technical_score,
        "proficiency_score": proficiency_score,
        "pain_points": pain_points,
        "summary": summary,
        "status": status,
        "projects": matched_projects,
        "extraction_stats": {
            "total_projects_found": len(projects),
            "projects_with_matches": len(matched_projects),
            "mandatory_skills_required": len(mandatory_skills),
            "mandatory_skills_matched": len(matched_skills),
            "coverage_percentage": int((len(matched_skills) / max(len(mandatory_skills), 1)) * 100)
        }
    }
    logger.info(f"Enhanced technical analysis completed - Technical Score: {technical_score}, Proficiency Score: {proficiency_score}")
    return result

def generate_enhanced_technical_summary(score: int, matched_projects: int, total_projects: int, 
                                       matched_skills: int, total_mandatory: int, missing_skills: List[str]) -> str:
    summary_parts = []
    if score >= 70:
        summary_parts.append(f"Excellent technical alignment with {matched_skills}/{total_mandatory} mandatory skills in {matched_projects} projects.")
    elif score >= 50:
        summary_parts.append(f"Moderate technical fit with {matched_skills}/{total_mandatory} mandatory skills in {matched_projects} projects.")
    else:
        summary_parts.append(f"Limited technical alignment with {matched_skills}/{total_mandatory} mandatory skills in {matched_projects} projects.")
    if matched_projects > 0:
        summary_parts.append(f"Projects cover {matched_projects}/{total_projects} relevant technologies.")
    else:
        summary_parts.append("No projects found with mandatory skills, indicating skill gaps.")
    if missing_skills:
        summary_parts.append(f"Key gaps: {', '.join(missing_skills[:3])}{'...' if len(missing_skills) > 3 else ''}.")
    if score >= 70:
        summary_parts.append("Recommend advanced technical interview to validate project depth.")
    elif score >= 50:
        summary_parts.append("Interview should focus on missing skills and project claims.")
    else:
        summary_parts.append("Extensive assessment needed to evaluate skill gaps.")
    return " ".join(summary_parts)

def analyze_resume(resume_text: str, job_description: str, projects: List[Dict] = None) -> Dict:
    try:
        return enhanced_technical_analysis(resume_text, job_description)
    except Exception as e:
        logger.error(f"Enhanced analysis failed, falling back to original: {e}")
        if projects is None:
            projects = extract_projects_from_resume(resume_text)
        return fallback_technical_score(resume_text, job_description, projects)

def fallback_technical_score(resume_text: str, job_description: str, projects: List[Dict]) -> Dict:
    score = 0
    pain_points_list = []
    matched_projects = []
    mandatory_skills = extract_mandatory_skills_fallback(job_description)
    logger.debug(f"Extracted mandatory technical skills: {mandatory_skills}")
    matched_tech_skills = []
    skill_weights = {
        'python': 3, 'r': 3, 'tensorflow': 3, 'pytorch': 3, 'scikit-learn': 3, 'pandas': 3, 'numpy': 3,
        'matplotlib': 3, 'nlp': 3, 'computer vision': 3, 'gpt': 3, 'bert': 3, 'dall-e': 3,
        'aws': 2, 'azure': 2, 'gcp': 2, 'sql': 2
    }
    total_possible_weight = sum(skill_weights.get(normalize_skill(skill), 1) for skill in mandatory_skills)
    current_weight = 0
    for project in projects:
        project_skills = project.get("skills", [])
        matched_skills = []
        for skill in mandatory_skills:
            normalized_mandatory = normalize_skill(skill)
            for ps in project_skills:
                if fuzz.ratio(normalized_mandatory, normalize_skill(ps)) > 80:
                    matched_skills.append(skill)
                    current_weight += skill_weights.get(normalized_mandatory, 1)
                    break
        if matched_skills:
            matched_projects.append({
                "name": project.get("name", "Unnamed Project"),
                "description": project.get("description", "No description provided"),
                "skills": project_skills,
                "relevance": f"Matches skills: {', '.join(matched_skills)}"
            })
            matched_tech_skills.extend(matched_skills)
    matched_tech_skills = list(set(matched_tech_skills))
    score = int((current_weight / max(total_possible_weight, 1)) * 100) if mandatory_skills else 0
    score = min(100, max(0, score))
    proficiency_score = calculate_proficiency_score(resume_text, job_description, projects)
    pain_points = {
        "critical": [f"Missing mandatory technical skills: {', '.join([skill for skill in mandatory_skills if skill not in matched_tech_skills])}"] if len(matched_tech_skills) < len(mandatory_skills) else [],
        "major": ["Low proficiency score indicates gaps in experience or resume quality"] if proficiency_score < 10 else [],
        "minor": ["Needs further evaluation in interview"] if score >= 50 else []
    }
    status = "Shortlisted" if score >= 70 and proficiency_score >= 20 else "Under Consideration" if score >= 50 else "Rejected"
    summary = f"Technical score: {score}/100, Proficiency: {proficiency_score}/30. Covers {len(matched_tech_skills)}/{len(mandatory_skills)} mandatory skills. {'Strong fit; recommend interview.' if score >= 70 else 'Moderate fit; assess gaps.' if score >= 50 else 'Significant gaps; may need training.'}"
    result = {
        "technical_score": score,
        "proficiency_score": proficiency_score,
        "pain_points": pain_points,
        "summary": summary,
        "status": status,
        "projects": matched_projects
    }
    logger.info(f"Fallback technical assessment result: Technical Score: {score}, Proficiency Score: {proficiency_score}, Matched Skills: {matched_tech_skills}")
    return result

def generate_technical_summary(score: int, pain_points: Dict) -> str:
    summary_parts = []
    if score >= 70:
        summary_parts.append("Strong alignment with mandatory technical skills in projects.")
    elif score >= 50:
        summary_parts.append("Moderate alignment with some gaps in project skills.")
    else:
        summary_parts.append("Significant gaps in project-based mandatory skills.")
    if pain_points.get("critical"):
        summary_parts.append(f"Critical issues: {'; '.join(pain_points['critical'][:2])}.")
    if pain_points.get("minor"):
        summary_parts.append(f"Minor improvements: {'; '.join(pain_points['minor'][:2])}.")
    summary_parts.append("Technical interview recommended to validate project skills.")
    return " ".join(summary_parts)