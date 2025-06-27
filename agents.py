import logging
import re
import json
from dotenv import load_dotenv
import os
import time
from openai import AzureOpenAI
from typing import Dict, List, Tuple
from collections import deque
from fuzzywuzzy import fuzz

# Setup logging
logging.basicConfig(level=logging.DEBUG, filename='logs.txt', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
try:
    load_dotenv()
    azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_version = os.getenv("AZURE_OPENAI_VERSION")
    azure_openai_model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")
    
    client = AzureOpenAI(
        api_key=azure_openai_key,
        api_version=azure_openai_version,
        azure_endpoint=azure_openai_endpoint
    )
    logger.info("Azure OpenAI API configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Azure OpenAI API: {e}")
    raise

# Rate limiter for LLM calls (30 RPM = 2 seconds minimum per request, adjust as needed)
class RateLimiter:
    def __init__(self, requests_per_minute: int = 30, period: int = 60):
        self.requests_per_minute = requests_per_minute
        self.period = period
        self.requests = deque(maxlen=requests_per_minute)
        self.last_request_time = 0
        logger.info(f"Initialized RateLimiter with {requests_per_minute} requests per {period} seconds")

    def acquire(self):
        current_time = time.time()
        while self.requests and current_time - self.requests[0] > self.period:
            self.requests.popleft()
        if len(self.requests) >= self.requests_per_minute:
            sleep_time = max(2, self.period - (current_time - self.requests[0]))  # Minimum 2s
            logger.debug(f"Rate limit reached, waiting {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            current_time = time.time()
            while self.requests and current_time - self.requests[0] > self.period:
                self.requests.popleft()
        if self.last_request_time > 0:
            elapsed = current_time - self.last_request_time
            if elapsed < 2:
                time.sleep(2 - elapsed)
                current_time = time.time()
        self.requests.append(current_time)
        self.last_request_time = current_time
        logger.debug(f"LLM call allowed, {len(self.requests)}/{self.requests_per_minute} requests in queue")

# Initialize rate limiter
rate_limiter = RateLimiter(requests_per_minute=30, period=60)

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
                    "all_skills": ["<skill1>", "<skill2>", "<skill3>", ...]
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
                response = client.chat.completions.create(
                    model=azure_openai_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                output = response.choices[0].message.content.strip()
                if output.startswith("```json"):
                    output = output[len("```json"):].rstrip("```").strip()
                elif output.startswith("```"):
                    output = output[len("```"):].rstrip("```").strip()
                result = json.loads(output)
                projects = result.get("projects", [])
                for project in projects:
                    project['all_skills'] = [normalize_skill(skill) for skill in project.get('all_skills', [])]
                logger.info(f"Extracted {len(projects)} projects from resume")
                return projects
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise
            except Exception as e:
                logger.error(f"Failed to extract projects using LLM: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
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
    projects = list(dict.fromkeys(projects))
    for project in projects:
        project['all_skills'] = [normalize_skill(skill) for skill in project.get('all_skills', [])]
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
                "all_skills": skills
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
                response = client.chat.completions.create(
                    model=azure_openai_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                output = response.choices[0].message.content.strip()
                if output.startswith("```json"):
                    output = output[len("```json"):].rstrip("```").strip()
                elif output.startswith("```"):
                    output = output[len("```"):].rstrip("```").strip()
                result = json.loads(output)
                skills = result.get("nice_to_have_skills", [])
                normalized_skills = [normalize_skill(skill) for skill in skills]
                logger.info(f"Extracted {len(normalized_skills)} nice-to-have skills from JD: {normalized_skills}")
                return normalized_skills
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise
            except Exception as e:
                logger.error(f"Failed to extract nice-to-have skills using LLM: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
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
        response = client.chat.completions.create(
            model=azure_openai_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        output = response.choices[0].message.content.strip()
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
        response = client.chat.completions.create(
            model=azure_openai_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        output = response.choices[0].message.content.strip()
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

def calculate_proficiency_score(resume_text: str, job_description: str, projects: List[Dict]) -> Tuple[int, Dict]:
    try:
        # Nice-to-have skills (8 points)
        nice_to_have_skills = extract_nice_to_have_skills_from_jd(job_description)
        matched_nice_skills = []
        for project in projects:
            for skill in project.get("all_skills", []):
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
                response = client.chat.completions.create(
                    model=azure_openai_model_name,
                    messages=[{"role": "user", "content": resume_quality_prompt}],
                    temperature=0.0
                )
                output = response.choices[0].message.content.strip()
                if output.startswith("```json"):
                    output = output[len("```json"):].rstrip("```").strip()
                elif output.startswith("```"):
                    output = output[len("```"):].rstrip("```").strip()
                quality_result = json.loads(output)
                resume_quality_score = min(8, quality_result.get("total", 0))
                logger.debug(f"Resume quality score: {resume_quality_score}, Details: {quality_result}")
                break
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise
            except Exception as e:
                logger.error(f"Failed to calculate resume quality score: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
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
            "years_details": "<description of years match>",
            "role_relevance": <score>,
            "role_details": "<description of role match>",
            "industry_fit": <score>,
            "industry_details": "<description of industry match>",
            "total": <sum of scores>
        }}
        """
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=azure_openai_model_name,
                    messages=[{"role": "user", "content": experience_prompt}],
                    temperature=0.0
                )
                output = response.choices[0].message.content.strip()
                if output.startswith("```json"):
                    output = output[len("```json"):].rstrip("```").strip()
                elif output.startswith("```"):
                    output = output[len("```"):].rstrip("```").strip()
                experience_result = json.loads(output)
                experience_score = min(8, experience_result.get("total", 0))
                logger.debug(f"Experience alignment score: {experience_score}, Details: {experience_result}")
                break
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise
            except Exception as e:
                logger.error(f"Failed to calculate experience alignment score: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise
        else:
            experience_result = {
                "years": 0,
                "years_details": "No experience data available",
                "role_relevance": 0,
                "role_details": "No role data available",
                "industry_fit": 0,
                "industry_details": "No industry data available",
                "total": 0
            }
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
        cert_score = min(3, int((len(relevant_certs) / 2) * 3))
        logger.debug(f"Certifications score: {cert_score}, Relevant: {relevant_certs}")

        total_proficiency_score = nice_skill_score + resume_quality_score + experience_score + soft_skill_score + cert_score
        logger.info(f"Proficiency score: {total_proficiency_score}/30 (Nice-to-have: {nice_skill_score}, Quality: {resume_quality_score}, Experience: {experience_score}, Soft Skills: {soft_skill_score}, Certifications: {cert_score})")
        return total_proficiency_score, experience_result
    except Exception as e:
        logger.error(f"Failed to calculate proficiency score: {e}")
        return 0, {
            "years": 0,
            "years_details": "Error in experience analysis",
            "role_relevance": 0,
            "role_details": "Error in role analysis",
            "industry_fit": 0,
            "industry_details": "Error in industry analysis",
            "total": 0
        }

def extract_mandatory_skills_from_jd(job_description: str) -> List[str]:
    try:
        rate_limiter.acquire()
        prompt = f"""
        **TASK:** Extract ALL mandatory, required, and must-have technical skills from the job description.
        
        **JOB DESCRIPTION:**
        {job_description}
        
        **EXTRACTION RULES:**
        1. Look for sections with keywords indicating importance:
           - "required", "must have", "essential", "mandatory", "must-have"
           - "required skills", "technical requirements", "minimum qualifications"
           - Skills mentioned in job responsibilities/duties (these are implicitly required)
        
        2. Extract ONLY technical skills including:
           - Programming languages (Python, R, Java, JavaScript, etc.)
           - Frameworks and libraries (TensorFlow, PyTorch, React, Django, etc.)
           - Tools and platforms (Git, Docker, Kubernetes, AWS, Azure, etc.)
           - Databases (MySQL, MongoDB, PostgreSQL, etc.)
           - AI/ML specific terms (NLP, Computer Vision, Deep Learning, MLOps, etc.)
           - Data analysis tools (Pandas, NumPy, Matplotlib, etc.)
        
        3. Ignore non-technical requirements:
           - Soft skills, years of experience, degrees, certifications
           - Business domain knowledge
        
        4. Handle compound skills:
           - Split "Python and R" → ["Python", "R"]
           - Extract from parentheses: "ML frameworks (TensorFlow, PyTorch)" → ["TensorFlow", "PyTorch"]
           - Handle alternatives: "Python or R" → ["Python", "R"]
        
        5. For single-character skills like "R", ensure they refer to the programming language
        
        **OUTPUT FORMAT - Return ONLY valid JSON:**
        {{
            "mandatory_skills": ["<skill1>", "<skill2>", "<skill3>", ...]
        }}
        
        **IMPORTANT:** Be comprehensive - include ALL technical skills that would be needed to perform the job successfully.
        """
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=azure_openai_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                output = response.choices[0].message.content.strip()
                if output.startswith("```json"):
                    output = output[len("```json"):].rstrip("```").strip()
                elif output.startswith("```"):
                    output = output[len("```"):].rstrip("```").strip()
                result = json.loads(output)
                skills = result.get("mandatory_skills", [])
                normalized_skills = [normalize_skill(skill) for skill in skills]
                logger.info(f"Extracted {len(normalized_skills)} mandatory skills from JD: {normalized_skills}")
                return normalized_skills
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise
            except Exception as e:
                logger.error(f"Failed to extract mandatory skills using LLM: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise
    except Exception as e:
        logger.error(f"Failed to extract mandatory skills after retries: {e}")
        return extract_mandatory_skills_fallback(job_description)

def extract_mandatory_skills_fallback(job_description: str) -> List[str]:
    mandatory_skills = []
    mandatory_patterns = [
        r'(?:required|must have|essential|mandatory|must-have|required skills|technical requirements|minimum qualifications)\s*[:\s]*(.*?)(?=\n[A-Z][a-zA-Z\s]+:|\n\n|\Z)',
        r'(?:responsibilities|duties|qualifications)\s*[:\s]*(.*?)(?=\n[A-Z][a-zA-Z\s]+:|\n\n|\Z)'
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
                skills = re.split(r',\s*|\s+and\s+|\s+or\s+', line, flags=re.IGNORECASE)
                skills = [skill.strip() for skill in skills if skill.strip() and len(skill) > 1]
                mandatory_skills.extend(skills)
    known_technical_skills = set(SKILL_ALIASES.keys())
    technical_skills = [normalize_skill(skill) for skill in mandatory_skills if normalize_skill(skill) in known_technical_skills]
    logger.debug(f"Fallback extracted mandatory skills: {technical_skills}")
    return list(set(technical_skills))

def calculate_technical_score_from_projects(projects: List[Dict], mandatory_skills: List[str]) -> Tuple[int, List[Dict], Dict]:
    if not mandatory_skills:
        logger.warning("No mandatory skills found - returning 0 score")
        return 0, [], {"matched_skills": [], "missing_skills": [], "match_details": {}}
    
    matched_skills = set()
    matched_projects = []
    skill_match_details = {}
    
    # Track which projects contain each skill
    skill_to_projects = {}
    
    for project_idx, project in enumerate(projects):
        project_skills = project.get("all_skills", [])
        project_matched_skills = []
        highlighted_skills = []
        
        for mandatory_skill in mandatory_skills:
            normalized_mandatory = normalize_skill(mandatory_skill)
            
            # Find matching skills in this project
            for project_skill in project_skills:
                normalized_project_skill = normalize_skill(project_skill)
                
                # Use fuzzy matching for better accuracy
                if fuzz.ratio(normalized_mandatory, normalized_project_skill) > 85:
                    project_matched_skills.append(mandatory_skill)
                    highlighted_skills.append(project_skill)  # Keep original casing
                    matched_skills.add(mandatory_skill)
                    
                    # Track skill-to-project mapping
                    if mandatory_skill not in skill_to_projects:
                        skill_to_projects[mandatory_skill] = []
                    skill_to_projects[mandatory_skill].append(project.get("name", f"Project {project_idx + 1}"))
                    break
        
        # Only include projects that have matched skills
        if project_matched_skills:
            project_with_highlights = {
                "name": project.get("name", f"Project {project_idx + 1}"),
                "description": project.get("description", "No description provided"),
                "all_skills": project_skills,
                "matched_mandatory_skills": list(set(project_matched_skills)),
                "highlighted_skills": list(set(highlighted_skills)),
                "skill_match_count": len(set(project_matched_skills))
            }
            matched_projects.append(project_with_highlights)
    
    # Calculate technical score as percentage of mandatory skills covered
    matched_skills_list = list(matched_skills)
    missing_skills = [skill for skill in mandatory_skills if skill not in matched_skills]
    
    # Technical score is simply the percentage of mandatory skills matched
    technical_score = int((len(matched_skills_list) / len(mandatory_skills)) * 100) if mandatory_skills else 0
    
    # Create detailed skill analysis
    skill_analysis = {
        "matched_skills": matched_skills_list,
        "missing_skills": missing_skills,
        "total_mandatory_skills": len(mandatory_skills),
        "total_matched_skills": len(matched_skills_list),
        "coverage_percentage": technical_score,
        "skill_to_projects": skill_to_projects,
        "match_details": {
            "projects_with_matches": len(matched_projects),
            "total_projects_analyzed": len(projects)
        }
    }
    
    logger.info(f"Technical Score: {technical_score}/100 - Matched {len(matched_skills_list)}/{len(mandatory_skills)} mandatory skills")
    logger.debug(f"Matched skills: {matched_skills_list}")
    logger.debug(f"Missing skills: {missing_skills}")
    
    return technical_score, matched_projects, skill_analysis

def enhanced_technical_analysis(resume_text: str, job_description: str) -> Dict:
    logger.info("Starting enhanced technical analysis...")
    
    # Extract projects from resume
    projects = extract_projects_from_resume(resume_text)
    logger.info(f"Extracted {len(projects)} projects from resume")
    
    # Extract mandatory skills from job description
    mandatory_skills = extract_mandatory_skills_from_jd(job_description)
    logger.info(f"Extracted {len(mandatory_skills)} mandatory skills from JD")
    
    # Calculate technical score based purely on project-skill matching
    technical_score, matched_projects, skill_analysis = calculate_technical_score_from_projects(projects, mandatory_skills)
    
    # Calculate proficiency score and experience details
    proficiency_score, experience_result = calculate_proficiency_score(resume_text, job_description, projects)
    
    # Extract additional information for strengths and weaknesses
    nice_to_have_skills = extract_nice_to_have_skills_from_jd(job_description)
    soft_skills = extract_soft_skills(resume_text)
    certifications = extract_certifications(resume_text)
    
    strengths = []
    weaknesses = []
    
    # Strengths based on skill matching
    if skill_analysis["matched_skills"]:
        strengths.append(f"Technical alignment: Matched {len(skill_analysis['matched_skills'])}/{len(mandatory_skills)} mandatory skills: {', '.join(skill_analysis['matched_skills'])}")
    
    # Experience-based strengths
    if experience_result["total"] >= 6:
        strengths.append(f"Strong experience alignment: {experience_result['years_details']} (Score: {experience_result['years']}/3)")
        strengths.append(f"Relevant roles: {experience_result['role_details']} (Score: {experience_result['role_relevance']}/3)")
        strengths.append(f"Industry fit: {experience_result['industry_details']} (Score: {experience_result['industry_fit']}/2)")
    elif experience_result["total"] >= 4:
        strengths.append(f"Moderate experience alignment: {experience_result['years_details']} (Score: {experience_result['years']}/3)")
        if experience_result["role_relevance"] > 0:
            strengths.append(f"Role relevance: {experience_result['role_details']} (Score: {experience_result['role_relevance']}/3)")
        if experience_result["industry_fit"] > 0:
            strengths.append(f"Industry fit: {experience_result['industry_details']} (Score: {experience_result['industry_fit']}/2)")
    
    if matched_projects:
        strengths.append(f"Relevant project experience: {len(matched_projects)} project(s) demonstrate required skills")
    
    # Nice-to-have skills
    matched_nice_skills = []
    for skill in nice_to_have_skills:
        for project in projects:
            if any(fuzz.ratio(normalize_skill(skill), normalize_skill(ps)) > 85 for ps in project.get("all_skills", [])):
                matched_nice_skills.append(skill)
                break
    if matched_nice_skills:
        strengths.append(f"Additional nice-to-have skills: {', '.join(matched_nice_skills)}")
    
    # Soft skills
    matched_soft_skills = [skill for skill in soft_skills if skill.lower() in ['problem-solving', 'communication', 'collaboration', 'analytical thinking']]
    if matched_soft_skills:
        strengths.append(f"Relevant soft skills: {', '.join(matched_soft_skills)}")
    
    # Certifications
    relevant_certs = [cert for cert in certifications if any(keyword in cert.lower() for keyword in ['ai', 'data science', 'machine learning', 'cloud', 'aws', 'azure', 'gcp'])]
    if relevant_certs:
        strengths.append(f"Relevant certifications: {', '.join(relevant_certs)}")
    
    # Weaknesses
    if skill_analysis["missing_skills"]:
        weaknesses.append(f"Missing skills: {', '.join(skill_analysis['missing_skills'])} ({len(skill_analysis['missing_skills'])}/{len(mandatory_skills)} skills not demonstrated)")
    
    if experience_result["total"] < 4:
        weaknesses.append(f"Lack of experience: {experience_result['years_details']} (Score: {experience_result['years']}/3)")
        weaknesses.append(f"Role gaps: {experience_result['role_details']} (Score: {experience_result['role_relevance']}/3)")
        weaknesses.append(f"Industry gaps: {experience_result['industry_details']} (Score: {experience_result['industry_fit']}/2)")
    
    if not matched_projects:
        weaknesses.append("No projects demonstrate mandatory technical skills from job requirements")
    
    if proficiency_score < 20:
        if proficiency_score < 10:
            weaknesses.append("Significant gaps in experience quality, certifications, and additional skills")
        else:
            weaknesses.append("Some gaps in experience depth or additional qualifications")
    
    if not strengths and not weaknesses:
        if technical_score > 50:
            strengths.append("Basic technical requirements partially met")
        else:
            weaknesses.append("Limited alignment with technical requirements")
    
    coverage_percentage = skill_analysis["coverage_percentage"]
    if coverage_percentage < 70 and mandatory_skills:
        weaknesses.append(f"Only {coverage_percentage}% of mandatory skills demonstrated")
    
    strengths_weaknesses = {
        "strengths": strengths,
        "weaknesses": weaknesses
    }
    
    # Determine status based on original logic
    if technical_score >= 80 and proficiency_score >= 20 and len(skill_analysis["missing_skills"]) <= 1:
        status = "Shortlisted"
    elif technical_score >= 60 and proficiency_score >= 15:
        status = "Under Consideration"
    else:
        status = "Rejected"
    
    result = {
        "technical_score": technical_score,
        "proficiency_score": proficiency_score,
        "status": status,
        "projects": matched_projects,
        "skill_analysis": skill_analysis,
        "strengths_weaknesses": strengths_weaknesses,
        "experience_result": experience_result,
        "extraction_stats": {
            "total_projects_found": len(projects),
            "projects_with_skill_matches": len(matched_projects),
            "mandatory_skills_required": len(mandatory_skills),
            "mandatory_skills_matched": len(skill_analysis["matched_skills"]),
            "coverage_percentage": technical_score
        }
    }
    
    logger.info(f"Enhanced technical analysis completed - Technical Score: {technical_score}/100, Proficiency Score: {proficiency_score}/30")
    return result

def fallback_technical_score(resume_text: str, job_description: str, projects: List[Dict]) -> Dict:
    logger.info("Starting fallback technical analysis...")
    
    # Extract mandatory skills using fallback method
    mandatory_skills = extract_mandatory_skills_fallback(job_description)
    logger.info(f"Extracted {len(mandatory_skills)} mandatory skills from JD (fallback)")
    
    # Calculate technical score using project-skill matching
    technical_score, matched_projects, skill_analysis = calculate_technical_score_from_projects(projects, mandatory_skills)
    
    # Calculate proficiency score and experience details
    proficiency_score, experience_result = calculate_proficiency_score(resume_text, job_description, projects)
    
    # Extract additional information for strengths and weaknesses
    nice_to_have_skills = extract_nice_to_have_skills_fallback(job_description)
    soft_skills = extract_soft_skills(resume_text)
    certifications = extract_certifications(resume_text)
    
    strengths = []
    weaknesses = []
    
    # Strengths based on skill matching
    if skill_analysis["matched_skills"]:
        strengths.append(f"Technical alignment: Matched {len(skill_analysis['matched_skills'])}/{len(mandatory_skills)} mandatory skills: {', '.join(skill_analysis['matched_skills'])}")
    
    # Experience-based strengths
    if experience_result["total"] >= 6:
        strengths.append(f"Strong experience alignment: {experience_result['years_details']} (Score: {experience_result['years']}/3)")
        strengths.append(f"Relevant roles: {experience_result['role_details']} (Score: {experience_result['role_relevance']}/3)")
        strengths.append(f"Industry fit: {experience_result['industry_details']} (Score: {experience_result['industry_fit']}/2)")
    elif experience_result["total"] >= 4:
        strengths.append(f"Moderate experience alignment: {experience_result['years_details']} (Score: {experience_result['years']}/3)")
        if experience_result["role_relevance"] > 0:
            strengths.append(f"Role relevance: {experience_result['role_details']} (Score: {experience_result['role_relevance']}/3)")
        if experience_result["industry_fit"] > 0:
            strengths.append(f"Industry fit: {experience_result['industry_details']} (Score: {experience_result['industry_fit']}/2)")
    
    if matched_projects:
        strengths.append(f"Relevant project experience: {len(matched_projects)} project(s) demonstrate required skills")
    
    # Nice-to-have skills
    matched_nice_skills = []
    for skill in nice_to_have_skills:
        for project in projects:
            if any(fuzz.ratio(normalize_skill(skill), normalize_skill(ps)) > 85 for ps in project.get("all_skills", [])):
                matched_nice_skills.append(skill)
                break
    if matched_nice_skills:
        strengths.append(f"Additional nice-to-have skills: {', '.join(matched_nice_skills)}")
    
    # Soft skills
    matched_soft_skills = [skill for skill in soft_skills if skill.lower() in ['problem-solving', 'communication', 'collaboration', 'analytical thinking']]
    if matched_soft_skills:
        strengths.append(f"Relevant soft skills: {', '.join(matched_soft_skills)}")
    
    # Certifications
    relevant_certs = [cert for cert in certifications if any(keyword in cert.lower() for keyword in ['ai', 'data science', 'machine learning', 'cloud', 'aws', 'azure', 'gcp'])]
    if relevant_certs:
        strengths.append(f"Relevant certifications: {', '.join(relevant_certs)}")
    
    # Weaknesses
    if skill_analysis["missing_skills"]:
        weaknesses.append(f"Missing skills: {', '.join(skill_analysis['missing_skills'])} ({len(skill_analysis['missing_skills'])}/{len(mandatory_skills)} skills not demonstrated)")
    
    if experience_result["total"] < 4:
        weaknesses.append(f"Lack of experience: {experience_result['years_details']} (Score: {experience_result['years']}/3)")
        weaknesses.append(f"Role gaps: {experience_result['role_details']} (Score: {experience_result['role_relevance']}/3)")
        weaknesses.append(f"Industry gaps: {experience_result['industry_details']} (Score: {experience_result['industry_fit']}/2)")
    
    if not matched_projects:
        weaknesses.append("No projects demonstrate mandatory technical skills from job requirements")
    
    if proficiency_score < 20:
        if proficiency_score < 10:
            weaknesses.append("Significant gaps in experience quality, certifications, and additional skills")
        else:
            weaknesses.append("Some gaps in experience depth or additional qualifications")
    
    if not strengths and not weaknesses:
        if technical_score > 50:
            strengths.append("Basic technical requirements partially met")
        else:
            weaknesses.append("Limited alignment with technical requirements")
    
    coverage_percentage = skill_analysis["coverage_percentage"]
    if coverage_percentage < 70 and mandatory_skills:
        weaknesses.append(f"Only {coverage_percentage}% of mandatory skills demonstrated")
    
    strengths_weaknesses = {
        "strengths": strengths,
        "weaknesses": weaknesses
    }
    
    # Determine status based on original logic
    if technical_score >= 70:
        status = "Shortlisted"
    elif technical_score >= 50:
        status = "Under Consideration"
    else:
        status = "Rejected"
    
    result = {
        "technical_score": technical_score,
        "proficiency_score": proficiency_score,
        "status": status,
        "projects": matched_projects,
        "skill_analysis": skill_analysis,
        "strengths_weaknesses": strengths_weaknesses,
        "experience_result": experience_result,
        "extraction_stats": {
            "total_projects_found": len(projects),
            "projects_with_skill_matches": len(matched_projects),
            "mandatory_skills_required": len(mandatory_skills),
            "mandatory_skills_matched": len(skill_analysis["matched_skills"]),
            "coverage_percentage": technical_score
        }
    }
    
    logger.info(f"Fallback technical analysis completed - Technical Score: {technical_score}/100, Proficiency Score: {proficiency_score}/30, Matched Skills: {skill_analysis['matched_skills']}")
    return result

def analyze_resume(resume_text: str, job_description: str, projects: List[Dict] = None) -> Dict:
    try:
        return enhanced_technical_analysis(resume_text, job_description)
    except Exception as e:
        logger.error(f"Enhanced analysis failed, falling back to original: {e}")
        if projects is None:
            projects = extract_projects_from_resume(resume_text)
        return fallback_technical_score(resume_text, job_description, projects)