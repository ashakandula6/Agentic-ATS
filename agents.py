import logging
import re
import json
from dotenv import load_dotenv
import os
import google.generativeai as genai
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, filename='logs.txt', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
try:
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    llm = genai.GenerativeModel("gemini-2.5-flash-lite-preview-06-17")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    raise

def extract_projects_from_resume(resume_text: str) -> List[Dict]:
    """
    Extract all projects from resume with enhanced accuracy using LLM.
    Returns list of projects with name, description, and skills.
    """
    try:
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
           - Extract frameworks (e.g., React, Django, Flask, Spring)
           - Extract databases (e.g., MySQL, MongoDB, PostgreSQL)
           - Extract cloud platforms (e.g., AWS, Azure, GCP)
           - Extract tools (e.g., Git, Docker, Kubernetes)
           - Extract libraries (e.g., TensorFlow, PyTorch, NumPy)
           - Extract methodologies (e.g., Agile, Scrum)
           - Include version numbers if mentioned (e.g., "Python 3.8", "React 18")
           - For single-character skills like "R", ensure they refer to the programming language, not part of other terms (e.g., not "R" in "BERT" or "research")
        
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
        
        **EXAMPLES:**
        - Correct: "Used Python, R, and TensorFlow" → ["Python", "R", "TensorFlow"]
        - Incorrect: "Research in AI" → [] (no "R")
        - Incorrect: "BERT model" → [] (no "R")
        
        **IMPORTANT:**
        - If no projects found, return empty projects array
        - Extract skills exactly as mentioned in the resume
        - Don't infer or add skills not explicitly mentioned
        - Include all technical skills, even if they seem basic
        """
        
        response = llm.generate_content(prompt)
        output = response.text.strip()
        
        # Clean JSON output
        if output.startswith("```json"):
            output = output[len("```json"):].rstrip("```").strip()
        elif output.startswith("```"):
            output = output[len("```"):].rstrip("```").strip()
        
        result = json.loads(output)
        projects = result.get("projects", [])
        
        logger.info(f"Extracted {len(projects)} projects from resume")
        for i, project in enumerate(projects):
            logger.debug(f"Project {i+1}: {project.get('name', 'Unnamed')} - Skills: {project.get('skills', [])}")
        
        return projects
        
    except Exception as e:
        logger.error(f"Failed to extract projects using LLM: {e}")
        return extract_projects_fallback(resume_text)

def extract_projects_fallback(resume_text: str) -> List[Dict]:
    """
    Fallback project extraction using regex patterns.
    """
    projects = []
    
    # Common project section headers
    project_patterns = [
        r'(?i)(?:^|\n)\s*(?:projects?|academic\s+projects?|personal\s+projects?|work\s+projects?|portfolio)\s*[:\-]?\s*\n',
        r'(?i)(?:^|\n)\s*(?:key\s+projects?|major\s+projects?|relevant\s+projects?)\s*[:\-]?\s*\n'
    ]
    
    for pattern in project_patterns:
        matches = list(re.finditer(pattern, resume_text))
        if matches:
            for match in matches:
                start_pos = match.end()
                # Find the end of the projects section
                next_section = re.search(r'(?i)\n\s*(?:experience|education|skills|certifications?|achievements?|awards?)\s*[:\-]?\s*\n', resume_text[start_pos:])
                end_pos = start_pos + next_section.start() if next_section else len(resume_text)
                
                projects_text = resume_text[start_pos:end_pos]
                projects.extend(parse_projects_text(projects_text))
    
    if not projects:
        logger.warning("No projects found using fallback extraction")
    
    return projects

def parse_projects_text(projects_text: str) -> List[Dict]:
    """
    Parse individual projects from projects section text.
    """
    projects = []
    
    # Split by bullet points or project indicators
    project_blocks = re.split(r'(?:\n\s*[•▪▫\-\*]\s*|\n\s*\d+\.\s*)', projects_text)
    
    for block in project_blocks:
        if len(block.strip()) < 20:  # Skip very short blocks
            continue
            
        # Extract project name (usually first line or bolded text)
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if not lines:
            continue
            
        project_name = lines[0]
        description = ' '.join(lines[1:]) if len(lines) > 1 else lines[0]
        
        # Extract skills using common patterns
        skills = extract_skills_from_text(description)
        
        if skills:  # Only include projects with identifiable skills
            projects.append({
                "name": project_name,
                "description": description,
                "skills": skills
            })
    
    return projects

def extract_skills_from_text(text: str) -> List[str]:
    """
    Extract technical skills from project description text with stricter matching.
    """
    skills = []
    
    # Updated skill patterns with stricter matching for "R"
    skill_patterns = {
        'languages': r'\b(?:Python|Java|JavaScript|TypeScript|C\+\+|C#|PHP|Ruby|Go|Rust|Swift|Kotlin|Scala|R(?!\w)|MATLAB|SQL)\b',
        'frameworks': r'\b(?:React|Angular|Vue|Django|Flask|Spring|Express|Laravel|Rails|ASP\.NET|Bootstrap|Tailwind)\b',
        'databases': r'\b(?:MySQL|PostgreSQL|MongoDB|Redis|Cassandra|DynamoDB|SQLite|Oracle|SQL Server)\b',
        'cloud': r'\b(?:AWS|Azure|GCP|Google Cloud|Heroku|DigitalOcean|Docker|Kubernetes)\b',
        'tools': r'\b(?:Git|Jenkins|Travis|CircleCI|Jira|Slack|Postman|Swagger|Webpack|Babel)\b',
        'libraries': r'\b(?:TensorFlow|PyTorch|NumPy|Pandas|Scikit-learn|OpenCV|React Native|jQuery|Lodash)\b'
    }
    
    for category, pattern in skill_patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        skills.extend(matches)
    
    # Remove duplicates while preserving order
    skills = list(dict.fromkeys(skills))
    
    # Validate skills against known technical skills
    known_technical_skills = {
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin', 'scala', 'r',
        'matlab', 'sql', 'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'laravel', 'rails', 'asp.net',
        'bootstrap', 'tailwind', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'dynamodb', 'sqlite', 'oracle',
        'sql server', 'aws', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean', 'docker', 'kubernetes', 'git',
        'jenkins', 'travis', 'circleci', 'jira', 'slack', 'postman', 'swagger', 'webpack', 'babel', 'tensorflow', 'pytorch',
        'numpy', 'pandas', 'scikit-learn', 'opencv', 'react native', 'jquery', 'lodas'
    }
    
    validated_skills = [skill for skill in skills if skill.lower() in known_technical_skills]
    
    logger.debug(f"Extracted skills: {validated_skills}")
    return validated_skills

def extract_mandatory_skills_from_jd(job_description: str) -> List[str]:
    """
    Extract mandatory technical skills from job description with enhanced accuracy.
    """
    try:
        prompt = f"""
        **TASK:** Extract ONLY mandatory/required technical skills from the job description.
        
        **JOB DESCRIPTION:**
        {job_description}
        
        **EXTRACTION RULES:**
        1. Look for sections with keywords: "required", "must have", "essential", "mandatory", "must-have"
        2. Extract ONLY technical skills (programming languages, frameworks, tools, databases, etc.)
        3. Ignore soft skills, experience years, degree requirements, or business skills
        4. Extract skills from parentheses (e.g., "AI frameworks (TensorFlow, PyTorch)" → "TensorFlow", "PyTorch")
        5. Split compound requirements (e.g., "Python and R" → "Python", "R")
        6. Remove skill level indicators (e.g., "Advanced Python" → "Python")
        7. For single-character skills like "R", ensure they refer to the programming language, not part of other terms (e.g., not "R" in "BERT")
        
        **OUTPUT FORMAT - Return ONLY valid JSON:**
        {{
            "mandatory_skills": ["<skill1>", "<skill2>", "<skill3>", ...]
        }}
        
        **EXAMPLES:**
        - Correct: "Must have Python, R, and SQL" → ["Python", "R", "SQL"]
        - Incorrect: "Experience with BERT" → [] (no "R")
        - Incorrect: "Research skills" → [] (no "R")
        
        **IMPORTANT:**
        - Extract exact skill names as mentioned
        - Don't include experience requirements or proficiency levels
        - Focus only on technical skills that can be learned/demonstrated
        """
        
        response = llm.generate_content(prompt)
        output = response.text.strip()
        
        # Clean JSON output
        if output.startswith("```json"):
            output = output[len("```json"):].rstrip("```").strip()
        elif output.startswith("```"):
            output = output[len("```"):].rstrip("```").strip()
        
        result = json.loads(output)
        skills = result.get("mandatory_skills", [])
        
        logger.info(f"Extracted {len(skills)} mandatory skills from JD: {skills}")
        return skills
        
    except Exception as e:
        logger.error(f"Failed to extract mandatory skills using LLM: {e}")
        return extract_mandatory_skills_fallback(job_description)

def extract_mandatory_skills_fallback(job_description: str) -> List[str]:
    """
    Fallback extraction of mandatory skills using regex patterns.
    """
    mandatory_skills = []
    
    # Find mandatory sections
    mandatory_patterns = [
        r'(?:required|must have|essential|mandatory|must-have)\s*[:\s]*(.*?)(?=\n[A-Z][a-zA-Z\s]+:|\n\n|\Z)',
        r'(?:requirements?|qualifications?)\s*[:\s]*(.*?)(?=\n[A-Z][a-zA-Z\s]+:|\n\n|\Z)'
    ]
    
    for pattern in mandatory_patterns:
        matches = re.findall(pattern, job_description, re.IGNORECASE | re.DOTALL)
        for match in matches:
            # Extract skills from the matched text
            lines = [line.strip() for line in match.split('\n') if line.strip()]
            for line in lines:
                # Remove common prefixes
                line = re.sub(r'^(?:Proficiency in|Strong|Experience with|Knowledge of)\s+', '', line, flags=re.IGNORECASE)
                
                # Extract skills in parentheses
                paren_skills = re.findall(r'\((.*?)\)', line)
                for skill_group in paren_skills:
                    skills = [s.strip() for s in skill_group.split(',') if s.strip()]
                    mandatory_skills.extend(skills)
                
                # Remove parentheses and split by common delimiters
                line = re.sub(r'\s*\(.*?\)', '', line)
                skills = re.split(r',\s*|\s+and\s+', line, flags=re.IGNORECASE)
                skills = [skill.strip() for skill in skills if skill.strip() and len(skill) > 1]
                mandatory_skills.extend(skills)
    
    # Filter out non-technical terms
    technical_skills = []
    non_technical_patterns = [
        r'(?i)\b(?:years?|experience|degree|bachelor|master|phd|communication|leadership|team|management)\b'
    ]
    
    known_technical_skills = {
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin', 'scala', 'r',
        'matlab', 'sql', 'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'laravel', 'rails', 'asp.net',
        'bootstrap', 'tailwind', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'dynamodb', 'sqlite', 'oracle',
        'sql server', 'aws', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean', 'docker', 'kubernetes', 'git',
        'jenkins', 'travis', 'circleci', 'jira', 'slack', 'postman', 'swagger', 'webpack', 'babel', 'tensorflow', 'pytorch',
        'numpy', 'pandas', 'scikit-learn', 'opencv', 'react native', 'jquery', 'lodas'
    }
    
    for skill in mandatory_skills:
        if (not any(re.search(pattern, skill) for pattern in non_technical_patterns) and
            skill.lower() in known_technical_skills):
            technical_skills.append(skill)
    
    return list(set(technical_skills))

def match_project_skills_to_mandatory(projects: List[Dict], mandatory_skills: List[str]) -> Tuple[List[Dict], List[str], int]:
    """
    Match project skills to mandatory skills with stricter matching logic.
    """
    matched_projects = []
    all_matched_skills = set()
    
    for project in projects:
        project_skills = project.get("skills", [])
        matched_skills_for_project = []
        
        # Stricter matching: exact match or word-boundary-aware substring match
        for mandatory_skill in mandatory_skills:
            for project_skill in project_skills:
                # Exact match or ensure single-character skills like "R" are standalone
                if (mandatory_skill.lower() == project_skill.lower() or
                    (len(mandatory_skill) > 1 and re.search(rf'\b{re.escape(mandatory_skill)}\b', project_skill, re.IGNORECASE)) or
                    (len(project_skill) > 1 and re.search(rf'\b{re.escape(project_skill)}\b', mandatory_skill, re.IGNORECASE))):
                    matched_skills_for_project.append(mandatory_skill)
                    all_matched_skills.add(mandatory_skill)
                    break
                # Special handling for single-character skills like "R"
                elif mandatory_skill.lower() == 'r':
                    if re.search(r'\bR\b', project_skill, re.IGNORECASE):
                        matched_skills_for_project.append(mandatory_skill)
                        all_matched_skills.add(mandatory_skill)
                        break
        
        if matched_skills_for_project:
            matched_projects.append({
                "name": project.get("name", "Unnamed Project"),
                "description": project.get("description", "No description provided"),
                "skills": project_skills,
                "relevance": f"Matches mandatory skills: {', '.join(matched_skills_for_project)}"
            })
            logger.debug(f"Project '{project.get('name', 'Unnamed')}' matched skills: {matched_skills_for_project}")
    
    # Calculate score
    score = int((len(all_matched_skills) / max(len(mandatory_skills), 1)) * 100) if mandatory_skills else 0
    
    logger.debug(f"Matched skills: {all_matched_skills}, Score: {score}")
    return matched_projects, list(all_matched_skills), score

def enhanced_technical_analysis(resume_text: str, job_description: str) -> Dict:
    """
    Enhanced technical analysis focusing 100% on project-based skills matching.
    """
    logger.info("Starting enhanced technical analysis...")
    
    # Step 1: Extract ALL projects from resume
    projects = extract_projects_from_resume(resume_text)
    logger.info(f"Extracted {len(projects)} projects from resume")
    
    # Step 2: Extract mandatory skills from JD
    mandatory_skills = extract_mandatory_skills_from_jd(job_description)
    logger.info(f"Extracted {len(mandatory_skills)} mandatory skills from JD")
    
    # Step 3: Match project skills to mandatory skills
    matched_projects, matched_skills, score = match_project_skills_to_mandatory(projects, mandatory_skills)
    
    # Step 4: Identify missing skills
    missing_skills = [skill for skill in mandatory_skills if skill not in matched_skills]
    
    # Step 5: Generate pain points
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
    
    if score >= 70:
        pain_points["minor"].append("Strong technical alignment - recommend advanced technical interview")
    elif score >= 50:
        pain_points["minor"].append("Moderate alignment - focus interview on missing skills")
    else:
        pain_points["minor"].append("Significant gaps - consider extensive technical assessment")
    
    # Step 6: Generate status
    if score >= 70:
        status = "Shortlisted"
    elif score >= 50:
        status = "Under Consideration"
    else:
        status = "Rejected"
    
    # Step 7: Generate summary
    summary = generate_enhanced_technical_summary(score, len(matched_projects), len(projects), len(matched_skills), len(mandatory_skills), missing_skills)
    
    result = {
        "score": score,
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
    
    logger.info(f"Enhanced technical analysis completed - Score: {score}, Projects: {len(matched_projects)}/{len(projects)}")
    return result

def generate_enhanced_technical_summary(score: int, matched_projects: int, total_projects: int, 
                                       matched_skills: int, total_mandatory: int, missing_skills: List[str]) -> str:
    """
    Generate enhanced technical summary based on project analysis.
    """
    summary_parts = []
    
    # Opening statement
    if score >= 70:
        summary_parts.append(f"Candidate demonstrates excellent technical alignment with {matched_skills}/{total_mandatory} mandatory skills covered across {matched_projects} relevant projects.")
    elif score >= 50:
        summary_parts.append(f"Candidate shows moderate technical fit with {matched_skills}/{total_mandatory} mandatory skills demonstrated in {matched_projects} projects.")
    else:
        summary_parts.append(f"Candidate exhibits limited technical alignment with only {matched_skills}/{total_mandatory} mandatory skills found in {matched_projects} projects.")
    
    # Project analysis
    if matched_projects > 0:
        summary_parts.append(f"Project portfolio analysis reveals {matched_projects} out of {total_projects} projects contain relevant mandatory technologies.")
    else:
        summary_parts.append("No projects found containing mandatory technical skills, indicating potential skill gaps.")
    
    # Missing skills highlight
    if missing_skills:
        critical_missing = missing_skills[:3]
        summary_parts.append(f"Key gaps identified in: {', '.join(critical_missing)}{'...' if len(missing_skills) > 3 else ''}.")
    
    # Recommendation
    if score >= 70:
        summary_parts.append("Strong project-based technical foundation warrants immediate technical interview focusing on implementation depth and best practices.")
    elif score >= 50:
        summary_parts.append("Technical interview should validate project claims and assess proficiency in missing mandatory skills.")
    else:
        summary_parts.append("Extensive technical assessment recommended to evaluate skill gaps and learning potential before proceeding.")
    
    return " ".join(summary_parts)

def analyze_resume(resume_text: str, job_description: str, projects: List[Dict] = None) -> Dict:
    """
    Main resume analysis function - uses enhanced project-focused analysis.
    """
    try:
        return enhanced_technical_analysis(resume_text, job_description)
    except Exception as e:
        logger.error(f"Enhanced analysis failed, falling back to original: {e}")
        if projects is None:
            projects = extract_projects_from_resume(resume_text)
        return fallback_technical_score(resume_text, job_description, projects)

def fallback_technical_score(resume_text: str, job_description: str, projects: List[Dict]) -> Dict:
    """
    Fallback scoring logic if enhanced analysis fails.
    """
    score = 0
    pain_points_list = []
    matched_projects = []

    # Extract mandatory technical skills from JD
    mandatory_skills = extract_mandatory_skills_fallback(job_description)
    logger.debug(f"Extracted mandatory technical skills: {mandatory_skills}")

    # Match project skills to mandatory skills
    matched_tech_skills = []
    for project in projects:
        project_skills = project.get("skills", [])
        matched_skills = []
        
        for skill in mandatory_skills:
            for ps in project_skills:
                if (skill.lower() == ps.lower() or
                    (len(skill) > 1 and re.search(rf'\b{re.escape(skill)}\b', ps, re.IGNORECASE)) or
                    (len(ps) > 1 and re.search(rf'\b{re.escape(ps)}\b', skill, re.IGNORECASE))):
                    matched_skills.append(skill)
                    break
                elif skill.lower() == 'r' and re.search(r'\bR\b', ps, re.IGNORECASE):
                    matched_skills.append(skill)
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
    
    # Calculate score
    score = (len(matched_tech_skills) / max(len(mandatory_skills), 1)) * 100 if mandatory_skills else 0
    logger.debug(f"Technical skills score: {score}, matched_skills: {matched_tech_skills}")

    # Identify missing skills
    if len(matched_tech_skills) < len(mandatory_skills):
        missing_skills = [skill for skill in mandatory_skills if skill not in matched_tech_skills]
        pain_points_list.append(f"Missing mandatory technical skills: {', '.join(missing_skills)}")

    # Categorize pain points
    pain_points = {
        "critical": [p for p in pain_points_list if "Missing mandatory technical skills" in p],
        "major": [],
        "minor": ["Needs further evaluation in interview"] if not pain_points_list else []
    }

    # Determine status
    status = "Shortlisted" if score >= 70 else "Under Consideration" if score >= 50 else "Rejected"

    # Generate summary
    summary = (
        f"Candidate scored {score:.1f} based on alignment of project skills with mandatory technical requirements. "
        f"Projects cover {len(matched_tech_skills)}/{len(mandatory_skills)} mandatory skills. "
        f"{'Strong technical alignment; recommend interview.' if score >= 70 else 'Moderate technical fit; assess gaps in interview.' if score >= 50 else 'Significant technical gaps; may need training.'}"
    )

    result = {
        "score": int(score),
        "pain_points": pain_points,
        "summary": summary,
        "status": status,
        "projects": matched_projects
    }

    logger.info(f"Fallback technical assessment result: {result}")
    return result

def generate_technical_summary(score: int, pain_points: Dict) -> str:
    """
    Generate a technical-focused summary based on score and pain points.
    """
    summary_parts = []
    
    if score >= 70:
        summary_parts.append("Candidate demonstrates strong alignment with mandatory technical skills in project experience.")
    elif score >= 50:
        summary_parts.append("Candidate shows moderate alignment with some gaps in project-based mandatory technical skills.")
    else:
        summary_parts.append("Candidate exhibits significant gaps in project-based mandatory technical skills.")
    
    if pain_points.get("critical"):
        summary_parts.append(f"Critical issues: {'; '.join(pain_points['critical'][:2])}.")
    
    if pain_points.get("minor"):
        summary_parts.append(f"Minor improvements: {'; '.join(pain_points['minor'][:2])}.")
    
    summary_parts.append("Technical interview recommended to validate project skills and assess technical proficiency.")
    
    return " ".join(summary_parts)