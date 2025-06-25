from flask import Flask, request, render_template, jsonify, send_from_directory
import uuid
import os
import logging
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from document_parser import parse_document
from masking_agent import mask_text
from agents import analyze_resume
from pymongo import MongoClient
import google.generativeai as genai

app = Flask(__name__, static_folder='frontend/build/static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    filename="app_logs.txt",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded: GOOGLE_API_KEY present=%s", "GOOGLE_API_KEY" in os.environ)

# Initialize MongoDB connection
mongo_client = None
try:
    mongo_client = MongoClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=5000)
    db = mongo_client["ats_system"]
    resume_collection = db["resumes"]
    logger.info("MongoDB connection successful.")
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    # Continue without MongoDB instead of exiting

# Configure Google Gemini API
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash-lite-preview-06-17")
    logger.info("Google Gemini API configured successfully.")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    exit(1)

def store_resume_in_mongo(resume_id: str, masked_text: str, mappings: dict, collection_id: str) -> dict:
    if mongo_client is None:
        logger.warning("MongoDB not available, skipping storage")
        return {"resume_id": resume_id, "status": "skipped"}
    resume_data = {
        "resume_id": resume_id,
        "masked_text": masked_text,
        "pii_mappings": mappings,
        "pii_collection_id": collection_id,
    }
    try:
        resume_collection.insert_one(resume_data)
        logger.info(f"Stored resume with ID: {resume_id}")
        return resume_data
    except Exception as e:
        logger.error(f"Failed to store resume in MongoDB: {e}")
        return {"resume_id": resume_id, "status": "failed"}

def get_candidate_name(masked_text: str) -> str:
    try:
        prompt = (
            "You are an expert in resume analysis. The following text is a resume with sensitive information masked (e.g., [ADDRESS], [PHONE], [EMAIL]). "
            "Your task is to identify and extract only the candidate's full name (first and last name, and optionally middle name or initial). "
            "The name is typically found at the top of the resume or in a 'Name' field. "
            "Do not extract any other information, such as job titles, technical terms, or masked data. "
            "If no clear name is found, return 'Unknown Candidate'. "
            "Return only the name as a string, nothing else.\n\n"
            f"{masked_text[:2000]}"
        )
        response = model.generate_content(prompt)
        candidate_name = response.text.strip()
        if not candidate_name or len(candidate_name.split()) < 2 or len(candidate_name) > 30:
            logger.warning("No valid candidate name found by LLM, using 'Unknown Candidate'")
            return "Unknown Candidate"
        logger.info(f"Candidate name extracted via LLM: {candidate_name}")
        return candidate_name
    except Exception as e:
        logger.error(f"Error extracting name with LLM: {e}")
        return "Unknown Candidate"

@app.route('/')
def index():
    logger.info("Attempting to render index.html")
    template_path = os.path.join(app.template_folder, 'index.html')
    if not os.path.exists(template_path):
        logger.error(f"Template not found at: {template_path}")
        return jsonify({"error": "Template index.html not found"}), 500
    
    js_files = []
    css_files = []
    static_js_path = os.path.join(app.static_folder, 'js')
    static_css_path = os.path.join(app.static_folder, 'css')
    
    if os.path.exists(static_js_path):
        js_files = [f for f in os.listdir(static_js_path) if f.endswith('.js')]
        if not js_files:
            logger.warning("No .js files found in static/js, falling back to default")
            js_files = ['main.js']
    if os.path.exists(static_css_path):
        css_files = [f for f in os.listdir(static_css_path) if f.endswith('.css')]
    
    if not js_files:
        logger.error("No JavaScript files found in static/js")
        return jsonify({"error": "No JavaScript files found"}), 500
    
    import time
    cache_buster = int(time.time())
    js_files = [f"{f}?v={cache_buster}" for f in js_files]
    css_files = [f"{f}?v={cache_buster}" for f in css_files]
    
    return render_template('index.html', js_files=js_files, css_files=css_files)

@app.route('/analyze', methods=['POST'])
def analyze_resumes():
    try:
        if 'resumes' not in request.files or (not request.form.get('job_description') and 'job_description_file' not in request.files):
            return jsonify({"error": "Missing resumes or job description (text or file)"}), 400

        resumes = request.files.getlist('resumes')
        job_description_file = request.files.get('job_description_file')
        job_description_text = request.form.get('job_description', '')

        # Process job description
        if job_description_file and job_description_file.filename.endswith(('.pdf', '.docx')):
            filename = secure_filename(job_description_file.filename)
            job_description_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            job_description_file.save(job_description_path)
            logger.info(f"Processing job description file: {filename}")
            
            with open(job_description_path, 'rb') as f:
                jd_data = parse_document(f, filename)
            job_description = jd_data["full_text"]
            job_description, _, _ = mask_text(job_description)
            logger.info("Job description extracted and masked from uploaded file")
        else:
            job_description = job_description_text
            if job_description.strip():
                job_description, _, _ = mask_text(job_description)
                logger.info("Job description used from text input and masked")
            else:
                return jsonify({"error": "No valid job description provided"}), 400

        results = []

        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        for resume in resumes:
            if resume and resume.filename.endswith(('.pdf', '.docx')):
                filename = secure_filename(resume.filename)
                resume_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                resume.save(resume_path)
                logger.info(f"Processing resume: {filename}")
                
                with open(resume_path, 'rb') as f:
                    resume_data = parse_document(f, filename)
                    resume_text = resume_data["full_text"]
                    projects = resume_data["projects"]
                masked_text, mappings, collection_id = mask_text(resume_text)
                candidate_name = get_candidate_name(masked_text)
                resume_id = str(uuid.uuid4())
                store_resume_in_mongo(resume_id, masked_text, mappings, collection_id)  # Non-critical operation
                result = analyze_resume(masked_text, job_description, projects)
                
                results.append({
                    "resume_name": filename,
                    "candidate_name": candidate_name,
                    "score": result["score"],
                    "pain_points": result["pain_points"],
                    "summary": result["summary"],
                    "status": result["status"],
                    "projects": result["projects"],
                    "resume_path": f"/uploads/{filename}"
                })

        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in analyze_resumes: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/uploads/<filename>')
def download_resume(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/static/<path:path>')
def serve_static(path):
    response = send_from_directory(app.static_folder, path)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    logger.info(f"Serving static file: {path}")
    return response

if __name__ == '__main__':
    app.run(debug=True)