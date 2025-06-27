ATS Resume Analysis System
The ATS Resume Analysis System is a web-based application designed to streamline the recruitment process by analyzing resumes against job descriptions. It leverages AI to extract candidate information, mask sensitive data, evaluate technical and proficiency scores, and provide detailed insights into candidate suitability. The system supports PDF and DOCX file formats for both resumes and job descriptions, ensuring compatibility with common document types. It uses Azure OpenAI for advanced natural language processing and MongoDB for storing resume data securely.
Features

Resume Parsing: Extracts structured content (e.g., projects, skills, experience) from PDF or DOCX resumes.
PII Masking: Identifies and masks personally identifiable information (PII) such as addresses, phone numbers, and emails using Presidio and regex patterns, storing mappings in MongoDB.
Candidate Scoring:
Technical Score (0-100): Measures alignment of resume skills with job description requirements.
Proficiency Score (0-30): Evaluates resume quality, experience alignment, soft skills, certifications, and nice-to-have skills.


Strengths and Weaknesses Analysis: Provides detailed feedback on candidate strengths (e.g., matched skills, relevant projects) and weaknesses (e.g., missing skills, experience gaps).
Job Description Processing: Supports job description input via text or file upload, extracting mandatory and nice-to-have skills.
Frontend Interface: A React-based UI with drag-and-drop file uploads, real-time analysis feedback, and sortable results table.
Secure Storage: Stores masked resume data and PII mappings in MongoDB.
Rate Limiting: Implements rate limiting for Azure OpenAI API calls to ensure compliance with usage quotas.

Architecture
The application follows a client-server architecture:

Frontend: Built with React and Tailwind CSS, served from frontend/build/static. The main component (App.jsx) handles file uploads, job description input, and result visualization.
Backend: A Flask-based server (app.py) processes file uploads, parses documents, masks PII, and performs AI-driven analysis using Azure OpenAI.
Database: MongoDB stores masked resume data and PII mappings.
External Services: Azure OpenAI for natural language processing tasks (e.g., skill extraction, resume quality evaluation).


Key Files

app.py:

Main Flask application.
Routes:
/: Serves the React frontend (index.html).
/analyze: Processes resume and job description files/text, performs analysis, and returns results.
/uploads/<filename>: Serves uploaded files for download.
/static/<path>: Serves static assets with cache control.


Integrates document parsing, PII masking, MongoDB storage, and Azure OpenAI analysis.


pii_store_mongo.py:

Handles MongoDB interactions for storing PII mappings.
Functions:
store_mapping_with_id: Stores masked and original PII values.
does_collection_id_exist: Checks for existing collection IDs.



document_parser.py:

Parses PDF (PyPDF2) and DOCX (docx2txt) files.
Extracts full text and structured project data (name, description, skills).
Uses regex to identify sections (e.g., Projects, Skills, Experience).


masking_agent.py:

Implements PIIMasker class using Presidio for address detection and regex for phone numbers/emails.
Generates unique masked values (e.g., <ADDRESS_1234>) and stores mappings in MongoDB.


agents.py:

Core analysis logic for resume evaluation.
Functions:
extract_projects_from_resume: Extracts project details using Azure OpenAI or regex fallback.
extract_mandatory_skills_from_jd: Identifies required skills from job descriptions.
extract_nice_to_have_skills_from_jd: Extracts optional skills.
extract_soft_skills and extract_certifications: Extract non-technical skills and certifications.
calculate_technical_score_from_projects: Scores resume based on skill matches.
calculate_proficiency_score: Evaluates resume quality, experience, and additional qualifications.
enhanced_technical_analysis: Combines all analyses for comprehensive scoring.
fallback_technical_score: Fallback method using regex if LLM fails.


Uses rate limiting to manage Azure OpenAI API calls.


App.jsx:

React frontend component.
Features:
Drag-and-drop file uploads for resumes and job descriptions.
Textarea for job description input.
Analysis results table with sortable technical scores.
Toggleable strengths/weaknesses and project details.
Responsive design with Tailwind CSS and Lucide icons.



requirements.txt:

Lists Python dependencies (e.g., Flask, PyMongo, PyPDF2, Presidio, Azure OpenAI SDK).



Setup Instructions
Prerequisites

Python 3.8+
Node.js 14+ (for React frontend)
MongoDB (local or cloud instance, e.g., MongoDB Atlas)
Azure OpenAI API credentials
spaCy model: en_core_web_sm (for Presidio PII detection)

Installation

Clone the Repository:
git clone <repository-url>
cd ATS-Resume-Analysis


Set Up Python Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Install spaCy Model:
python -m spacy download en_core_web_sm


Set Up Environment Variables:Create a .env file in the root directory:
MONGO_URI=mongodb://localhost:27017
AZURE_OPENAI_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_VERSION=your_api_version
AZURE_OPENAI_MODEL_NAME=your_model_name


Build the Frontend:
cd frontend
npm install
npm run build
cd ..


Run the Application:
python app.py

The app will be available at http://localhost:5000.


MongoDB Setup

Ensure MongoDB is running locally or use a cloud service like MongoDB Atlas.
The database ats_system will be created automatically with collections:
resumes: Stores masked resume data and PII mappings.
pii_mappings: Stores individual PII mappings with collection IDs.



Azure OpenAI Configuration

Obtain credentials from Azure OpenAI service.
Update .env with your API key, endpoint, version, and model name (e.g., gpt-4).

Usage

Access the Web Interface:

Open http://localhost:5000 in a browser.
Upload one or more resumes (PDF/DOCX).
Enter a job description via text or upload a PDF/DOCX file.
Click "Analyze Resumes" to process.


View Results:

Results are displayed in a table with:
Candidate name (extracted or "Unknown Candidate").
Technical score (0-100).
Proficiency score (0-30).
Strengths/weaknesses (toggleable).
Project details (toggleable).
Download link for original resume.


Sort results by technical score using the "Sort by Technical Score" button.


Candidate Status:

Shortlisted: Technical score ≥ 80, proficiency score ≥ 20, ≤ 1 missing skill.
Under Consideration: Technical score ≥ 60, proficiency score ≥ 15.
Rejected: Below thresholds.



Technical Details
PII Masking

Uses Presidio for address detection and regex for phone numbers (Indian format) and emails.
Stores PII mappings in MongoDB with unique collection IDs.
Masked values (e.g., <PHONE_1234>) ensure privacy while maintaining analysis integrity.

Scoring Mechanism

Technical Score:
Based on the percentage of mandatory skills matched in resume projects.
Uses fuzzy matching (via fuzzywuzzy) for skill comparison.


Proficiency Score:
Nice-to-have skills (0-8 points).
Resume quality (clarity, relevance, complexity; 0-8 points).
Experience alignment (years, role, industry; 0-8 points).
Soft skills (0-3 points).
Certifications (0-3 points).



Error Handling

Logs errors to logs.txt for debugging.
Fallback methods (regex-based) for skill and project extraction if LLM fails.
Input validation for file types and job description presence.

Security

PII is masked before analysis and storage.
MongoDB stores only masked resume text and PII mappings.
File uploads are secured with secure_filename and size limits (16MB).

Limitations

File Formats: Only PDF and DOCX are supported.
Language: Optimized for English resumes and job descriptions.
API Dependency: Requires Azure OpenAI, which may incur costs and rate limits.
PII Detection: Limited to addresses, phone numbers (Indian format), and emails.
Resume Structure: Assumes standard resume formats; unconventional layouts may affect parsing accuracy.

Future Improvements

Support additional file formats (e.g., plain text, RTF).
Expand PII detection to include more entities (e.g., names, dates).
Add multi-language support for non-English resumes.
Implement caching for faster analysis of previously processed resumes.
Enhance frontend with advanced visualizations (e.g., skill match charts).

Contributing

Fork the repository.
Create a feature branch (git checkout -b feature/YourFeature).
Commit changes (git commit -m "Add YourFeature").
Push to the branch (git push origin feature/YourFeature).
Open a pull request.

Contact
For issues or suggestions, please open an issue on GitHub or contact the repository owner.