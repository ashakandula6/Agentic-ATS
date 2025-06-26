import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, FileText, BarChart3, Download, Target, Eye, Trash2 } from 'lucide-react';

function App() {
  const [resumes, setResumes] = useState([]);
  const [jobDescription, setJobDescription] = useState('');
  const [jobDescriptionFile, setJobDescriptionFile] = useState(null);
  const [results, setResults] = useState([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const [visibleContent, setVisibleContent] = useState({});
  const [analyzeClicked, setAnalyzeClicked] = useState(false);
  const [isResumeDragOver, setIsResumeDragOver] = useState(false);
  const [isJDUploadDragOver, setIsJDUploadDragOver] = useState(false);
  const jdInputRef = useRef(null);

  const handleFileChange = (e) => {
    const newFiles = Array.from(e.target.files);
    const uniqueNewFiles = newFiles.filter(
      (newFile) => !resumes.some((existingFile) => existingFile.name === newFile.name)
    );
    setResumes((prev) => [...prev, ...uniqueNewFiles]);
    e.target.value = '';
  };

  const handleJDFileChange = (e) => {
    console.log('JD file input changed:', e.target.files);
    const file = e.target.files[0];
    if (!file) {
      setError('No file selected.');
      return;
    }
    const extension = file.name.split('.').pop().toLowerCase();
    if (extension === 'pdf' || extension === 'docx') {
      setJobDescriptionFile(file);
      setError(null);
      e.target.value = '';
      if (jdInputRef.current) {
        jdInputRef.current.value = '';
      }
    } else {
      setError('Please upload a valid PDF or DOCX file for the job description.');
    }
  };

  const removeJDFile = () => {
    setJobDescriptionFile(null);
    if (jdInputRef.current) {
      jdInputRef.current.value = '';
    }
  };

  const handleResumeDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsResumeDragOver(true);
  };

  const handleResumeDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsResumeDragOver(false);
  };

  const handleResumeDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsResumeDragOver(false);
    const droppedFiles = Array.from(e.dataTransfer.files);
    const validFiles = droppedFiles.filter(
      (file) => {
        const extension = file.name.split('.').pop().toLowerCase();
        return extension === 'pdf' || extension === 'docx';
      }
    );
    const uniqueValidFiles = validFiles.filter(
      (newFile) => !resumes.some((existingFile) => existingFile.name === newFile.name)
    );
    setResumes((prev) => [...prev, ...uniqueValidFiles]);
  };

  const handleJDDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsJDUploadDragOver(true);
  };

  const handleJDDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsJDUploadDragOver(false);
  };

  const handleJDDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsJDUploadDragOver(false);
    console.log('JD files dropped:', e.dataTransfer.files);
    const droppedFiles = Array.from(e.dataTransfer.files);
    const validFiles = droppedFiles.filter(
      (file) => {
        const extension = file.name.split('.').pop().toLowerCase();
        return extension === 'pdf' || extension === 'docx';
      }
    );
    if (validFiles.length > 0) {
      setJobDescriptionFile({ ...validFiles[0] });
      setError(null);
      if (jdInputRef.current) {
        jdInputRef.current.value = '';
      }
    } else {
      setError('Please drop a valid PDF or DOCX file for the job description.');
    }
  };

  const removeFile = (fileName) => {
    setResumes((prev) => prev.filter((file) => file.name !== fileName));
  };

  const handleAnalyze = async () => {
    setAnalyzeClicked(true);
    if (resumes.length === 0 || (!jobDescription.trim() && !jobDescriptionFile)) {
      if (resumes.length === 0) {
        setError('Please upload at least one resume file.');
      } else {
        setError('Please provide a job description via text or file upload.');
      }
      return;
    }
    setError(null);
    setAnalyzing(true);
    setResults([]);
    setVisibleContent({});

    try {
      const formData = new FormData();
      resumes.forEach((file) => {
        formData.append('resumes', file);
      });
      if (jobDescriptionFile) {
        formData.append('job_description_file', jobDescriptionFile);
      } else {
        formData.append('job_description', jobDescription);
      }

      const response = await axios.post('/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const results = response.data;
      setResults(results);
      const initialVisibility = {};
      results.forEach((_, index) => {
        initialVisibility[`strengths-weaknesses-${index}`] = false;
        initialVisibility[`projects-${index}`] = false;
      });
      setVisibleContent(initialVisibility);
    } catch (err) {
      setError('An error occurred during analysis. Please try again.');
      console.error('Analysis error:', err);
    } finally {
      setAnalyzing(false);
    }
  };

  const toggleContent = (id) => {
    setVisibleContent((prev) => ({
      ...prev,
      [id]: !prev[id] || false,
    }));
  };

  const getStatusColor = (status) => {
    switch (status.toLowerCase()) {
      case 'shortlisted':
        return 'bg-green-100 text-green-800';
      case 'under consideration':
        return 'bg-yellow-100 text-yellow-800';
      case 'rejected':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getScoreColor = (score) => {
    if (score >= 70) return 'text-green-600 bg-green-50';
    else if (score >= 50) return 'text-yellow-600 bg-yellow-50';
    else return 'text-red-600 bg-red-50';
  };

  const getProficiencyColor = (score) => {
    if (score >= 20) return 'text-green-600 bg-green-50';
    else if (score >= 10) return 'text-yellow-600 bg-yellow-50';
    else return 'text-red-600 bg-red-50';
  };

  const sortByScore = () => {
    setResults((prevResults) => {
      const sortedResults = [...prevResults].sort((a, b) => b.technical_score - a.technical_score);
      const newVisibleContent = {};
      sortedResults.forEach((_, index) => {
        newVisibleContent[`strengths-weaknesses-${index}`] = visibleContent[`strengths-weaknesses-${index}`] || false;
        newVisibleContent[`projects-${index}`] = visibleContent[`projects-${index}`] || false;
      });
      setVisibleContent(newVisibleContent);
      return sortedResults;
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-white to-blue-50 p-4">
      <div className="max-w-7xl mx-auto space-y-8">
        <div className="text-center space-y-2">
          <div className="flex items-center justify-center gap-2 mb-4">
            <div className="w-8 h-8 bg-red-500 rounded-full flex items-center justify-center">
              <Target className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-blue-600">ATS Resume Analysis System</h1>
          </div>
          <p className="text-lg text-gray-600 font-medium">
            Streamline Your Recruitment Process with AI-Powered Resume Analysis
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-lg border border-blue-100 overflow-hidden">
              <div className="bg-gradient-to-r from-blue-50 to-blue-100 px-6 py-4 border-b border-blue-200">
                <div className="flex items-center gap-2">
                  <FileText className="w-5 h-5 text-blue-600" />
                  <h2 className="text-xl font-bold text-blue-600">Upload Resumes</h2>
                </div>
              </div>
              <div className="p-6">
                <div
                  className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors duration-300 relative ${
                    isResumeDragOver 
                      ? "border-blue-400 bg-blue-50" 
                      : "border-blue-300 bg-gradient-to-br from-blue-50 to-blue-100 hover:border-blue-400 hover:bg-blue-50"
                  }`}
                  onDragOver={handleResumeDragOver}
                  onDragLeave={handleResumeDragLeave}
                  onDrop={handleResumeDrop}
                >
                  <input
                    type="file"
                    multiple
                    accept=".pdf,.docx"
                    onChange={handleFileChange}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
                    id="resume-upload"
                    name="resume-upload"
                  />
                  <Upload className="w-8 h-8 mx-auto mb-4 text-blue-600" />
                  <p className="text-lg font-semibold text-blue-600 mb-2">
                    Click to upload or drag and drop files here
                  </p>
                  <p className="text-sm text-gray-600">
                    Upload multiple PDF or DOCX resume files
                  </p>
                </div>

                {resumes.length > 0 && (
                  <div className="mt-4 bg-green-50 border border-green-200 rounded-xl p-4">
                    <p className="text-green-800 font-semibold mb-2 flex items-center gap-2">
                      <div className="w-5 h-5 bg-green-500 rounded-full flex items-center justify-center">
                        <span className="text-white text-xs">✓</span>
                      </div>
                      {resumes.length} file(s) uploaded successfully!
                    </p>
                    <div className="space-y-2">
                      {resumes.map((file, index) => (
                        <div key={file.name} className="flex items-center justify-between gap-2 text-sm text-gray-700">
                          <div className="flex items-center gap-2">
                            <FileText className="w-4 h-4 text-blue-600" />
                            <span className="font-medium">{index + 1}.</span>
                            <span>{file.name}</span>
                            <span className="text-gray-500">({(file.size / 1024).toFixed(0)} KB)</span>
                          </div>
                          <button
                            onClick={() => removeFile(file.name)}
                            className="text-red-600 hover:text-red-800 transition-colors"
                            title="Remove file"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-lg border border-blue-100 overflow-hidden">
              <div className="bg-gradient-to-r from-orange-50 to-orange-100 px-6 py-4 border-b border-orange-200">
                <div className="flex items-center gap-2">
                  <div className="w-5 h-5 bg-orange-500 rounded flex items-center justify-center">
                    <FileText className="w-3 h-3 text-white" />
                  </div>
                  <h2 className="text-xl font-bold text-orange-600">Job Requirements</h2>
                </div>
              </div>
              <div className="p-6 space-y-6">
                <div>
                  <label className="block text-gray-700 font-semibold mb-2">
                    Job Description
                  </label>
                  <textarea
                    value={jobDescription}
                    onChange={(e) => setJobDescription(e.target.value)}
                    placeholder="Enter detailed job description including:
• Required skills and technologies
• Key responsibilities  
• Required years of experience (e.g., '3+ years of experience')
• Educational qualifications
• Preferred certifications
• Company culture fit criteria..."
                    className="w-full h-72 bg-white border-2 border-blue-200 rounded-lg p-3 text-gray-700 focus:border-blue-400 focus:ring-2 focus:ring-blue-200 focus:outline-none resize-none"
                  />
                </div>

                <div className="bg-white rounded-lg p-4">
                  <label className="block text-gray-700 font-semibold mb-2">
                    Upload Job Description (PDF or DOCX)
                  </label>
                  <div
                    className={`border-2 border-dashed rounded-xl p-4 text-center transition-colors duration-300 relative ${
                      isJDUploadDragOver 
                        ? "border-blue-400 bg-blue-50" 
                        : "border-blue-300 bg-gradient-to-br from-blue-50 to-blue-100 hover:border-blue-400 hover:bg-blue-50"
                    }`}
                    onDragOver={handleJDDragOver}
                    onDragLeave={handleJDDragLeave}
                    onDrop={handleJDDrop}
                  >
                    <input
                      type="file"
                      accept=".pdf,.docx"
                      onChange={handleJDFileChange}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
                      id="jd-upload"
                      name="jd-upload"
                      ref={jdInputRef}
                    />
                    <Upload className="w-6 h-6 mx-auto mb-2 text-blue-600" />
                    <p className="text-sm font-semibold text-blue-600 mb-1">
                      Click to upload or drag and drop a job description file
                    </p>
                    <p className="text-xs text-gray-600">
                      Supports PDF or DOCX files
                    </p>
                  </div>
                  {jobDescriptionFile && (
                    <div className="mt-4 bg-green-50 border border-green-200 rounded-xl p-4">
                      <p className="text-green-800 font-semibold mb-2 flex items-center gap-2">
                        <div className="w-5 h-5 bg-green-500 rounded-full flex items-center justify-center">
                          <span className="text-white text-xs">✓</span>
                        </div>
                        Job description file uploaded successfully!
                      </p>
                      <div className="flex items-center justify-between gap-2 text-sm text-gray-700">
                        <div className="flex items-center gap-2">
                          <FileText className="w-4 h-4 text-blue-600" />
                          <span>{jobDescriptionFile.name}</span>
                          <span className="text-gray-500">({(jobDescriptionFile.size / 1024).toFixed(0)} KB)</span>
                        </div>
                        <button
                          onClick={removeJDFile}
                          className="text-red-600 hover:text-red-800 transition-colors"
                          title="Remove file"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                <button
                  onClick={handleAnalyze}
                  disabled={analyzing}
                  className={`w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white font-bold py-3 px-6 rounded-lg shadow-lg hover:from-blue-700 hover:to-blue-800 hover:shadow-xl hover:-translate-y-1 transition-all duration-300 flex items-center justify-center gap-2 ${
                    analyzing ? 'opacity-70 cursor-not-allowed' : ''
                  }`}
                >
                  {analyzing ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <BarChart3 className="w-4 h-4" />
                      ANALYZE RESUMES
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
            <div className="flex items-center gap-2">
              <span className="text-red-500">❌</span>
              {error}
            </div>
          </div>
        )}

        <div className="space-y-6">
          <div className="bg-gradient-to-r from-blue-50 to-blue-100 border-2 border-blue-200 rounded-xl p-6">
            <div className="text-center">
              <div className="flex items-center justify-center gap-2 mb-2">
                <BarChart3 className="w-6 h-6 text-blue-600" />
                <h2 className="text-2xl font-bold text-blue-700">Analysis Results</h2>
              </div>
              <p className="text-blue-600">Comprehensive resume analysis and candidate ranking</p>
            </div>
          </div>

          {results.length > 0 ? (
            <div className="bg-white rounded-xl shadow-lg border border-blue-100 overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-blue-50 border-b border-blue-200">
                      <th className="px-6 py-4 text-left text-sm font-bold text-blue-700">Name</th>
                      <th className="px-6 py-4 text-left text-sm font-bold text-blue-700">Technical Score /100</th>
                      <th className="px-6 py-4 text-left text-sm font-bold text-blue-700">Proficiency Score /30</th>
                      <th className="px-6 py-4 text-left text-sm font-bold text-blue-700">Strengths/Weakness</th>
                      <th className="px-6 py-4 text-left text-sm font-bold text-blue-700">Projects</th>
                      <th className="px-6 py-4 text-left text-sm font-bold text-blue-700">Status</th>
                      <th className="px-6 py-4 text-left text-sm font-bold text-blue-700">Resume</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((result, index) => (
                      <tr key={index} className="border-b border-blue-100 hover:bg-blue-50 transition-colors">
                        <td className="px-6 py-4 font-medium text-gray-900">{result.candidate_name}</td>
                        <td className="px-6 py-4">
                          <span className={`px-3 py-1 rounded-full text-sm font-bold ${getScoreColor(result.technical_score)}`}>
                            {result.technical_score}%
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          <span className={`px-3 py-1 rounded-full text-sm font-bold ${getProficiencyColor(result.proficiency_score)}`}>
                            {result.proficiency_score}/30
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          <button
                            className="bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-lg px-3 py-1 text-sm hover:from-blue-700 hover:to-blue-800 transition-all duration-300 flex items-center gap-1"
                            onClick={() => toggleContent(`strengths-weaknesses-${index}`)}
                          >
                            <Eye className="w-3 h-3" />
                            View
                          </button>
                          {visibleContent[`strengths-weaknesses-${index}`] && (
                            <div className="mt-2 bg-gray-50 rounded-lg p-3">
                              <div className="text-sm text-gray-700 space-y-2">
                                <div>
                                  <strong className="text-green-600">Strengths:</strong>
                                  <ul className="list-disc pl-5">
                                    {result.strengths_weaknesses.strengths.length > 0 ? (
                                      result.strengths_weaknesses.strengths.map((strength, i) => (
                                        <li key={i}>{strength}</li>
                                      ))
                                    ) : (
                                      <li>No strengths identified.</li>
                                    )}
                                  </ul>
                                </div>
                                <div>
                                  <strong className="text-red-600">Weaknesses:</strong>
                                  <ul className="list-disc pl-5">
                                    {result.strengths_weaknesses.weaknesses.length > 0 ? (
                                      result.strengths_weaknesses.weaknesses.map((weakness, i) => (
                                        <li key={i}>{weakness}</li>
                                      ))
                                    ) : (
                                      <li>No weaknesses identified.</li>
                                    )}
                                  </ul>
                                </div>
                              </div>
                            </div>
                          )}
                        </td>
                        <td className="px-6 py-4">
                          <button
                            className="bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-lg px-3 py-1 text-sm hover:from-blue-700 hover:to-blue-800 transition-all duration-300 flex items-center gap-1"
                            onClick={() => toggleContent(`projects-${index}`)}
                          >
                            <Eye className="w-3 h-3" />
                            View
                          </button>
                          {visibleContent[`projects-${index}`] && (
                            <div className="mt-2 bg-gray-50 rounded-lg p-3">
                              <div className="text-sm text-gray-700 space-y-2">
                                {result.projects.length > 0 ? (
                                  result.projects.map((project, i) => (
                                    <div key={i}>
                                      <strong className="text-blue-600">{project.name}</strong>
                                      <p><strong>Description:</strong> {project.description}</p>
                                      <p><strong>Skills:</strong> {project.skills.join(", ") || "None listed"}</p>
                                      <p><strong>Relevance:</strong> {project.relevance}</p>
                                    </div>
                                  ))
                                ) : (
                                  <p>No projects identified in the resume.</p>
                                )}
                              </div>
                            </div>
                          )}
                        </td>
                        <td className="px-6 py-4">
                          <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(result.status)}`}>
                            {result.status}
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          <a
                            href={`/uploads/${encodeURIComponent(result.resume_name)}`}
                            download
                            className="text-blue-600 font-semibold hover:text-blue-800 hover:underline flex items-center gap-1"
                          >
                            <Download className="w-3 h-3" />
                            Download
                          </a>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="p-6">
                <button
                  onClick={sortByScore}
                  className="bg-gradient-to-r from-blue-600 to-blue-700 text-white font-bold py-3 px-6 rounded-lg shadow-lg hover:from-blue-700 hover:to-blue-800 hover:shadow-xl hover:-translate-y-1 transition-all duration-300 flex items-center justify-center gap-2"
                >
                  <BarChart3 className="w-4 h-4" />
                  Sort by Score (High to Low)
                </button>
              </div>
            </div>
          ) : (
            <div className="bg-white border-2 border-dashed border-blue-300 rounded-xl p-12">
              <div className="text-center">
                <div className="flex items-center justify-center gap-2 mb-4">
                  <FileText className="w-8 h-8 text-gray-400" />
                  <h3 className="text-xl font-semibold text-gray-700">Detailed Results Table</h3>
                </div>
                <p className="text-gray-600">
                  {analyzeClicked
                    ? "No results to display. Please ensure valid resumes and job description are provided."
                    : 'Upload resumes and click "Analyze" to see comprehensive candidate scoring and ranking'}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;