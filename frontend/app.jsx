// File: src/app.jsx
import { useState, useRef } from 'react';
// import LossChart from '/components/Chart';
import { useEffect } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import axios from 'axios';
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

export default function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [techData, setTechData] = useState([]);
  const [inputs, setInputs] = useState({
    scope: 0.5,
    prospects: 0.5,
    potential: 0.5,
    economy: 0.5,
    efficiency: 0.5
  });
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [techComparison, setTechComparison] = useState([]);
  const [file, setFile] = useState(null);
  const [fileData, setFileData] = useState(null);
  const fileInputRef = useRef(null);

  const sampleTechData = [
    { name: 'AI', criticality: 0.85, status: 'Critical' },
    { name: 'Blockchain', criticality: 0.72, status: 'Critical' },
    { name: 'VR', criticality: 0.63, status: 'Critical' },
    { name: '5G', criticality: 0.58, status: 'Critical' },
    { name: 'IoT', criticality: 0.45, status: 'Non-Critical' }
  ];

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    
    if (selectedFile) {
      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const data = JSON.parse(event.target.result);
          setFileData(data);
        } catch (error) {
          toast.error('File format is invalid');
        }
      };
      reader.readAsText(selectedFile);
    }
  };

  const handleInputChange = (key, value) => {
    setInputs(prev => ({
      ...prev,
      [key]: parseFloat(value)
    }));
  };

  const handlePredict = async () => {
    setIsLoading(true);
    try {
      const response = await axios.post(`${API_URL}/predict`, inputs);
      setResult(response.data);
      toast.success('Prediction successful!');
    } catch (error) {
      console.error('Prediction error:', error);
      toast.error(error.response?.data?.error || 'Prediction failed');
    } finally {
      setIsLoading(false);
    }
  };

  const analyzeFileData = async () => {
    if (!fileData) {
      toast.error('Please upload a file first');
      return;
    }

    setIsLoading(true);
    try {
      const response = await axios.post(`${API_URL}/predict`, fileData);
      setResult(response.data);
      toast.success('Analysis successful!');
    } catch (error) {
      console.error('Analysis error:', error);
      setResult({ criticality: 0, status: 'Error' });
      toast.error(error.response?.data?.error || 'Analysis failed');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    const fetchTechData = async () => {
      try {
        const response = await fetch(`${API_URL}/analysis`);
        const data = await response.json();
        setTechData(data);
        setTechComparison(sampleTechData);
      } catch (error) {
        toast.error('Failed to load technology data');
      }
    };
    
    fetchTechData();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      <ToastContainer position="top-right" autoClose={3000} />

      {/* Modern Navigation */}
      <nav className="bg-white shadow-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-20 items-center">
            <div className="flex items-center space-x-2">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">TechCritical</span>
            </div>
            <div className="hidden md:flex space-x-8">
              <button
                onClick={() => setActiveTab('home')}
                className={`py-2 px-1 font-medium text-sm transition-colors ${activeTab === 'home' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-blue-600'}`}
              >
                Home
              </button>
              <button
                onClick={() => setActiveTab('predictor')}
                className={`py-2 px-1 font-medium text-sm transition-colors ${activeTab === 'predictor' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-blue-600'}`}
              >
                Predictor
              </button>
              <button
                onClick={() => setActiveTab('analysis')}
                className={`py-2 px-1 font-medium text-sm transition-colors ${activeTab === 'analysis' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-blue-600'}`}
              >
                Analysis
              </button>
              <button
                onClick={() => setActiveTab('about')}
                className={`py-2 px-1 font-medium text-sm transition-colors ${activeTab === 'about' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-blue-600'}`}
              >
                About
              </button>
            </div>
            <button className="md:hidden text-gray-500">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>
      </nav>

      <main>
        {activeTab === 'home' ? (
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
            {/* Hero Section */}
            <div className="text-center py-16">
              <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
                Advanced Technology Criticality Prediction with <span className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">ANFIS</span>
              </h1>
              <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-10">
                Our system evaluates technology criticality using Adaptive Neuro-Fuzzy Inference System for accurate and reliable predictions.
              </p>
              <div className="flex justify-center gap-4">
                <button 
                  onClick={() => setActiveTab('predictor')}
                  className="px-8 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg font-medium shadow-lg hover:shadow-xl transition-all hover:-translate-y-1"
                >
                  Try Predictor
                </button>
                <button 
                  onClick={() => setActiveTab('analysis')}
                  className="px-8 py-3 bg-white text-blue-600 border border-blue-600 rounded-lg font-medium shadow hover:shadow-md transition-all"
                >
                  View Analysis
                </button>
              </div>
            </div>

            {/* Features Section */}
            <div className="py-16">
              <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">Key Features</h2>
              <div className="grid md:grid-cols-3 gap-8">
                <div className="bg-white p-8 rounded-xl shadow-md hover:shadow-lg transition-shadow">
                  <div className="w-14 h-14 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-semibold mb-3">ANFIS Technology</h3>
                  <p className="text-gray-600">Combines neural networks and fuzzy logic for superior prediction accuracy in technology assessment.</p>
                </div>
                <div className="bg-white p-8 rounded-xl shadow-md hover:shadow-lg transition-shadow">
                  <div className="w-14 h-14 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-semibold mb-3">Multi-Dimensional Analysis</h3>
                  <p className="text-gray-600">Evaluates technology across five critical dimensions: scope, prospects, potential, economy, and efficiency.</p>
                </div>
                <div className="bg-white p-8 rounded-xl shadow-md hover:shadow-lg transition-shadow">
                  <div className="w-14 h-14 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-semibold mb-3">Data Visualization</h3>
                  <p className="text-gray-600">Interactive charts and graphs to help you understand technology criticality at a glance.</p>
                </div>
              </div>
            </div>

            {/* How It Works */}
            <div className="py-16 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl px-8">
              <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">How It Works</h2>
              <div className="max-w-4xl mx-auto">
                <div className="relative">
                  {/* Timeline */}
                  <div className="hidden sm:block absolute h-full w-0.5 bg-blue-200 left-1/2 transform -translate-x-1/2"></div>
                  
                  {/* Steps */}
                  <div className="space-y-8 sm:space-y-16">
                    {[
                      {
                        title: "Input Parameters",
                        description: "Adjust the sliders for each critical dimension or upload your JSON data file",
                        icon: (
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z" />
                          </svg>
                        )
                      },
                      {
                        title: "ANFIS Processing",
                        description: "Our system processes your inputs through trained ANFIS models",
                        icon: (
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                          </svg>
                        )
                      },
                      {
                        title: "Get Results",
                        description: "Receive detailed criticality analysis with visualizations",
                        icon: (
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                          </svg>
                        )
                      }
                    ].map((step, index) => (
                      <div key={index} className={`relative flex flex-col sm:flex-row items-center ${index % 2 === 0 ? 'sm:text-right' : 'sm:text-left'}`}>
                        <div className={`sm:absolute mx-auto sm:mx-0 w-12 h-12 rounded-full bg-white border-4 border-blue-500 flex items-center justify-center text-blue-500 z-10 ${index % 2 === 0 ? 'sm:left-1/2 sm:ml-6' : 'sm:right-1/2 sm:mr-6'}`}>
                          {step.icon}
                        </div>
                        <div className={`mt-4 sm:mt-0 sm:w-5/12 ${index % 2 === 0 ? 'sm:mr-auto sm:pr-8' : 'sm:ml-auto sm:pl-8'}`}>
                          <h3 className="text-xl font-semibold text-gray-900">{step.title}</h3>
                          <p className="text-gray-600 mt-2">{step.description}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Call to Action */}
            <div className="py-16 text-center">
              <h2 className="text-3xl font-bold text-gray-900 mb-6">Ready to Analyze Your Technology?</h2>
              <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">Get started with our ANFIS-based criticality prediction system today.</p>
              <button 
                onClick={() => setActiveTab('predictor')}
                className="px-8 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg font-medium shadow-lg hover:shadow-xl transition-all hover:-translate-y-1"
              >
                Start Predicting Now
              </button>
            </div>
          </div>
        ) : activeTab === 'predictor' ? (
          <div className="max-w-4xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
            <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
              <div className="p-6 sm:p-8 bg-gradient-to-r from-blue-600 to-indigo-600 text-white">
                <h2 className="text-2xl font-bold">Technology Criticality Predictor</h2>
                <p className="opacity-90">Adjust the parameters below to evaluate technology criticality</p>
              </div>
              
              <div className="p-6 sm:p-8">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                  {Object.entries(inputs).map(([key, value]) => (
                    <div key={key} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <label className="block text-sm font-medium text-gray-700">
                          {key.charAt(0).toUpperCase() + key.slice(1)}
                        </label>
                        <span className="text-sm font-mono bg-gray-100 px-2 py-1 rounded">
                          {value.toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={value}
                        onChange={(e) => handleInputChange(key, e.target.value)}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                      />
                    </div>
                  ))}
                </div>

                <div className="flex flex-wrap gap-4">
                  <button
                    onClick={handlePredict}
                    disabled={isLoading}
                    className={`px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg font-medium shadow hover:shadow-md transition-all flex items-center ${isLoading ? 'opacity-70' : 'hover:-translate-y-0.5'}`}
                  >
                    {isLoading ? (
                      <>
                        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Predicting...
                      </>
                    ) : 'Predict Criticality'}
                  </button>

                  <div className="relative flex-grow">
                    <input
                      type="file"
                      ref={fileInputRef}
                      onChange={handleFileChange}
                      accept=".json"
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    />
                    <button
                      className="w-full px-6 py-3 bg-white border border-gray-300 text-gray-700 rounded-lg font-medium shadow-sm hover:bg-gray-50 transition-colors flex items-center justify-center"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                      </svg>
                      {file ? file.name : 'Upload JSON Data'}
                    </button>
                  </div>

                  {fileData && (
                    <button
                      onClick={analyzeFileData}
                      disabled={isLoading}
                      className={`px-6 py-3 bg-gradient-to-r from-green-600 to-teal-600 text-white rounded-lg font-medium shadow hover:shadow-md transition-all flex items-center ${isLoading ? 'opacity-70' : 'hover:-translate-y-0.5'}`}
                    >
                      {isLoading ? 'Analyzing...' : 'Analyze File Data'}
                    </button>
                  )}
                </div>

                {result && (
                  <div className="mt-8 bg-gray-50 rounded-xl p-6">
                    <h3 className="text-xl font-semibold mb-4 flex items-center">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                      Criticality Analysis
                    </h3>
                    
                    <div className="flex flex-col md:flex-row items-center gap-6">
                      <div className="relative w-48 h-48">
                        <svg className="w-full h-full" viewBox="0 0 100 100">
                          <circle cx="50" cy="50" r="45" fill="none" stroke="#e2e8f0" strokeWidth="10"/>
                          <circle 
                            cx="50" cy="50" r="45" fill="none" 
                            stroke={result.criticality > 0.7 ? "#ef4444" : result.criticality > 0.5 ? "#3b82f6" : "#10b981"}
                            strokeWidth="10" 
                            strokeDasharray={`${result.criticality * 283} 283`}
                            strokeLinecap="round"
                            transform="rotate(-90 50 50)"
                          />
                          <text x="50" y="50" textAnchor="middle" dominantBaseline="middle" className="text-2xl font-bold">
                            {Math.round(result.criticality * 100)}%
                          </text>
                          <text x="50" y="65" textAnchor="middle" dominantBaseline="middle" className="text-sm">
                            {result.status}
                          </text>
                        </svg>
                      </div>
                      
                      <div className="flex-1">
                        <h4 className="text-lg font-medium mb-2">Recommendation:</h4>
                        <p className="mb-4 text-gray-700">
                          {result.status === 'Critical' 
                            ? 'This technology shows strong potential for future implementation based on our ANFIS analysis.'
                            : 'This technology may not be optimal for future implementation based on current metrics.'}
                        </p>
                        <button 
                          className="text-blue-600 hover:text-blue-800 font-medium flex items-center"
                          onClick={() => setShowAdvanced(!showAdvanced)}
                        >
                          {showAdvanced ? (
                            <>
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                              </svg>
                              Hide Details
                            </>
                          ) : (
                            <>
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                              </svg>
                              Show Detailed Analysis
                            </>
                          )}
                        </button>
                      </div>
                    </div>
                    
                    {showAdvanced && (
                      <div className="mt-6 grid grid-cols-2 md:grid-cols-5 gap-4">
                        {Object.entries(inputs).map(([key, value]) => (
                          <div key={key} className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
                            <h5 className="text-sm font-medium text-gray-700 mb-2 capitalize">{key}</h5>
                            <div className="h-2 bg-gray-200 rounded-full mb-1">
                              <div 
                                className="h-2 rounded-full" 
                                style={{ 
                                  width: `${value * 100}%`,
                                  backgroundColor: value > 0.7 ? "#ef4444" : value > 0.5 ? "#3b82f6" : "#10b981"
                                }}
                              ></div>
                            </div>
                            <p className="text-xs text-right text-gray-500">{value.toFixed(2)}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : activeTab === 'analysis' ? (
          <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
            <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
              <div className="p-6 sm:p-8 bg-gradient-to-r from-blue-600 to-indigo-600 text-white">
                <h2 className="text-2xl font-bold">Technology Analysis Dashboard</h2>
                <p className="opacity-90">Comprehensive view of technology criticality assessments</p>
              </div>
              
              <div className="p-6 sm:p-8 space-y-8">
                <div>
                  <h3 className="text-lg font-semibold mb-4 flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    Model Training Progress
                  </h3>
                  <div className="bg-white p-4 rounded-lg border border-gray-200">
                    <div className="h-64">
                      <LossChart losses={[0.8, 0.6, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15]} />
                    </div>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold mb-4 flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                    </svg>
                    Technology Comparison
                  </h3>
                  <div className="overflow-hidden border border-gray-200 rounded-lg">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Technology</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Criticality</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Details</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {techComparison.map((tech) => (
                          <tr key={tech.name} className="hover:bg-gray-50">
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="flex items-center">
                                <div className="flex-shrink-0 h-10 w-10 bg-blue-100 rounded-full flex items-center justify-center">
                                  <span className="text-blue-600 font-medium">{tech.name.charAt(0)}</span>
                                </div>
                                <div className="ml-4">
                                  <div className="text-sm font-medium text-gray-900">{tech.name}</div>
                                </div>
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="flex items-center">
                                <div className="w-24 h-2 bg-gray-200 rounded-full mr-2">
                                  <div 
                                    className="h-2 rounded-full" 
                                    style={{ 
                                      width: `${tech.criticality * 100}%`,
                                      backgroundColor: tech.status === 'Critical' ? "#ef4444" : "#10b981"
                                    }}
                                  ></div>
                                </div>
                                <span className={`text-sm font-mono ${tech.status === 'Critical' ? 'text-red-600' : 'text-green-600'}`}>
                                  {tech.criticality.toFixed(3)}
                                </span>
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${tech.status === 'Critical' ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}`}>
                                {tech.status}
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                              <button className="text-blue-600 hover:text-blue-900">
                                View
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : activeTab === 'about' ? (
          <div className="max-w-4xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
            <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
              <div className="p-6 sm:p-8 bg-gradient-to-r from-blue-600 to-indigo-600 text-white">
                <h2 className="text-2xl font-bold">About TechCritical</h2>
                <p className="opacity-90">ANFIS-based Technology Criticality Prediction System</p>
              </div>
              
              <div className="p-6 sm:p-8 space-y-6">
                <div>
                  <h3 className="text-xl font-semibold mb-3">Our Technology</h3>
                  <p className="text-gray-700">
                    TechCritical utilizes Adaptive Neuro-Fuzzy Inference System (ANFIS) to evaluate and predict the criticality of emerging technologies. 
                    Our system combines the learning capabilities of neural networks with the reasoning power of fuzzy logic to provide accurate assessments.
                  </p>
                </div>
                
                <div>
                  <h3 className="text-xl font-semibold mb-3">Evaluation Parameters</h3>
                  <p className="text-gray-700 mb-4">
                    We assess technologies across five key dimensions:
                  </p>
                  <ul className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {['Scope', 'Prospects', 'Potential', 'Economy', 'Efficiency'].map((param) => (
                      <li key={param} className="flex items-start">
                        <svg className="h-5 w-5 text-green-500 mr-2 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        <span className="text-gray-700">{param}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div>
                  <h3 className="text-xl font-semibold mb-3">Development Team</h3>
                  <p className="text-gray-700">
                    This system was developed as part of a research project in intelligent systems and fuzzy logic applications.
                  </p>
                </div>
              </div>
            </div>
          </div>
        ) : null}
      </main>

      <footer className="bg-gray-50 border-t mt-12 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="md:flex md:items-center md:justify-between">
            <div className="flex justify-center md:order-2 space-x-6">
              <a href="#" className="text-gray-400 hover:text-gray-500">
                <span className="sr-only">GitHub</span>
                <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 24 24">
                  <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
                </svg>
              </a>
            </div>
            <div className="mt-8 md:mt-0 md:order-1">
              <p className="text-center text-base text-gray-500">
                &copy; {new Date().getFullYear()} TechCritical. ANFIS-Fuzzy Criticality Analysis System.
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}