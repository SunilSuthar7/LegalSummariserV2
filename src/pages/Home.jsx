import React, { useState } from 'react';
import { Upload, FileText, Zap, BarChart3, Settings, BookOpen } from 'lucide-react';
import '../styles/Home.css';

export default function Home() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [selectedDataset, setSelectedDataset] = useState('ilc');
  const [isProcessing, setIsProcessing] = useState(false);

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (file && file.type === 'application/pdf') {
      setUploadedFile(file);
    }
  };

  const handleSummarize = async () => {
    if (!uploadedFile) return;
    setIsProcessing(true);
    // Add your API call here
    setTimeout(() => setIsProcessing(false), 2000);
  };

  const features = [
    {
      icon: <FileText className="w-6 h-6" />,
      title: "Two-Stage Pipeline",
      description: "Chunk-level + global refinement for handling long legal documents"
    },
    {
      icon: <Zap className="w-6 h-6" />,
      title: "LegalBERT Extraction",
      description: "Supervised extractive model identifies summary-worthy sentences"
    },
    {
      icon: <BarChart3 className="w-6 h-6" />,
      title: "Superior Results",
      description: "ROUGE-1 ≈ 47.3% — outperforming prior benchmarks"
    },
    {
      icon: <Settings className="w-6 h-6" />,
      title: "QLoRA Optimization",
      description: "Memory-efficient T5 fine-tuning on low-resource GPUs"
    },
    {
      icon: <BookOpen className="w-6 h-6" />,
      title: "Indian Legal Focus",
      description: "Trained on ILC & IN-ABS datasets for Indian judgments"
    },
    {
      icon: <Zap className="w-6 h-6" />,
      title: "Modular & Scalable",
      description: "Reproducible pipeline designed for future legal NLP research"
    }
  ];

  return (
    <div className="home-container">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title">Legal Document Summarizer</h1>
          <p className="hero-subtitle">
            Powered by AI | Built for Indian Legal Judgments
          </p>
        </div>
      </section>

      {/* Upload Section */}
      <section className="upload-section">
        <div className="upload-container">
          <div className="upload-wrapper">
            <h2 className="section-title">Upload Your Document</h2>
            
            <div className="upload-area">
              <label htmlFor="file-input" className="upload-label">
                <Upload className="upload-icon" />
                <span className="upload-text">
                  {uploadedFile ? (
                    <>✓ {uploadedFile.name}</>
                  ) : (
                    <>Drag & drop your PDF or click to browse</>
                  )}
                </span>
              </label>
              <input
                id="file-input"
                type="file"
                accept=".pdf"
                onChange={handleFileUpload}
                className="file-input"
              />
            </div>

            {/* Dataset Selection */}
            <div className="dataset-section">
              <label className="dataset-label">Select Dataset Source:</label>
              <div className="dataset-buttons">
                <button
                  className={`dataset-btn ${selectedDataset === 'ilc' ? 'active' : ''}`}
                  onClick={() => setSelectedDataset('ilc')}
                >
                  ILC Dataset
                </button>
                <button
                  className={`dataset-btn ${selectedDataset === 'inabs' ? 'active' : ''}`}
                  onClick={() => setSelectedDataset('inabs')}
                >
                  IN-ABS Dataset
                </button>
              </div>
            </div>

            {/* Summarize Button */}
            <button
              className={`summarize-btn ${isProcessing ? 'processing' : ''}`}
              onClick={handleSummarize}
              disabled={!uploadedFile || isProcessing}
            >
              {isProcessing ? 'Summarizing...' : 'Summarize Document'}
            </button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <div className="features-container">
          <h2 className="features-title">Why Choose Our Summarizer?</h2>
          <p className="features-subtitle">
            State-of-the-art abstractive summarization designed specifically for legal documents
          </p>
          
          <div className="features-grid">
            {features.map((feature, idx) => (
              <div key={idx} className="feature-card">
                <div className="feature-icon">{feature.icon}</div>
                <h3 className="feature-title">{feature.title}</h3>
                <p className="feature-description">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="cta-content">
          <h2>Ready to Summarize?</h2>
          <p>Upload a PDF document above to get started</p>
        </div>
      </section>
    </div>
  );
}
