import React, { useState, useCallback } from 'react';
import './App.css';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFile = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => setSelectedImage(e.target.result);
    reader.readAsDataURL(file);
    onPredict(file);
  };

  const onPredict = async (file) => {
    setLoading(true);
    setPrediction(null);
    
    // Mock the API call
    const formData = new FormData();
    formData.append('file', file);

    try {
      // In a real scenario:
      // const response = await fetch('http://localhost:8000/predict', { method: 'POST', body: formData });
      // const data = await response.json();
      
      // Mocked delay
      await new Promise(r => setTimeout(r, 2000));
      
      const mockResult = {
        prediction: Math.random() > 0.5 ? "Malignant" : "Benign",
        confidence: (0.85 + Math.random() * 0.14).toFixed(4),
        details: {
          asymmetry: "High",
          border: "Irregular",
          color: "Mixed"
        }
      };
      setPrediction(mockResult);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header style={{ marginBottom: '4rem', textAlign: 'center' }}>
        <h1 className="gradient-text">SkinSync AI</h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '1.2rem' }}>
          Advanced ML-Based Skin Cancer Classification System
        </p>
      </header>

      <main className="glass-card">
        <div style={{ maxWidth: '600px', margin: '0 auto' }}>
          <div 
            className={`upload-zone ${dragActive ? 'active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-upload').click()}
          >
            <input 
              id="file-upload" 
              type="file" 
              style={{ display: 'none' }} 
              onChange={(e) => handleFile(e.target.files[0])}
              accept="image/*"
            />
            {selectedImage ? (
              <img src={selectedImage} alt="Selected" style={{ maxWidth: '100%', borderRadius: '12px', maxHeight: '300px' }} />
            ) : (
              <div style={{ padding: '2rem' }}>
                <p style={{ fontSize: '1.25rem', fontWeight: '600' }}>Drop image here or click to browse</p>
                <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>Support JPEG, PNG formats</p>
              </div>
            )}
          </div>

          {loading && (
            <div style={{ textAlign: 'center', padding: '2rem' }}>
              <div className="loader"></div>
              <p style={{ marginTop: '1rem', color: 'var(--primary)', fontWeight: '600' }}>Analyzing Dermoscopic Patterns...</p>
            </div>
          )}

          {prediction && (
            <div className="result-section" style={{ animation: 'fadeIn 0.5s ease' }}>
              <div style={{ 
                padding: '1.5rem', 
                borderRadius: '16px', 
                background: prediction.prediction === 'Malignant' ? 'rgba(239, 68, 68, 0.1)' : 'rgba(34, 197, 94, 0.1)',
                border: `1px solid ${prediction.prediction === 'Malignant' ? 'var(--danger)' : 'var(--success)'}`,
                textAlign: 'center',
                marginBottom: '2rem'
              }}>
                <h2 style={{ color: prediction.prediction === 'Malignant' ? 'var(--danger)' : 'var(--success)' }}>
                  {prediction.prediction}
                </h2>
                <p style={{ marginTop: '0.5rem' }}>Confidence Score: {prediction.confidence}</p>
              </div>

              <div className="stats-grid">
                <div className="stat-item">
                  <div className="stat-value">{prediction.details.asymmetry}</div>
                  <div className="stat-label">Asymmetry</div>
                </div>
                <div className="stat-item">
                  <div className="stat-value">{prediction.details.border}</div>
                  <div className="stat-label">Border</div>
                </div>
                <div className="stat-item">
                  <div className="stat-value">{prediction.details.color}</div>
                  <div className="stat-label">Color</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      <footer style={{ marginTop: '4rem', textAlign: 'center', color: 'var(--text-muted)', paddingBottom: '2rem' }}>
        <p>© 2026 SkinSync AI. For medical research purposes only.</p>
      </footer>

      <style>{`
        .loader {
          border: 4px solid rgba(255, 255, 255, 0.1);
          border-left: 4px solid var(--primary);
          border-radius: 50%;
          width: 40px;
          height: 40px;
          animation: spin 1s linear infinite;
          margin: 0 auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .upload-zone.active {
          border-color: var(--primary);
          background: rgba(13, 148, 136, 0.1);
        }
      `}</style>
    </div>
  );
}

export default App;
