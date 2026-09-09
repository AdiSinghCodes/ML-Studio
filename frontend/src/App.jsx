import { useState } from "react";
import {
  uploadDataset,
  fetchDatasetColumns,
  analyzeEDA,
  getAIAgentRecommendation
} from "./services/api";
import "./index.css";

function App() {
  const [file, setFile] = useState(null);
  const [storedFilename, setStoredFilename] = useState("");
  const [columns, setColumns] = useState([]);
  const [targetColumn, setTargetColumn] = useState("");
  
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [runningAgent, setRunningAgent] = useState(false);
  
  const [statusMessage, setStatusMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  const [edaResults, setEdaResults] = useState(null);
  const [agentResults, setAgentResults] = useState(null);
  const [showJsonPayload, setShowJsonPayload] = useState(false);

  // Handle file selection and auto upload to retrieve columns
  const handleFileSelect = async (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setEdaResults(null);
    setAgentResults(null);
    setErrorMessage("");
    setTargetColumn("");
    setStatusMessage("Uploading dataset and detecting columns...");

    try {
      setUploading(true);
      const uploadRes = await uploadDataset(selectedFile);
      const filename = uploadRes.stored_filename;
      setStoredFilename(filename);

      // Fetch columns for target dropdown
      const colRes = await fetchDatasetColumns(filename);
      setColumns(colRes.columns || []);

      setStatusMessage(`Dataset uploaded successfully (${uploadRes.analysis.rows} rows, ${uploadRes.analysis.columns} columns). Select target or run EDA.`);
    } catch (err) {
      setErrorMessage(err.message || "Failed to upload dataset.");
      setStatusMessage("");
    } finally {
      setUploading(false);
    }
  };

  // Run EDA Analysis
  const handleRunEDA = async () => {
    if (!storedFilename) {
      setErrorMessage("Please select and upload a dataset first.");
      return;
    }

    try {
      setAnalyzing(true);
      setErrorMessage("");
      setStatusMessage("Running comprehensive backend EDA analysis...");

      const res = await analyzeEDA(storedFilename, targetColumn);
      setEdaResults(res);
      setStatusMessage("EDA Analysis completed successfully.");
    } catch (err) {
      setErrorMessage(err.message || "EDA Analysis failed.");
      setStatusMessage("");
    } finally {
      setAnalyzing(false);
    }
  };

  // Ask AI Agent & Benchmark 4 Models
  const handleAskAIAgent = async () => {
    if (!storedFilename) {
      setErrorMessage("Please select and upload a dataset first.");
      return;
    }

    try {
      setRunningAgent(true);
      setErrorMessage("");
      setStatusMessage("Connecting EDA JSON to AI Agent & Benchmarking Top 4 Models...");

      const agentRes = await getAIAgentRecommendation(storedFilename, targetColumn);
      setAgentResults(agentRes);
      
      // Also set EDA results if not already loaded
      if (!edaResults && agentRes.eda_summary) {
        const edaRes = await analyzeEDA(storedFilename, targetColumn);
        setEdaResults(edaRes);
      }

      setStatusMessage("AI Agent evaluation completed successfully.");
    } catch (err) {
      setErrorMessage(err.message || "AI Agent recommendation failed.");
      setStatusMessage("");
    } finally {
      setRunningAgent(false);
    }
  };

  const getHealthColorClass = (score) => {
    if (score >= 80) return "health-good";
    if (score >= 50) return "health-warn";
    return "health-bad";
  };

  return (
    <div className="app-container">
      {/* HEADER */}
      <header className="header">
        <div className="header-content">
          <div className="logo-section">
            <h1>📊 ModelScope ML Studio</h1>
            <p>Automated Exploratory Data Analysis & AI Agent Model Recommendation Platform</p>
          </div>
          <span className="badge-tag">AI Agent Enabled</span>
        </div>
      </header>

      <main className="main-content">
        {/* FILE UPLOAD & TARGET SELECTION CARD */}
        <section className="card">
          <h2 className="card-title">📁 Upload Dataset & Select Target</h2>
          <p className="card-subtitle">
            Upload your CSV/XLSX file. Optionally select a target column for <strong>Supervised Learning</strong>, or leave as <strong>None</strong> for <strong>Unsupervised Learning</strong>.
          </p>

          <div className="upload-grid">
            <div className="form-group">
              <label>1. Select Dataset File (.csv, .xlsx, .tsv)</label>
              <label htmlFor="file-upload" className="file-dropzone">
                {file ? (
                  <div>
                    <strong>📄 {file.name}</strong>
                    <div style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "4px" }}>
                      {(file.size / 1024).toFixed(1)} KB
                    </div>
                  </div>
                ) : (
                  <div>
                    <span>Drag & Drop or Click to Browse</span>
                  </div>
                )}
              </label>
              <input
                id="file-upload"
                type="file"
                className="file-input"
                accept=".csv, .xlsx, .tsv"
                onChange={handleFileSelect}
                disabled={uploading || analyzing || runningAgent}
              />
            </div>

            <div className="form-group">
              <label>2. Target Feature (Optional)</label>
              <select
                className="select-input"
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                disabled={!storedFilename || uploading || analyzing || runningAgent}
              >
                <option value="">-- None (Unsupervised Learning Mode) --</option>
                {columns.map((col) => (
                  <option key={col} value={col}>
                    🎯 {col}
                  </option>
                ))}
              </select>
              <small style={{ color: "var(--text-muted)", fontSize: "12px", marginTop: "4px" }}>
                {targetColumn
                  ? `Supervised Mode: Model will target [${targetColumn}]`
                  : "Unsupervised Mode: Evaluating feature variance & clustering readiness"}
              </small>
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", marginTop: "16px" }}>
            <button
              className="btn-primary"
              onClick={handleRunEDA}
              disabled={!storedFilename || uploading || analyzing || runningAgent}
            >
              {analyzing ? "🔍 Analyzing Dataset..." : "⚡ 1. Run Exploratory Data Analysis (EDA)"}
            </button>

            <button
              className="btn-primary"
              style={{ background: "linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)" }}
              onClick={handleAskAIAgent}
              disabled={!storedFilename || uploading || analyzing || runningAgent}
            >
              {runningAgent ? "🤖 AI Agent Benchmarking..." : "🤖 2. Ask AI Agent & Benchmark Top 4 Models"}
            </button>
          </div>

          {statusMessage && <div className="status-alert info">ℹ️ {statusMessage}</div>}
          {errorMessage && <div className="status-alert error">⚠️ {errorMessage}</div>}
        </section>

        {/* AI AGENT RECOMMENDATION CARD */}
        {agentResults && (
          <section className="card" style={{ border: "1px solid rgba(99, 102, 241, 0.4)", background: "linear-gradient(135deg, #1e293b 0%, #1e1b4b 100%)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
              <h2 className="card-title" style={{ margin: 0, color: "#a5b4fc" }}>
                🤖 AI Agent Recommendation & Model Rationale
              </h2>
              <span className="badge-tag" style={{ background: "rgba(99, 102, 241, 0.25)", color: "#c7d2fe", fontSize: "13px" }}>
                AI Confidence: {agentResults.ai_confidence}%
              </span>
            </div>

            <div style={{ background: "rgba(15, 23, 42, 0.7)", padding: "20px", borderRadius: "10px", border: "1px solid var(--border-color)", marginBottom: "20px" }}>
              <div style={{ fontSize: "18px", fontWeight: "700", color: "#38bdf8", marginBottom: "12px" }}>
                🥇 Recommended Best Model: <span style={{ color: "#34d399" }}>{agentResults.best_model}</span>
              </div>
              <div style={{ fontSize: "14px", fontWeight: "600", color: "var(--text-main)", marginBottom: "8px" }}>
                🧠 AI Justification & Rationale (Based on EDA JSON):
              </div>
              <ul style={{ paddingLeft: "20px", fontSize: "14px", color: "var(--text-muted)", display: "flex", flexDirection: "column", gap: "6px" }}>
                {agentResults.ai_justification && agentResults.ai_justification.map((point, idx) => (
                  <li key={idx}>🔹 {point}</li>
                ))}
              </ul>
            </div>

            {/* 4 CANDIDATE MODELS LEADERBOARD TABLE */}
            {agentResults.leaderboard && agentResults.leaderboard.length > 0 && (
              <div>
                <h3 style={{ fontSize: "16px", marginBottom: "12px", color: "#c7d2fe" }}>
                  🏆 Top 4 Candidate Models Benchmark Leaderboard
                </h3>
                <div className="table-container">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Rank</th>
                        <th>Model Name</th>
                        <th>Validation Score</th>
                        <th>Benchmark Metrics</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {agentResults.leaderboard.map((m, idx) => (
                        <tr key={idx} style={{ background: idx === 0 ? "rgba(16, 185, 129, 0.1)" : "transparent" }}>
                          <td>
                            <strong>#{idx + 1}</strong>
                          </td>
                          <td>
                            <strong>{m.model_name}</strong>
                            {idx === 0 && (
                              <span className="pill pill-target" style={{ marginLeft: "8px" }}>
                                🥇 Best AI Pick
                              </span>
                            )}
                          </td>
                          <td>
                            <strong style={{ color: idx === 0 ? "#34d399" : "var(--text-main)", fontSize: "16px" }}>
                              {m.primary_score !== undefined ? m.primary_score : "N/A"}
                            </strong>
                          </td>
                          <td style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                            {agentResults.problem_type === "classification" ? (
                              `Acc: ${m.accuracy} | F1: ${m.f1_score} | Prec: ${m.precision} | Rec: ${m.recall}`
                            ) : (
                              `R²: ${m.r2_score} | MAE: ${m.mae} | RMSE: ${m.rmse}`
                            )}
                          </td>
                          <td>
                            <span className="pill" style={{ background: m.status === "success" ? "rgba(16, 185, 129, 0.2)" : "rgba(239, 68, 68, 0.2)", color: m.status === "success" ? "#34d399" : "#fca5a5" }}>
                              {m.status}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </section>
        )}

        {/* EDA RESULTS DASHBOARD */}
        {edaResults && (
          <section className="card">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
              <h2 className="card-title" style={{ margin: 0 }}>📈 EDA Results & Data Health Dashboard</h2>
              <span className="badge-tag" style={{ fontSize: "13px" }}>
                Mode: {edaResults.mode.toUpperCase()}
              </span>
            </div>

            {/* OVERVIEW STATS GRID */}
            <div className="stats-grid">
              <div className="stat-card">
                <span className="stat-label">Total Rows</span>
                <span className="stat-value">{edaResults.summary.rows.toLocaleString()}</span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Total Columns</span>
                <span className="stat-value">{edaResults.summary.columns}</span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Memory Footprint</span>
                <span className="stat-value">{edaResults.summary.memory_mb} MB</span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Duplicate Rows</span>
                <span className="stat-value">{edaResults.summary.duplicate_rows} ({edaResults.summary.duplicate_percentage}%)</span>
              </div>
            </div>

            {/* HEALTH SCORE BANNER */}
            <div className="health-box" style={{ marginBottom: "24px" }}>
              <div>
                <span className="stat-label">Data Health Score</span>
                <div className={`health-meter ${getHealthColorClass(edaResults.summary.health_score)}`}>
                  {edaResults.summary.health_score} / 100
                </div>
              </div>
              <div style={{ borderLeft: "1px solid var(--border-color)", paddingLeft: "20px", fontSize: "14px" }}>
                <div><strong>Feature Types Breakdown:</strong></div>
                <div style={{ color: "var(--text-muted)", marginTop: "4px" }}>
                  🔢 Numeric: <strong>{edaResults.summary.numeric_column_count}</strong> | 
                  🏷️ Categorical: <strong>{edaResults.summary.categorical_column_count}</strong> | 
                  📝 Text/NLP: <strong>{edaResults.summary.text_column_count}</strong> | 
                  📅 Datetime: <strong>{edaResults.summary.datetime_column_count}</strong>
                </div>
              </div>
            </div>

            {/* TARGET & IMBALANCE ANALYSIS CARD */}
            {edaResults.target_analysis && (
              <div style={{ background: "rgba(15, 23, 42, 0.6)", padding: "16px", borderRadius: "10px", border: "1px solid var(--border-color)", marginBottom: "24px" }}>
                <h3 style={{ fontSize: "16px", marginBottom: "10px", color: "var(--primary)" }}>
                  🎯 Target Column Analysis ({edaResults.target_column || "Unsupervised"})
                </h3>
                {edaResults.target_analysis.mode === "supervised" ? (
                  <div>
                    <p style={{ fontSize: "14px", marginBottom: "8px" }}>
                      Task Type: <strong>{edaResults.target_analysis.problem_type.toUpperCase()}</strong>
                    </p>
                    {edaResults.target_analysis.problem_type === "classification" && (
                      <div>
                        <p style={{ fontSize: "14px", marginBottom: "8px" }}>
                          Imbalance Status:{" "}
                          <strong style={{ color: edaResults.target_analysis.imbalance_status === "balanced" ? "var(--success)" : "var(--warning)" }}>
                            {edaResults.target_analysis.imbalance_status.replace("_", " ").toUpperCase()} (Ratio: {edaResults.target_analysis.imbalance_ratio}:1)
                          </strong>
                        </p>
                        <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", marginTop: "10px" }}>
                          {Object.entries(edaResults.target_analysis.class_distribution).map(([cls, info]) => (
                            <span key={cls} className="badge-tag" style={{ background: "rgba(99, 102, 241, 0.2)", color: "#a5b4fc" }}>
                              Class '{cls}': {info.count} ({info.percentage}%)
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    {edaResults.target_analysis.problem_type === "regression" && (
                      <div style={{ fontSize: "14px", color: "var(--text-muted)" }}>
                        Range: [{edaResults.target_analysis.min} to {edaResults.target_analysis.max}] | Mean: {edaResults.target_analysis.mean} | Median: {edaResults.target_analysis.median} | Skewness: {edaResults.target_analysis.skewness}
                      </div>
                    )}
                  </div>
                ) : (
                  <p style={{ fontSize: "14px", color: "var(--text-muted)" }}>
                    {edaResults.target_analysis.message}
                  </p>
                )}
              </div>
            )}

            {/* COLUMN PROFILES TABLE */}
            <h3 style={{ fontSize: "16px", marginBottom: "12px" }}>📊 Detailed Feature Profiles</h3>
            <div className="table-container" style={{ marginBottom: "24px" }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Column Name</th>
                    <th>Type Category</th>
                    <th>Missing Values</th>
                    <th>Unique Values</th>
                    <th>Outliers (IQR)</th>
                    <th>Key Statistics</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(edaResults.column_profiles).map(([colName, prof]) => (
                    <tr key={colName}>
                      <td>
                        <strong>{colName}</strong>
                        {prof.is_target && <span className="pill pill-target" style={{ marginLeft: "8px" }}>Target</span>}
                      </td>
                      <td>
                        <span className={`pill pill-${prof.type_category}`}>
                          {prof.type_category}
                        </span>
                      </td>
                      <td>{prof.missing_values} ({prof.missing_percentage}%)</td>
                      <td>{prof.unique_values}</td>
                      <td>
                        {prof.numeric_stats ? (
                          <span style={{ color: prof.numeric_stats.outliers_count > 0 ? "var(--warning)" : "var(--text-muted)" }}>
                            {prof.numeric_stats.outliers_count} ({prof.numeric_stats.outliers_percentage}%)
                          </span>
                        ) : "N/A"}
                      </td>
                      <td style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                        {prof.numeric_stats && (
                          `Mean: ${prof.numeric_stats.mean} | Std: ${prof.numeric_stats.std} | Skew: ${prof.numeric_stats.skewness}`
                        )}
                        {prof.nlp_stats && (
                          `Avg Words: ${prof.nlp_stats.avg_word_count} | Vocab: ${prof.nlp_stats.vocabulary_size}`
                        )}
                        {prof.categorical_stats && (
                          `Top: ${Object.keys(prof.categorical_stats.top_frequencies)[0] || 'N/A'}`
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* HIGH CORRELATIONS CARD */}
            {edaResults.high_correlations && edaResults.high_correlations.length > 0 && (
              <div style={{ background: "rgba(245, 158, 11, 0.1)", border: "1px solid rgba(245, 158, 11, 0.3)", padding: "16px", borderRadius: "10px", marginBottom: "24px" }}>
                <h4 style={{ color: "var(--warning)", marginBottom: "8px" }}>⚠️ High Feature Correlation Warnings (|r| ≥ 0.85)</h4>
                <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
                  {edaResults.high_correlations.map((pair, idx) => (
                    <span key={idx} className="badge-tag" style={{ background: "rgba(245, 158, 11, 0.2)", color: "#fcd34d" }}>
                      {pair.feature_1} ↔ {pair.feature_2} (r = {pair.correlation})
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* AI AGENT PAYLOAD VIEWER */}
            <div>
              <button
                style={{ background: "transparent", border: "1px solid var(--border-color)", color: "var(--text-muted)", padding: "8px 16px", borderRadius: "6px", cursor: "pointer" }}
                onClick={() => setShowJsonPayload(!showJsonPayload)}
              >
                {showJsonPayload ? "🙈 Hide AI Agent Payload" : "🤖 View AI Agent Summary Payload (JSON)"}
              </button>
              {showJsonPayload && (
                <div className="json-box" style={{ marginTop: "12px" }}>
                  <pre>{JSON.stringify(edaResults.eda_agent_summary, null, 2)}</pre>
                </div>
              )}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;