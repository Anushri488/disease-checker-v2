import { useState, useEffect, useMemo, useCallback } from "react";
import axios from "axios";

const API = "http://localhost:5000";

// ─── Severity helpers ───────────────────────────────────────────────────────
const SEV = {
  mild:     { bg: "#dcfce7", text: "#166534", border: "#86efac", label: "Mild",     icon: "🟢" },
  moderate: { bg: "#fef3c7", text: "#92400e", border: "#fcd34d", label: "Moderate", icon: "🟡" },
  severe:   { bg: "#fee2e2", text: "#991b1b", border: "#fca5a5", label: "Severe",   icon: "🔴" },
};

// ─── Components ─────────────────────────────────────────────────────────────
function SeverityBadge({ level, small }) {
  const s = SEV[level] || SEV.mild;
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 3,
      background: s.bg, color: s.text, border: `1px solid ${s.border}`,
      padding: small ? "1px 7px" : "3px 10px",
      borderRadius: 99, fontSize: small ? 10 : 11, fontWeight: 700,
    }}>
      {s.icon} {s.label}
    </span>
  );
}

function Badge({ label, onRemove }) {
  return (
    <span onClick={onRemove} style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      background: "#e0f2fe", color: "#0369a1",
      padding: "3px 10px", borderRadius: 99, fontSize: 12,
      fontWeight: 500, cursor: "pointer", border: "1px solid #bae6fd",
    }}>
      {label.replace(/_/g, " ")} ×
    </span>
  );
}

function SymptomChip({ symptom, severity, selected, onToggle }) {
  return (
    <button onClick={onToggle} style={{
      padding: "7px 12px", borderRadius: 9, fontSize: 12,
      cursor: "pointer", fontFamily: "inherit", textAlign: "left",
      background: selected ? "#0ea5e9" : "#f8fafc",
      color: selected ? "#fff" : "#334155",
      border: `1.5px solid ${selected ? "#0ea5e9" : "#e2e8f0"}`,
      transition: "all 0.15s", fontWeight: selected ? 600 : 400,
      display: "flex", alignItems: "center", gap: 6,
    }}>
      <span style={{ fontSize: 9, opacity: selected ? 0.8 : 0.5 }}>
        {SEV[severity]?.icon || "🟢"}
      </span>
      {symptom.replace(/_/g, " ")}
    </button>
  );
}

function HistoryItem({ item, onReload, idx }) {
  const sev = item.severity || "moderate";
  return (
    <div style={{
      background: "#fff", border: "1px solid #e2e8f0", borderRadius: 12,
      padding: "12px 16px", cursor: "pointer", transition: "box-shadow 0.15s",
    }} onClick={() => onReload(item)}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <span style={{ fontWeight: 700, fontSize: 14, color: "#0f172a" }}>{item.disease}</span>
        <SeverityBadge level={sev} small />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#94a3b8" }}>
        <span>{item.symptoms?.length} symptoms · {item.confidence}% confidence</span>
        <span>{new Date(item.timestamp).toLocaleDateString()}</span>
      </div>
    </div>
  );
}

function ResultCard({ result, onDownloadPDF, pdfLoading }) {
  const conf = result.confidence;
  const sev  = result.severity || "moderate";
  const barColor = conf >= 70 ? "#10b981" : conf >= 40 ? "#f59e0b" : "#ef4444";

  return (
    <div style={{
      marginTop: 28, borderRadius: 16, overflow: "hidden",
      border: "1px solid #e2e8f0", boxShadow: "0 4px 24px rgba(0,0,0,0.07)"
    }}>
      {/* Emergency banner */}
      {result.emergency && (
        <div style={{
          background: "#fee2e2", borderBottom: "2px solid #fca5a5",
          padding: "12px 24px", display: "flex", alignItems: "center", gap: 10
        }}>
          <span style={{ fontSize: 20 }}>🚨</span>
          <div>
            <div style={{ fontWeight: 800, fontSize: 14, color: "#991b1b" }}>EMERGENCY — Seek Immediate Medical Attention</div>
            <div style={{ fontSize: 12, color: "#b91c1c" }}>Call 102 or 112 right now</div>
          </div>
        </div>
      )}

      {/* Header */}
      <div style={{
        background: "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)",
        padding: "22px 28px", color: "#fff"
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
          <div>
            <p style={{ fontSize: 10, letterSpacing: 2, opacity: 0.5, margin: "0 0 4px", textTransform: "uppercase" }}>
              Primary Diagnosis
            </p>
            <h2 style={{ margin: 0, fontSize: 24, fontWeight: 800, letterSpacing: -0.5 }}>{result.disease}</h2>
          </div>
          <SeverityBadge level={sev} />
        </div>
        {conf && (
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
              <span style={{ fontSize: 11, opacity: 0.6 }}>Model Confidence</span>
              <span style={{ fontSize: 13, fontWeight: 700, color: barColor }}>{conf}%</span>
            </div>
            <div style={{ background: "rgba(255,255,255,0.1)", borderRadius: 99, height: 5 }}>
              <div style={{
                width: `${conf}%`, height: 5, borderRadius: 99,
                background: barColor, transition: "width 1s ease"
              }} />
            </div>
          </div>
        )}
      </div>

      {/* Description */}
      {result.description && (
        <div style={{ padding: "18px 28px", background: "#fff", borderBottom: "1px solid #f1f5f9" }}>
          <p style={{ margin: "0 0 6px", fontSize: 11, fontWeight: 700, color: "#94a3b8", letterSpacing: 1, textTransform: "uppercase" }}>About this condition</p>
          <p style={{ margin: 0, fontSize: 13, color: "#374151", lineHeight: 1.65 }}>{result.description}</p>
        </div>
      )}

      {/* Doctor advice */}
      {result.doctor_advice && (
        <div style={{
          padding: "14px 28px", background: "#fffbeb",
          borderBottom: "1px solid #fde68a", display: "flex", gap: 10, alignItems: "flex-start"
        }}>
          <span style={{ fontSize: 18, flexShrink: 0 }}>⚕️</span>
          <div>
            <p style={{ margin: "0 0 3px", fontSize: 11, fontWeight: 700, color: "#92400e", letterSpacing: 1, textTransform: "uppercase" }}>When to see a Doctor</p>
            <p style={{ margin: 0, fontSize: 13, color: "#78350f", lineHeight: 1.55 }}>{result.doctor_advice}</p>
          </div>
        </div>
      )}

      {/* Symptoms with severity */}
      {result.matched_symptoms?.length > 0 && (
        <div style={{ padding: "16px 28px", background: "#f8fafc", borderBottom: "1px solid #f1f5f9" }}>
          <p style={{ margin: "0 0 10px", fontSize: 11, fontWeight: 700, color: "#94a3b8", letterSpacing: 1, textTransform: "uppercase" }}>Matched Symptoms</p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {result.matched_symptoms.map(s => {
              const sev = result.symptom_severities?.[s] || "mild";
              const c = SEV[sev];
              return (
                <span key={s} style={{
                  background: c.bg, color: c.text, border: `1px solid ${c.border}`,
                  padding: "3px 10px", borderRadius: 99, fontSize: 11, fontWeight: 600,
                  display: "flex", alignItems: "center", gap: 4
                }}>
                  {c.icon} {s.replace(/_/g, " ")}
                </span>
              );
            })}
          </div>
        </div>
      )}

      {/* Top 3 */}
      {result.top3?.length > 1 && (
        <div style={{ padding: "16px 28px", background: "#fff", borderBottom: "1px solid #f1f5f9" }}>
          <p style={{ margin: "0 0 12px", fontSize: 11, fontWeight: 700, color: "#94a3b8", letterSpacing: 1, textTransform: "uppercase" }}>Other Possibilities</p>
          <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
            {result.top3.slice(1).map((item, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div style={{ flex: 1, fontSize: 13, color: "#334155", fontWeight: 500 }}>{item.disease}</div>
                <div style={{ width: 100, background: "#f1f5f9", borderRadius: 99, height: 4 }}>
                  <div style={{ width: `${item.confidence}%`, height: 4, borderRadius: 99, background: "#94a3b8" }} />
                </div>
                <span style={{ fontSize: 12, color: "#64748b", minWidth: 34, textAlign: "right" }}>{item.confidence}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Precautions */}
      {result.precautions?.length > 0 && (
        <div style={{ padding: "16px 28px", background: "#f0fdf4", borderBottom: "1px solid #bbf7d0" }}>
          <p style={{ margin: "0 0 10px", fontSize: 11, fontWeight: 700, color: "#166534", letterSpacing: 1, textTransform: "uppercase" }}>Precautions & Self-care</p>
          <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            {result.precautions.map((p, i) => (
              <div key={i} style={{ display: "flex", gap: 8, fontSize: 13, color: "#15803d" }}>
                <span>✓</span><span>{p}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* PDF download */}
      <div style={{ padding: "14px 28px", background: "#fff", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: 11, color: "#94a3b8" }}>
          ⚠️ Educational tool only — not a substitute for medical advice.
        </span>
        <button onClick={onDownloadPDF} disabled={pdfLoading} style={{
          padding: "8px 18px", borderRadius: 8, border: "none", cursor: pdfLoading ? "wait" : "pointer",
          background: pdfLoading ? "#e2e8f0" : "#0f172a", color: pdfLoading ? "#94a3b8" : "#fff",
          fontSize: 13, fontWeight: 700, fontFamily: "inherit",
          display: "flex", alignItems: "center", gap: 6
        }}>
          {pdfLoading ? "⏳ Generating..." : "📄 Download PDF Report"}
        </button>
      </div>
    </div>
  );
}

// ─── Main App ────────────────────────────────────────────────────────────────
export default function App() {
  const [allSymptoms, setAllSymptoms] = useState([]);
  const [symptomMeta, setSymptomMeta] = useState({});
  const [selected, setSelected]   = useState([]);
  const [search, setSearch]       = useState("");
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [error, setError]         = useState(null);
  const [apiStatus, setApiStatus] = useState("checking");
  const [activeCategory, setActiveCategory] = useState("All");
  const [activeTab, setActiveTab] = useState("checker"); // checker | history
  const [history, setHistory]     = useState(() => {
    try { return JSON.parse(localStorage.getItem("symptomHistory") || "[]"); }
    catch { return []; }
  });

  useEffect(() => {
    axios.get(`${API}/health`)
      .then(() => { setApiStatus("online"); return axios.get(`${API}/symptoms`); })
      .then(res => {
        const names = res.data.symptom_names || res.data.symptoms.map(s => s.name || s);
        setAllSymptoms(names);
        const meta = {};
        (res.data.symptoms || []).forEach(s => {
          if (typeof s === "object") meta[s.name] = s.severity;
          else meta[s] = "mild";
        });
        setSymptomMeta(meta);
      })
      .catch(() => setApiStatus("offline"));
  }, []);

  const categories = useMemo(() => ({
    "All":        allSymptoms,
    "Skin":       allSymptoms.filter(s => ["itching","skin_rash","blister","blackheads","pus_filled_pimples","skin_peeling","nodal_skin_eruptions","red_spots_over_body","silver_like_dusting","scurring"].includes(s)),
    "Digestive":  allSymptoms.filter(s => ["vomiting","nausea","diarrhoea","constipation","abdominal_pain","stomach_pain","indigestion","loss_of_appetite","acidity","passage_of_gases","bloody_stool"].includes(s)),
    "Fever":      allSymptoms.filter(s => ["high_fever","mild_fever","fever","chills","sweating","shivering"].includes(s)),
    "Pain":       allSymptoms.filter(s => s.includes("pain") || s.includes("ache") || s.includes("cramp")),
    "Respiratory":allSymptoms.filter(s => ["cough","breathlessness","chest_pain","runny_nose","congestion","mucoid_sputum","blood_in_sputum"].includes(s)),
    "Neuro":      allSymptoms.filter(s => ["headache","dizziness","loss_of_balance","slurred_speech","altered_sensorium","spinning_movements","unsteadiness","depression","anxiety","irritability"].includes(s)),
    "Fatigue":    allSymptoms.filter(s => ["fatigue","lethargy","weakness_in_limbs","malaise","muscle_wasting","muscle_weakness"].includes(s)),
    "Severe":     allSymptoms.filter(s => symptomMeta[s] === "severe"),
  }), [allSymptoms, symptomMeta]);

  const filtered = useMemo(() => {
    const base = categories[activeCategory] || allSymptoms;
    if (!search) return base;
    return base.filter(s => s.toLowerCase().includes(search.toLowerCase().replace(/ /g, "_")));
  }, [allSymptoms, search, activeCategory, categories]);

  const toggle = useCallback((sym) => {
    setSelected(prev => prev.includes(sym) ? prev.filter(s => s !== sym) : [...prev, sym]);
    setResult(null);
  }, []);

  const predict = async () => {
    if (!selected.length) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await axios.post(`${API}/predict`, { symptoms: selected });
      const data = { ...res.data, symptoms: selected };
      setResult(data);
      // Save to history
      const entry = { ...data, id: Date.now() };
      const newHistory = [entry, ...history].slice(0, 20);
      setHistory(newHistory);
      localStorage.setItem("symptomHistory", JSON.stringify(newHistory));
    } catch {
      setError("Could not connect to Flask API. Make sure it's running on port 5000.");
    }
    setLoading(false);
  };

  const downloadPDF = async () => {
    if (!result) return;
    setPdfLoading(true);
    try {
      const res = await axios.post(`${API}/pdf-report`, result, { responseType: "blob" });
      const url = URL.createObjectURL(new Blob([res.data], { type: "application/pdf" }));
      const a = document.createElement("a");
      a.href = url; a.download = `SymptomAI_Report.pdf`; a.click();
      URL.revokeObjectURL(url);
    } catch {
      alert("PDF generation failed. Make sure Flask server is running.");
    }
    setPdfLoading(false);
  };

  const reloadHistory = (item) => {
    setSelected(item.symptoms || []);
    setResult(item);
    setActiveTab("checker");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem("symptomHistory");
  };

  return (
    <div style={{ minHeight: "100vh", background: "#f8fafc", fontFamily: "'DM Sans','Segoe UI',sans-serif" }}>
      {/* Topbar */}
      <div style={{ background: "#0f172a", padding: "13px 28px", display: "flex", alignItems: "center", justifyContent: "space-between", position: "sticky", top: 0, zIndex: 100 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 22 }}>🩺</span>
          <span style={{ color: "#fff", fontWeight: 700, fontSize: 18, letterSpacing: -0.3 }}>SymptomAI</span>
          <span style={{ background: "#1e40af", color: "#93c5fd", fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 99, letterSpacing: 0.5 }}>v2 · RF 99%</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          {/* Tabs */}
          {["checker","history"].map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)} style={{
              background: "none", border: "none", cursor: "pointer", fontFamily: "inherit",
              color: activeTab === tab ? "#fff" : "#64748b", fontWeight: activeTab === tab ? 700 : 400,
              fontSize: 13, padding: "4px 0", borderBottom: activeTab === tab ? "2px solid #38bdf8" : "2px solid transparent"
            }}>
              {tab === "checker" ? "🔍 Checker" : `📋 History ${history.length > 0 ? `(${history.length})` : ""}`}
            </button>
          ))}
          <div style={{
            display: "flex", alignItems: "center", gap: 5, fontSize: 11,
            color: apiStatus === "online" ? "#34d399" : apiStatus === "offline" ? "#f87171" : "#94a3b8"
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: "50%", display: "inline-block",
              background: apiStatus === "online" ? "#34d399" : apiStatus === "offline" ? "#f87171" : "#94a3b8"
            }} />
            API {apiStatus}
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 920, margin: "0 auto", padding: "28px 20px" }}>

        {/* ── HISTORY TAB ── */}
        {activeTab === "history" && (
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
              <h2 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: "#0f172a" }}>Prediction History</h2>
              {history.length > 0 && (
                <button onClick={clearHistory} style={{
                  background: "none", border: "1px solid #fecaca", color: "#ef4444",
                  borderRadius: 8, padding: "6px 14px", cursor: "pointer", fontSize: 13, fontFamily: "inherit"
                }}>🗑 Clear All</button>
              )}
            </div>
            {history.length === 0 ? (
              <div style={{ textAlign: "center", color: "#94a3b8", padding: "60px 0", fontSize: 14 }}>
                No predictions yet. Run the checker to get started!
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {history.map((item, i) => (
                  <HistoryItem key={item.id || i} item={item} onReload={reloadHistory} idx={i} />
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── CHECKER TAB ── */}
        {activeTab === "checker" && (
          <>
            <div style={{ marginBottom: 24, textAlign: "center" }}>
              <h1 style={{ fontSize: 30, fontWeight: 800, color: "#0f172a", margin: "0 0 6px", letterSpacing: -1 }}>
                What symptoms do you have?
              </h1>
              <p style={{ color: "#64748b", fontSize: 14, margin: 0 }}>
                Select all that apply — Random Forest model predicts the most likely disease
              </p>
              {/* Severity legend */}
              <div style={{ display: "flex", gap: 12, justifyContent: "center", marginTop: 10 }}>
                {Object.entries(SEV).map(([k, v]) => (
                  <span key={k} style={{
                    display: "inline-flex", alignItems: "center", gap: 4,
                    background: v.bg, color: v.text, border: `1px solid ${v.border}`,
                    padding: "2px 10px", borderRadius: 99, fontSize: 11, fontWeight: 600
                  }}>
                    {v.icon} {v.label} symptom
                  </span>
                ))}
              </div>
            </div>

            {/* Selected badges */}
            {selected.length > 0 && (
              <div style={{
                background: "#fff", borderRadius: 12, padding: "11px 14px",
                border: "1px solid #e2e8f0", marginBottom: 14,
                display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center"
              }}>
                <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 700, marginRight: 4 }}>
                  {selected.length} selected:
                </span>
                {selected.map(s => <Badge key={s} label={s} onRemove={() => toggle(s)} />)}
                <button onClick={() => { setSelected([]); setResult(null); }} style={{
                  marginLeft: "auto", fontSize: 12, color: "#ef4444",
                  background: "none", border: "none", cursor: "pointer", fontFamily: "inherit"
                }}>Clear all</button>
              </div>
            )}

            {/* Search + Categories + Grid */}
            <div style={{ background: "#fff", borderRadius: 14, border: "1px solid #e2e8f0", padding: 18, marginBottom: 18 }}>
              <input
                placeholder="🔍  Search symptoms..."
                value={search}
                onChange={e => setSearch(e.target.value)}
                style={{
                  width: "100%", padding: "9px 13px", borderRadius: 9,
                  border: "1.5px solid #e2e8f0", fontSize: 13, fontFamily: "inherit",
                  outline: "none", boxSizing: "border-box", marginBottom: 12,
                  color: "#0f172a", background: "#f8fafc"
                }}
              />
              <div style={{ display: "flex", gap: 5, flexWrap: "wrap", marginBottom: 14 }}>
                {Object.keys(categories).map(cat => (
                  <button key={cat} onClick={() => setActiveCategory(cat)} style={{
                    padding: "4px 13px", borderRadius: 99, fontSize: 11,
                    cursor: "pointer", fontFamily: "inherit", fontWeight: 600,
                    background: activeCategory === cat ? "#0f172a" : "#f1f5f9",
                    color: activeCategory === cat ? "#fff" : "#64748b",
                    border: "none", transition: "all 0.15s"
                  }}>{cat}</button>
                ))}
              </div>
              <div style={{
                display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(170px, 1fr))",
                gap: 7, maxHeight: 300, overflowY: "auto", paddingRight: 2
              }}>
                {filtered.map(s => (
                  <SymptomChip
                    key={s} symptom={s}
                    severity={symptomMeta[s] || "mild"}
                    selected={selected.includes(s)}
                    onToggle={() => toggle(s)}
                  />
                ))}
                {filtered.length === 0 && (
                  <p style={{ color: "#94a3b8", fontSize: 13, gridColumn: "1/-1" }}>No symptoms found.</p>
                )}
              </div>
            </div>

            {/* Predict button */}
            <button onClick={predict}
              disabled={selected.length === 0 || loading || apiStatus !== "online"}
              style={{
                width: "100%", padding: "14px", borderRadius: 12,
                background: selected.length === 0 || apiStatus !== "online" ? "#e2e8f0" : "linear-gradient(135deg,#0f172a,#1e3a5f)",
                color: selected.length === 0 || apiStatus !== "online" ? "#94a3b8" : "#fff",
                border: "none", fontSize: 15, fontWeight: 700, cursor: selected.length === 0 ? "not-allowed" : "pointer",
                fontFamily: "inherit", transition: "all 0.2s", letterSpacing: -0.3
              }}>
              {loading ? "🧠 Analyzing symptoms..." : `🔬 Predict Disease${selected.length ? ` (${selected.length} symptoms)` : ""}`}
            </button>

            {apiStatus === "offline" && (
              <div style={{ marginTop: 14, padding: "11px 15px", background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 10, fontSize: 12, color: "#b91c1c" }}>
                <strong>Backend offline.</strong> Run <code style={{ background: "#fee2e2", padding: "1px 5px", borderRadius: 3 }}>python app.py</code> in the <code style={{ background: "#fee2e2", padding: "1px 5px", borderRadius: 3 }}>backend/</code> folder.
              </div>
            )}

            {error && (
              <div style={{ marginTop: 14, padding: "11px 15px", background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 10, fontSize: 12, color: "#b91c1c" }}>
                {error}
              </div>
            )}

            {result && <ResultCard result={result} onDownloadPDF={downloadPDF} pdfLoading={pdfLoading} />}
          </>
        )}
      </div>
    </div>
  );
}
