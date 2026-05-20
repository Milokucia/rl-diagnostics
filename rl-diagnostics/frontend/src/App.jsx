import { useState, useRef, useEffect } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

const API = "http://localhost:7842";

const CHART_COLORS = [
  "#60a5fa","#f472b6","#34d399","#fbbf24",
  "#a78bfa","#fb923c","#2dd4bf","#f87171",
];

const SEVERITY_STYLE = {
  critical: { bg: "#1a0505", border: "#7f1d1d", accent: "#ef4444", text: "#fca5a5" },
  warning:  { bg: "#1a1205", border: "#78350f", accent: "#f59e0b", text: "#fcd34d" },
  info:     { bg: "#060f1a", border: "#1e3a5f", accent: "#3b82f6", text: "#93c5fd" },
};

const ACTION_COLOR = {
  keep_training: "#22c55e",
  kill:          "#ef4444",
  tune:          "#f59e0b",
  investigate:   "#60a5fa",
};

const STATUS_COLOR = {
  healthy:    "#22c55e",
  promising:  "#34d399",
  plateau:    "#f59e0b",
  diverging:  "#f97316",
  collapsed:  "#ef4444",
};

// ── small components ────────────────────────────────────────────────────────

function Sparkline({ data, color }) {
  if (!data?.length) return null;
  return (
    <ResponsiveContainer width="100%" height={52}>
      <LineChart data={data}>
        <Line type="monotone" dataKey="value" stroke={color} strokeWidth={1.5} dot={false} />
        <XAxis dataKey="step" hide />
        <YAxis hide />
        <Tooltip
          contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6, fontSize: 10 }}
          formatter={v => [v.toFixed(4), ""]}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

function MetricCard({ tag, data, color }) {
  if (!data?.length) return null;
  const vals = data.map(d => d.value);
  const delta = vals[vals.length - 1] - vals[0];
  return (
    <div style={{ background:"#0f172a", border:"1px solid #1e293b", borderRadius:8, padding:"12px 14px", marginBottom:10 }}>
      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:6 }}>
        <span style={{ fontFamily:"monospace", fontSize:11, color:"#94a3b8", wordBreak:"break-all" }}>{tag}</span>
        <span style={{ fontSize:11, color: delta>0?"#22c55e":delta<0?"#f87171":"#64748b", marginLeft:8, whiteSpace:"nowrap" }}>
          {delta>0?"▲":delta<0?"▼":"—"} {Math.abs(delta).toFixed(3)}
        </span>
      </div>
      <Sparkline data={data} color={color} />
    </div>
  );
}

function HealthBar({ score }) {
  const color = score >= 70 ? "#22c55e" : score >= 40 ? "#f59e0b" : "#ef4444";
  return (
    <div style={{ marginBottom:16 }}>
      <div style={{ display:"flex", justifyContent:"space-between", fontSize:11, color:"#64748b", marginBottom:5 }}>
        <span style={{ letterSpacing:2, textTransform:"uppercase" }}>Health Score</span>
        <span style={{ color, fontWeight:700, fontSize:16 }}>{score}</span>
      </div>
      <div style={{ background:"#1e293b", borderRadius:4, height:6 }}>
        <div style={{ width:`${score}%`, height:"100%", borderRadius:4, background:color, transition:"width 0.4s" }} />
      </div>
    </div>
  );
}

function AnalysisPanel({ analysis }) {
  if (!analysis) return null;
  const { summary, failures, positives, next_steps, health_score, algo } = analysis;
  return (
    <div style={{ display:"flex", flexDirection:"column", gap:14 }}>
      {algo && (
        <div style={{ fontSize:10, color:"#475569", letterSpacing:2, textTransform:"uppercase" }}>
          Algorithm: <span style={{ color:"#60a5fa" }}>{algo}</span>
        </div>
      )}
      {health_score != null && <HealthBar score={health_score} />}
      <div style={{ background:"#0f172a", border:"1px solid #1e293b", borderRadius:8, padding:14 }}>
        <div style={{ fontSize:10, color:"#64748b", letterSpacing:2, marginBottom:8, textTransform:"uppercase" }}>Summary</div>
        <p style={{ margin:0, color:"#cbd5e1", fontSize:12, lineHeight:1.7 }}>{summary}</p>
      </div>

      {failures?.length > 0 && (
        <div>
          <div style={{ fontSize:10, letterSpacing:2, marginBottom:10, textTransform:"uppercase", color:"#ef4444" }}>
            ⚠ Failures ({failures.length})
          </div>
          {failures.map((f,i) => {
            const s = SEVERITY_STYLE[f.severity] || SEVERITY_STYLE.warning;
            return (
              <div key={i} style={{ background:s.bg, border:`1px solid ${s.border}`, borderLeft:`3px solid ${s.accent}`, borderRadius:6, padding:12, marginBottom:8 }}>
                <div style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
                  <span style={{ fontWeight:600, color:s.text, fontSize:12 }}>{f.name}</span>
                  {f.severity && (
                    <span style={{ fontSize:10, color:s.accent, textTransform:"uppercase", letterSpacing:1 }}>{f.severity}</span>
                  )}
                </div>
                <div style={{ color:"#94a3b8", fontSize:11, marginBottom:5 }}>
                  <span style={{ color:"#475569" }}>evidence: </span>{f.evidence}
                </div>
                <div style={{ color:"#86efac", fontSize:11 }}>
                  <span style={{ color:"#475569" }}>fix: </span>{f.fix}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {positives?.length > 0 && (
        <div>
          <div style={{ fontSize:10, color:"#22c55e", letterSpacing:2, marginBottom:8, textTransform:"uppercase" }}>✓ Positives</div>
          {positives.map((p,i) => (
            <div key={i} style={{ background:"#071a0f", border:"1px solid #14532d", borderLeft:"3px solid #22c55e", borderRadius:6, padding:"7px 12px", marginBottom:6, color:"#86efac", fontSize:11 }}>{p}</div>
          ))}
        </div>
      )}

      {next_steps?.length > 0 && (
        <div>
          <div style={{ fontSize:10, color:"#60a5fa", letterSpacing:2, marginBottom:8, textTransform:"uppercase" }}>→ Next Steps</div>
          {next_steps.map((s,i) => (
            <div key={i} style={{ background:"#060f1a", border:"1px solid #1e3a5f", borderLeft:"3px solid #3b82f6", borderRadius:6, padding:"7px 12px", marginBottom:6, color:"#93c5fd", fontSize:11, display:"flex", gap:8 }}>
              <span style={{ color:"#3b82f6", fontWeight:700 }}>{i+1}.</span>{s}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function BatchPanel({ analysis, chartData }) {
  if (!analysis) return null;
  const { runs, batch_summary } = analysis;
  const [selected, setSelected] = useState(null);

  return (
    <div>
      <div style={{ background:"#0f172a", border:"1px solid #1e293b", borderRadius:8, padding:14, marginBottom:16 }}>
        <div style={{ fontSize:10, color:"#64748b", letterSpacing:2, marginBottom:8, textTransform:"uppercase" }}>Batch Summary</div>
        <p style={{ margin:0, color:"#cbd5e1", fontSize:12, lineHeight:1.7 }}>{batch_summary}</p>
      </div>

      <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
        {[...runs].sort((a,b) => a.health_score - b.health_score).map((r,i) => (
          <div
            key={i}
            onClick={() => setSelected(selected === r.name ? null : r.name)}
            style={{
              background: selected === r.name ? "#1e293b" : "#0f172a",
              border:`1px solid ${selected===r.name?"#334155":"#1e293b"}`,
              borderLeft:`3px solid ${STATUS_COLOR[r.status]||"#475569"}`,
              borderRadius:8,
              padding:"10px 14px",
              cursor:"pointer",
            }}
          >
            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
              <span style={{ fontFamily:"monospace", fontSize:12, color:"#e2e8f0" }}>{r.name}</span>
              <div style={{ display:"flex", gap:8, alignItems:"center" }}>
                <span style={{ fontSize:10, color: STATUS_COLOR[r.status]||"#475569", textTransform:"uppercase", letterSpacing:1 }}>{r.status}</span>
                <span style={{ fontSize:11, fontWeight:700, color: ACTION_COLOR[r.action]||"#64748b",
                  background:"#020817", border:`1px solid ${ACTION_COLOR[r.action]||"#334155"}`, borderRadius:4, padding:"2px 7px" }}>
                  {r.action?.replace("_"," ")}
                </span>
                <span style={{ fontSize:13, fontWeight:700, color: r.health_score>=70?"#22c55e":r.health_score>=40?"#f59e0b":"#ef4444" }}>
                  {r.health_score}
                </span>
              </div>
            </div>
            {r.top_issue && r.top_issue !== "none" && (
              <div style={{ fontSize:11, color:"#64748b", marginTop:4 }}>↳ {r.top_issue}</div>
            )}
            {r.note && <div style={{ fontSize:11, color:"#475569", marginTop:3 }}>{r.note}</div>}

            {selected === r.name && chartData[r.name] && (
              <div style={{ marginTop:12, display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
                {Object.entries(chartData[r.name]).slice(0,4).map(([tag,data],ci) => (
                  <MetricCard key={tag} tag={tag} data={data} color={CHART_COLORS[ci%CHART_COLORS.length]} />
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function ChatPanel({ analysisContext, algo }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior:"smooth" }); }, [messages]);

  const send = async () => {
    if (!input.trim()) return;
    const question = input.trim();
    setInput("");
    const newMsgs = [...messages, { role:"user", content:question }];
    setMessages(newMsgs);
    setLoading(true);
    try {
      const res = await fetch(`${API}/api/chat`, {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({ question, context:JSON.stringify(analysisContext), algo,
          history:newMsgs.slice(0,-1).map(m=>({role:m.role,content:m.content})) }),
      });
      const data = await res.json();
      setMessages(prev => [...prev, { role:"assistant", content:data.reply||data.error }]);
    } catch(e) {
      setMessages(prev => [...prev, { role:"assistant", content:"Error: "+e.message }]);
    } finally { setLoading(false); }
  };

  return (
    <div style={{ display:"flex", flexDirection:"column", height:400 }}>
      <div style={{ flex:1, overflowY:"auto", padding:"10px 0", display:"flex", flexDirection:"column", gap:8 }}>
        {messages.length===0 && (
          <div style={{ color:"#475569", fontSize:12, textAlign:"center", marginTop:40 }}>
            Ask anything — why is entropy collapsing? what should I tune first?
          </div>
        )}
        {messages.map((m,i) => (
          <div key={i} style={{
            alignSelf: m.role==="user"?"flex-end":"flex-start",
            maxWidth:"85%",
            background: m.role==="user"?"#1e3a5f":"#1e293b",
            border:`1px solid ${m.role==="user"?"#2563eb":"#334155"}`,
            borderRadius:8, padding:"8px 12px", fontSize:12,
            color: m.role==="user"?"#bfdbfe":"#cbd5e1",
            lineHeight:1.6, whiteSpace:"pre-wrap",
          }}>{m.content}</div>
        ))}
        {loading && <div style={{ alignSelf:"flex-start", color:"#475569", fontSize:12, padding:"4px 8px" }}>thinking...</div>}
        <div ref={bottomRef} />
      </div>
      <div style={{ display:"flex", gap:8, paddingTop:10, borderTop:"1px solid #1e293b" }}>
        <input value={input} onChange={e=>setInput(e.target.value)} onKeyDown={e=>e.key==="Enter"&&!e.shiftKey&&send()}
          placeholder="Why is value loss still high at step 200k?"
          style={{ flex:1, background:"#0f172a", border:"1px solid #334155", borderRadius:6, padding:"8px 12px",
            color:"#e2e8f0", fontSize:12, outline:"none", fontFamily:"inherit" }} />
        <button onClick={send} disabled={loading||!input.trim()}
          style={{ background:loading?"#1e293b":"#2563eb", border:"none", borderRadius:6,
            padding:"8px 16px", color:"#fff", fontSize:12, cursor:loading?"default":"pointer" }}>
          send
        </button>
      </div>
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────

const BUILTIN_ALGOS = ["auto", "ppo", "sac", "td3", "ddpg", "dqn", "custom"];

export default function App() {
  const [mode, setMode]       = useState("single");   // "single" | "batch"
  const [logdir, setLogdir]   = useState("");
  const [algo, setAlgo]       = useState("auto");
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");
  const [result, setResult]   = useState(null);
  const [tab, setTab]         = useState("analysis");

  // custom algo registration
  const [showRegister, setShowRegister] = useState(false);
  const [customKey, setCustomKey]       = useState("");
  const [customName, setCustomName]     = useState("");
  const [customFamily, setCustomFamily] = useState("");
  const [customFailures, setCustomFailures] = useState("");
  const [customHealthy, setCustomHealthy]   = useState("");
  const [extraAlgos, setExtraAlgos]     = useState([]);

  const scan = async () => {
    setLoading(true); setError(""); setResult(null);
    const endpoint = mode === "batch" ? "/api/batch_scan" : "/api/scan";
    const body = mode === "batch"
      ? { parent_dir: logdir, algo: algo === "auto" ? "" : algo }
      : { logdir, algo: algo === "auto" ? "" : algo };
    try {
      const res = await fetch(`${API}${endpoint}`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(data);
      setTab("analysis");
    } catch(e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const registerAlgo = async () => {
    try {
      const res = await fetch(`${API}/api/register_algo`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({
          key: customKey,
          name: customName,
          family: customFamily,
          failures: customFailures.split("\n").filter(Boolean),
          healthy: customHealthy,
        }),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setExtraAlgos(prev => [...prev, customKey.toLowerCase()]);
      setAlgo(customKey.toLowerCase());
      setShowRegister(false);
    } catch(e) { setError(e.message); }
  };

  const allAlgos = [...BUILTIN_ALGOS, ...extraAlgos];

  return (
    <div style={{
      minHeight:"100vh", background:"#020817", color:"#e2e8f0",
      fontFamily:"'JetBrains Mono','Fira Code',monospace",
      padding:"28px 22px", maxWidth:1120, margin:"0 auto",
    }}>
      {/* Header */}
      <div style={{ marginBottom:26 }}>
        <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:4 }}>
          <div style={{ width:7, height:7, borderRadius:"50%", background:"#22c55e", boxShadow:"0 0 8px #22c55e" }} />
          <span style={{ fontSize:10, letterSpacing:3, color:"#475569", textTransform:"uppercase" }}>RL Diagnostics Agent</span>
        </div>
        <h1 style={{ margin:0, fontSize:20, fontWeight:700, color:"#f1f5f9", letterSpacing:-0.5 }}>Training Inspector</h1>
        <p style={{ margin:"4px 0 0", fontSize:11, color:"#475569" }}>
          Point at a TensorBoard logdir. Claude diagnoses failures — no graphs needed.
        </p>
      </div>

      {/* Mode + algo row */}
      <div style={{ display:"flex", gap:10, marginBottom:12, flexWrap:"wrap" }}>
        {["single","batch"].map(m => (
          <button key={m} onClick={() => { setMode(m); setResult(null); }}
            style={{ background: mode===m?"#1e3a5f":"#0f172a",
              border:`1px solid ${mode===m?"#2563eb":"#334155"}`,
              borderRadius:6, padding:"6px 14px", color: mode===m?"#60a5fa":"#475569",
              fontSize:11, cursor:"pointer", fontFamily:"inherit", textTransform:"uppercase", letterSpacing:1 }}>
            {m === "single" ? "Single Run" : "Batch Scan"}
          </button>
        ))}
        <div style={{ flex:1 }} />
        <select value={algo} onChange={e => setAlgo(e.target.value)}
          style={{ background:"#0f172a", border:"1px solid #334155", borderRadius:6,
            padding:"6px 12px", color:"#94a3b8", fontSize:11, fontFamily:"inherit", cursor:"pointer" }}>
          {allAlgos.map(a => <option key={a} value={a}>{a === "auto" ? "auto-detect algo" : a.toUpperCase()}</option>)}
        </select>
        <button onClick={() => setShowRegister(!showRegister)}
          style={{ background:"#0f172a", border:"1px solid #334155", borderRadius:6,
            padding:"6px 12px", color:"#475569", fontSize:11, cursor:"pointer", fontFamily:"inherit" }}>
          + custom algo
        </button>
      </div>

      {/* Custom algo registration */}
      {showRegister && (
        <div style={{ background:"#0f172a", border:"1px solid #334155", borderRadius:8, padding:16, marginBottom:14 }}>
          <div style={{ fontSize:10, color:"#60a5fa", letterSpacing:2, marginBottom:12, textTransform:"uppercase" }}>Register Custom Algorithm</div>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8, marginBottom:8 }}>
            {[["key (e.g. drq)", customKey, setCustomKey], ["full name", customName, setCustomName],
              ["family (e.g. off-policy model-based)", customFamily, setCustomFamily],
              ["healthy training description", customHealthy, setCustomHealthy]].map(([ph,val,set],i) => (
              <input key={i} value={val} onChange={e=>set(e.target.value)} placeholder={ph}
                style={{ background:"#020817", border:"1px solid #334155", borderRadius:6,
                  padding:"7px 10px", color:"#e2e8f0", fontSize:11, fontFamily:"inherit", outline:"none" }} />
            ))}
          </div>
          <textarea value={customFailures} onChange={e=>setCustomFailures(e.target.value)}
            placeholder={"Known failure modes (one per line):\nQ-value divergence: ...\nPolicy collapse: ..."}
            rows={4} style={{ width:"100%", background:"#020817", border:"1px solid #334155", borderRadius:6,
              padding:"7px 10px", color:"#e2e8f0", fontSize:11, fontFamily:"inherit", outline:"none",
              resize:"vertical", boxSizing:"border-box", marginBottom:8 }} />
          <button onClick={registerAlgo}
            style={{ background:"#2563eb", border:"none", borderRadius:6, padding:"7px 18px",
              color:"#fff", fontSize:11, cursor:"pointer", fontFamily:"inherit" }}>
            Register
          </button>
        </div>
      )}

      {/* Input */}
      <div style={{ display:"flex", gap:10, marginBottom:22 }}>
        <input value={logdir} onChange={e=>setLogdir(e.target.value)} onKeyDown={e=>e.key==="Enter"&&scan()}
          placeholder={mode==="batch" ? "~/runs/experiment_01  (parent dir with run subdirs)" : "~/runs/ppo_hand_v3"}
          style={{ flex:1, background:"#0f172a", border:"1px solid #334155", borderRadius:7,
            padding:"9px 13px", color:"#e2e8f0", fontSize:12, outline:"none", fontFamily:"inherit" }} />
        <button onClick={scan} disabled={loading||!logdir.trim()}
          style={{ background:loading?"#1e293b":"#2563eb", border:"none", borderRadius:7,
            padding:"9px 20px", color:"#fff", fontSize:12, cursor:loading?"default":"pointer",
            fontFamily:"inherit", letterSpacing:0.5 }}>
          {loading ? "scanning..." : "→ scan"}
        </button>
      </div>

      {error && (
        <div style={{ background:"#1a0a0a", border:"1px solid #7f1d1d", borderRadius:7,
          padding:"9px 13px", color:"#fca5a5", fontSize:12, marginBottom:18 }}>
          {error}
        </div>
      )}

      {/* Results */}
      {result && mode === "single" && (
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1.15fr", gap:18 }}>
          <div>
            <div style={{ fontSize:10, color:"#475569", letterSpacing:2, marginBottom:12, textTransform:"uppercase" }}>
              Metrics ({result.tags?.length})
            </div>
            <div style={{ maxHeight:"80vh", overflowY:"auto", paddingRight:4 }}>
              {result.tags?.map((tag,i) => (
                <MetricCard key={tag} tag={tag} data={result.chart_data?.[tag]} color={CHART_COLORS[i%CHART_COLORS.length]} />
              ))}
            </div>
          </div>
          <div>
            <div style={{ display:"flex", gap:0, marginBottom:14, borderBottom:"1px solid #1e293b" }}>
              {["analysis","chat"].map(t => (
                <button key={t} onClick={()=>setTab(t)}
                  style={{ background:"none", border:"none",
                    borderBottom: tab===t?"2px solid #3b82f6":"2px solid transparent",
                    padding:"7px 14px", color: tab===t?"#60a5fa":"#475569",
                    fontSize:11, cursor:"pointer", fontFamily:"inherit",
                    letterSpacing:1, textTransform:"uppercase", marginBottom:-1 }}>
                  {t}
                </button>
              ))}
            </div>
            {tab==="analysis" && <AnalysisPanel analysis={result.analysis} />}
            {tab==="chat" && <ChatPanel analysisContext={result.analysis} algo={result.algo} />}
          </div>
        </div>
      )}

      {result && mode === "batch" && (
        <div style={{ display:"grid", gridTemplateColumns:"1.2fr 1fr", gap:18 }}>
          <div>
            <div style={{ fontSize:10, color:"#475569", letterSpacing:2, marginBottom:12, textTransform:"uppercase" }}>
              Runs ({result.runs?.length}) — sorted by health score
            </div>
            <div style={{ maxHeight:"80vh", overflowY:"auto" }}>
              <BatchPanel analysis={result.analysis} chartData={result.chart_data||{}} />
            </div>
          </div>
          <div>
            <div style={{ borderBottom:"1px solid #1e293b", marginBottom:14, paddingBottom:7 }}>
              <span style={{ fontSize:11, color:"#60a5fa", letterSpacing:1, textTransform:"uppercase" }}>Chat</span>
            </div>
            <ChatPanel analysisContext={result.analysis} algo={algo} />
          </div>
        </div>
      )}
    </div>
  );
}
