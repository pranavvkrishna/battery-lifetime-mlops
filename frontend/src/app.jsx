import { useState, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from "recharts";

const API_BASE = "https://ca-battery-serve-38a4bf.calmrock-176121a2.westus.azurecontainerapps.io";
const NOMINAL_CAPACITY = 1.1;

const MODEL_RESULTS = [
  { model: "MLP", type: "Engineered features", mae: 80.6, r2: 0.85, status: "production" },
  { model: "XGBoost", type: "Engineered features", mae: 78.9, r2: 0.84, status: "evaluated" },
  { model: "Random Forest", type: "Engineered features", mae: 83.4, r2: 0.81, status: "evaluated" },
  { model: "LSTM", type: "Raw sequences", mae: 262.0, r2: -0.01, status: "rejected" },
  { model: "1D CNN", type: "Raw sequences", mae: 260.5, r2: -0.01, status: "rejected" },
];

function featuresForCycles(cycles) {
  const fadeFraction = cycles / 1200;
  return {
    qd_current: +(1.1 - fadeFraction * 0.25).toFixed(4),
    qd_slope: -0.00015 - fadeFraction * 0.0003,
    qd_min: +(1.08 - fadeFraction * 0.28).toFixed(4),
    qd_std: 0.008 + fadeFraction * 0.01,
    qc_slope: -0.0001 - fadeFraction * 0.0002,
    qc_mean: +(1.09 - fadeFraction * 0.2).toFixed(4),
    ir_current: +(0.014 + fadeFraction * 0.006).toFixed(5),
    ir_slope: 0.00001 + fadeFraction * 0.00002,
    ir_mean: +(0.014 + fadeFraction * 0.005).toFixed(5),
    tavg_mean: 32.5,
    tavg_std: 1.2,
    tmax_mean: 35.0,
    tmin_mean: 30.0,
    chargetime_mean: 10.5 + fadeFraction * 2,
    chargetime_slope: 0.01 + fadeFraction * 0.01,
    window_size: 50,
  };
}

function buildChartData(cyclesObserved, predictedRul) {
  const observed = Array.from({ length: cyclesObserved }, (_, i) => {
    const f = featuresForCycles(i);
    return { cycle: i, capacity: f.qd_current };
  });
  const lastCapacity = observed[observed.length - 1]?.capacity ?? 1.1;
  const steps = 12;
  const projected = predictedRul
    ? Array.from({ length: steps + 1 }, (_, i) => ({
        cycle: cyclesObserved + (i * predictedRul) / steps,
        projected: lastCapacity - (i / steps) * (lastCapacity - 0.88),
      }))
    : [];
  return [...observed, ...projected];
}

function StatusLabel({ status }) {
  const label = { production: "Production", evaluated: "Evaluated", rejected: "Rejected" }[status];
  return <span className={`status-text status-${status}`}>{label}</span>;
}

function BatteryGauge({ percent }) {
  const clamped = Math.max(0, Math.min(100, percent));
  const width = 60, height = 26, pad = 3, nubWidth = 5, nubHeight = 12;
  const innerWidth = width - pad * 2;
  const innerHeight = height - pad * 2;
  const fillWidth = innerWidth * (clamped / 100);

  return (
    <svg width={width + nubWidth} height={height} viewBox={`0 0 ${width + nubWidth} ${height}`}>
      <rect x={0.75} y={0.75} width={width - 1.5} height={height - 1.5} rx={4} className="battery-case" />
      <rect x={pad} y={pad} width={innerWidth} height={innerHeight} rx={1.5} className="battery-track" />
      <rect x={pad} y={pad} width={fillWidth} height={innerHeight} rx={1.5} className="battery-fill" />
      <rect x={width} y={(height - nubHeight) / 2} width={nubWidth} height={nubHeight} rx={1.5} className="battery-nub" />
    </svg>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [cyclesObserved, setCyclesObserved] = useState(50);

  useEffect(() => {
    fetch(`${API_BASE}/health`).then((r) => r.json()).then(setHealth).catch(() => setHealth({ status: "unreachable" }));
    fetch(`${API_BASE}/model-info`).then((r) => r.json()).then(setModelInfo).catch(() => {});
  }, []);

  async function runPrediction() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(featuresForCycles(cyclesObserved)),
      });
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      setPrediction(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const currentFeatures = featuresForCycles(cyclesObserved);
  const soh = ((currentFeatures.qd_current / NOMINAL_CAPACITY) * 100).toFixed(1);
  const chartData = buildChartData(cyclesObserved, prediction?.predicted_rul);

  return (
    <div className="page">
      <header className="header">
        <div className="header-row">
          <div>
            <div className="eyebrow">Battery Intelligence</div>
            <h1>Remaining Useful Life</h1>
          </div>
          <div className="header-meta">
            <span>{modelInfo?.registry_name ?? "—"}</span>
            <span>v{modelInfo?.version ?? "—"}</span>
            <span>Azure ML MLflow</span>
            <span>Azure Container Apps</span>
          </div>
        </div>
      </header>

      <main className="layout">
        <section className="panel">
          <div className="row-top">
            <div>
              <div className="label">Predicted RUL</div>
              <div className="big-number">
                {prediction ? Math.round(prediction.predicted_rul) : "—"}
                <span className="unit">cycles</span>
              </div>
            </div>
            <span className={`status-badge ${health?.status === "healthy" ? "ok" : "off"}`}>
              {health?.status === "healthy" ? "Connected" : "Connecting…"}
            </span>
          </div>

          <div className="stats-row">
            <div>
              <span className="label">State of health</span>
              <div className="soh-row">
                <BatteryGauge percent={Number(soh)} />
                <span className="value">{soh}%</span>
              </div>
            </div>
            <div><span className="label">Cycles observed</span><div className="value">{cyclesObserved}</div></div>
          </div>

          <div className="slider-row">
            <label htmlFor="cycles">Cycles observed</label>
            <input
              id="cycles" type="range" min="20" max="300" step="10"
              value={cyclesObserved}
              onChange={(e) => setCyclesObserved(Number(e.target.value))}
            />
            <span className="slider-value">{cyclesObserved}</span>
          </div>

          <button onClick={runPrediction} disabled={loading} className="predict-btn">
            {loading ? "Predicting…" : "Predict"}
          </button>
          {error && <div className="error">Couldn't reach the API — {error}</div>}
        </section>

        <section className="panel">
          <div className="panel-title">Capacity vs. cycle</div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                <XAxis dataKey="cycle" tick={{ fontSize: 11, fill: "#66806F" }} tickLine={false} axisLine={{ stroke: "#263229" }} />
                <YAxis
                  domain={[0.85, 1.12]}
                  tick={{ fontSize: 11, fill: "#66806F" }}
                  tickLine={false}
                  axisLine={false}
                  width={40}
                  tickFormatter={(v) => v.toFixed(2)}
                />
                <Tooltip contentStyle={{ fontSize: 12, background: "#0D1310", border: "1px solid #263229", borderRadius: 4, color: "#D6E8DC" }} />
                <Line type="monotone" dataKey="capacity" stroke="#3ECF6E" strokeWidth={2} dot={false} name="Observed" isAnimationActive={false} />
                <Line type="monotone" dataKey="projected" stroke="#3C4E43" strokeWidth={2} strokeDasharray="4 4" dot={false} name="Projected" isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="panel panel-wide">
          <div className="panel-title">Model comparison</div>
          <table className="table">
            <thead>
              <tr><th>Model</th><th>MAE</th><th>R²</th><th></th></tr>
            </thead>
            <tbody>
              {MODEL_RESULTS.map((m) => (
                <tr key={m.model}>
                  <td>{m.model}</td>
                  <td>{m.mae.toFixed(1)}</td>
                  <td>{m.r2.toFixed(2)}</td>
                  <td><StatusLabel status={m.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </main>
    </div>
  );
}