"use client";

import {
  Activity,
  Archive,
  Beaker,
  BookOpen,
  Box,
  Braces,
  Check,
  ChevronRight,
  CircleDot,
  Clock3,
  Copy,
  Download,
  FileBarChart,
  FlaskConical,
  FolderKanban,
  Gauge,
  GitCommitHorizontal,
  Grid3X3,
  Library,
  LoaderCircle,
  MessageSquareText,
  PanelRight,
  Play,
  Plus,
  Save,
  Settings2,
  ShieldAlert,
  Sparkles,
  TerminalSquare,
  TriangleAlert,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { API_URL, apiRequest } from "../lib/api";
import type {
  ExperimentSpecification,
  ExperimentStatus,
  MetricSummary,
  PlannerConfiguration,
  PlannerId,
  ProtocolCompilation,
  SystemContext,
} from "../lib/types";

const starterPrompts = [
  "Compare all four planners over 30 deterministic seeds.",
  "Test how lambda_irr = 8 affects irreversible failure rate.",
  "Analyse the trade-off between path length and escape options.",
  "Generate a reproducible four-planner benchmark report.",
];

const plannerNames: Record<PlannerId, string> = {
  shortest: "Shortest path",
  risk_aware: "Risk-aware",
  recoverability_aware: "Recoverability-aware",
  proposed: "Joint objective",
};

const navigation = [
  ["Research Workspace", MessageSquareText],
  ["Experiments", FlaskConical],
  ["Scenarios", Grid3X3],
  ["Results", FileBarChart],
  ["Research Library", Library],
  ["System & Reproducibility", Settings2],
] as const;

function evidenceLabel(status?: ExperimentStatus["evidence_status"]) {
  if (status === "completed") return "Executed evidence";
  if (status === "partial") return "Partial evidence";
  if (status === "running" || status === "queued") return "Execution in progress";
  if (status === "failed" || status === "cancelled") return "No complete evidence";
  return "Configured · unexecuted";
}

function formatNumber(value: number, digits = 3) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: digits }).format(value);
}

function updatePlanner(
  specification: ExperimentSpecification,
  plannerId: PlannerId,
  key: "risk_weight" | "irreversibility_weight",
  value: number,
) {
  return {
    ...specification,
    planners: specification.planners.map((planner) =>
      planner.planner_id === plannerId ? { ...planner, [key]: value } : planner,
    ),
  };
}

export function ResearchWorkspace() {
  const [request, setRequest] = useState(starterPrompts[0]);
  const [compilation, setCompilation] = useState<ProtocolCompilation | null>(null);
  const [specification, setSpecification] = useState<ExperimentSpecification | null>(null);
  const [editorValue, setEditorValue] = useState("");
  const [showJson, setShowJson] = useState(false);
  const [status, setStatus] = useState<ExperimentStatus | null>(null);
  const [system, setSystem] = useState<SystemContext | null>(null);
  const [busy, setBusy] = useState<"compile" | "run" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    apiRequest<SystemContext>("/v1/system")
      .then(setSystem)
      .catch(() => setSystem(null));
    return () => {
      if (pollTimer.current) clearTimeout(pollTimer.current);
    };
  }, []);

  const plannedRuns = specification ? specification.seeds.length * specification.planners.length : 0;
  const progress = status ? ((status.completed_runs + status.failed_runs) / Math.max(status.total_runs, 1)) * 100 : 0;
  const latestEvents = status?.events.slice(-5).reverse() ?? [];

  function replaceSpecification(next: ExperimentSpecification) {
    setSpecification(next);
    setEditorValue(JSON.stringify(next, null, 2));
    setStatus(null);
  }

  async function compileProtocol() {
    setBusy("compile");
    setError(null);
    try {
      const result = await apiRequest<ProtocolCompilation>("/v1/protocols", {
        method: "POST",
        body: JSON.stringify({ request }),
      });
      setCompilation(result);
      replaceSpecification(result.specification);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Protocol compilation failed.");
    } finally {
      setBusy(null);
    }
  }

  function applyJson() {
    try {
      const parsed = JSON.parse(editorValue) as ExperimentSpecification;
      replaceSpecification(parsed);
      setShowJson(false);
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? `Configuration JSON: ${cause.message}` : "Invalid configuration JSON.");
    }
  }

  async function pollExperiment(experimentId: string) {
    try {
      const nextStatus = await apiRequest<ExperimentStatus>(`/v1/experiments/${experimentId}`);
      setStatus(nextStatus);
      if (["queued", "running"].includes(nextStatus.evidence_status)) {
        pollTimer.current = setTimeout(() => void pollExperiment(experimentId), 650);
      } else {
        setBusy(null);
      }
    } catch (cause) {
      setBusy(null);
      setError(cause instanceof Error ? cause.message : "Could not read experiment status.");
    }
  }

  async function runExperiment() {
    if (!specification) return;
    setBusy("run");
    setError(null);
    try {
      const created = await apiRequest<ExperimentStatus>("/v1/experiments", {
        method: "POST",
        body: JSON.stringify(specification),
      });
      setStatus(created);
      if (["completed", "partial"].includes(created.evidence_status)) {
        setBusy(null);
        return;
      }
      const queued = await apiRequest<ExperimentStatus>(`/v1/experiments/${created.experiment_id}/run`, {
        method: "POST",
      });
      setStatus(queued);
      await pollExperiment(created.experiment_id);
    } catch (cause) {
      setBusy(null);
      setError(cause instanceof Error ? cause.message : "Experiment execution failed.");
    }
  }

  function setSeedCount(count: number) {
    if (!specification) return;
    const bounded = Math.max(1, Math.min(500, Math.round(count)));
    replaceSpecification({ ...specification, seeds: Array.from({ length: bounded }, (_, index) => index) });
  }

  function setWeight(plannerId: PlannerId, key: "risk_weight" | "irreversibility_weight", value: number) {
    if (!specification) return;
    replaceSpecification(updatePlanner(specification, plannerId, key, Math.max(0, value)));
  }

  return (
    <main className="app-shell">
      <header className="product-header">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><CircleDot size={18} /></div>
          <div>
            <h1>DynNav Researcher</h1>
            <p>Risk- and recoverability-aware autonomous navigation research</p>
          </div>
        </div>
        <div className="system-strip" aria-label="Project and execution context">
          <span><FolderKanban size={14} /> DynNav / researcher</span>
          <span title={system?.git_commit_sha ?? "API unavailable"}><GitCommitHorizontal size={14} /> {system?.git_commit_sha.slice(0, 8) ?? "unresolved"}</span>
          <span><TerminalSquare size={14} /> Python {system?.python_version ?? "—"}</span>
          <span className={`status-dot status-${status?.evidence_status ?? "configured"}`}><Activity size={14} /> {status?.evidence_status ?? "idle"}</span>
        </div>
        <button className="button button-secondary" disabled={!status?.artifacts.length} onClick={() => {
          const report = status?.artifacts.find((item) => item.filename === "report.md");
          if (report) window.open(`${API_URL}${report.download_url}`, "_blank", "noopener,noreferrer");
        }}>
          <Download size={15} /> Export report
        </button>
      </header>

      <nav className="primary-nav" aria-label="Primary navigation">
        {navigation.map(([label, Icon], index) => (
          <button key={label} className={index === 0 ? "active" : ""} title={index === 0 ? label : `${label} · planned workspace module`}>
            <Icon size={15} /> {label}
          </button>
        ))}
      </nav>

      <div className="workspace-grid">
        <aside className="left-rail" aria-label="Research sessions">
          <button className="new-session"><Plus size={16} /> New research session</button>
          <section className="rail-section">
            <div className="section-label"><span>Saved sessions</span><ChevronRight size={14} /></div>
            <button className="session-row active-session">
              <span className="session-icon"><Beaker size={14} /></span>
              <span><strong>Four-planner study</strong><small>{evidenceLabel(status?.evidence_status)}</small></span>
            </button>
          </section>
          <section className="rail-section compact-list">
            <div className="section-label"><span>Research assets</span></div>
            <button><Clock3 size={14} /><span>Recent experiments</span><em>{status ? "1" : "0"}</em></button>
            <button><Gauge size={14} /><span>Experiment status</span><em>{status?.completed_runs ?? 0}/{status?.total_runs ?? plannedRuns}</em></button>
            <button><Grid3X3 size={14} /><span>Scenario library</span><em>1</em></button>
            <button><Box size={14} /><span>Planner presets</span><em>4</em></button>
            <button><FileBarChart size={14} /><span>Report history</span><em>{status?.artifacts.some((a) => a.kind === "report") ? "1" : "0"}</em></button>
          </section>
          <div className="rail-spacer" />
          <div className="rail-actions">
            <button disabled={!specification}><Save size={14} /> Rename</button>
            <button disabled={!specification}><Copy size={14} /> Duplicate</button>
            <button disabled={!specification}><Archive size={14} /> Archive</button>
          </div>
          <div className="boundary-note"><ShieldAlert size={15} /><p><strong>Evidence boundary</strong>Synthetic software experiments only. No hardware or safety claim.</p></div>
        </aside>

        <section className="research-stream" aria-label="AI researcher workspace">
          <div className="workflow-steps" aria-label="Research workflow">
            {[
              ["01", "Define"], ["02", "Design"], ["03", "Execute"], ["04", "Analyse"], ["05", "Report"],
            ].map(([number, label], index) => (
              <div key={number} className={(index === 0 || specification) && index < 2 ? "complete" : status && index === 2 ? "current" : status?.summary && index > 2 ? "complete" : ""}>
                <span>{number}</span><strong>{label}</strong>
              </div>
            ))}
          </div>

          <div className="stream-scroll">
            {!specification ? (
              <section className="empty-state">
                <div className="eyebrow"><Sparkles size={14} /> Evidence-bound protocol compiler</div>
                <h2>What should DynNav investigate?</h2>
                <p>Describe a comparison. The researcher will make the protocol explicit before any simulation is executed.</p>
                <div className="prompt-grid">
                  {starterPrompts.map((prompt) => <button key={prompt} onClick={() => setRequest(prompt)}>{prompt}<ChevronRight size={14} /></button>)}
                </div>
                <div className="capability-row">
                  <span><Check size={13} /> 4 canonical planners</span>
                  <span><Check size={13} /> Paired deterministic seeds</span>
                  <span><Check size={13} /> Bootstrap uncertainty</span>
                  <span><Check size={13} /> Provenance bundle</span>
                </div>
              </section>
            ) : (
              <>
                <article className="research-block question-block">
                  <div className="block-kicker"><MessageSquareText size={14} /> Research question <span className="evidence-tag configured">Configured</span></div>
                  <h2>{specification.title}</h2>
                  <p>{specification.research_question}</p>
                </article>

                <article className="research-block hypothesis-block">
                  <div className="block-kicker"><Beaker size={14} /> Hypotheses <span className="evidence-tag hypothesis">Hypothesis · not evidence</span></div>
                  <ol>{specification.hypotheses.map((hypothesis) => <li key={hypothesis}>{hypothesis}</li>)}</ol>
                </article>

                <article className="research-block canvas-block">
                  <div className="block-kicker"><FlaskConical size={14} /> Experiment canvas <span className="evidence-tag configured">{plannedRuns} planned runs</span></div>
                  <div className="experiment-canvas">
                    {[
                      ["Research question", "Defined", MessageSquareText],
                      ["Scenario set", `${specification.seeds.length} seeded maps`, Grid3X3],
                      ["Planner variants", `${specification.planners.length} objectives`, Box],
                      ["Execution matrix", `${plannedRuns} runs`, Gauge],
                      ["Statistics", `${specification.analysis_plan.bootstrap_resamples} resamples`, FileBarChart],
                      ["Report", status?.artifacts.some((a) => a.kind === "report") ? "Available" : "Pending evidence", BookOpen],
                    ].map(([label, detail, Icon], index) => (
                      <div className="canvas-node" key={String(label)}><Icon size={16} /><span><strong>{String(label)}</strong><small>{String(detail)}</small></span>{index < 5 && <ChevronRight size={13} className="node-arrow" />}</div>
                    ))}
                  </div>
                </article>

                {compilation?.capability_notices.map((notice) => (
                  <article className="notice-block" key={notice}><TriangleAlert size={16} /><p>{notice}</p></article>
                ))}

                {status && (
                  <article className="research-block execution-block">
                    <div className="block-kicker"><Activity size={14} /> Tool action <span className={`evidence-tag ${status.evidence_status}`}>{evidenceLabel(status.evidence_status)}</span></div>
                    <div className="execution-title"><div><h3>{status.message}</h3><p>{status.experiment_id} · configuration {status.configuration_hash.slice(0, 12)}</p></div><strong>{status.completed_runs + status.failed_runs}/{status.total_runs}</strong></div>
                    <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
                    <div className="progress-meta"><span>{formatNumber(progress, 0)}% matrix processed</span><span>{formatNumber(status.elapsed_seconds, 1)} s elapsed</span><span>{status.failed_runs} failed</span></div>
                    {!!latestEvents.length && <div className="event-log">{latestEvents.map((event) => <div key={event.sequence}><span>{String(event.sequence).padStart(2, "0")}</span><p>{event.message}</p></div>)}</div>}
                  </article>
                )}

                {status?.summary && (
                  <ResultsBlock summary={status.summary} status={status} />
                )}
              </>
            )}
          </div>

          <div className="composer">
            <div className="composer-top"><span><Sparkles size={14} /> Research request</span><span className="model-policy"><ShieldAlert size={13} /> Tools cannot invent results</span></div>
            <textarea aria-label="Research request" value={request} onChange={(event) => setRequest(event.target.value)} rows={3} />
            <div className="composer-actions">
              <span>Enter a question, comparison, metric, or parameter study.</span>
              <button className="button button-primary" onClick={compileProtocol} disabled={busy !== null || request.trim().length < 10}>
                {busy === "compile" ? <LoaderCircle className="spin" size={15} /> : <Sparkles size={15} />} Compile protocol
              </button>
            </div>
            {error && <div className="error-message" role="alert"><TriangleAlert size={14} /> {error}</div>}
          </div>
        </section>

        <aside className="inspector" aria-label="Experiment inspector">
          <div className="inspector-header"><div><PanelRight size={15} /><strong>Inspector</strong></div><span className={`evidence-tag ${status?.evidence_status ?? "configured"}`}>{evidenceLabel(status?.evidence_status)}</span></div>
          {!specification ? (
            <div className="inspector-empty"><Braces size={22} /><h3>No active protocol</h3><p>Compile a research question to inspect its exact configuration.</p></div>
          ) : (
            <div className="inspector-scroll">
              <InspectorSection title="Execution matrix" icon={Gauge}>
                <div className="inline-metrics"><div><strong>{plannedRuns}</strong><span>planned runs</span></div><div><strong>{specification.seeds.length}</strong><span>paired seeds</span></div></div>
                <label>Seed count<input type="number" min={1} max={500} value={specification.seeds.length} onChange={(event) => setSeedCount(Number(event.target.value))} /></label>
              </InspectorSection>
              <InspectorSection title="Scenario" icon={Grid3X3}>
                <div className="definition-row"><span>Family</span><code>{specification.scenario.family}</code></div>
                <div className="definition-row"><span>Grid</span><code>{specification.scenario.parameters.width} × {specification.scenario.parameters.height}</code></div>
                <div className="definition-row"><span>Obstacle probability</span><code>{specification.scenario.parameters.obstacle_probability}</code></div>
                <p className="field-note">Same generator and seed are used for every planner in a paired comparison.</p>
              </InspectorSection>
              <InspectorSection title="Objective weights" icon={Settings2}>
                {specification.planners.map((planner) => <PlannerControls key={planner.planner_id} planner={planner} onWeight={setWeight} />)}
                <div className="objective-formula"><span>J(path)</span><code>L + λ<sub>risk</sub>R + λ<sub>irr</sub>I</code></div>
              </InspectorSection>
              <InspectorSection title="Analysis" icon={FileBarChart}>
                <div className="definition-row"><span>Confidence</span><code>{specification.analysis_plan.confidence * 100}%</code></div>
                <div className="definition-row"><span>Bootstrap</span><code>{specification.analysis_plan.bootstrap_resamples}</code></div>
                <div className="definition-row"><span>Design</span><code>paired / exploratory</code></div>
              </InspectorSection>
              <button className="json-toggle" onClick={() => setShowJson((value) => !value)}><Braces size={14} /> {showJson ? "Close JSON editor" : "Edit protocol JSON"}<ChevronRight size={14} /></button>
              {showJson && <div className="json-editor"><textarea aria-label="Experiment JSON" value={editorValue} onChange={(event) => setEditorValue(event.target.value)} spellCheck={false} /><button onClick={applyJson}><Check size={14} /> Apply configuration</button></div>}
              {!!status?.artifacts.length && <InspectorSection title="Artifact provenance" icon={ShieldAlert}>
                <div className="artifact-list">{status.artifacts.map((artifact) => <a key={artifact.artifact_id} href={`${API_URL}${artifact.download_url}`} target="_blank" rel="noreferrer"><span><strong>{artifact.filename}</strong><small>{artifact.kind} · {(artifact.bytes / 1024).toFixed(1)} KB</small></span><code>{artifact.sha256.slice(0, 10)}</code><Download size={13} /></a>)}</div>
              </InspectorSection>}
            </div>
          )}
          <div className="inspector-footer">
            <button className="button button-primary run-button" disabled={!specification || busy !== null || ["queued", "running"].includes(status?.evidence_status ?? "")} onClick={runExperiment}>
              {busy === "run" ? <LoaderCircle className="spin" size={15} /> : <Play size={15} />} Run {plannedRuns || ""} planner executions
            </button>
            <p>Execution requires explicit confirmation. Completed experiments are immutable.</p>
          </div>
        </aside>
      </div>
    </main>
  );
}

function InspectorSection({ title, icon: Icon, children }: { title: string; icon: typeof Gauge; children: React.ReactNode }) {
  return <section className="inspector-section"><h3><Icon size={14} /> {title}</h3>{children}</section>;
}

function PlannerControls({ planner, onWeight }: { planner: PlannerConfiguration; onWeight: (id: PlannerId, key: "risk_weight" | "irreversibility_weight", value: number) => void }) {
  const riskEditable = planner.planner_id === "risk_aware" || planner.planner_id === "proposed";
  const irrEditable = planner.planner_id === "recoverability_aware" || planner.planner_id === "proposed";
  return <div className="planner-control"><strong>{plannerNames[planner.planner_id]}</strong><div><label>λ risk<input aria-label={`${plannerNames[planner.planner_id]} risk weight`} type="number" min={0} step={0.5} disabled={!riskEditable} value={planner.risk_weight} onChange={(event) => onWeight(planner.planner_id, "risk_weight", Number(event.target.value))} /></label><label>λ irr<input aria-label={`${plannerNames[planner.planner_id]} irreversibility weight`} type="number" min={0} step={0.5} disabled={!irrEditable} value={planner.irreversibility_weight} onChange={(event) => onWeight(planner.planner_id, "irreversibility_weight", Number(event.target.value))} /></label></div></div>;
}

function ResultsBlock({ summary, status }: { summary: Record<string, MetricSummary>; status: ExperimentStatus }) {
  const ordered = (["shortest", "risk_aware", "recoverability_aware", "proposed"] as PlannerId[]).filter((id) => summary[id]);
  return <article className="research-block results-block">
    <div className="block-kicker"><FileBarChart size={14} /> Results summary <span className={`evidence-tag ${status.evidence_status}`}>{evidenceLabel(status.evidence_status)}</span></div>
    <div className="result-callouts"><div><span>Executed runs</span><strong>{status.completed_runs}</strong></div><div><span>Failed runs</span><strong>{status.failed_runs}</strong></div><div><span>Paired seeds</span><strong>{Math.min(...ordered.map((id) => summary[id].trials))}</strong></div><div><span>Artifacts</span><strong>{status.artifacts.length}</strong></div></div>
    <div className="table-scroll"><table><thead><tr><th>Planner</th><th>n</th><th>Success</th><th>Irrev. failure</th><th>Path length</th><th>Cumulative risk</th><th>Planning time</th></tr></thead><tbody>{ordered.map((id) => {
      const row = summary[id];
      return <tr key={id}><td><span className={`planner-swatch swatch-${id}`} />{plannerNames[id]}</td><td>{row.trials}</td><td>{formatNumber(row.success_rate * 100, 1)}%</td><td>{formatNumber(row.irreversible_failure_rate * 100, 1)}%</td><td>{formatNumber(row.path_length.summary.mean)}</td><td>{formatNumber(row.cumulative_risk.summary.mean)}</td><td>{formatNumber(row.planning_time_ms.summary.mean)} ms</td></tr>;
    })}</tbody></table></div>
    <div className="interpretation-note"><ShieldAlert size={16} /><p><strong>Observation, not causal conclusion.</strong> Review confidence intervals, paired effects, failures, and scenario assumptions in the report before interpreting planner differences.</p></div>
  </article>;
}
