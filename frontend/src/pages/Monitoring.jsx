import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import {
  Activity,
  AlertTriangle,
  ShieldCheck,
  CheckCircle2,
  RotateCcw,
  Play,
  Layers,
  Sparkles,
  ArrowRight,
  RefreshCw,
  BarChart3,
  Sliders,
  TrendingUp,
  Briefcase,
  Settings,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import { useViewMode } from '../context/ViewModeContext'
import ProjectPicker from '../components/common/ProjectPicker'
import Spinner from '../components/common/Spinner'
import { getLastProjectId, setLastProjectId } from '../lib/lastProject'
import { Badge } from '../components/ui/Badge'
import { ProgressBar } from '../components/ui/ProgressBar'
import { EmptyState } from '../components/ui/EmptyState'
import { SkeletonCard, SkeletonStat } from '../components/ui/Skeleton'

function pct(v) {
  if (v == null || Number.isNaN(Number(v))) return '—'
  return `${Math.round(Number(v) * 100)}%`
}

function scoreLabel(project) {
  const ready = project?.status === 'ready' || project?.status === 'trained'
  const m = project?.active_model
  if (!m) {
    if (ready) {
      return {
        label: 'Ready',
        detail: 'Select to inspect ranking quality and learning.',
        tone: 'teal',
      }
    }
    return { label: 'Not prepared', detail: 'Train this project to see performance.', tone: 'muted' }
  }
  if (project.problem_type === 'regression') {
    const r2 = m.r2_score != null ? Number(m.r2_score) : null
    if (r2 == null) return { label: 'Ready', detail: 'Regression model is live.', tone: 'teal' }
    if (r2 >= 0.5) return { label: 'Strong guide', detail: `R² ${r2.toFixed(2)} — patterns look reliable.`, tone: 'teal' }
    if (r2 >= 0.2) return { label: 'Useful guide', detail: `R² ${r2.toFixed(2)} — directionally helpful.`, tone: 'ink' }
    return { label: 'Rough guide', detail: `R² ${r2.toFixed(2)} — use as a hint only.`, tone: 'coral' }
  }
  const auc = m.auc_roc != null ? Number(m.auc_roc) : null
  const acc = m.accuracy != null ? Number(m.accuracy) : null
  const score = auc ?? acc
  if (score == null) return { label: 'Ready', detail: 'Model is live.', tone: 'teal' }
  if (score >= 0.8) return { label: 'Strong guide', detail: 'Good at ranking who needs attention.', tone: 'teal' }
  if (score >= 0.7) return { label: 'Solid guide', detail: 'Solid enough for a daily priority list.', tone: 'teal' }
  if (score >= 0.6) return { label: 'Useful guide', detail: 'Helpful direction — confirm before big spends.', tone: 'ink' }
  return { label: 'Rough guide', detail: 'Treat as a first filter; add human judgment.', tone: 'coral' }
}

/**
 * Top-level Monitoring — model quality + learning without opening project settings.
 */
export default function Monitoring() {
  const { token } = useAuth()
  const { plainEnglish, term } = useViewMode()
  const [searchParams, setSearchParams] = useSearchParams()
  const [projects, setProjects] = useState([])
  const [loading, setLoading] = useState(true)
  const [projectId, setProjectId] = useState('')
  const [project, setProject] = useState(null)
  const [spotCheck, setSpotCheck] = useState(null)
  const [feedback, setFeedback] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [spotLoading, setSpotLoading] = useState(false)
  const [driftReport, setDriftReport] = useState(null)
  const [modelHistory, setModelHistory] = useState([])
  const [candidateInfo, setCandidateInfo] = useState(null)
  const [candidateTraining, setCandidateTraining] = useState(false)
  const [candidatePromoting, setCandidatePromoting] = useState(false)
  const [rollbackLoading, setRollbackLoading] = useState(false)
  const [actionMsg, setActionMsg] = useState(null)

  const readyProjects = useMemo(
    () => projects.filter((p) => p.status === 'ready' || p.status === 'trained'),
    [projects]
  )

  useEffect(() => {
    if (!token) return
    ;(async () => {
      try {
        const res = await fetch('/api/projects', {
          headers: { Authorization: `Bearer ${token}` },
        })
        if (res.ok) {
          const data = await res.json()
          setProjects(data)
          const ready = data.filter((p) => p.status === 'ready' || p.status === 'trained')
          const fromQuery = searchParams.get('project')
          const last = getLastProjectId()
          const pick =
            ready.find((p) => p.id === fromQuery)?.id ||
            ready.find((p) => p.id === last)?.id ||
            ready[0]?.id ||
            ''
          setProjectId(pick)

          // Enrich cards with active_model metrics (list endpoint omits them)
          if (ready.length) {
            const details = await Promise.all(
              ready.slice(0, 12).map(async (p) => {
                try {
                  const r = await fetch(`/api/projects/${p.id}`, {
                    headers: { Authorization: `Bearer ${token}` },
                  })
                  return r.ok ? await r.json() : p
                } catch {
                  return p
                }
              })
            )
            const byId = Object.fromEntries(details.map((d) => [d.id, d]))
            setProjects((prev) => prev.map((p) => byId[p.id] || p))
          }
        }
      } catch (err) {
        console.error(err)
      }
      setLoading(false)
    })()
  }, [token])

  const loadDetail = async (id) => {
    if (!token || !id) {
      setProject(null)
      setSpotCheck(null)
      setFeedback(null)
      return
    }
    setDetailLoading(true)
    setLastProjectId(id)
    try {
      const [pRes, sRes, fRes, dRes, hRes] = await Promise.all([
        fetch(`/api/projects/${id}`, { headers: { Authorization: `Bearer ${token}` } }),
        fetch(`/api/projects/${id}/spot-check?limit=50`, {
          headers: { Authorization: `Bearer ${token}` },
        }),
        fetch(`/api/projects/${id}/feedback-summary`, {
          headers: { Authorization: `Bearer ${token}` },
        }),
        fetch(`/api/monitoring/projects/${id}/drift`, {
          headers: { Authorization: `Bearer ${token}` },
        }),
        fetch(`/api/monitoring/projects/${id}/history`, {
          headers: { Authorization: `Bearer ${token}` },
        }),
      ])
      setProject(pRes.ok ? await pRes.json() : null)
      setSpotCheck(sRes.ok ? await sRes.json() : null)
      setFeedback(fRes.ok ? await fRes.json() : null)
      setDriftReport(dRes.ok ? await dRes.json() : null)
      setModelHistory(hRes.ok ? await hRes.json() : [])
    } catch {
      setProject(null)
      setSpotCheck(null)
      setFeedback(null)
      setDriftReport(null)
      setModelHistory([])
    }
    setDetailLoading(false)
  }

  const handleTrainCandidate = async (triggerReason = 'Evidence-based drift trigger') => {
    if (!token || !projectId) return
    setCandidateTraining(true)
    setActionMsg(null)
    try {
      const res = await fetch(`/api/monitoring/projects/${projectId}/candidate/train`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ trigger_reason: triggerReason }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Candidate training failed')
      setCandidateInfo(data)
      setActionMsg({ type: 'success', text: `Candidate ${data.candidate_version} trained in isolation!` })
      loadDetail(projectId)
    } catch (err) {
      setActionMsg({ type: 'error', text: err.message })
    } finally {
      setCandidateTraining(false)
    }
  }

  const handlePromoteCandidate = async (candId, force = false) => {
    if (!token || !projectId || !candId) return
    setCandidatePromoting(true)
    setActionMsg(null)
    try {
      const res = await fetch(`/api/monitoring/projects/${projectId}/candidate/${candId}/promote`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ force }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Promotion failed')
      setCandidateInfo(null)
      setActionMsg({ type: 'success', text: data.message })
      loadDetail(projectId)
    } catch (err) {
      setActionMsg({ type: 'error', text: err.message })
    } finally {
      setCandidatePromoting(false)
    }
  }

  const handleRollback = async () => {
    if (!token || !projectId) return
    setRollbackLoading(true)
    setActionMsg(null)
    try {
      const res = await fetch(`/api/monitoring/projects/${projectId}/rollback`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Rollback failed')
      setActionMsg({ type: 'success', text: data.message })
      loadDetail(projectId)
    } catch (err) {
      setActionMsg({ type: 'error', text: err.message })
    } finally {
      setRollbackLoading(false)
    }
  }

  useEffect(() => {
    loadDetail(projectId)
  }, [token, projectId])

  const handleProjectChange = (id) => {
    setProjectId(id)
    const next = new URLSearchParams(searchParams)
    if (id) next.set('project', id)
    else next.delete('project')
    setSearchParams(next, { replace: true })
  }

  const refreshSpot = async () => {
    if (!projectId) return
    setSpotLoading(true)
    try {
      const res = await fetch(`/api/projects/${projectId}/spot-check?limit=50`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (res.ok) setSpotCheck(await res.json())
    } catch {
      /* ignore */
    }
    setSpotLoading(false)
  }

  const health = scoreLabel(project)
  const m = project?.active_model
  const isReg = project?.problem_type === 'regression'

  if (loading) {
    return (
      <div className="page flex justify-center items-center min-h-[40vh]">
        <Spinner />
      </div>
    )
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <p className="page-kicker">Operational Trust</p>
          <h1 className="page-title">Model Health & Reliability</h1>
          <p className="page-sub max-w-xl">
            Evaluate whether prediction guidance remains reliable over time, verify ranking quality against known cases, and inspect feedback learning from recorded business outcomes.
          </p>
        </div>
        {projectId && (
          <div className="flex flex-wrap gap-2">
            <Link to={`/cases?project=${projectId}`} className="btn-secondary text-xs inline-flex items-center gap-1.5">
              <Briefcase size={13} />
              <span>Open cases</span>
            </Link>
            <Link to={`/projects/${projectId}`} className="btn-ghost text-xs inline-flex items-center gap-1.5">
              <Settings size={13} />
              <span>Project settings</span>
            </Link>
          </div>
        )}
      </div>

      {readyProjects.length === 0 ? (
        <EmptyState
          icon={Activity}
          title="Nothing to monitor yet"
          description="Prepare a project first — then model quality, operational drift, and feedback learning show up here."
          actionLabel="Go to Projects"
          actionHref="/projects"
        />
      ) : (
        <>
          {/* Org overview cards */}
          <section className="mb-10">
            <h2 className="font-display text-lg font-semibold text-ink mb-3">All ready projects</h2>
            <ul className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {readyProjects.map((p) => {
                const h = scoreLabel(p)
                const selected = p.id === projectId
                return (
                  <li key={p.id}>
                    <button
                      type="button"
                      onClick={() => handleProjectChange(p.id)}
                      className={`w-full text-left border px-4 py-4 rounded-control transition-colors ${
                        selected
                          ? 'border-teal bg-teal-soft/20'
                          : 'border-mist hover:border-teal/40'
                      }`}
                    >
                      <div className="font-medium text-ink truncate">{p.name}</div>
                      <div
                        className={`mt-2 text-sm font-medium ${
                          h.tone === 'coral'
                            ? 'text-coral'
                            : h.tone === 'teal'
                              ? 'text-teal'
                              : 'text-ink'
                        }`}
                      >
                        {h.label}
                      </div>
                      <p className="mt-1 text-xs text-[var(--muted)] leading-relaxed">{h.detail}</p>
                    </button>
                  </li>
                )
              })}
            </ul>
          </section>

          <div className="mb-6">
            <ProjectPicker
              projects={readyProjects}
              value={projectId}
              onChange={handleProjectChange}
              id="monitoring-project"
              label="Inspect project"
            />
          </div>

          {detailLoading ? (
            <div className="flex items-center gap-3 text-sm text-[var(--muted)] py-10">
              <Spinner className="h-4 w-4" /> Loading performance…
            </div>
          ) : !project ? (
            <p className="text-sm text-[var(--muted)]">Could not load this project.</p>
          ) : (
            <div className="space-y-10">
              <section>
                <div className="flex flex-wrap items-start justify-between gap-3 mb-4">
                  <div>
                    <h2 className="font-display text-lg font-semibold text-ink">{project.name}</h2>
                    <p className="text-sm text-[var(--muted)] mt-1">
                      <span className="text-ink font-medium">{health.label}</span>
                      {' — '}
                      {health.detail}
                    </p>
                  </div>
                  {m?.trained_at && (
                    <p className="text-xs text-[var(--muted)]">
                      Last prepared {new Date(m.trained_at).toLocaleDateString()}
                      {m.version != null ? ` · v${m.version}` : ''}
                    </p>
                  )}
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-mist border border-mist">
                  {isReg ? (
                    <>
                      <MetricCell
                        label={plainEnglish ? 'Pattern Reliability (R²)' : 'R² Score'}
                        value={m?.r2_score != null ? Number(m.r2_score).toFixed(2) : '—'}
                      />
                      <MetricCell
                        label={plainEnglish ? 'Avg Prediction Error (MAE)' : 'Mean Absolute Error (MAE)'}
                        value={m?.mae != null ? Number(m.mae).toFixed(2) : '—'}
                      />
                      <MetricCell
                        label={plainEnglish ? 'Severe Miss Penalty (RMSE)' : 'Root Mean Sq Error (RMSE)'}
                        value={m?.rmse != null ? Number(m.rmse).toFixed(2) : '—'}
                      />
                      <MetricCell
                        label={plainEnglish ? 'Usability Guidance' : 'Readiness Status'}
                        value={health.label}
                      />
                    </>
                  ) : (
                    <>
                      <MetricCell
                        label={plainEnglish ? 'Ranking Reliability (AUC)' : 'ROC-AUC Score'}
                        value={pct(m?.auc_roc)}
                        accent="text-teal"
                        progress={m?.auc_roc}
                        color="teal"
                      />
                      <MetricCell
                        label={plainEnglish ? 'Overall Match Rate' : 'Accuracy Score'}
                        value={pct(m?.accuracy)}
                        progress={m?.accuracy}
                        color="teal"
                      />
                      <MetricCell
                        label={plainEnglish ? 'False Alarm Filter (Precision)' : 'Precision (PPV)'}
                        value={pct(m?.precision)}
                        progress={m?.precision}
                        color="teal"
                      />
                      <MetricCell
                        label={plainEnglish ? 'Catch Rate (Recall)' : 'Recall / Sensitivity'}
                        value={pct(m?.recall)}
                        progress={m?.recall}
                        color="teal"
                      />
                    </>
                  )}
                </div>
                <p className="mt-3 text-xs text-[var(--muted)] max-w-2xl leading-relaxed">
                  {plainEnglish
                    ? 'These scores measure how well the AI performed on real test data it had never seen before.'
                    : 'Training holdout validation metrics evaluated on an out-of-sample test partition.'}
                </p>
              </section>

              {!isReg && (
                <section className="border border-mist">
                  <div className="px-5 py-4 border-b border-mist flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <h2 className="font-display text-lg font-semibold text-ink">
                        Does the ranking help?
                      </h2>
                      <p className="text-sm text-[var(--muted)] mt-1">
                        Spot-check against known outcomes in your data
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={refreshSpot}
                      disabled={spotLoading}
                      className="btn-ghost text-sm"
                    >
                      {spotLoading ? 'Checking…' : 'Refresh'}
                    </button>
                  </div>
                  <div className="px-5 py-5">
                    {spotCheck?.supported === false && (
                      <p className="text-sm text-[var(--muted)]">{spotCheck.message}</p>
                    )}
                    {spotCheck && spotCheck.n === 0 && (
                      <p className="text-sm text-[var(--muted)]">
                        {spotCheck.message || 'Not enough known outcomes to check yet.'}
                      </p>
                    )}
                    {spotCheck && spotCheck.n > 0 && (
                      <>
                        <p className="text-sm text-ink max-w-2xl leading-relaxed mb-4">
                          {spotCheck.plain_summary}
                        </p>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-px bg-mist border border-mist">
                          <MetricCell
                            label={plainEnglish ? 'Matched Known Outcomes' : 'Validation Concordance Rate'}
                            value={pct(spotCheck.agree_rate)}
                            accent="text-ink"
                          />
                          <MetricCell
                            label={plainEnglish ? 'Right When Flagged High' : 'Precision @ High Risk Cutoff'}
                            value={
                              spotCheck.high_risk_precision != null
                                ? pct(spotCheck.high_risk_precision)
                                : '—'
                            }
                          />
                          <MetricCell
                            label={plainEnglish ? 'Right When Flagged Low' : 'Specificity @ Low Risk Cutoff'}
                            value={
                              spotCheck.low_risk_true_negative_rate != null
                                ? pct(spotCheck.low_risk_true_negative_rate)
                                : '—'
                            }
                          />
                        </div>
                      </>
                    )}
                    {!spotCheck && (
                      <p className="text-sm text-[var(--muted)]">Spot-check unavailable.</p>
                    )}
                  </div>
                </section>
              )}

              <section className="border border-mist">
                <div className="px-5 py-4 border-b border-mist">
                  <h2 className="font-display text-lg font-semibold text-ink">
                    Outcomes your team recorded
                  </h2>
                  <p className="text-sm text-[var(--muted)] mt-1">
                    Learning from logged results and follow-up check-ins
                  </p>
                </div>
                <div className="px-5 py-5 space-y-4">
                  {feedback ? (
                    <>
                      <p className="text-sm text-[var(--muted)] leading-relaxed max-w-2xl">
                        {feedback.plain_summary}
                      </p>
                      {feedback.learning?.plain && (
                        <p className="text-sm text-teal leading-relaxed max-w-2xl">
                          {feedback.learning.plain}
                        </p>
                      )}
                      <div className="grid sm:grid-cols-3 gap-px bg-mist border border-mist">
                        <MetricCell
                          label="Recorded"
                          value={String(feedback.with_feedback || 0)}
                          hint={`of ${feedback.total_predictions || 0} cases reviewed`}
                        />
                        <MetricCell
                          label="Estimate matched"
                          value={
                            feedback.model_match_rate != null
                              ? pct(feedback.model_match_rate)
                              : '—'
                          }
                        />
                        <MetricCell
                          label="Reshaping rankings"
                          value={String(feedback.learning?.actions_reshaping_rankings ?? 0)}
                          hint="actions with 3+ outcomes"
                        />
                      </div>
                      {Array.isArray(feedback.action_effectiveness_ranked) &&
                        feedback.action_effectiveness_ranked.length > 0 && (
                          <ul className="mt-3 space-y-2.5">
                            {feedback.action_effectiveness_ranked.slice(0, 6).map((a) => {
                              const rate = a.n > 0 ? (a.success_n / a.n) * 100 : 0
                              return (
                                <li
                                  key={a.action_code}
                                  className="p-3 bg-paper/60 border border-mist/60 rounded-control"
                                >
                                  <div className="flex flex-wrap items-center justify-between gap-2 text-sm mb-1.5">
                                    <span className="text-ink font-semibold flex items-center gap-2">
                                      {a.action_name || a.action_code}
                                      {a.reliable && (
                                        <Badge variant="safe" size="xs">
                                          Reshapes ranking
                                        </Badge>
                                      )}
                                    </span>
                                    <span className="tabular-nums text-muted text-xs">
                                      <span className="text-teal font-semibold">{a.success_n}</span>/{a.n} favorable ({Math.round(rate)}%)
                                    </span>
                                  </div>
                                  <ProgressBar value={rate} max={100} color="teal" size="xs" />
                                </li>
                              )
                            })}
                          </ul>
                        )}
                    </>
                  ) : (
                    <p className="text-sm text-[var(--muted)]">
                      No outcome log yet. Log results from Cases or follow-up check-ins.
                    </p>
                  )}
                </div>
              </section>

              {/* Status Message */}
              {actionMsg && (
                <div
                  className={`p-4 rounded-xl border text-xs flex items-center justify-between ${
                    actionMsg.type === 'success'
                      ? 'border-teal/30 bg-teal/10 text-teal'
                      : 'border-coral/30 bg-coral/10 text-coral'
                  }`}
                >
                  <span>{actionMsg.text}</span>
                  <button onClick={() => setActionMsg(null)} className="text-muted hover:text-ink">
                    ✕
                  </button>
                </div>
              )}

              {/* Feature & Distribution Drift Monitoring */}
              <section className="border border-mist rounded-panel bg-surface overflow-hidden">
                <div className="px-5 py-4 border-b border-mist flex flex-col md:flex-row md:items-center justify-between gap-3">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="p-1 rounded bg-teal/10 text-teal">
                        <Activity className="h-4 w-4" />
                      </span>
                      <h3 className="font-display text-base font-semibold text-ink">
                        Distribution Drift & Model Health
                      </h3>
                    </div>
                    <p className="text-xs text-[var(--muted)] mt-1">
                      Continuous statistical checks comparing training baseline vs incoming production cases.
                    </p>
                  </div>

                  {driftReport && (
                    <div className="flex items-center gap-2">
                      <span
                        className={`text-xs px-2.5 py-1 rounded font-semibold uppercase tracking-wider ${
                          driftReport.health_status === 'HEALTHY'
                            ? 'bg-teal/10 text-teal'
                            : driftReport.health_status === 'WARNING'
                              ? 'bg-ink/10 text-ink'
                              : 'bg-coral/10 text-coral'
                        }`}
                      >
                        {driftReport.health_status}
                      </span>
                    </div>
                  )}
                </div>

                <div className="p-5 space-y-4">
                  {driftReport ? (
                    <>
                      {/* Retraining evidence trigger alert */}
                      <div
                        className={`p-4 rounded-xl border text-xs space-y-1 ${
                          driftReport.retrain_recommended
                            ? 'border-coral/30 bg-coral/5 text-coral'
                            : 'border-teal/30 bg-teal/5 text-teal'
                        }`}
                      >
                        <div className="flex items-center gap-2 font-semibold">
                          {driftReport.retrain_recommended ? (
                            <AlertTriangle className="h-4 w-4" />
                          ) : (
                            <CheckCircle2 className="h-4 w-4" />
                          )}
                          <span>
                            {driftReport.retrain_recommended
                              ? 'Evidence-Based Retraining Trigger Recommended'
                              : 'No Retraining Required'}
                          </span>
                        </div>
                        <p className="text-muted leading-relaxed pl-6">
                          {driftReport.retrain_reason}
                        </p>
                      </div>

                      {/* Summary stats */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                        <div className="p-3 rounded-lg bg-paper border border-mist">
                          <span className="text-muted block text-[11px]">Monitored Features</span>
                          <span className="text-lg font-bold text-ink mt-0.5 block">
                            {driftReport.summary?.total_features_monitored || 0}
                          </span>
                        </div>
                        <div className="p-3 rounded-lg bg-paper border border-mist">
                          <span className="text-muted block text-[11px]">Severe Drift</span>
                          <span className="text-lg font-bold text-coral mt-0.5 block">
                            {driftReport.summary?.features_with_drift || 0}
                          </span>
                        </div>
                        <div className="p-3 rounded-lg bg-paper border border-mist">
                          <span className="text-muted block text-[11px]">Moderate Warnings</span>
                          <span className="text-lg font-bold text-ink mt-0.5 block">
                            {driftReport.summary?.features_with_warnings || 0}
                          </span>
                        </div>
                        <div className="p-3 rounded-lg bg-paper border border-mist">
                          <span className="text-muted block text-[11px]">Stable Features</span>
                          <span className="text-lg font-bold text-teal mt-0.5 block">
                            {driftReport.summary?.features_normal || 0}
                          </span>
                        </div>
                      </div>

                      {/* Features table */}
                      {driftReport.features && driftReport.features.length > 0 && (
                        <div className="overflow-x-auto rounded-xl border border-mist">
                          <table className="w-full text-left text-xs">
                            <thead className="bg-paper text-muted uppercase text-[10px] tracking-wider border-b border-mist">
                              <tr>
                                <th className="py-2.5 px-3">Feature Name</th>
                                <th className="py-2.5 px-3">Type</th>
                                <th className="py-2.5 px-3">PSI Metric</th>
                                <th className="py-2.5 px-3">KS p-value</th>
                                <th className="py-2.5 px-3">Missing Shift</th>
                                <th className="py-2.5 px-3 text-right">Drift Status</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-mist text-ink">
                              {driftReport.features.map((f, idx) => (
                                <tr key={idx} className="hover:bg-white/[0.02]">
                                  <td className="py-2.5 px-3 font-medium text-teal">{f.feature}</td>
                                  <td className="py-2.5 px-3 text-muted uppercase text-[10px]">
                                    {f.type}
                                  </td>
                                  <td className="py-2.5 px-3 font-mono">
                                    {f.psi != null ? f.psi.toFixed(3) : '—'}
                                  </td>
                                  <td className="py-2.5 px-3 font-mono">
                                    {f.ks_p_value != null ? f.ks_p_value.toFixed(4) : '—'}
                                  </td>
                                  <td className="py-2.5 px-3 text-muted">
                                    {f.null_rate_change != null
                                      ? `${(f.null_rate_change * 100).toFixed(1)}%`
                                      : '0.0%'}
                                  </td>
                                  <td className="py-2.5 px-3 text-right">
                                    <span
                                      className={`px-2 py-0.5 rounded text-[11px] font-medium ${
                                        f.status === 'drift_detected'
                                          ? 'bg-coral/10 text-coral'
                                          : f.status === 'warning'
                                            ? 'bg-ink/10 text-ink'
                                            : 'bg-teal/10 text-teal'
                                      }`}
                                    >
                                      {f.status === 'drift_detected'
                                        ? 'Drift Detected'
                                        : f.status === 'warning'
                                          ? 'Warning'
                                          : 'Stable'}
                                    </span>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </>
                  ) : (
                    <p className="text-xs text-[var(--muted)]">Loading statistical drift analysis...</p>
                  )}
                </div>
              </section>

              {/* Safe Candidate Model Retraining & Lifecycle */}
              <section className="border border-mist rounded-panel bg-surface overflow-hidden">
                <div className="px-5 py-4 border-b border-mist flex flex-col md:flex-row md:items-center justify-between gap-3">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="p-1 rounded bg-teal/10 text-teal">
                        <ShieldCheck className="h-4 w-4" />
                      </span>
                      <h3 className="font-display text-base font-semibold text-ink">
                        Safe Model Retraining & Deployment Lifecycle
                      </h3>
                    </div>
                    <p className="text-xs text-[var(--muted)] mt-1">
                      Candidates train in isolation. Promotions are gated by acceptance criteria; previous versions support instant rollback.
                    </p>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleTrainCandidate()}
                      disabled={candidateTraining}
                      className="px-3.5 py-1.5 rounded-lg bg-teal text-paper font-semibold text-xs hover:bg-teal/90 transition-all disabled:opacity-50 flex items-center gap-1.5"
                    >
                      {candidateTraining ? (
                        <>
                          <Spinner className="h-3.5 w-3.5 text-paper" /> Training Candidate...
                        </>
                      ) : (
                        <>
                          <Play className="h-3.5 w-3.5" /> Train Isolated Candidate
                        </>
                      )}
                    </button>

                    <button
                      onClick={handleRollback}
                      disabled={rollbackLoading || modelHistory.length < 2}
                      className="px-3 py-1.5 rounded-lg border border-mist bg-paper text-ink font-medium text-xs hover:border-coral/50 hover:text-coral transition-colors disabled:opacity-40 flex items-center gap-1.5"
                      title="Roll back active model to previous production version"
                    >
                      <RotateCcw className={`h-3.5 w-3.5 ${rollbackLoading ? 'animate-spin' : ''}`} />
                      Rollback Version
                    </button>
                  </div>
                </div>

                <div className="p-5 space-y-5">
                  {/* Active candidate evaluation card if one is being tested */}
                  {candidateInfo?.evaluation_comparison && (
                    <div className="p-4 rounded-xl border border-teal/30 bg-teal/5 space-y-3 text-xs">
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-ink text-sm">
                          Candidate Evaluation: {candidateInfo.candidate_version} vs Active (
                          {candidateInfo.evaluation_comparison.active_version})
                        </span>
                        <span
                          className={`px-2 py-0.5 rounded font-semibold text-[11px] ${
                            candidateInfo.evaluation_comparison.passes_acceptance_criteria
                              ? 'bg-teal/10 text-teal'
                              : 'bg-coral/10 text-coral'
                          }`}
                        >
                          {candidateInfo.evaluation_comparison.passes_acceptance_criteria
                            ? 'Acceptance Criteria Passed'
                            : 'Acceptance Criteria Failed'}
                        </span>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                        <div className="p-2.5 rounded bg-paper border border-mist">
                          <span className="text-muted block text-[10px]">Active AUC-ROC</span>
                          <span className="font-bold text-ink">
                            {candidateInfo.evaluation_comparison.active_metrics?.auc_roc ?? '—'}
                          </span>
                        </div>
                        <div className="p-2.5 rounded bg-paper border border-mist">
                          <span className="text-muted block text-[10px]">Candidate AUC-ROC</span>
                          <span className="font-bold text-teal">
                            {candidateInfo.evaluation_comparison.candidate_metrics?.auc_roc ?? '—'}
                          </span>
                        </div>
                        <div className="p-2.5 rounded bg-paper border border-mist">
                          <span className="text-muted block text-[10px]">Active Accuracy</span>
                          <span className="font-bold text-ink">
                            {candidateInfo.evaluation_comparison.active_metrics?.accuracy ?? '—'}
                          </span>
                        </div>
                        <div className="p-2.5 rounded bg-paper border border-mist">
                          <span className="text-muted block text-[10px]">Candidate Accuracy</span>
                          <span className="font-bold text-teal">
                            {candidateInfo.evaluation_comparison.candidate_metrics?.accuracy ?? '—'}
                          </span>
                        </div>
                      </div>

                      {/* Criteria list */}
                      <div className="space-y-1.5 pt-1">
                        {candidateInfo.evaluation_comparison.acceptance_criteria?.map((c, i) => (
                          <div key={i} className="flex items-center gap-2 text-[11px]">
                            {c.passed ? (
                              <CheckCircle2 className="h-3.5 w-3.5 text-teal" />
                            ) : (
                              <AlertTriangle className="h-3.5 w-3.5 text-coral" />
                            )}
                            <span className={c.passed ? 'text-ink' : 'text-coral'}>{c.criterion}</span>
                            <span className="text-muted font-mono">({c.delta >= 0 ? `+${c.delta}` : c.delta})</span>
                          </div>
                        ))}
                      </div>

                      {/* Promotion actions */}
                      <div className="flex items-center justify-end gap-2 pt-2 border-t border-mist/50">
                        <button
                          onClick={() =>
                            handlePromoteCandidate(candidateInfo.candidate_model_id, false)
                          }
                          disabled={
                            candidatePromoting ||
                            !candidateInfo.evaluation_comparison.passes_acceptance_criteria
                          }
                          className="px-4 py-1.5 rounded-lg bg-teal text-paper font-bold text-xs hover:bg-teal/90 transition-all disabled:opacity-40"
                        >
                          Promote to Production
                        </button>

                        {!candidateInfo.evaluation_comparison.passes_acceptance_criteria && (
                          <button
                            onClick={() =>
                              handlePromoteCandidate(candidateInfo.candidate_model_id, true)
                            }
                            disabled={candidatePromoting}
                            className="px-3 py-1.5 rounded-lg border border-coral text-coral font-medium text-xs hover:bg-coral/10 transition-all"
                          >
                            Force Promote Override
                          </button>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Model version history */}
                  <div className="space-y-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-muted">
                      Model Version Audit Trail ({modelHistory.length} Versions)
                    </span>
                    <div className="overflow-x-auto rounded-xl border border-mist">
                      <table className="w-full text-left text-xs">
                        <thead className="bg-paper text-muted uppercase text-[10px] tracking-wider border-b border-mist">
                          <tr>
                            <th className="py-2 px-3">Version</th>
                            <th className="py-2 px-3">Status</th>
                            <th className="py-2 px-3">Trained At</th>
                            <th className="py-2 px-3">Samples</th>
                            <th className="py-2 px-3">Accuracy</th>
                            <th className="py-2 px-3 text-right">AUC-ROC</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-mist text-ink">
                          {modelHistory.map((m, idx) => (
                            <tr key={idx} className={m.is_active ? 'bg-teal/5 font-semibold' : 'hover:bg-white/[0.02]'}>
                              <td className="py-2 px-3 text-teal font-mono">{m.version}</td>
                              <td className="py-2 px-3">
                                <span
                                  className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                                    m.is_active
                                      ? 'bg-teal/10 text-teal'
                                      : 'bg-paper text-muted border border-mist'
                                  }`}
                                >
                                  {m.is_active ? 'Active Production' : 'Archived / Rollback Ready'}
                                </span>
                              </td>
                              <td className="py-2 px-3 text-muted">
                                {m.trained_at ? new Date(m.trained_at).toLocaleString() : '—'}
                              </td>
                              <td className="py-2 px-3 text-muted">{m.training_samples?.toLocaleString() || '—'}</td>
                              <td className="py-2 px-3">{m.accuracy != null ? `${(m.accuracy * 100).toFixed(1)}%` : '—'}</td>
                              <td className="py-2 px-3 text-right font-medium">
                                {m.auc_roc != null ? m.auc_roc.toFixed(3) : '—'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </section>
            </div>
          )}
        </>
      )}
    </div>
  )
}

function MetricCell({ label, value, accent = 'text-ink', hint, progress, color = 'teal' }) {
  return (
    <div className="bg-paper px-4 py-4 flex flex-col justify-between">
      <div>
        <div className="text-[11px] uppercase tracking-wide text-[var(--muted)] font-medium">{label}</div>
        <div className={`mt-1 font-display text-2xl font-semibold tabular-nums ${accent}`}>
          {value}
        </div>
      </div>
      {progress != null && !Number.isNaN(Number(progress)) && (
        <div className="mt-2.5">
          <ProgressBar value={Number(progress) * 100} max={100} color={color} size="xs" />
        </div>
      )}
      {hint && <p className="text-xs text-[var(--muted)] mt-1">{hint}</p>}
    </div>
  )
}
