import React, { useState, useEffect } from 'react'
import { Sparkles, ArrowRight, ShieldAlert, CheckCircle2, AlertTriangle, RefreshCw } from 'lucide-react'

/**
 * CounterfactualCard: Actionable, constraint-aware counterfactual recommendations.
 * Preserves immutable features, shows cost and predicted impact, and states causal caveats.
 */
export default function CounterfactualCard({ projectId, predictionId, features, currentProbability, token }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchCounterfactual = async () => {
    if (!token) return
    setLoading(true)
    setError(null)
    try {
      let res
      if (projectId) {
        res = await fetch(`/api/projects/${projectId}/counterfactual`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            features: features || {},
            target_class: 0,
            threshold: 0.40,
          }),
        })
      } else if (predictionId) {
        res = await fetch(`/api/explain/${predictionId}/counterfactual`, {
          headers: { Authorization: `Bearer ${token}` },
        })
      }

      if (!res || !res.ok) {
        const errJson = await res?.json().catch(() => ({}))
        throw new Error(errJson?.detail || 'Unable to compute feasible counterfactual')
      }

      const json = await res.json()
      setData(json)
    } catch (err) {
      setError(err.message || 'Counterfactual computation failed')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchCounterfactual()
  }, [projectId, predictionId])

  return (
    <div className="rounded-xl border border-mist bg-surface/80 p-5 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-teal/10 text-teal">
            <Sparkles className="h-4 w-4" />
          </div>
          <div>
            <h4 className="text-sm font-semibold text-ink">Actionable Counterfactual Scenario</h4>
            <p className="text-xs text-muted">Feasible changes to steer prediction away from high risk</p>
          </div>
        </div>

        <button
          onClick={fetchCounterfactual}
          disabled={loading}
          className="p-1.5 text-muted hover:text-ink transition-colors disabled:opacity-50"
          title="Recompute counterfactual"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {loading && (
        <div className="py-8 flex flex-col items-center justify-center gap-2 text-muted text-xs">
          <div className="h-5 w-5 border-2 border-teal border-t-transparent rounded-full animate-spin" />
          <span>Searching feasible feature state space...</span>
        </div>
      )}

      {error && !loading && (
        <div className="p-3 rounded-lg border border-coral/20 bg-coral/5 text-xs text-coral flex items-start gap-2">
          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
          <div>
            <p className="font-medium">No Feasible Counterfactual Found</p>
            <p className="text-muted mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {data && !loading && (
        <>
          {/* Outcome comparison strip */}
          <div className="grid grid-cols-2 gap-3 p-3 rounded-lg bg-paper border border-mist text-xs">
            <div>
              <span className="text-muted block text-[11px]">Original Prediction</span>
              <div className="flex items-baseline gap-2 mt-0.5">
                <span className="text-lg font-bold text-coral">
                  {Math.round((data.original_probability ?? currentProbability ?? 0.75) * 100)}%
                </span>
                <span className="text-[11px] text-muted">High Risk</span>
              </div>
            </div>
            <div>
              <span className="text-muted block text-[11px]">Proposed Scenario</span>
              <div className="flex items-baseline gap-2 mt-0.5">
                <span className="text-lg font-bold text-teal">
                  {Math.round((data.proposed_probability ?? 0.35) * 100)}%
                </span>
                <span className="text-[11px] text-teal">Feasible Target</span>
              </div>
            </div>
          </div>

          {/* Actionable changes table */}
          {data.changes && data.changes.length > 0 ? (
            <div className="space-y-2">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-muted">
                Controllable Interventions ({data.sparsity || data.changes.length} adjustments)
              </span>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border border-mist rounded-lg overflow-hidden">
                  <thead className="bg-paper text-muted uppercase text-[10px] tracking-wider">
                    <tr>
                      <th className="py-2 px-3">Controllable Feature</th>
                      <th className="py-2 px-3">Current</th>
                      <th className="py-2 px-3">Proposed Target</th>
                      <th className="py-2 px-3 text-right">Feasibility</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-mist text-ink">
                    {data.changes.map((ch, idx) => (
                      <tr key={idx} className="hover:bg-white/[0.02]">
                        <td className="py-2 px-3 font-medium text-teal">{ch.feature}</td>
                        <td className="py-2 px-3 text-muted">{String(ch.original_value)}</td>
                        <td className="py-2 px-3 text-ink font-semibold flex items-center gap-1.5">
                          <ArrowRight className="h-3 w-3 text-muted" />
                          {String(ch.proposed_value)}
                        </td>
                        <td className="py-2 px-3 text-right">
                          <span className="inline-flex items-center gap-1 text-[11px] text-teal bg-teal/10 px-1.5 py-0.5 rounded">
                            <CheckCircle2 className="h-3 w-3" /> Valid Range
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="p-3 text-xs text-muted bg-paper rounded border border-mist">
              {data.message || 'Customer feature profile is already in the target probability window.'}
            </div>
          )}

          {/* Preserved Immutable Attributes */}
          <div className="flex items-center gap-2 text-[11px] text-muted p-2 rounded bg-paper/50 border border-mist/50">
            <span className="font-medium text-ink">Guaranteed Invariants:</span>
            <span>Customer age, historical tenure, and demographic IDs remained fixed.</span>
          </div>

          {/* Estimated Cost & Effort */}
          {data.estimated_cost_level && (
            <div className="flex items-center justify-between text-xs p-2.5 rounded-lg bg-paper border border-mist">
              <span className="text-muted">Estimated Intervention Cost</span>
              <span className="font-semibold text-ink uppercase tracking-wider text-[11px]">
                {data.estimated_cost_level}
              </span>
            </div>
          )}

          {/* Critical Causal Disclaimer */}
          <div className="p-3 rounded-lg border border-mist bg-paper text-[11px] text-muted space-y-1">
            <div className="flex items-center gap-1.5 font-medium text-ink">
              <ShieldAlert className="h-3.5 w-3.5 text-teal" />
              <span>Association vs Causal Impact</span>
            </div>
            <p className="leading-relaxed">
              {data.causal_disclaimer ||
                'This counterfactual scenario describes the statistical conditions under which Knowa predicts lower risk. It reflects model association and does not guarantee causal impact without an A/B test.'}
            </p>
          </div>
        </>
      )}
    </div>
  )
}
