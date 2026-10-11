import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
  Sliders,
  DollarSign,
  TrendingUp,
  ShieldCheck,
  CheckCircle,
  AlertCircle,
  ArrowRight,
  Layers,
  Sparkles,
  Info,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import ProjectPicker from '../components/common/ProjectPicker'
import Spinner from '../components/common/Spinner'

export default function DecisionStudio() {
  const { token } = useAuth()
  const [projects, setProjects] = useState([])
  const [selectedProjectId, setSelectedProjectId] = useState('')
  const [loadingProjects, setLoadingProjects] = useState(true)

  // Optimization parameters
  const [budget, setBudget] = useState(50000)
  const [maxCapacity, setMaxCapacity] = useState(25)
  const [valuePerRetention, setValuePerRetention] = useState(10000)

  // Optimization results
  const [optimizing, setOptimizing] = useState(false)
  const [optimizationData, setOptimizationData] = useState(null)
  const [selectedStrategy, setSelectedStrategy] = useState('ilp_optimal_knapsack')
  const [committing, setCommitting] = useState(false)
  const [commitSuccess, setCommitSuccess] = useState(null)
  const [error, setError] = useState(null)

  // Fetch projects
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
          if (data.length > 0 && !selectedProjectId) {
            setSelectedProjectId(data[0].id)
          }
        }
      } catch (err) {
        console.error('Failed to load projects', err)
      } finally {
        setLoadingProjects(false)
      }
    })()
  }, [token])

  // Run optimization when parameters change or project selected
  const runOptimization = async () => {
    if (!token) return
    setOptimizing(true)
    setError(null)
    setCommitSuccess(null)

    try {
      let res
      if (selectedProjectId) {
        res = await fetch(`/api/optimizer/projects/${selectedProjectId}/optimize`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            budget: Number(budget),
            max_capacity: Number(maxCapacity),
            value_per_retention: Number(valuePerRetention),
          }),
        })
      } else {
        // Fallback demo optimizer
        res = await fetch('/api/optimizer/demo-optimize', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            budget: Number(budget),
            max_capacity: Number(maxCapacity),
            value_per_retention: Number(valuePerRetention),
          }),
        })
      }

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}))
        throw new Error(errJson.detail || 'Optimization failed')
      }

      const data = await res.json()
      setOptimizationData(data)
    } catch (err) {
      setError(err.message || 'Optimization failed')
    } finally {
      setOptimizing(false)
    }
  }

  useEffect(() => {
    if (selectedProjectId) {
      runOptimization()
    }
  }, [selectedProjectId])

  // Commit selected strategy to Decision Ledger
  const handleCommitToLedger = async () => {
    if (!token || !selectedProjectId) return
    setCommitting(true)
    setError(null)

    try {
      const res = await fetch(`/api/optimizer/projects/${selectedProjectId}/optimize/commit`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          strategy_name: selectedStrategy,
          budget: Number(budget),
          max_capacity: Number(maxCapacity),
          value_per_retention: Number(valuePerRetention),
        }),
      })

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}))
        throw new Error(errJson.detail || 'Failed to commit decisions to ledger')
      }

      const data = await res.json()
      setCommitSuccess(data)
    } catch (err) {
      setError(err.message || 'Commit failed')
    } finally {
      setCommitting(false)
    }
  }

  const currentStrategyObj = optimizationData?.strategies?.[selectedStrategy] || {}

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-mist pb-6">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-teal/10 text-teal">
              <Sliders className="h-5 w-5" />
            </span>
            <h1 className="text-2xl font-bold text-ink">Decision Studio</h1>
          </div>
          <p className="text-sm text-muted mt-1">
            Constraint-aware optimization: allocate business interventions under budgets, capacity, and ROI limits.
          </p>
        </div>

        {/* Project Selector */}
        {projects.length > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted">Project:</span>
            <select
              value={selectedProjectId}
              onChange={(e) => setSelectedProjectId(e.target.value)}
              className="bg-surface text-ink text-xs border border-mist rounded-lg px-3 py-2 outline-none focus:border-teal"
            >
              {projects.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Control Panel: Budget & Capacity Sliders */}
      <div className="p-6 rounded-xl bg-surface border border-mist space-y-6">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-teal flex items-center gap-2">
          <Layers className="h-4 w-4" /> Real-World Business Constraints
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Budget Slider */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted">Total Intervention Budget</span>
              <span className="font-bold text-ink text-sm">₹{Number(budget).toLocaleString()}</span>
            </div>
            <input
              type="range"
              min="10000"
              max="200000"
              step="5000"
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value))}
              className="w-full accent-teal cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-muted">
              <span>₹10,000</span>
              <span>₹100,000</span>
              <span>₹200,000</span>
            </div>
          </div>

          {/* Max Capacity Input */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted">Max Intervention Capacity</span>
              <span className="font-bold text-ink text-sm">{maxCapacity} accounts</span>
            </div>
            <input
              type="range"
              min="5"
              max="100"
              step="5"
              value={maxCapacity}
              onChange={(e) => setMaxCapacity(Number(e.target.value))}
              className="w-full accent-teal cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-muted">
              <span>5</span>
              <span>50</span>
              <span>100</span>
            </div>
          </div>

          {/* Value per retention */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted">Value per Saved Customer</span>
              <span className="font-bold text-ink text-sm">₹{Number(valuePerRetention).toLocaleString()}</span>
            </div>
            <input
              type="range"
              min="2000"
              max="30000"
              step="1000"
              value={valuePerRetention}
              onChange={(e) => setValuePerRetention(Number(e.target.value))}
              className="w-full accent-teal cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-muted">
              <span>₹2,000</span>
              <span>₹15,000</span>
              <span>₹30,000</span>
            </div>
          </div>
        </div>

        <div className="flex justify-end pt-2">
          <button
            onClick={runOptimization}
            disabled={optimizing}
            className="px-4 py-2 rounded-lg bg-teal text-paper font-semibold text-xs hover:bg-teal/90 transition-colors disabled:opacity-50 flex items-center gap-2"
          >
            {optimizing ? (
              <>
                <Spinner className="h-3.5 w-3.5 text-paper" /> Re-optimizing...
              </>
            ) : (
              'Recalculate Optimal Strategies'
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl border border-coral/30 bg-coral/10 text-coral text-xs flex items-center gap-2">
          <AlertCircle className="h-4 w-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Strategies Comparison Cards */}
      {optimizationData?.strategies && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-ink">Strategy Comparison</h3>
            <span className="text-xs text-muted">
              {optimizationData.candidate_actions_evaluated || 0} candidate interventions analyzed
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {Object.entries(optimizationData.strategies).map(([key, s]) => {
              const isSelected = selectedStrategy === key
              return (
                <div
                  key={key}
                  onClick={() => setSelectedStrategy(key)}
                  className={`p-5 rounded-xl border cursor-pointer transition-all ${
                    isSelected
                      ? 'border-teal bg-teal/5 shadow-lg shadow-teal/5'
                      : 'border-mist bg-surface hover:border-muted/40'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-muted uppercase tracking-wider">{s.name}</span>
                    {isSelected && <CheckCircle className="h-4 w-4 text-teal" />}
                  </div>

                  <div className="mt-3">
                    <span className="text-2xl font-bold text-ink">
                      ₹{Math.round(s.net_benefit || 0).toLocaleString()}
                    </span>
                    <span className="text-xs text-muted block mt-0.5">Expected Net Benefit</span>
                  </div>

                  <div className="mt-4 pt-4 border-t border-mist/50 space-y-1.5 text-xs">
                    <div className="flex justify-between">
                      <span className="text-muted">Total Cost</span>
                      <span className="text-ink font-medium">₹{Math.round(s.total_cost || 0).toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted">Actions Selected</span>
                      <span className="text-ink font-medium">{s.selected_count || 0} accounts</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted">Expected ROI</span>
                      <span className="text-teal font-medium">
                        {s.roi_percent ? `${Math.round(s.roi_percent)}%` : '0%'}
                      </span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>

          {/* Detailed Selected Strategy Breakdown */}
          <div className="p-6 rounded-xl bg-surface border border-mist space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <h4 className="text-base font-bold text-ink">{currentStrategyObj.name}</h4>
                  <span className="text-xs px-2 py-0.5 rounded bg-teal/10 text-teal font-medium">
                    {currentStrategyObj.algorithm || 'Highs ILP Knapsack'}
                  </span>
                </div>
                <p className="text-xs text-muted mt-1">{currentStrategyObj.description}</p>
              </div>

              {/* Commit button */}
              <button
                onClick={handleCommitToLedger}
                disabled={committing || selectedStrategy === 'no_action'}
                className="px-5 py-2.5 rounded-lg bg-teal text-paper font-bold text-xs hover:bg-teal/90 transition-all disabled:opacity-40 flex items-center gap-2 shadow-md"
              >
                {committing ? (
                  <>
                    <Spinner className="h-4 w-4 text-paper" /> Committing to Ledger...
                  </>
                ) : (
                  <>
                    <ShieldCheck className="h-4 w-4" /> Commit Strategy to Ledger
                  </>
                )}
              </button>
            </div>

            {commitSuccess && (
              <div className="p-4 rounded-xl border border-teal/30 bg-teal/10 text-teal text-xs flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4" />
                  <span>
                    Successfully committed {commitSuccess.committed_count} decisions into the durable ledger with
                    30-day rechecks.
                  </span>
                </div>
                <Link to="/follow-ups" className="underline font-semibold hover:text-ink">
                  View in Follow-ups →
                </Link>
              </div>
            )}

            {/* Constraint Attribution Explanation */}
            <div className="p-4 rounded-xl bg-paper border border-mist space-y-2 text-xs">
              <span className="font-semibold text-ink flex items-center gap-1.5 text-xs">
                <Info className="h-4 w-4 text-teal" /> Constraint & Allocation Rationale
              </span>
              <p className="text-muted leading-relaxed">
                {currentStrategyObj.binding_constraint_explanation ||
                  'Optimization prioritized actions maximizing expected net value under budget and capacity limits.'}
              </p>
              {currentStrategyObj.assumptions && (
                <p className="text-[11px] text-muted border-t border-mist/40 pt-2">
                  <span className="text-ink font-medium">Model Assumptions: </span>
                  {currentStrategyObj.assumptions}
                </p>
              )}
            </div>

            {/* Selected Action Items List */}
            {currentStrategyObj.selected_actions && currentStrategyObj.selected_actions.length > 0 ? (
              <div className="space-y-3">
                <h5 className="text-xs font-semibold uppercase tracking-wider text-muted">
                  Prioritized Action Roster ({currentStrategyObj.selected_actions.length} Accounts)
                </h5>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border border-mist rounded-xl overflow-hidden">
                    <thead className="bg-paper text-muted uppercase text-[10px] tracking-wider">
                      <tr>
                        <th className="py-2.5 px-3">Entity ID</th>
                        <th className="py-2.5 px-3">Recommended Action</th>
                        <th className="py-2.5 px-3">Action Cost</th>
                        <th className="py-2.5 px-3">Initial Risk</th>
                        <th className="py-2.5 px-3">Expected Lift</th>
                        <th className="py-2.5 px-3 text-right">Expected Benefit</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-mist text-ink">
                      {currentStrategyObj.selected_actions.map((act, idx) => (
                        <tr key={idx} className="hover:bg-white/[0.02]">
                          <td className="py-2.5 px-3 font-medium text-teal">{act.entity_id || `Case #${idx + 1}`}</td>
                          <td className="py-2.5 px-3 font-semibold text-ink">{act.action_name}</td>
                          <td className="py-2.5 px-3 text-muted">₹{Math.round(act.cost || 0).toLocaleString()}</td>
                          <td className="py-2.5 px-3 text-coral font-medium">
                            {act.current_risk ? `${Math.round(act.current_risk * 100)}%` : '—'}
                          </td>
                          <td className="py-2.5 px-3 text-teal">
                            {act.expected_lift ? `${Math.round(act.expected_lift * 100)}% risk reduction` : '—'}
                          </td>
                          <td className="py-2.5 px-3 text-right font-bold text-ink">
                            ₹{Math.round(act.expected_benefit || 0).toLocaleString()}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="p-6 text-center text-xs text-muted bg-paper rounded-xl border border-mist">
                No interventions selected under this strategy.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
