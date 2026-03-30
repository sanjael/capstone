import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  BarChart3, Activity, Brain, TrendingUp, CheckCircle, 
  XCircle, Clock, Zap, AlertTriangle, Loader2 
} from 'lucide-react'
import toast from 'react-hot-toast'
import Navbar from '../components/Navbar'
import { analyticsService } from '../services/api'

const TUMOR_COLORS = {
  glioma: '#ef4444',
  meningioma: '#f97316',
  pituitary: '#eab308',
}

function StatCard({ icon: Icon, label, value, sub, color = 'text-violet-600', bg = 'bg-violet-50' }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-2xl p-5 border border-slate-100 shadow-sm flex items-center gap-4"
    >
      <div className={`w-12 h-12 ${bg} rounded-xl flex items-center justify-center shrink-0`}>
        <Icon className={`w-6 h-6 ${color}`} />
      </div>
      <div>
        <p className="text-xs text-slate-500 font-medium uppercase tracking-wide">{label}</p>
        <p className="text-2xl font-black text-slate-800">{value}</p>
        {sub && <p className="text-xs text-slate-400 mt-0.5">{sub}</p>}
      </div>
    </motion.div>
  )
}

function MiniBar({ label, value, color, max }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-slate-600">
        <span className="capitalize font-medium">{label}</span>
        <span className="font-bold">{value}</span>
      </div>
      <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
        />
      </div>
    </div>
  )
}

function ActivityChart({ data }) {
  if (!data || data.length === 0) {
    return <p className="text-sm text-slate-400 text-center py-6">No activity in the last 7 days.</p>
  }
  const max = Math.max(...data.map(d => d.count))
  return (
    <div className="space-y-2">
      {data.map(({ date, count }) => (
        <div key={date} className="flex items-center gap-3">
          <span className="text-xs text-slate-400 w-20 shrink-0">{date.slice(5)}</span>
          <div className="flex-1 h-5 bg-slate-50 rounded overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.round((count / max) * 100)}%` }}
              transition={{ duration: 0.6 }}
              className="h-full bg-violet-500 rounded"
            />
          </div>
          <span className="text-xs font-bold text-slate-600 w-6 text-right">{count}</span>
        </div>
      ))}
    </div>
  )
}

export default function AnalyticsPage() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true)
      try {
        const analytics = await analyticsService.getUserAnalytics()
        setData(analytics)
      } catch {
        toast.error('Failed to load analytics.')
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-50">
        <Navbar />
        <div className="flex items-center justify-center h-[60vh]">
          <Loader2 className="w-8 h-8 text-violet-500 animate-spin" />
        </div>
      </div>
    )
  }

  const typeMax = data ? Math.max(...Object.values(data.tumor_type_distribution || {}), 1) : 1

  return (
    <div className="min-h-screen bg-slate-50">
      <Navbar />
      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-10">

        <div className="mb-8">
          <h1 className="font-display text-3xl sm:text-4xl text-slate-900 mb-1">Analytics</h1>
          <p className="text-slate-500">Your personal scan activity and AI performance metrics.</p>
        </div>

        {/* Personal Stats */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatCard icon={Activity} label="Total Scans" value={data?.total_scans ?? 0} color="text-violet-600" bg="bg-violet-50" />
          <StatCard
            icon={AlertTriangle}
            label="Tumors Detected"
            value={data?.tumor_detected ?? 0}
            sub={`${((data?.detection_rate ?? 0) * 100).toFixed(1)}% detection rate`}
            color="text-red-600"
            bg="bg-red-50"
          />
          <StatCard
            icon={CheckCircle}
            label="Confirmed by Dr"
            value={data?.feedback.confirmed ?? 0}
            sub={`${data?.feedback.rejected ?? 0} rejected`}
            color="text-emerald-600"
            bg="bg-emerald-50"
          />
          <StatCard
            icon={Zap}
            label="Avg. Confidence"
            value={`${((data?.average_confidence ?? 0) * 100).toFixed(1)}%`}
            color="text-amber-600"
            bg="bg-amber-50"
          />
        </div>

        <div className="grid lg:grid-cols-2 gap-6 mb-8">
          {/* Type Distribution */}
          <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm">
            <h2 className="font-semibold text-slate-700 mb-5 flex items-center gap-2">
              <Brain className="w-4 h-4 text-violet-500" />
              Tumor Type Distribution
            </h2>
            {Object.keys(data?.tumor_type_distribution || {}).length === 0 ? (
              <p className="text-sm text-slate-400 text-center py-8">No tumors detected yet.</p>
            ) : (
              <div className="space-y-4">
                {Object.entries(data.tumor_type_distribution).map(([type, count]) => (
                  <MiniBar key={type} label={type} value={count} color={TUMOR_COLORS[type] || '#6366f1'} max={typeMax} />
                ))}
              </div>
            )}
          </div>

          {/* 7-day Activity */}
          <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm">
            <h2 className="font-semibold text-slate-700 mb-5 flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-violet-500" />
              Activity — Last 7 Days
            </h2>
            <ActivityChart data={data?.daily_activity_7d} />
          </div>
        </div>

        {/* Feedback Summary */}
        <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm mb-8">
          <h2 className="font-semibold text-slate-700 mb-5 flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-violet-500" />
            Doctor Feedback Summary
          </h2>
          <div className="grid sm:grid-cols-3 gap-6">
            {[
              { label: 'Confirmed', count: data?.feedback.confirmed ?? 0, icon: CheckCircle, color: 'text-emerald-600', bg: 'bg-emerald-50' },
              { label: 'Rejected', count: data?.feedback.rejected ?? 0, icon: XCircle, color: 'text-red-600', bg: 'bg-red-50' },
              { label: 'Pending', count: data?.feedback.pending ?? 0, icon: Clock, color: 'text-amber-600', bg: 'bg-amber-50' },
            ].map(({ label, count, icon: Icon, color, bg }) => (
              <div key={label} className={`${bg} rounded-xl p-4 flex items-center gap-4`}>
                <Icon className={`w-7 h-7 ${color}`} />
                <div>
                  <p className="text-2xl font-black text-slate-800">{count}</p>
                  <p className="text-xs text-slate-500">{label}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  )
}
