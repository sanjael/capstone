import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  History, Brain, AlertTriangle, CheckCircle, Clock,
  Trash2, MessageSquare, Eye, Filter, Loader2,
  ChevronLeft, ChevronRight, X, Download, ShieldAlert, FileText
} from 'lucide-react'
import toast from 'react-hot-toast'
import Navbar from '../components/Navbar'
import { historyService, predictionService } from '../services/api'

const RISK_COLORS = {
  'Critical': 'text-red-600 bg-red-50 border-red-200',
  'High': 'text-orange-600 bg-orange-50 border-orange-200',
  'Medium': 'text-yellow-600 bg-yellow-50 border-yellow-200',
  'Low': 'text-green-600 bg-green-50 border-green-200',
  'None': 'text-slate-600 bg-slate-50 border-slate-200',
}

const FEEDBACK_BADGE = {
  confirmed: 'text-emerald-700 bg-emerald-50 border border-emerald-200',
  rejected: 'text-red-700 bg-red-50 border border-red-200',
}

function FeedbackModal({ scan, onClose, onSubmit }) {
  const [feedback, setFeedback] = useState(scan.doctor_feedback || '')
  const [notes, setNotes] = useState(scan.feedback_notes || '')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async () => {
    if (!feedback) { toast.error('Select a feedback type.'); return }
    setLoading(true)
    try {
      await historyService.submitFeedback(scan.id, feedback, notes || null)
      onSubmit(scan.id, feedback, notes)
      toast.success('Feedback saved.')
      onClose()
    } catch {
      toast.error('Failed to save feedback.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="bg-white rounded-2xl p-6 max-w-md w-full shadow-2xl"
      >
        <div className="flex items-center justify-between mb-5">
          <h3 className="font-semibold text-slate-800 flex items-center gap-2">
            <MessageSquare className="w-4 h-4 text-violet-500" />
            Doctor Feedback
          </h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600">
            <X className="w-5 h-5" />
          </button>
        </div>

        <p className="text-sm text-slate-500 mb-4">
          Scan #{scan.id} — {scan.tumor_detected ? `${scan.tumor_type || 'Tumor'} detected` : 'No tumor'}
        </p>

        <div className="grid grid-cols-2 gap-3 mb-4">
          {['confirmed', 'rejected'].map(f => (
            <button
              key={f}
              onClick={() => setFeedback(f)}
              className={`py-3 rounded-xl border-2 font-medium text-sm capitalize transition-all ${
                feedback === f
                  ? f === 'confirmed'
                    ? 'border-emerald-500 bg-emerald-50 text-emerald-700'
                    : 'border-red-500 bg-red-50 text-red-700'
                  : 'border-slate-200 text-slate-600 hover:border-slate-300'
              }`}
            >
              {f === 'confirmed' ? '✅ ' : '❌ '}{f}
            </button>
          ))}
        </div>

        <textarea
          value={notes}
          onChange={e => setNotes(e.target.value)}
          placeholder="Clinical notes (optional)…"
          rows={3}
          className="w-full border border-slate-200 rounded-xl px-3 py-2 text-sm text-slate-700 resize-none focus:outline-none focus:ring-2 focus:ring-violet-400 mb-4"
        />

        <button
          onClick={handleSubmit}
          disabled={loading || !feedback}
          className="w-full py-3 bg-violet-600 hover:bg-violet-500 disabled:opacity-50 text-white rounded-xl font-semibold transition-all"
        >
          {loading ? 'Saving…' : 'Save Feedback'}
        </button>
      </motion.div>
    </div>
  )
}

function ScanDetailModal({ scan, onClose }) {
  const [downloading, setDownloading] = useState(false)

  const handleDownload = async () => {
    setDownloading(true)
    try {
      const blob = await predictionService.getReport(scan)
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `NeuroScan_Report_${scan.id}.pdf`)
      document.body.appendChild(link)
      link.click()
      link.remove()
      toast.success('Report downloaded.')
    } catch {
      toast.error('Failed to generate report.')
    } finally {
      setDownloading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="bg-white rounded-2xl max-w-lg w-full shadow-2xl max-h-[90vh] overflow-y-auto"
      >
        <div className="flex items-center justify-between p-6 border-b border-slate-100">
          <h3 className="font-semibold text-slate-800">Scan #{scan.id} Details</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-5">
          {/* Status */}
          <div className={`flex items-center gap-3 p-4 rounded-xl border ${scan.tumor_detected ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'}`}>
            {scan.tumor_detected
              ? <AlertTriangle className="w-5 h-5 text-red-600" />
              : <CheckCircle className="w-5 h-5 text-green-600" />
            }
            <div>
              <p className={`font-semibold text-sm ${scan.tumor_detected ? 'text-red-700' : 'text-green-700'}`}>
                {scan.tumor_detected ? `Tumor Detected — ${scan.tumor_type?.charAt(0).toUpperCase() + scan.tumor_type?.slice(1)}` : 'No Tumor Detected'}
              </p>
              <p className="text-xs text-slate-500">{scan.reliability}</p>
            </div>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-50 rounded-xl p-3">
              <p className="text-xs text-slate-400 uppercase font-medium mb-1">Confidence</p>
              <p className="text-xl font-black text-slate-800">{(scan.confidence * 100).toFixed(1)}%</p>
            </div>
            <div className="bg-slate-50 rounded-xl p-3">
              <p className="text-xs text-slate-400 uppercase font-medium mb-1">MC Uncertainty</p>
              <p className={`text-xl font-black ${scan.uncertainty > 0.1 ? 'text-amber-600' : 'text-emerald-600'}`}>
                ±{scan.uncertainty.toFixed(4)}
              </p>
            </div>
          </div>

          {/* Heatmap */}
          {scan.heatmap_image && (
            <div>
              <p className="text-xs font-semibold text-slate-500 uppercase mb-2">Grad-CAM++ Heatmap</p>
              <img src={scan.heatmap_image} alt="Heatmap" className="w-full rounded-xl border border-slate-100" />
            </div>
          )}

          {/* Notes */}
          <div className="space-y-3">
            <div>
              <p className="text-xs font-semibold text-slate-500 uppercase mb-1">Clinical Note</p>
              <p className="text-sm text-slate-600 leading-relaxed">{scan.clinical_note}</p>
            </div>
            <div>
              <p className="text-xs font-semibold text-slate-500 uppercase mb-1">Recommendation</p>
              <p className="text-sm text-slate-600 leading-relaxed">{scan.recommendation}</p>
            </div>
          </div>

          {scan.feedback_notes && (
            <div className="bg-violet-50 rounded-xl p-3 border border-violet-100">
              <p className="text-xs font-semibold text-violet-600 uppercase mb-1">Doctor Notes</p>
              <p className="text-sm text-violet-800">{scan.feedback_notes}</p>
            </div>
          )}

          <button
            onClick={handleDownload}
            disabled={downloading}
            className="w-full flex items-center justify-center gap-2 py-3 border border-violet-200 text-violet-700 hover:bg-violet-50 rounded-xl font-medium text-sm transition-all"
          >
            <Download className="w-4 h-4" />
            {downloading ? 'Generating PDF…' : 'Download Clinical Report'}
          </button>
        </div>
      </motion.div>
    </div>
  )
}

export default function HistoryPage() {
  const [scans, setScans] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(true)
  const [filterType, setFilterType] = useState('')
  const [feedbackScan, setFeedbackScan] = useState(null)
  const [detailScan, setDetailScan] = useState(null)
  const PAGE_SIZE = 10

  const fetchHistory = useCallback(async () => {
    setLoading(true)
    try {
      const res = await historyService.list({ page, pageSize: PAGE_SIZE, tumorType: filterType || null })
      setScans(res.items)
      setTotal(res.total)
    } catch {
      toast.error('Failed to load history.')
    } finally {
      setLoading(false)
    }
  }, [page, filterType])

  useEffect(() => { fetchHistory() }, [fetchHistory])

  const handleDelete = async (id) => {
    if (!confirm('Delete this scan record?')) return
    try {
      await historyService.deleteScan(id)
      setScans(prev => prev.filter(s => s.id !== id))
      setTotal(prev => prev - 1)
      toast.success('Scan deleted.')
    } catch {
      toast.error('Failed to delete scan.')
    }
  }

  const handleFeedbackSubmit = (scanId, feedback, notes) => {
    setScans(prev => prev.map(s => s.id === scanId ? { ...s, doctor_feedback: feedback, feedback_notes: notes } : s))
  }

  const totalPages = Math.ceil(total / PAGE_SIZE)

  return (
    <div className="min-h-screen bg-slate-50">
      <Navbar />

      <AnimatePresence>
        {feedbackScan && <FeedbackModal scan={feedbackScan} onClose={() => setFeedbackScan(null)} onSubmit={handleFeedbackSubmit} />}
        {detailScan && <ScanDetailModal scan={detailScan} onClose={() => setDetailScan(null)} />}
      </AnimatePresence>

      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-10">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-8">
          <div>
            <h1 className="font-display text-3xl sm:text-4xl text-slate-900 mb-1">Scan History</h1>
            <p className="text-slate-500">{total} scan{total !== 1 ? 's' : ''} recorded</p>
          </div>

          {/* Filter */}
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-slate-400" />
            <select
              value={filterType}
              onChange={e => { setFilterType(e.target.value); setPage(1) }}
              className="border border-slate-200 rounded-xl px-3 py-2 text-sm text-slate-600 focus:outline-none focus:ring-2 focus:ring-violet-400 bg-white"
            >
              <option value="">All Types</option>
              <option value="glioma">Glioma</option>
              <option value="meningioma">Meningioma</option>
              <option value="pituitary">Pituitary</option>
            </select>
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center h-40">
            <Loader2 className="w-7 h-7 text-violet-500 animate-spin" />
          </div>
        ) : scans.length === 0 ? (
          <div className="bg-white rounded-2xl border border-dashed border-slate-200 flex flex-col items-center justify-center py-24">
            <History className="w-10 h-10 text-slate-300 mb-3" />
            <p className="font-medium text-slate-400">No scans found</p>
            <p className="text-sm text-slate-300 mt-1">Start analysing MRIs from the Dashboard</p>
          </div>
        ) : (
          <div className="space-y-3">
            {scans.map((scan, i) => (
              <motion.div
                key={scan.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.04 }}
                className="bg-white rounded-2xl border border-slate-100 shadow-sm p-5 flex flex-wrap gap-4 items-center"
              >
                {/* Status */}
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${scan.tumor_detected ? 'bg-red-50' : 'bg-green-50'}`}>
                  {scan.tumor_detected
                    ? <Brain className="w-5 h-5 text-red-500" />
                    : <CheckCircle className="w-5 h-5 text-green-500" />
                  }
                </div>

                {/* Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex flex-wrap items-center gap-2 mb-1">
                    <span className="font-semibold text-slate-800">
                      {scan.tumor_detected ? (scan.tumor_type ? scan.tumor_type.charAt(0).toUpperCase() + scan.tumor_type.slice(1) : 'Tumor') : 'No Tumor'}
                    </span>
                    <span className={`text-xs px-2 py-0.5 rounded-full border ${RISK_COLORS[scan.risk_level] || RISK_COLORS['None']}`}>
                      {scan.risk_level}
                    </span>
                    {scan.doctor_feedback && (
                      <span className={`text-xs px-2 py-0.5 rounded-full capitalize ${FEEDBACK_BADGE[scan.doctor_feedback]}`}>
                        {scan.doctor_feedback}
                      </span>
                    )}
                  </div>
                  <div className="flex flex-wrap gap-4 text-xs text-slate-400">
                    <span>Confidence: <strong className="text-slate-600">{(scan.confidence * 100).toFixed(1)}%</strong></span>
                    <span>Uncertainty: <strong className={scan.uncertainty > 0.1 ? 'text-amber-600' : 'text-emerald-600'}>±{scan.uncertainty.toFixed(4)}</strong></span>
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {new Date(scan.created_at).toLocaleString()}
                    </span>
                    {scan.original_filename && <span className="truncate max-w-[150px]">{scan.original_filename}</span>}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center gap-2 shrink-0">
                  <button
                    onClick={() => setDetailScan(scan)}
                    className="w-8 h-8 rounded-lg bg-violet-50 hover:bg-violet-100 flex items-center justify-center text-violet-600 transition-colors"
                    title="View Details"
                  >
                    <Eye className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setFeedbackScan(scan)}
                    className="w-8 h-8 rounded-lg bg-blue-50 hover:bg-blue-100 flex items-center justify-center text-blue-600 transition-colors"
                    title="Add Feedback"
                  >
                    <MessageSquare className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => handleDelete(scan.id)}
                    className="w-8 h-8 rounded-lg bg-red-50 hover:bg-red-100 flex items-center justify-center text-red-500 transition-colors"
                    title="Delete Scan"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </motion.div>
            ))}
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-center gap-3 mt-8">
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
              className="w-9 h-9 rounded-xl border border-slate-200 flex items-center justify-center text-slate-600 hover:bg-slate-50 disabled:opacity-40 transition"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="text-sm text-slate-600 font-medium">
              Page {page} of {totalPages}
            </span>
            <button
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
              className="w-9 h-9 rounded-xl border border-slate-200 flex items-center justify-center text-slate-600 hover:bg-slate-50 disabled:opacity-40 transition"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        )}
      </main>
    </div>
  )
}
