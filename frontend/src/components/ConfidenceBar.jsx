import { useEffect, useState } from 'react'

const getColor = (value) => {
  if (value >= 0.85) return 'bg-emerald-500'
  if (value >= 0.70) return 'bg-yellow-500'
  return 'bg-red-500'
}

export default function ConfidenceBar({ value, label = 'Tumor Detection Confidence' }) {
  const [width, setWidth] = useState(0)
  const pct = Math.round((value ?? 0) * 100)

  useEffect(() => {
    const timer = setTimeout(() => setWidth(pct), 100)
    return () => clearTimeout(timer)
  }, [pct])

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <span className="text-sm font-medium text-slate-600">{label}</span>
        <span className="font-mono font-bold text-slate-800">{pct}%</span>
      </div>
      <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full confidence-fill ${getColor(value ?? 0)}`}
          style={{ width: `${width}%` }}
        />
      </div>
      <p className="text-xs text-slate-400">
        {(value ?? 0) >= 0.75 ? '✅ Above reliability threshold (75%)' : '⚠️ Below reliability threshold — further evaluation advised'}
      </p>
    </div>
  )
}
