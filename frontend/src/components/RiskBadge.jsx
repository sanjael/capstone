import { AlertTriangle, AlertCircle, Info, CheckCircle } from 'lucide-react'

const RISK_CONFIG = {
  Critical: { icon: AlertTriangle, bg: 'bg-red-100', text: 'text-red-800', border: 'border-red-400', dot: 'bg-red-600' },
  High:     { icon: AlertTriangle, bg: 'bg-red-50',    text: 'text-red-700',    border: 'border-red-200',    dot: 'bg-red-500'    },
  Moderate: { icon: AlertCircle,   bg: 'bg-orange-50', text: 'text-orange-700', border: 'border-orange-200', dot: 'bg-orange-500' },
  None:     { icon: CheckCircle,   bg: 'bg-green-50',  text: 'text-green-700',  border: 'border-green-200',  dot: 'bg-green-500'  },
  default:  { icon: Info,          bg: 'bg-slate-50',  text: 'text-slate-700',  border: 'border-slate-200',  dot: 'bg-slate-400'  },
}

export default function RiskBadge({ level }) {
  const cfg = RISK_CONFIG[level] || RISK_CONFIG.default
  const Icon = cfg.icon

  return (
    <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl border ${cfg.bg} ${cfg.border}`}>
      <span className={`w-2 h-2 rounded-full ${cfg.dot} animate-pulse`} />
      <Icon className={`w-4 h-4 ${cfg.text}`} />
      <span className={`text-sm font-semibold ${cfg.text}`}>{level} Risk</span>
    </div>
  )
}
