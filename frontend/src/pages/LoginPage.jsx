import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Brain, Eye, EyeOff, Loader2 } from 'lucide-react'
import { useAuth } from '../components/AuthContext'
import toast from 'react-hot-toast'

export default function LoginPage() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const [form, setForm] = useState({ email: '', password: '' })
  const [showPwd, setShowPwd] = useState(false)
  const [loading, setLoading] = useState(false)

  const handleChange = (e) => setForm(f => ({ ...f, [e.target.name]: e.target.value }))

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.email || !form.password) {
      toast.error('Please fill in all fields.')
      return
    }
    setLoading(true)
    try {
      await login(form)
      toast.success('Welcome back!')
      navigate('/dashboard')
    } catch (err) {
      const msg = err.response?.data?.detail || 'Login failed. Please try again.'
      toast.error(msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex">
      {/* Left panel — brand */}
      <div className="hidden lg:flex flex-col justify-between w-1/2 bg-neural-700 p-12 text-white relative overflow-hidden">
        {/* Background pattern */}
        <div className="absolute inset-0 opacity-10"
          style={{ backgroundImage: 'radial-gradient(circle at 20% 50%, white 1px, transparent 1px), radial-gradient(circle at 80% 20%, white 1px, transparent 1px)', backgroundSize: '60px 60px' }} />

        <div className="relative">
          <div className="flex items-center gap-3 mb-12">
            <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <span className="font-display text-2xl">NeuroScan AI</span>
          </div>

          <h1 className="font-display text-4xl leading-tight mb-4">
            Intelligent Brain<br />
            <em>Tumor Detection</em>
          </h1>
          <p className="text-neural-200 text-lg leading-relaxed max-w-sm">
            Powered by ResNet101 and Grad-CAM++ explainability — designed for clinical insight.
          </p>
        </div>

        <div className="relative space-y-4">
          {[
            { stat: '97%+', label: 'Detection Accuracy' },
            { stat: 'Grad-CAM++', label: 'Explainable AI' },
            { stat: '3 Types', label: 'Tumor Classification' },
          ].map(({ stat, label }) => (
            <div key={label} className="flex items-center gap-4 bg-white/10 rounded-2xl px-5 py-3">
              <span className="font-mono font-bold text-xl text-white">{stat}</span>
              <span className="text-neural-200 text-sm">{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Right panel — form */}
      <div className="flex-1 flex items-center justify-center p-8 bg-slate-50">
        <div className="w-full max-w-md">
          {/* Mobile brand */}
          <div className="lg:hidden flex items-center gap-2 mb-8">
            <div className="w-9 h-9 bg-neural-600 rounded-xl flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <span className="font-display text-xl text-slate-900">NeuroScan AI</span>
          </div>

          <h2 className="font-display text-3xl text-slate-900 mb-1">Sign in</h2>
          <p className="text-slate-500 text-sm mb-8">Access your diagnostic dashboard</p>

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="label">Email address</label>
              <input
                name="email" type="email" autoComplete="email"
                value={form.email} onChange={handleChange}
                className="input" placeholder="doctor@hospital.com"
              />
            </div>

            <div>
              <label className="label">Password</label>
              <div className="relative">
                <input
                  name="password" type={showPwd ? 'text' : 'password'}
                  autoComplete="current-password"
                  value={form.password} onChange={handleChange}
                  className="input pr-11" placeholder="••••••••"
                />
                <button type="button"
                  onClick={() => setShowPwd(v => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600">
                  {showPwd ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <button type="submit" disabled={loading} className="btn-primary w-full justify-center py-3">
              {loading
                ? <><Loader2 className="w-4 h-4 animate-spin" /> Signing in…</>
                : 'Sign in'
              }
            </button>
          </form>

          <p className="mt-6 text-center text-sm text-slate-500">
            Don't have an account?{' '}
            <Link to="/register" className="text-neural-600 font-medium hover:underline">Register here</Link>
          </p>

          {/* Demo hint */}
          <div className="mt-8 p-4 bg-neural-50 border border-neural-100 rounded-xl">
            <p className="text-xs text-neural-600 font-medium mb-1">Demo credentials</p>
            <p className="text-xs text-slate-500 font-mono">Register a new account to get started</p>
          </div>
        </div>
      </div>
    </div>
  )
}
