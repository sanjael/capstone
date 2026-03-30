import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Brain, Eye, EyeOff, Loader2, ShieldCheck, User } from 'lucide-react'
import { useAuth } from '../components/AuthContext'
import toast from 'react-hot-toast'

export default function RegisterPage() {
  const { register } = useAuth()
  const navigate = useNavigate()
  const [form, setForm] = useState({ full_name: '', email: '', password: '', role: 'user' })
  const [showPwd, setShowPwd] = useState(false)
  const [loading, setLoading] = useState(false)

  const handleChange = (e) => setForm(f => ({ ...f, [e.target.name]: e.target.value }))

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.full_name || !form.email || !form.password) {
      toast.error('Please fill in all fields.')
      return
    }
    if (form.password.length < 8) {
      toast.error('Password must be at least 8 characters.')
      return
    }
    setLoading(true)
    try {
      await register(form)
      toast.success('Account created! Please log in.')
      navigate('/login')
    } catch (err) {
      const msg = err.response?.data?.detail || 'Registration failed.'
      toast.error(msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-6 bg-slate-50">
      <div className="w-full max-w-lg">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 bg-neural-600 rounded-2xl mb-4">
            <Brain className="w-8 h-8 text-white" />
          </div>
          <h1 className="font-display text-3xl text-slate-900">Create account</h1>
          <p className="text-slate-500 text-sm mt-1">Join NeuroScan AI to access diagnostic tools</p>
        </div>

        <div className="card">
          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="label">Full name</label>
              <input
                name="full_name" type="text" autoComplete="name"
                value={form.full_name} onChange={handleChange}
                className="input" placeholder="Dr. Priya Sharma"
              />
            </div>

            <div>
              <label className="label">Email address</label>
              <input
                name="email" type="email" autoComplete="email"
                value={form.email} onChange={handleChange}
                className="input" placeholder="you@hospital.com"
              />
            </div>

            <div>
              <label className="label">Password</label>
              <div className="relative">
                <input
                  name="password" type={showPwd ? 'text' : 'password'}
                  value={form.password} onChange={handleChange}
                  className="input pr-11" placeholder="Min. 8 characters"
                />
                <button type="button"
                  onClick={() => setShowPwd(v => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600">
                  {showPwd ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {/* Role selector */}
            <div>
              <label className="label">Account type</label>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { value: 'user', label: 'Doctor / Student', icon: User },
                  { value: 'admin', label: 'Administrator', icon: ShieldCheck },
                ].map(({ value, label, icon: Icon }) => (
                  <label key={value}
                    className={`flex items-center gap-3 p-3.5 rounded-xl border-2 cursor-pointer transition-all
                      ${form.role === value
                        ? 'border-neural-500 bg-neural-50'
                        : 'border-slate-200 hover:border-slate-300'
                      }`}>
                    <input type="radio" name="role" value={value}
                      checked={form.role === value} onChange={handleChange}
                      className="sr-only" />
                    <Icon className={`w-4 h-4 ${form.role === value ? 'text-neural-600' : 'text-slate-400'}`} />
                    <span className={`text-sm font-medium ${form.role === value ? 'text-neural-700' : 'text-slate-600'}`}>
                      {label}
                    </span>
                  </label>
                ))}
              </div>
            </div>

            <button type="submit" disabled={loading} className="btn-primary w-full justify-center py-3 mt-2">
              {loading
                ? <><Loader2 className="w-4 h-4 animate-spin" /> Creating account…</>
                : 'Create account'
              }
            </button>
          </form>
        </div>

        <p className="mt-5 text-center text-sm text-slate-500">
          Already have an account?{' '}
          <Link to="/login" className="text-neural-600 font-medium hover:underline">Sign in</Link>
        </p>
      </div>
    </div>
  )
}
