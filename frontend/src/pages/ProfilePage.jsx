import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  User, Mail, Shield, Lock, Save, Edit3, CheckCircle, Loader2, Eye, EyeOff
} from 'lucide-react'
import toast from 'react-hot-toast'
import Navbar from '../components/Navbar'
import { userService } from '../services/api'
import { useAuth } from '../components/AuthContext'

function Section({ title, children }) {
  return (
    <div className="bg-white rounded-2xl border border-slate-100 shadow-sm overflow-hidden mb-6">
      <div className="px-6 py-4 border-b border-slate-50">
        <h2 className="font-semibold text-slate-700">{title}</h2>
      </div>
      <div className="p-6">{children}</div>
    </div>
  )
}

export default function ProfilePage() {
  const { user } = useAuth()
  const [profile, setProfile] = useState(null)
  const [editing, setEditing] = useState(false)
  const [saving, setSaving] = useState(false)
  const [form, setForm] = useState({ full_name: '', bio: '' })
  const [pwForm, setPwForm] = useState({ current: '', newPw: '', confirm: '' })
  const [showPw, setShowPw] = useState(false)
  const [pwLoading, setPwLoading] = useState(false)

  useEffect(() => {
    userService.getProfile().then(p => {
      setProfile(p)
      setForm({ full_name: p.full_name, bio: p.bio || '' })
    }).catch(() => toast.error('Failed to load profile.'))
  }, [])

  const handleSaveProfile = async () => {
    setSaving(true)
    try {
      const updated = await userService.updateProfile(form)
      setProfile(updated)
      setEditing(false)
      toast.success('Profile updated!')
      // Update local storage name
      const stored = JSON.parse(localStorage.getItem('user') || '{}')
      stored.full_name = updated.full_name
      localStorage.setItem('user', JSON.stringify(stored))
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Failed to update profile.')
    } finally {
      setSaving(false)
    }
  }

  const handleChangePassword = async () => {
    if (pwForm.newPw !== pwForm.confirm) {
      toast.error('New passwords do not match.')
      return
    }
    if (pwForm.newPw.length < 8) {
      toast.error('Password must be at least 8 characters.')
      return
    }
    setPwLoading(true)
    try {
      await userService.changePassword(pwForm.current, pwForm.newPw)
      toast.success('Password changed successfully!')
      setPwForm({ current: '', newPw: '', confirm: '' })
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Failed to change password.')
    } finally {
      setPwLoading(false)
    }
  }

  if (!profile) {
    return (
      <div className="min-h-screen bg-slate-50">
        <Navbar />
        <div className="flex items-center justify-center h-[60vh]">
          <Loader2 className="w-7 h-7 text-violet-500 animate-spin" />
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <Navbar />
      <main className="max-w-3xl mx-auto px-4 sm:px-6 py-10">

        {/* Avatar & Name */}
        <div className="flex flex-col sm:flex-row items-center sm:items-start gap-6 mb-8">
          <div className="w-20 h-20 bg-gradient-to-br from-violet-500 to-purple-600 rounded-2xl flex items-center justify-center text-white text-3xl font-black select-none shadow-lg">
            {profile.full_name.charAt(0).toUpperCase()}
          </div>
          <div>
            <h1 className="font-display text-3xl text-slate-900">{profile.full_name}</h1>
            <p className="text-slate-500 flex items-center gap-1.5 mt-1">
              <Mail className="w-4 h-4" />
              {profile.email}
            </p>
            <span className={`mt-2 inline-flex items-center gap-1 text-xs px-3 py-1 rounded-full font-semibold ${
              profile.role === 'admin'
                ? 'bg-violet-100 text-violet-700'
                : 'bg-slate-100 text-slate-600'
            }`}>
              <Shield className="w-3 h-3" />
              {profile.role === 'admin' ? 'Administrator' : 'Clinician'}
            </span>
          </div>
        </div>

        {/* Profile Details */}
        <Section title="Profile Information">
          <div className="space-y-4">
            <div>
              <label className="text-xs font-semibold text-slate-500 uppercase block mb-1">Full Name</label>
              {editing ? (
                <input
                  value={form.full_name}
                  onChange={e => setForm(f => ({ ...f, full_name: e.target.value }))}
                  className="border border-slate-200 rounded-xl px-3 py-2 text-sm w-full focus:outline-none focus:ring-2 focus:ring-violet-400"
                />
              ) : (
                <p className="text-slate-800 font-medium">{profile.full_name}</p>
              )}
            </div>

            <div>
              <label className="text-xs font-semibold text-slate-500 uppercase block mb-1">Email</label>
              <p className="text-slate-600 text-sm flex items-center gap-2">
                {profile.email}
                <CheckCircle className="w-3.5 h-3.5 text-emerald-500" />
              </p>
            </div>

            <div>
              <label className="text-xs font-semibold text-slate-500 uppercase block mb-1">Bio</label>
              {editing ? (
                <textarea
                  value={form.bio}
                  onChange={e => setForm(f => ({ ...f, bio: e.target.value }))}
                  rows={3}
                  placeholder="Brief professional bio…"
                  className="border border-slate-200 rounded-xl px-3 py-2 text-sm w-full resize-none focus:outline-none focus:ring-2 focus:ring-violet-400"
                />
              ) : (
                <p className="text-slate-600 text-sm">{profile.bio || <span className="text-slate-300 italic">No bio set</span>}</p>
              )}
            </div>

            <div>
              <label className="text-xs font-semibold text-slate-500 uppercase block mb-1">Member Since</label>
              <p className="text-slate-600 text-sm">{new Date(profile.created_at).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}</p>
            </div>

            <div className="flex items-center gap-3 pt-2">
              {editing ? (
                <>
                  <button
                    onClick={handleSaveProfile}
                    disabled={saving}
                    className="flex items-center gap-2 px-5 py-2 bg-violet-600 hover:bg-violet-500 text-white rounded-xl text-sm font-semibold transition-all disabled:opacity-50"
                  >
                    {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                    {saving ? 'Saving…' : 'Save Changes'}
                  </button>
                  <button
                    onClick={() => { setEditing(false); setForm({ full_name: profile.full_name, bio: profile.bio || '' }) }}
                    className="px-5 py-2 border border-slate-200 text-slate-600 hover:bg-slate-50 rounded-xl text-sm font-semibold transition-all"
                  >
                    Cancel
                  </button>
                </>
              ) : (
                <button
                  onClick={() => setEditing(true)}
                  className="flex items-center gap-2 px-5 py-2 border border-slate-200 text-slate-600 hover:bg-slate-50 rounded-xl text-sm font-semibold transition-all"
                >
                  <Edit3 className="w-4 h-4" />
                  Edit Profile
                </button>
              )}
            </div>
          </div>
        </Section>

        {/* Change Password */}
        <Section title="Change Password">
          <div className="space-y-4 max-w-sm">
            {[
              { key: 'current', label: 'Current Password' },
              { key: 'newPw', label: 'New Password' },
              { key: 'confirm', label: 'Confirm New Password' },
            ].map(({ key, label }) => (
              <div key={key}>
                <label className="text-xs font-semibold text-slate-500 uppercase block mb-1">{label}</label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-300" />
                  <input
                    type={showPw ? 'text' : 'password'}
                    value={pwForm[key]}
                    onChange={e => setPwForm(f => ({ ...f, [key]: e.target.value }))}
                    className="border border-slate-200 rounded-xl pl-9 pr-9 py-2 text-sm w-full focus:outline-none focus:ring-2 focus:ring-violet-400"
                    placeholder="••••••••"
                  />
                </div>
              </div>
            ))}
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="showPw"
                checked={showPw}
                onChange={e => setShowPw(e.target.checked)}
                className="accent-violet-600"
              />
              <label htmlFor="showPw" className="text-xs text-slate-500 cursor-pointer">Show passwords</label>
            </div>
            <button
              onClick={handleChangePassword}
              disabled={pwLoading || !pwForm.current || !pwForm.newPw || !pwForm.confirm}
              className="flex items-center gap-2 px-5 py-2 bg-violet-600 hover:bg-violet-500 disabled:opacity-50 text-white rounded-xl text-sm font-semibold transition-all"
            >
              {pwLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Lock className="w-4 h-4" />}
              {pwLoading ? 'Changing…' : 'Change Password'}
            </button>
          </div>
        </Section>

      </main>
    </div>
  )
}
