import { useState, useEffect, useCallback } from 'react'
import { motion } from 'framer-motion'
import {
  Users, Search, Trash2, Shield, ShieldCheck, Loader2,
  ChevronLeft, ChevronRight, AlertTriangle, RefreshCw
} from 'lucide-react'
import toast from 'react-hot-toast'
import Navbar from '../components/Navbar'
import { userService, analyticsService } from '../services/api'
import { useAuth } from '../components/AuthContext'
import { useNavigate } from 'react-router-dom'

function StatPill({ label, value, color = 'bg-violet-100 text-violet-700' }) {
  return (
    <div className={`${color} rounded-xl px-4 py-3 text-center`}>
      <p className="text-2xl font-black">{value}</p>
      <p className="text-xs font-medium mt-0.5 opacity-80">{label}</p>
    </div>
  )
}

export default function AdminPage() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const [users, setUsers] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(true)
  const [platformStats, setPlatformStats] = useState(null)
  const PAGE_SIZE = 20

  useEffect(() => {
    if (user?.role !== 'admin') navigate('/dashboard')
  }, [user, navigate])

  const fetchUsers = useCallback(async () => {
    setLoading(true)
    try {
      const res = await userService.listUsers({ page, pageSize: PAGE_SIZE, search: search || null })
      setUsers(res.items)
      setTotal(res.total)
    } catch {
      toast.error('Failed to load users.')
    } finally {
      setLoading(false)
    }
  }, [page, search])

  useEffect(() => { fetchUsers() }, [fetchUsers])

  useEffect(() => {
    analyticsService.getAdminStats()
      .then(setPlatformStats)
      .catch(() => {})
  }, [])

  const handleDelete = async (userId) => {
    if (!confirm('Permanently delete this user and all their data?')) return
    try {
      await userService.deleteUser(userId)
      setUsers(prev => prev.filter(u => u.id !== userId))
      setTotal(prev => prev - 1)
      toast.success('User deleted.')
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Failed to delete user.')
    }
  }

  const handleRoleChange = async (userId, currentRole) => {
    const newRole = currentRole === 'admin' ? 'user' : 'admin'
    if (!confirm(`Change this user's role to "${newRole}"?`)) return
    try {
      await userService.changeUserRole(userId, newRole)
      setUsers(prev => prev.map(u => u.id === userId ? { ...u, role: newRole } : u))
      toast.success(`Role changed to ${newRole}.`)
    } catch {
      toast.error('Failed to change role.')
    }
  }

  const totalPages = Math.ceil(total / PAGE_SIZE)

  return (
    <div className="min-h-screen bg-slate-50">
      <Navbar />
      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-10">

        <div className="mb-8">
          <div className="flex items-center gap-2 mb-1">
            <div className="w-7 h-7 bg-red-100 rounded-lg flex items-center justify-center">
              <Shield className="w-4 h-4 text-red-600" />
            </div>
            <h1 className="font-display text-3xl text-slate-900">Admin Panel</h1>
          </div>
          <p className="text-slate-500">Manage users, roles, and platform-wide settings.</p>
        </div>

        {/* Platform Stats */}
        {platformStats && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
            <StatPill label="Total Users" value={platformStats.total_users} color="bg-violet-100 text-violet-700" />
            <StatPill label="Total Scans" value={platformStats.total_scans} color="bg-blue-100 text-blue-700" />
            <StatPill label="Tumors Found" value={platformStats.total_detected} color="bg-red-100 text-red-700" />
            <StatPill label="Avg Confidence" value={`${((platformStats.average_confidence || 0) * 100).toFixed(1)}%`} color="bg-emerald-100 text-emerald-700" />
          </div>
        )}

        {/* Users Table */}
        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-4 px-6 py-4 border-b border-slate-100">
            <div className="flex items-center gap-2">
              <Users className="w-4 h-4 text-slate-400" />
              <h2 className="font-semibold text-slate-700">All Users ({total})</h2>
            </div>
            <div className="flex items-center gap-3">
              {/* Search */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                <input
                  value={search}
                  onChange={e => { setSearch(e.target.value); setPage(1) }}
                  placeholder="Search name or email…"
                  className="border border-slate-200 rounded-xl pl-9 pr-3 py-2 text-sm w-52 focus:outline-none focus:ring-2 focus:ring-violet-400"
                />
              </div>
              <button
                onClick={fetchUsers}
                className="w-9 h-9 rounded-xl border border-slate-200 flex items-center justify-center text-slate-500 hover:bg-slate-50 transition"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            </div>
          </div>

          {loading ? (
            <div className="flex items-center justify-center h-32">
              <Loader2 className="w-6 h-6 text-violet-500 animate-spin" />
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-xs uppercase font-semibold text-slate-400 border-b border-slate-50">
                    <th className="text-left px-6 py-3">User</th>
                    <th className="text-left px-6 py-3">Email</th>
                    <th className="text-left px-6 py-3">Role</th>
                    <th className="text-left px-6 py-3">Joined</th>
                    <th className="text-right px-6 py-3">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((u, i) => (
                    <motion.tr
                      key={u.id}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: i * 0.04 }}
                      className="border-b border-slate-50 hover:bg-slate-25 transition-colors last:border-none"
                    >
                      <td className="px-6 py-3">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 bg-gradient-to-br from-violet-500 to-purple-600 rounded-lg flex items-center justify-center text-white text-xs font-bold">
                            {u.full_name.charAt(0).toUpperCase()}
                          </div>
                          <div>
                            <p className="font-medium text-slate-800 text-sm">{u.full_name}</p>
                            <p className="text-xs text-slate-400">ID #{u.id}</p>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-3 text-sm text-slate-600">{u.email}</td>
                      <td className="px-6 py-3">
                        <span className={`inline-flex items-center gap-1 text-xs px-2.5 py-1 rounded-full font-semibold ${
                          u.role === 'admin' ? 'bg-violet-100 text-violet-700' : 'bg-slate-100 text-slate-600'
                        }`}>
                          {u.role === 'admin' ? <ShieldCheck className="w-3 h-3" /> : <Shield className="w-3 h-3" />}
                          {u.role}
                        </span>
                      </td>
                      <td className="px-6 py-3 text-xs text-slate-400">
                        {new Date(u.created_at).toLocaleDateString()}
                      </td>
                      <td className="px-6 py-3">
                        <div className="flex items-center justify-end gap-2">
                          {u.id !== user?.id && (
                            <>
                              <button
                                onClick={() => handleRoleChange(u.id, u.role)}
                                className="px-3 py-1.5 text-xs border border-slate-200 rounded-lg text-slate-600 hover:bg-slate-50 transition font-medium"
                                title={`Change to ${u.role === 'admin' ? 'user' : 'admin'}`}
                              >
                                {u.role === 'admin' ? 'Demote' : 'Promote'}
                              </button>
                              <button
                                onClick={() => handleDelete(u.id)}
                                className="w-7 h-7 rounded-lg bg-red-50 hover:bg-red-100 flex items-center justify-center text-red-500 transition"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                              </button>
                            </>
                          )}
                          {u.id === user?.id && (
                            <span className="text-xs text-slate-300 italic">You</span>
                          )}
                        </div>
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-3 px-6 py-4 border-t border-slate-50">
              <button
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page === 1}
                className="w-9 h-9 rounded-xl border border-slate-200 flex items-center justify-center text-slate-600 hover:bg-slate-50 disabled:opacity-40 transition"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="text-sm text-slate-600 font-medium">Page {page} of {totalPages}</span>
              <button
                onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
                className="w-9 h-9 rounded-xl border border-slate-200 flex items-center justify-center text-slate-600 hover:bg-slate-50 disabled:opacity-40 transition"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>

        <div className="mt-6 p-4 bg-amber-50 border border-amber-200 rounded-2xl flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
          <p className="text-sm text-amber-800">
            <strong>Admin Actions are Irreversible.</strong> Deleting a user will permanently remove their account and all associated scan records. Promote/demote roles with care.
          </p>
        </div>
      </main>
    </div>
  )
}
