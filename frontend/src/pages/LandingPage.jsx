import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Brain, Zap, Shield, Activity, ArrowRight, CheckCircle, ChevronRight
} from 'lucide-react'

const FEATURES = [
  {
    icon: Brain,
    title: 'ResNet101 Detection',
    desc: 'State-of-the-art deep residual network trained on thousands of MRI scans for binary tumor detection.',
    color: 'from-violet-500 to-purple-600',
  },
  {
    icon: Zap,
    title: 'Grad-CAM++ Explainability',
    desc: 'Visual heatmaps highlight the exact regions that influenced the AI decision — clinical transparency.',
    color: 'from-amber-500 to-orange-600',
  },
  {
    icon: Shield,
    title: 'Bayesian Uncertainty',
    desc: 'Monte Carlo Dropout quantifies prediction confidence — flag uncertain cases for mandatory review.',
    color: 'from-emerald-500 to-teal-600',
  },
]

const TUMOR_TYPES = [
  { name: 'Glioma', desc: 'Most common malignant brain tumor in adults', color: 'bg-red-500', pct: '45%' },
  { name: 'Meningioma', desc: 'Arises from the meninges, often benign', color: 'bg-orange-500', pct: '30%' },
  { name: 'Pituitary', desc: 'Adenoma of the pituitary gland', color: 'bg-yellow-500', pct: '25%' },
]

const STATS = [
  { value: '96.4%', label: 'Detection Accuracy' },
  { value: '3', label: 'Tumor Types Classified' },
  { value: '<3s', label: 'Analysis Time' },
  { value: '100%', label: 'HIPAA Conscious Design' },
]

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white overflow-x-hidden">
      {/* ── Navbar ── */}
      <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4 bg-[#0a0a0f]/80 backdrop-blur-xl border-b border-white/5">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-gradient-to-br from-violet-500 to-purple-600 rounded-lg flex items-center justify-center">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <span className="font-bold text-lg tracking-tight">NeuroScan AI</span>
        </div>
        <div className="hidden md:flex items-center gap-8 text-sm text-gray-400">
          <a href="#features" className="hover:text-white transition-colors">Features</a>
          <a href="#how-it-works" className="hover:text-white transition-colors">How It Works</a>
          <a href="#tumor-types" className="hover:text-white transition-colors">Tumor Types</a>
        </div>
        <div className="flex items-center gap-3">
          <Link to="/dashboard" className="px-4 py-2 bg-violet-600 hover:bg-violet-500 rounded-lg text-sm font-medium transition-colors">
            Analyse Now →
          </Link>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="relative min-h-screen flex flex-col items-center justify-center text-center px-6 pt-20">
        {/* Background glow */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-violet-600/20 rounded-full blur-[120px]" />
          <div className="absolute top-1/3 left-1/4 w-[300px] h-[300px] bg-purple-600/10 rounded-full blur-[80px]" />
          <div className="absolute top-1/3 right-1/4 w-[300px] h-[300px] bg-blue-600/10 rounded-full blur-[80px]" />
        </div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="relative z-10 max-w-4xl"
        >
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-violet-500/10 border border-violet-500/20 text-violet-400 text-sm mb-8">
            <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />
            AI-Powered Clinical Decision Support
          </div>

          <h1 className="text-5xl sm:text-7xl font-black tracking-tight mb-6 leading-tight">
            Brain Tumor Detection
            <span className="block bg-gradient-to-r from-violet-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">
              Reimagined with AI
            </span>
          </h1>

          <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-10 leading-relaxed">
            Upload an MRI scan and get an instant AI analysis with explainable Grad-CAM++ heatmaps,
            Bayesian uncertainty quantification, and a professional clinical PDF report.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-4">
            <Link
              to="/dashboard"
              className="group flex items-center gap-2 px-8 py-4 bg-violet-600 hover:bg-violet-500 rounded-2xl font-semibold text-lg transition-all hover:scale-105 hover:shadow-2xl hover:shadow-violet-500/30"
            >
              Start Analysing Free
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <a
              href="#how-it-works"
              className="flex items-center gap-2 px-8 py-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-2xl font-semibold text-lg transition-all"
            >
              See How It Works
            </a>
          </div>
        </motion.div>

        {/* Stats row */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="relative z-10 grid grid-cols-2 sm:grid-cols-4 gap-8 mt-20 max-w-3xl mx-auto w-full"
        >
          {STATS.map(({ value, label }) => (
            <div key={label} className="text-center">
              <p className="text-3xl font-black text-white mb-1">{value}</p>
              <p className="text-sm text-gray-500">{label}</p>
            </div>
          ))}
        </motion.div>
      </section>

      {/* ── Features ── */}
      <section id="features" className="py-28 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-black mb-4">Everything You Need</h2>
            <p className="text-gray-400 text-lg">A complete clinical AI platform designed for medical professionals</p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {FEATURES.map(({ icon: Icon, title, desc, color }, i) => (
              <motion.div
                key={title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="group p-6 bg-white/3 border border-white/8 rounded-2xl hover:border-white/20 hover:bg-white/6 transition-all"
              >
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                  <Icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="font-bold text-lg mb-2">{title}</h3>
                <p className="text-gray-400 text-sm leading-relaxed">{desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How It Works ── */}
      <section id="how-it-works" className="py-28 px-6 bg-white/2">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-black mb-4">How It Works</h2>
            <p className="text-gray-400 text-lg">Four steps from MRI scan to clinical insight</p>
          </div>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { n: '01', title: 'Upload MRI', desc: 'Drag & drop any JPEG/PNG brain MRI image. Supports up to 10 MB.' },
              { n: '02', title: 'AI Detection', desc: 'ResNet101 with MC Dropout classifies: tumor present or absent.' },
              { n: '03', title: 'Classification', desc: 'If detected, secondary model identifies glioma, meningioma, or pituitary.' },
              { n: '04', title: 'Report & Review', desc: 'Get heatmaps, risk levels, and download a clinical PDF report.' },
            ].map(({ n, title, desc }) => (
              <div key={n} className="relative p-6 bg-white/3 border border-white/8 rounded-2xl">
                <span className="text-5xl font-black text-white/5 absolute top-4 right-4">{n}</span>
                <span className="text-xs font-mono font-bold text-violet-400 mb-3 block">{n}</span>
                <h3 className="font-bold text-lg mb-2">{title}</h3>
                <p className="text-gray-400 text-sm leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Tumor Types ── */}
      <section id="tumor-types" className="py-28 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-black mb-4">3 Tumor Types Classified</h2>
            <p className="text-gray-400 text-lg">The model detects and distinguishes among the most common brain tumor types</p>
          </div>
          <div className="grid sm:grid-cols-3 gap-6">
            {TUMOR_TYPES.map(({ name, desc, color, pct }) => (
              <div key={name} className="p-6 bg-white/3 border border-white/8 rounded-2xl">
                <div className="flex items-center gap-3 mb-4">
                  <div className={`w-3 h-3 rounded-full ${color}`} />
                  <h3 className="font-bold text-xl">{name}</h3>
                </div>
                <p className="text-gray-400 text-sm mb-4">{desc}</p>
                <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                  <div className={`h-full ${color} rounded-full`} style={{ width: pct }} />
                </div>
                <p className="text-xs text-gray-500 mt-1">{pct} of detected cases</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="py-28 px-6">
        <div className="max-w-3xl mx-auto text-center">
          <div className="p-16 bg-gradient-to-br from-violet-600/20 to-purple-600/20 border border-violet-500/20 rounded-3xl">
            <h2 className="text-4xl font-black mb-4">Ready to Analyse?</h2>
            <p className="text-gray-400 text-lg mb-8">
              Join clinicians and researchers using NeuroScan AI for faster, explainable diagnostics.
            </p>
            <div className="flex flex-wrap gap-4 justify-center">
              <Link
                to="/dashboard"
                className="flex items-center gap-2 px-8 py-4 bg-violet-600 hover:bg-violet-500 rounded-2xl font-semibold transition-all hover:scale-105"
              >
                Analyse Now
                <ArrowRight className="w-5 h-5" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="border-t border-white/5 py-8 px-6 text-center text-sm text-gray-600">
        <div className="flex items-center justify-center gap-2 mb-2">
          <Brain className="w-4 h-4 text-violet-500" />
          <span className="font-semibold text-gray-400">NeuroScan AI</span>
        </div>
        <p>AI-assisted clinical decision support. Not a substitute for professional medical diagnosis.</p>
        <p className="mt-1">© {new Date().getFullYear()} NeuroScan AI. All rights reserved.</p>
      </footer>
    </div>
  )
}
