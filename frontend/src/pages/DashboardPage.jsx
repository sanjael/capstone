import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, ImageIcon, X, Loader2, Brain, Activity, Zap } from 'lucide-react'
import toast from 'react-hot-toast'
import Navbar from '../components/Navbar'
import ResultsPanel from '../components/ResultsPanel'
import { predictionService } from '../services/api'

const ACCEPTED_TYPES = { 'image/jpeg': [], 'image/png': [], 'image/bmp': [] }

export default function DashboardPage() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  const onDrop = useCallback((accepted, rejected) => {
    if (rejected.length > 0) {
      toast.error('Invalid file type. Please upload JPEG or PNG.')
      return
    }
    const f = accepted[0]
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024,
  })

  const handleClear = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
  }

  const handlePredict = async () => {
    if (!file) { toast.error('Please upload an MRI image first.'); return }
    setLoading(true)
    setResult(null)
    try {
      const res = await predictionService.predict(file)
      setResult(res)
      toast.success('Analysis complete!')
    } catch (err) {
      const msg = err.response?.data?.detail || 'Prediction failed. Please try again.'
      toast.error(msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <Navbar />

      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-10">
        {/* Page header */}
        <div className="mb-8">
          <h1 className="font-display text-3xl sm:text-4xl text-slate-900 mb-1">
            MRI Analysis Dashboard
          </h1>
          <p className="text-slate-500">
            Upload a brain MRI scan for AI-powered tumor detection and classification.
          </p>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-3 gap-4 mb-8">
          {[
            { icon: Brain,    label: 'Detection Model',    value: 'EfficientNet-B4', color: 'text-neural-600', bg: 'bg-neural-50' },
            { icon: Zap,      label: 'Classification Model', value: 'ResNet101',       color: 'text-amber-600',  bg: 'bg-amber-50'  },
            { icon: Activity, label: 'Explainability',       value: 'Grad-CAM++ · EigenCAM', color: 'text-emerald-600', bg: 'bg-emerald-50' },
          ].map(({ icon: Icon, label, value, color, bg }) => (
            <div key={label} className={`card flex items-center gap-3 py-4 ${bg} border-0`}>
              <Icon className={`w-5 h-5 ${color} shrink-0`} />
              <div>
                <p className="text-xs text-slate-500 font-medium">{label}</p>
                <p className={`font-semibold text-sm ${color}`}>{value}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left — Upload panel */}
          <div className="space-y-5">
            <div className="card">
              <h2 className="font-semibold text-slate-700 mb-4 flex items-center gap-2">
                <Upload className="w-4 h-4 text-neural-500" />
                Upload MRI Scan
              </h2>

              {!file ? (
                <div
                  {...getRootProps()}
                  className={`upload-zone border-2 border-dashed border-slate-200 rounded-xl p-10
                              flex flex-col items-center justify-center cursor-pointer text-center
                              transition-all duration-200 ${isDragActive ? 'drag-active' : 'hover:border-neural-300 hover:bg-slate-50'}`}
                >
                  <input {...getInputProps()} />
                  <div className="w-14 h-14 bg-neural-50 rounded-2xl flex items-center justify-center mb-4">
                    <ImageIcon className="w-7 h-7 text-neural-400" />
                  </div>
                  {isDragActive
                    ? <p className="text-neural-600 font-medium">Drop the MRI image here…</p>
                    : <>
                        <p className="font-medium text-slate-700 mb-1">Drag & drop MRI image</p>
                        <p className="text-sm text-slate-400">or click to browse</p>
                        <p className="text-xs text-slate-300 mt-3">JPEG, PNG, BMP · Max 10 MB</p>
                      </>
                  }
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="relative rounded-xl overflow-hidden bg-slate-100 aspect-square">
                    <img src={preview} alt="MRI preview" className="w-full h-full object-cover" />
                    <button
                      onClick={handleClear}
                      className="absolute top-2 right-2 w-8 h-8 bg-black/50 hover:bg-black/70
                                 rounded-full flex items-center justify-center transition-colors">
                      <X className="w-4 h-4 text-white" />
                    </button>
                  </div>
                  <div className="flex items-center justify-between text-xs text-slate-500 px-1">
                    <span className="font-medium truncate max-w-[70%]">{file.name}</span>
                    <span>{(file.size / 1024).toFixed(0)} KB</span>
                  </div>
                </div>
              )}
            </div>

            {/* Analyse button */}
            <button
              onClick={handlePredict}
              disabled={!file || loading}
              className="btn-primary w-full justify-center py-3.5 text-base"
            >
              {loading
                ? <><Loader2 className="w-5 h-5 animate-spin" /> Analysing MRI…</>
                : <><Brain className="w-5 h-5" /> Run AI Analysis</>
              }
            </button>

            {loading && (
              <div className="card py-5 text-center animate-fade-in">
                <div className="flex justify-center gap-1 mb-3">
                  {[0, 1, 2, 3, 4].map(i => (
                    <div key={i}
                      className="w-1.5 h-6 bg-neural-400 rounded-full animate-pulse"
                      style={{ animationDelay: `${i * 0.15}s` }} />
                  ))}
                </div>
                <p className="text-sm text-slate-500">Running EfficientNet-B4 + ResNet101 + Grad-CAM++ + EigenCAM…</p>
                <p className="text-xs text-slate-400 mt-1">This may take a few seconds</p>
              </div>
            )}
          </div>

          {/* Right — Results */}
          <div>
            {result ? (
              <ResultsPanel result={result} />
            ) : (
              <div className="card h-full flex flex-col items-center justify-center py-20 text-center border-dashed border-2 border-slate-100">
                <div className="w-16 h-16 bg-slate-50 rounded-2xl flex items-center justify-center mb-4">
                  <Activity className="w-8 h-8 text-slate-300" />
                </div>
                <p className="font-medium text-slate-400">Results will appear here</p>
                <p className="text-sm text-slate-300 mt-1">Upload an MRI and click Run Analysis</p>
              </div>
            )}
          </div>
        </div>

        {/* How it works */}
        <div className="mt-12 card">
          <h2 className="font-display text-xl text-slate-900 mb-5">How the AI Pipeline Works</h2>
          <div className="grid sm:grid-cols-4 gap-4">
            {[
              { step: '01', title: 'Preprocess', desc: 'CLAHE + Skull Strip + Resize 224×224 + ImageNet normalization' },
              { step: '02', title: 'Detect',     desc: 'EfficientNet-B4 + MC Dropout uncertainty estimation' },
              { step: '03', title: 'Classify',   desc: 'ResNet101 — glioma, meningioma, or pituitary' },
              { step: '04', title: 'Explain',    desc: 'Grad-CAM++ · EigenCAM heatmaps' },
            ].map(({ step, title, desc }) => (
              <div key={step} className="bg-slate-50 rounded-xl p-4">
                <span className="font-mono text-xs text-neural-400 font-bold">{step}</span>
                <h3 className="font-semibold text-slate-700 mt-1 mb-1">{title}</h3>
                <p className="text-xs text-slate-500 leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  )
}
