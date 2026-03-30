import axios from 'axios'

const api = axios.create({
  baseURL: '',  // proxied by Vite → localhost:8000
  timeout: 60000,
})

// ── Prediction ────────────────────────────────────────────────────────────────

export const predictionService = {
  async predict(file) {
    const formData = new FormData()
    formData.append('file', file)
    const { data } = await api.post('/predict/', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    console.log('[NeuroScan] API response:', JSON.stringify({
      prediction_entropy: data.prediction_entropy,
      uncertainty_profile: data.uncertainty_profile,
    }, null, 2))
    return data
  },

  async getReport(predictionData) {
    const { data } = await api.post('/report/', predictionData, {
      responseType: 'blob',
    })
    return data
  },
}

// ── Analytics ─────────────────────────────────────────────────────────────────

export const analyticsService = {
  async getSummary(scans) {
    const { data } = await api.post('/analytics/summary', { scans })
    return data
  },
}

export default api
