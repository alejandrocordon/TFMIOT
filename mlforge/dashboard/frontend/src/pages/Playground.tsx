import { useState, useCallback } from 'react'
import { Upload, Image as ImageIcon } from 'lucide-react'

interface Prediction {
  class_id: number
  confidence: number
}

export default function Playground() {
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [dragOver, setDragOver] = useState(false)

  const handleFile = useCallback(async (file: File) => {
    // Show preview
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target?.result as string)
    reader.readAsDataURL(file)

    // Run inference
    setLoading(true)
    setPredictions([])

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('/api/inference', { method: 'POST', body: formData })
      const data = await res.json()
      setPredictions(data.predictions || [])
    } catch (err) {
      console.error('Inference failed:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      handleFile(file)
    }
  }, [handleFile])

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }, [handleFile])

  return (
    <div className="p-8">
      <h2 className="text-2xl font-bold text-gray-900 mb-2">Playground</h2>
      <p className="text-gray-500 mb-6">Test your models with real images</p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Upload area */}
        <div>
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-xl p-12 text-center transition-colors ${
              dragOver ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
            }`}
          >
            {preview ? (
              <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded-lg" />
            ) : (
              <div className="text-gray-400">
                <ImageIcon className="w-16 h-16 mx-auto mb-4" />
                <p className="text-lg font-medium">Drop an image here</p>
                <p className="text-sm mt-1">or click to browse</p>
              </div>
            )}
            <input
              type="file"
              accept="image/*"
              onChange={handleInputChange}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              style={{ position: 'relative' }}
            />
          </div>
        </div>

        {/* Results */}
        <div>
          <h3 className="font-semibold text-lg mb-4">Predictions</h3>

          {loading && (
            <div className="flex items-center gap-3 text-gray-500">
              <div className="animate-spin w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full" />
              Running inference...
            </div>
          )}

          {predictions.length > 0 && (
            <div className="space-y-3">
              {predictions.map((pred, i) => (
                <div key={i} className="bg-white rounded-lg border p-4">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-medium">Class {pred.class_id}</span>
                    <span className="text-sm font-mono text-gray-500">
                      {(pred.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 rounded-full h-2 transition-all"
                      style={{ width: `${pred.confidence * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}

          {!loading && predictions.length === 0 && (
            <p className="text-gray-400">Upload an image to see predictions</p>
          )}
        </div>
      </div>
    </div>
  )
}
