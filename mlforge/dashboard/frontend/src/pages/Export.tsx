import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Download, Plus, Smartphone, Monitor, Globe, Cpu, Tablet, Check } from 'lucide-react'

interface ExportedModel {
  id: number
  training_run_id: number
  format: string
  file_path: string
  file_size_mb: number
  quantization: string
  created_at: string
}

interface TrainingRun {
  id: number
  project_id: number
  status: string
  architecture: string
  dataset: string
  best_accuracy: number
}

interface FormatCard {
  format: string
  label: string
  desc: string
  target: string
  icon: typeof Smartphone
  bg: string
  text: string
  border: string
  ring: string
}

const formatCards: FormatCard[] = [
  { format: 'coreml', label: 'CoreML', desc: 'Apple Neural Engine', target: 'iPhone, iPad, Mac', icon: Smartphone, bg: 'bg-purple-100', text: 'text-purple-700', border: 'border-purple-400', ring: 'ring-purple-300' },
  { format: 'tflite', label: 'TFLite', desc: 'Mobile and embedded', target: 'Android, RPi', icon: Tablet, bg: 'bg-green-100', text: 'text-green-700', border: 'border-green-400', ring: 'ring-green-300' },
  { format: 'onnx', label: 'ONNX', desc: 'Universal CPU/GPU', target: 'Server, Desktop', icon: Monitor, bg: 'bg-blue-100', text: 'text-blue-700', border: 'border-blue-400', ring: 'ring-blue-300' },
  { format: 'tfjs', label: 'TF.js', desc: 'Browser inference', target: 'Web apps', icon: Globe, bg: 'bg-orange-100', text: 'text-orange-700', border: 'border-orange-400', ring: 'ring-orange-300' },
  { format: 'edgetpu', label: 'Edge TPU', desc: 'Google Coral', target: 'Coral USB/PCIe', icon: Cpu, bg: 'bg-yellow-100', text: 'text-yellow-700', border: 'border-yellow-400', ring: 'ring-yellow-300' },
]

const formatBadgeColors: Record<string, string> = {
  onnx: 'bg-blue-100 text-blue-700',
  tflite: 'bg-green-100 text-green-700',
  edgetpu: 'bg-yellow-100 text-yellow-700',
  coreml: 'bg-purple-100 text-purple-700',
  tfjs: 'bg-orange-100 text-orange-700',
}

function ExportForm({ onClose }: { onClose: () => void }) {
  const queryClient = useQueryClient()
  const [runId, setRunId] = useState<number>(0)
  const [selectedFormats, setSelectedFormats] = useState<Set<string>>(new Set())

  const { data: runs = [] } = useQuery<TrainingRun[]>({
    queryKey: ['training-runs'],
    queryFn: () => fetch('/api/training/').then(r => r.json()),
  })

  const completedRuns = runs.filter(r => r.status === 'completed')
  const selectedRun = completedRuns.find(r => r.id === runId)

  const toggleFormat = (fmt: string) => {
    setSelectedFormats(prev => {
      const next = new Set(prev)
      if (next.has(fmt)) {
        next.delete(fmt)
      } else {
        next.add(fmt)
      }
      return next
    })
  }

  const exportMutation = useMutation({
    mutationFn: (data: { training_run_id: number; project_id: number; config_path: string; formats: string }) =>
      fetch('/api/exports/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      }).then(r => {
        if (!r.ok) throw new Error('Export failed')
        return r.json()
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['exports'] })
      onClose()
    },
  })

  const handleExport = () => {
    if (!selectedRun || selectedFormats.size === 0) return
    exportMutation.mutate({
      training_run_id: selectedRun.id,
      project_id: selectedRun.project_id,
      config_path: `configs/run_${selectedRun.id}.yaml`,
      formats: Array.from(selectedFormats).join(','),
    })
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border p-6 mb-6">
      <h3 className="font-semibold text-lg mb-4">Export Model</h3>

      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-1">Training Run</label>
        <select
          value={runId}
          onChange={e => setRunId(Number(e.target.value))}
          className="w-full border rounded-lg px-3 py-2"
        >
          <option value={0}>Select a completed run...</option>
          {completedRuns.map(r => (
            <option key={r.id} value={r.id}>
              Run #{r.id} - {r.architecture} ({r.dataset}) - {(r.best_accuracy * 100).toFixed(1)}%
            </option>
          ))}
        </select>
      </div>

      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Export Formats <span className="text-gray-400 font-normal">(click to select)</span>
        </label>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
          {formatCards.map((card) => {
            const selected = selectedFormats.has(card.format)
            const Icon = card.icon
            return (
              <button
                type="button"
                key={card.format}
                onClick={() => toggleFormat(card.format)}
                className={[
                  'relative p-3 rounded-xl border-2 text-left transition-all cursor-pointer',
                  selected
                    ? `${card.bg} ${card.text} ${card.border} ring-2 ${card.ring} ring-offset-1`
                    : 'bg-white border-gray-200 hover:border-gray-400 hover:shadow-sm',
                ].join(' ')}
              >
                {selected && (
                  <div className={`absolute top-2 right-2 w-5 h-5 rounded-full ${card.bg} flex items-center justify-center`}>
                    <Check className={`w-3 h-3 ${card.text}`} />
                  </div>
                )}
                <Icon className={`w-5 h-5 mb-1 ${selected ? card.text : 'text-gray-400'}`} />
                <p className={`font-semibold text-sm ${selected ? card.text : 'text-gray-700'}`}>{card.label}</p>
                <p className={`text-xs mt-0.5 ${selected ? '' : 'text-gray-400'}`}>{card.target}</p>
              </button>
            )
          })}
        </div>
      </div>

      <div className="flex gap-3 mt-4">
        <button
          type="button"
          onClick={handleExport}
          disabled={!runId || selectedFormats.size === 0 || exportMutation.isPending}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 font-medium"
        >
          {exportMutation.isPending
            ? 'Exporting...'
            : selectedFormats.size === 0
              ? 'Select formats above'
              : `Export to ${Array.from(selectedFormats).map(f => f.toUpperCase()).join(', ')}`}
        </button>
        <button type="button" onClick={onClose} className="px-4 py-2 text-gray-600 hover:text-gray-800">
          Cancel
        </button>
      </div>

      {exportMutation.isError && (
        <p className="text-red-600 text-sm mt-3">Export failed. Check server logs.</p>
      )}
    </div>
  )
}

export default function Export() {
  const [showForm, setShowForm] = useState(false)

  const { data: exports = [] } = useQuery<ExportedModel[]>({
    queryKey: ['exports'],
    queryFn: () => fetch('/api/exports/').then(r => r.json()),
    refetchInterval: 5000,
  })

  return (
    <div className="p-8">
      <div className="flex justify-between items-center mb-2">
        <h2 className="text-2xl font-bold text-gray-900">Exported Models</h2>
        <button
          type="button"
          onClick={() => setShowForm(!showForm)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          <Plus className="w-4 h-4" /> Export Model
        </button>
      </div>
      <p className="text-gray-500 mb-6">Models converted for deployment on different platforms</p>

      {showForm && <ExportForm onClose={() => setShowForm(false)} />}

      {!showForm && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
          {formatCards.map((card) => {
            const Icon = card.icon
            const count = exports.filter(e => e.format === card.format).length
            return (
              <div key={card.format} className={`rounded-xl border p-4 ${card.bg}`}>
                <div className="flex items-center gap-2 mb-1">
                  <Icon className={`w-5 h-5 ${card.text}`} />
                  <h4 className={`font-semibold ${card.text}`}>{card.label}</h4>
                  {count > 0 && (
                    <span className={`ml-auto text-xs font-medium ${card.text} opacity-70`}>
                      {count} model{count !== 1 ? 's' : ''}
                    </span>
                  )}
                </div>
                <p className="text-sm opacity-80">{card.desc}</p>
                <p className="text-xs opacity-60 mt-1">{card.target}</p>
              </div>
            )
          })}
        </div>
      )}

      {exports.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <p className="mb-2">No exported models yet.</p>
          <p className="text-sm">Click "Export Model" to convert a trained model for deployment.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {exports.map(exp => (
            <div key={exp.id} className="bg-white rounded-xl border p-4 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <span className={`px-3 py-1 rounded-lg text-sm font-medium ${formatBadgeColors[exp.format] || 'bg-gray-100'}`}>
                  {exp.format.toUpperCase()}
                </span>
                <div>
                  <p className="text-sm font-mono text-gray-700">{exp.file_path}</p>
                  <p className="text-xs text-gray-400">
                    {exp.file_size_mb > 0 ? `${exp.file_size_mb.toFixed(1)} MB` : ''}
                    {exp.quantization !== 'none' && ` | ${exp.quantization}`}
                    {' | Run #'}{exp.training_run_id}
                    {' | '}{new Date(exp.created_at).toLocaleDateString()}
                  </p>
                </div>
              </div>
              <a
                href={`/api/exports/${exp.id}/download`}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-50 text-blue-700 rounded-lg hover:bg-blue-100 text-sm font-medium transition-colors"
              >
                <Download className="w-4 h-4" />
                Download
              </a>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
