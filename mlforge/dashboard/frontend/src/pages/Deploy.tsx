import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Smartphone, Globe, Cpu, Plus, Download, Package, Trash2 } from 'lucide-react'

interface DeployedApp {
  id: number
  project_id: number
  training_run_id: number
  target: string
  labels: string
  input_size: number
  output_path: string
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

interface ExportedModel {
  id: number
  training_run_id: number
  format: string
}

const targetInfo: Record<string, { label: string; desc: string; tech: string; format: string; color: string; bgColor: string }> = {
  ios: {
    label: 'iOS',
    desc: 'iPhone, iPad, Mac',
    tech: 'SwiftUI + CoreML + Vision',
    format: 'coreml',
    color: 'text-purple-700',
    bgColor: 'bg-purple-50 border-purple-200',
  },
  android: {
    label: 'Android',
    desc: 'Phones & Tablets',
    tech: 'Kotlin + CameraX + TFLite',
    format: 'tflite',
    color: 'text-green-700',
    bgColor: 'bg-green-50 border-green-200',
  },
  web: {
    label: 'Web',
    desc: 'Browser apps',
    tech: 'ONNX Runtime Web',
    format: 'onnx',
    color: 'text-blue-700',
    bgColor: 'bg-blue-50 border-blue-200',
  },
  edge: {
    label: 'Edge / RPi',
    desc: 'Raspberry Pi + Coral',
    tech: 'Python + TFLite Runtime',
    format: 'tflite',
    color: 'text-yellow-700',
    bgColor: 'bg-yellow-50 border-yellow-200',
  },
}

const targetIcons: Record<string, typeof Smartphone> = {
  ios: Smartphone,
  android: Smartphone,
  web: Globe,
  edge: Cpu,
}

function DeployForm({ onClose }: { onClose: () => void }) {
  const queryClient = useQueryClient()
  const [runId, setRunId] = useState<number>(0)
  const [target, setTarget] = useState('ios')
  const [labels, setLabels] = useState('')
  const [inputSize, setInputSize] = useState(224)

  const { data: runs = [] } = useQuery<TrainingRun[]>({
    queryKey: ['training-runs'],
    queryFn: () => fetch('/api/training/').then(r => r.json()),
  })

  const { data: exports = [] } = useQuery<ExportedModel[]>({
    queryKey: ['exports'],
    queryFn: () => fetch('/api/exports/').then(r => r.json()),
  })

  const completedRuns = runs.filter(r => r.status === 'completed')

  // Check which targets have exports available for the selected run
  const runExportFormats = new Set(
    exports.filter(e => e.training_run_id === runId).map(e => e.format)
  )

  const deployMutation = useMutation({
    mutationFn: (data: { training_run_id: number; target: string; labels: string; input_size: number }) =>
      fetch('/api/deploy/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      }).then(r => {
        if (!r.ok) return r.json().then(err => { throw new Error(err.detail || 'Deploy failed') })
        return r.json()
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['deploys'] })
      onClose()
    },
  })

  const info = targetInfo[target]
  const requiredFormat = info?.format
  const hasExport = runId > 0 && (requiredFormat ? runExportFormats.has(requiredFormat) : false)

  return (
    <div className="bg-white rounded-xl shadow-sm border p-6 mb-6">
      <h3 className="font-semibold text-lg mb-4">Generate App</h3>

      <div className="grid grid-cols-2 gap-4">
        {/* Training Run */}
        <div>
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

        {/* Input Size */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Input Size</label>
          <select
            value={inputSize}
            onChange={e => setInputSize(Number(e.target.value))}
            className="w-full border rounded-lg px-3 py-2"
          >
            <option value={224}>224x224</option>
            <option value={256}>256x256</option>
            <option value={320}>320x320</option>
          </select>
        </div>

        {/* Labels */}
        <div className="col-span-2">
          <label className="block text-sm font-medium text-gray-700 mb-1">Class Labels (comma-separated)</label>
          <input
            value={labels}
            onChange={e => setLabels(e.target.value)}
            placeholder="e.g. airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck"
            className="w-full border rounded-lg px-3 py-2"
          />
          <p className="text-xs text-gray-400 mt-1">Leave empty for generic labels (Class 0, Class 1, ...)</p>
        </div>
      </div>

      {/* Target Selection */}
      <div className="mt-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">Platform</label>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {Object.entries(targetInfo).map(([key, info]) => {
            const Icon = targetIcons[key]
            const selected = target === key
            const exportAvailable = runId > 0 ? runExportFormats.has(info.format) : true
            return (
              <button
                key={key}
                onClick={() => setTarget(key)}
                className={`p-4 rounded-xl border-2 text-left transition-all ${
                  selected
                    ? `${info.bgColor} border-current ring-2 ring-offset-1 ${info.color}`
                    : 'bg-white border-gray-200 hover:border-gray-300'
                } ${!exportAvailable && runId > 0 ? 'opacity-50' : ''}`}
              >
                <Icon className={`w-6 h-6 mb-2 ${selected ? info.color : 'text-gray-400'}`} />
                <p className="font-semibold text-sm">{info.label}</p>
                <p className={`text-xs mt-0.5 ${selected ? 'opacity-80' : 'text-gray-400'}`}>{info.desc}</p>
                <p className={`text-xs mt-1 font-mono ${selected ? 'opacity-60' : 'text-gray-300'}`}>{info.format}</p>
                {runId > 0 && !exportAvailable && (
                  <p className="text-xs text-red-500 mt-1">No {info.format} export</p>
                )}
              </button>
            )
          })}
        </div>
      </div>

      {/* Info box */}
      {info && (
        <div className={`mt-4 p-3 rounded-lg ${info.bgColor} border`}>
          <p className={`text-sm font-medium ${info.color}`}>
            {info.label}: {info.tech}
          </p>
          <p className="text-xs mt-1 opacity-70">
            Requires {info.format.toUpperCase()} export.
            {!hasExport && runId > 0 && ' Export the model first from the Export page.'}
          </p>
        </div>
      )}

      <div className="flex gap-3 mt-4">
        <button
          onClick={() => deployMutation.mutate({ training_run_id: runId, target, labels, input_size: inputSize })}
          disabled={!runId || deployMutation.isPending}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 font-medium"
        >
          {deployMutation.isPending ? 'Generating...' : `Generate ${info?.label || ''} App`}
        </button>
        <button onClick={onClose} className="px-4 py-2 text-gray-600 hover:text-gray-800">
          Cancel
        </button>
      </div>

      {deployMutation.isError && (
        <p className="text-red-600 text-sm mt-3">
          {(deployMutation.error as Error)?.message || 'Failed to generate app. Check server logs.'}
        </p>
      )}
    </div>
  )
}

export default function Deploy() {
  const [showForm, setShowForm] = useState(false)

  const { data: deploys = [] } = useQuery<DeployedApp[]>({
    queryKey: ['deploys'],
    queryFn: () => fetch('/api/deploy/').then(r => r.json()),
  })

  return (
    <div className="p-8">
      <div className="flex justify-between items-center mb-2">
        <h2 className="text-2xl font-bold text-gray-900">Deploy Apps</h2>
        <button
          onClick={() => setShowForm(!showForm)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          <Plus className="w-4 h-4" /> Generate App
        </button>
      </div>
      <p className="text-gray-500 mb-6">Generate ready-to-build apps for mobile, web, and edge devices</p>

      {showForm && <DeployForm onClose={() => setShowForm(false)} />}

      {/* Platform cards overview */}
      {!showForm && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {Object.entries(targetInfo).map(([key, info]) => {
            const Icon = targetIcons[key]
            const count = deploys.filter(d => d.target === key).length
            return (
              <div key={key} className={`rounded-xl border p-4 ${info.bgColor}`}>
                <div className="flex items-center gap-2 mb-1">
                  <Icon className={`w-5 h-5 ${info.color}`} />
                  <h4 className={`font-semibold ${info.color}`}>{info.label}</h4>
                </div>
                <p className="text-xs opacity-70">{info.tech}</p>
                {count > 0 && (
                  <p className="text-xs font-medium mt-2 opacity-80">{count} app{count !== 1 ? 's' : ''} generated</p>
                )}
              </div>
            )
          })}
        </div>
      )}

      {/* Generated apps list */}
      {deploys.length === 0 && !showForm ? (
        <div className="text-center py-12 text-gray-500">
          <Package className="w-12 h-12 mx-auto mb-3 text-gray-300" />
          <p className="mb-2">No apps generated yet.</p>
          <p className="text-sm">Train a model, export it, then generate an app here.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {deploys.map(app => {
            const info = targetInfo[app.target]
            const Icon = targetIcons[app.target] || Package
            return (
              <div key={app.id} className="bg-white rounded-xl border p-4 flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className={`p-2 rounded-lg ${info?.bgColor || 'bg-gray-50'}`}>
                    <Icon className={`w-5 h-5 ${info?.color || 'text-gray-500'}`} />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className={`font-semibold ${info?.color || ''}`}>{info?.label || app.target}</span>
                      <span className="text-xs text-gray-400">Run #{app.training_run_id}</span>
                    </div>
                    <p className="text-xs text-gray-500 font-mono mt-0.5">{app.output_path}</p>
                    {app.labels && (
                      <p className="text-xs text-gray-400 mt-0.5 truncate max-w-md">{app.labels}</p>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-400">
                    {new Date(app.created_at).toLocaleDateString()}
                  </span>
                  <a
                    href={`/api/deploy/${app.id}/download`}
                    className="flex items-center gap-1 px-3 py-1.5 bg-blue-50 text-blue-700 rounded-lg hover:bg-blue-100 text-sm font-medium"
                  >
                    <Download className="w-4 h-4" />
                    .zip
                  </a>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
