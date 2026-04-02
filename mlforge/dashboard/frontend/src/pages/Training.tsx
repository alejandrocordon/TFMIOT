import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Plus, ChevronDown, ChevronUp, AlertCircle, Wrench } from 'lucide-react'

interface TrainingRun {
  id: number
  project_id: number
  status: string
  architecture: string
  epochs: number
  current_epoch: number
  best_accuracy: number
  metrics_json: string
  error_message: string
  created_at: string
}

interface RunLogs {
  run_id: number
  status: string
  error_message: string
  log_output: string
}

interface Project {
  id: number
  name: string
}

interface Architecture {
  name: string
  params: string
  size: string
  target: string
}

interface Dataset {
  name: string
  num_classes: number
  description: string
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    pending: 'bg-gray-100 text-gray-700',
    running: 'bg-yellow-100 text-yellow-700',
    completed: 'bg-green-100 text-green-700',
    failed: 'bg-red-100 text-red-700',
  }
  return (
    <span className={`px-2 py-1 rounded text-xs font-medium ${colors[status] || colors.pending}`}>
      {status}
    </span>
  )
}

function MetricsChart({ metricsJson }: { metricsJson: string }) {
  let metrics: any[] = []
  try {
    metrics = JSON.parse(metricsJson)
  } catch {
    return null
  }

  if (metrics.length === 0) return null

  return (
    <div className="h-64 mt-4">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={metrics}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="epoch" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="train_loss" stroke="#ef4444" name="Train Loss" dot={false} />
          <Line type="monotone" dataKey="val_loss" stroke="#3b82f6" name="Val Loss" dot={false} />
          <Line type="monotone" dataKey="val_accuracy" stroke="#22c55e" name="Val Accuracy" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function NewTrainingForm({ onClose }: { onClose: () => void }) {
  const queryClient = useQueryClient()

  const [projectId, setProjectId] = useState<number>(0)
  const [dataset, setDataset] = useState('cifar10')
  const [architecture, setArchitecture] = useState('mobilenet_v3_small')
  const [framework, setFramework] = useState('pytorch')
  const [epochs, setEpochs] = useState(10)
  const [batchSize, setBatchSize] = useState(32)
  const [learningRate, setLearningRate] = useState(0.001)

  const { data: projects = [] } = useQuery<Project[]>({
    queryKey: ['projects'],
    queryFn: () => fetch('/api/projects/').then(r => r.json()),
  })

  const { data: architectures = [] } = useQuery<Architecture[]>({
    queryKey: ['architectures'],
    queryFn: () => fetch('/api/architectures').then(r => r.json()),
  })

  const { data: datasets = [] } = useQuery<Dataset[]>({
    queryKey: ['datasets'],
    queryFn: () => fetch('/api/datasets').then(r => r.json()),
  })

  const selectedDataset = datasets.find(d => d.name === dataset)

  const createMutation = useMutation({
    mutationFn: (data: any) =>
      fetch('/api/training/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      }).then(r => {
        if (!r.ok) throw new Error('Failed to start training')
        return r.json()
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['training-runs'] })
      onClose()
    },
  })

  const handleSubmit = () => {
    if (!projectId) return
    createMutation.mutate({
      project_id: projectId,
      dataset,
      architecture,
      framework,
      epochs,
      batch_size: batchSize,
      learning_rate: learningRate,
      input_size: 224,
    })
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border p-6 mb-6">
      <h3 className="font-semibold text-lg mb-4">New Training Run</h3>

      <div className="grid grid-cols-2 gap-4">
        {/* Project */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Project</label>
          <select
            value={projectId}
            onChange={e => setProjectId(Number(e.target.value))}
            className="w-full border rounded-lg px-3 py-2"
          >
            <option value={0}>Select a project...</option>
            {projects.map(p => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>
        </div>

        {/* Dataset */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Dataset</label>
          <select
            value={dataset}
            onChange={e => setDataset(e.target.value)}
            className="w-full border rounded-lg px-3 py-2"
          >
            {datasets.map(d => (
              <option key={d.name} value={d.name}>
                {d.name} ({d.num_classes} classes)
              </option>
            ))}
          </select>
          {selectedDataset && (
            <p className="text-xs text-gray-500 mt-1">{selectedDataset.description}</p>
          )}
        </div>

        {/* Architecture */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Architecture</label>
          <select
            value={architecture}
            onChange={e => setArchitecture(e.target.value)}
            className="w-full border rounded-lg px-3 py-2"
          >
            {architectures.map(a => (
              <option key={a.name} value={a.name}>
                {a.name} ({a.params} params, ~{a.size})
              </option>
            ))}
          </select>
        </div>

        {/* Framework */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Framework</label>
          <select
            value={framework}
            onChange={e => setFramework(e.target.value)}
            className="w-full border rounded-lg px-3 py-2"
          >
            <option value="pytorch">PyTorch</option>
            <option value="tensorflow">TensorFlow</option>
          </select>
        </div>

        {/* Epochs */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Epochs</label>
          <input
            type="number"
            value={epochs}
            onChange={e => setEpochs(Number(e.target.value))}
            min={1}
            max={200}
            className="w-full border rounded-lg px-3 py-2"
          />
        </div>

        {/* Batch Size */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Batch Size</label>
          <select
            value={batchSize}
            onChange={e => setBatchSize(Number(e.target.value))}
            className="w-full border rounded-lg px-3 py-2"
          >
            <option value={8}>8</option>
            <option value={16}>16</option>
            <option value={32}>32</option>
            <option value={64}>64</option>
          </select>
        </div>

        {/* Learning Rate */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Learning Rate</label>
          <select
            value={learningRate}
            onChange={e => setLearningRate(Number(e.target.value))}
            className="w-full border rounded-lg px-3 py-2"
          >
            <option value={0.0001}>0.0001</option>
            <option value={0.0005}>0.0005</option>
            <option value={0.001}>0.001</option>
            <option value={0.005}>0.005</option>
            <option value={0.01}>0.01</option>
          </select>
        </div>
      </div>

      <div className="flex gap-3 mt-6">
        <button
          onClick={handleSubmit}
          disabled={!projectId || createMutation.isPending}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 font-medium"
        >
          {createMutation.isPending ? 'Starting...' : 'Start Training'}
        </button>
        <button
          onClick={onClose}
          className="px-4 py-2 text-gray-600 hover:text-gray-800"
        >
          Cancel
        </button>
      </div>

      {createMutation.isError && (
        <p className="text-red-600 text-sm mt-3">Failed to start training. Check server logs.</p>
      )}
    </div>
  )
}

function RunLogViewer({ runId }: { runId: number }) {
  const [expanded, setExpanded] = useState(false)

  const { data: logs } = useQuery<RunLogs>({
    queryKey: ['run-logs', runId],
    queryFn: () => fetch(`/api/training/${runId}/logs`).then(r => r.json()),
    enabled: expanded,
  })

  return (
    <div className="mt-3">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 text-sm text-blue-600 hover:text-blue-800"
      >
        {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        {expanded ? 'Hide Logs' : 'View Logs'}
      </button>

      {expanded && logs && (
        <div className="mt-2 space-y-2">
          {logs.error_message && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3">
              <p className="text-sm font-medium text-red-800 mb-1">Error:</p>
              <pre className="text-xs text-red-700 whitespace-pre-wrap font-mono">{logs.error_message}</pre>
            </div>
          )}
          {logs.log_output && (
            <div className="bg-gray-900 rounded-lg p-3 max-h-96 overflow-auto">
              <pre className="text-xs text-gray-300 whitespace-pre-wrap font-mono">{logs.log_output}</pre>
            </div>
          )}
          {!logs.error_message && !logs.log_output && (
            <p className="text-sm text-gray-500">No logs available yet.</p>
          )}
        </div>
      )}
    </div>
  )
}

function StuckRunFixer({ runId, epoch, total }: { runId: number; epoch: number; total: number }) {
  const queryClient = useQueryClient()
  const fixMutation = useMutation({
    mutationFn: () =>
      fetch(`/api/training/${runId}/fix-status`, { method: 'POST' }).then(r => r.json()),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['training-runs'] }),
  })

  return (
    <div className="mt-3 flex items-center gap-3 bg-yellow-50 border border-yellow-200 rounded-lg p-3">
      <Wrench className="w-4 h-4 text-yellow-600 flex-shrink-0" />
      <p className="text-sm text-yellow-800 flex-1">
        This run appears stuck ({epoch}/{total} epochs done but still "running").
      </p>
      <button
        onClick={() => fixMutation.mutate()}
        disabled={fixMutation.isPending}
        className="px-3 py-1 bg-yellow-600 text-white rounded text-xs font-medium hover:bg-yellow-700 disabled:opacity-50"
      >
        {fixMutation.isPending ? 'Fixing...' : 'Mark as Completed'}
      </button>
    </div>
  )
}

export default function Training() {
  const [showForm, setShowForm] = useState(false)

  const { data: runs = [] } = useQuery<TrainingRun[]>({
    queryKey: ['training-runs'],
    queryFn: () => fetch('/api/training/').then(r => r.json()),
    refetchInterval: 5000,
  })

  return (
    <div className="p-8">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Training Runs</h2>
        <button
          onClick={() => setShowForm(!showForm)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          <Plus className="w-4 h-4" /> New Training Run
        </button>
      </div>

      {showForm && <NewTrainingForm onClose={() => setShowForm(false)} />}

      {runs.length === 0 && !showForm && (
        <div className="text-center py-12 text-gray-500">
          <p className="mb-2">No training runs yet.</p>
          <p className="text-sm">Click "New Training Run" to get started!</p>
        </div>
      )}

      <div className="space-y-6">
        {runs.map(run => (
          <div key={run.id} className="bg-white rounded-xl shadow-sm border p-6">
            <div className="flex justify-between items-start">
              <div>
                <div className="flex items-center gap-3">
                  <h3 className="font-semibold text-lg">{run.architecture || 'Unknown'}</h3>
                  <StatusBadge status={run.status} />
                </div>
                <p className="text-sm text-gray-500 mt-1">
                  Epoch {run.current_epoch}/{run.epochs}
                  {run.best_accuracy > 0 && (
                    <> &middot; Best: <span className="text-green-600 font-medium">
                      {(run.best_accuracy * 100).toFixed(1)}%
                    </span></>
                  )}
                </p>
              </div>
              <span className="text-xs text-gray-400">
                {new Date(run.created_at).toLocaleDateString()}
              </span>
            </div>

            {/* Error banner */}
            {run.status === 'failed' && run.error_message && (
              <div className="mt-3 flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg p-3">
                <AlertCircle className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
                <p className="text-sm text-red-700 line-clamp-2">{run.error_message}</p>
              </div>
            )}

            {/* Stuck run detection + fix button */}
            {run.status === 'running' && run.current_epoch >= run.epochs && run.epochs > 0 && (
              <StuckRunFixer runId={run.id} epoch={run.current_epoch} total={run.epochs} />
            )}

            {/* Progress bar */}
            {run.status === 'running' && (
              <div className="mt-4 bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 rounded-full h-2 transition-all"
                  style={{ width: `${(run.current_epoch / run.epochs) * 100}%` }}
                />
              </div>
            )}

            <MetricsChart metricsJson={run.metrics_json} />

            {/* Logs viewer for failed or completed runs */}
            <RunLogViewer runId={run.id} />
          </div>
        ))}
      </div>
    </div>
  )
}
