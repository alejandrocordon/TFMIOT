import { useQuery } from '@tanstack/react-query'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface TrainingRun {
  id: number
  project_id: number
  status: string
  architecture: string
  epochs: number
  current_epoch: number
  best_accuracy: number
  metrics_json: string
  created_at: string
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

export default function Training() {
  const { data: runs = [] } = useQuery<TrainingRun[]>({
    queryKey: ['training-runs'],
    queryFn: () => fetch('/api/training/').then(r => r.json()),
    refetchInterval: 5000,
  })

  return (
    <div className="p-8">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Training Runs</h2>

      {runs.length === 0 && (
        <div className="text-center py-12 text-gray-500">
          <p className="mb-2">No training runs yet.</p>
          <p className="text-sm font-mono">mlforge train --config config.yaml</p>
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
          </div>
        ))}
      </div>
    </div>
  )
}
