import { useQuery } from '@tanstack/react-query'
import { Cpu, FlaskConical, Play, Upload } from 'lucide-react'

interface Summary {
  total_projects: number
  total_runs: number
  running_runs: number
  total_exports: number
  best_accuracy: number
  best_model_architecture: string
}

function StatCard({ icon: Icon, label, value, color }: {
  icon: any, label: string, value: string | number, color: string
}) {
  return (
    <div className="bg-white rounded-xl shadow-sm border p-6">
      <div className="flex items-center gap-4">
        <div className={`p-3 rounded-lg ${color}`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
        <div>
          <p className="text-sm text-gray-500">{label}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
        </div>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const { data: summary } = useQuery<Summary>({
    queryKey: ['summary'],
    queryFn: () => fetch('/api/metrics/summary').then(r => r.json()),
  })

  return (
    <div className="p-8">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Dashboard</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          icon={FlaskConical}
          label="Projects"
          value={summary?.total_projects ?? 0}
          color="bg-blue-500"
        />
        <StatCard
          icon={Play}
          label="Training Runs"
          value={summary?.total_runs ?? 0}
          color="bg-green-500"
        />
        <StatCard
          icon={Cpu}
          label="Running Now"
          value={summary?.running_runs ?? 0}
          color="bg-yellow-500"
        />
        <StatCard
          icon={Upload}
          label="Exported Models"
          value={summary?.total_exports ?? 0}
          color="bg-purple-500"
        />
      </div>

      {summary && summary.best_accuracy > 0 && (
        <div className="bg-white rounded-xl shadow-sm border p-6">
          <h3 className="text-lg font-semibold mb-2">Best Model</h3>
          <p className="text-gray-600">
            <span className="font-mono bg-gray-100 px-2 py-1 rounded">
              {summary.best_model_architecture}
            </span>
            {' '} with {' '}
            <span className="font-bold text-green-600">
              {(summary.best_accuracy * 100).toFixed(1)}% accuracy
            </span>
          </p>
        </div>
      )}

      <div className="mt-8 bg-white rounded-xl shadow-sm border p-6">
        <h3 className="text-lg font-semibold mb-4">Quick Start</h3>
        <div className="space-y-3 text-sm text-gray-600 font-mono">
          <p>$ mlforge train --config config.yaml</p>
          <p>$ mlforge export --config config.yaml --formats onnx,tflite,coreml</p>
          <p>$ mlforge benchmark --model-dir ./exported_models</p>
        </div>
      </div>
    </div>
  )
}
