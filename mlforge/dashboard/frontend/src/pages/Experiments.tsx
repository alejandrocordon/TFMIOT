import { useQuery } from '@tanstack/react-query'

interface TrainingRun {
  id: number
  architecture: string
  status: string
  epochs: number
  current_epoch: number
  best_accuracy: number
  best_loss: number
  created_at: string
}

export default function Experiments() {
  const { data: runs = [] } = useQuery<TrainingRun[]>({
    queryKey: ['training-runs'],
    queryFn: () => fetch('/api/training/').then(r => r.json()),
  })

  const completedRuns = runs.filter(r => r.status === 'completed')

  return (
    <div className="p-8">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Experiments</h2>
      <p className="text-gray-500 mb-6">Compare training runs side by side</p>

      {completedRuns.length === 0 ? (
        <p className="text-gray-400 text-center py-12">No completed experiments yet.</p>
      ) : (
        <div className="bg-white rounded-xl shadow-sm border overflow-hidden">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="text-left px-6 py-3 text-xs font-medium text-gray-500 uppercase">ID</th>
                <th className="text-left px-6 py-3 text-xs font-medium text-gray-500 uppercase">Architecture</th>
                <th className="text-right px-6 py-3 text-xs font-medium text-gray-500 uppercase">Epochs</th>
                <th className="text-right px-6 py-3 text-xs font-medium text-gray-500 uppercase">Best Accuracy</th>
                <th className="text-right px-6 py-3 text-xs font-medium text-gray-500 uppercase">Best Loss</th>
                <th className="text-right px-6 py-3 text-xs font-medium text-gray-500 uppercase">Date</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {completedRuns.map(run => (
                <tr key={run.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 text-sm text-gray-500">#{run.id}</td>
                  <td className="px-6 py-4 font-mono text-sm">{run.architecture}</td>
                  <td className="px-6 py-4 text-sm text-right">{run.current_epoch}</td>
                  <td className="px-6 py-4 text-sm text-right font-medium text-green-600">
                    {(run.best_accuracy * 100).toFixed(1)}%
                  </td>
                  <td className="px-6 py-4 text-sm text-right">{run.best_loss.toFixed(4)}</td>
                  <td className="px-6 py-4 text-sm text-right text-gray-400">
                    {new Date(run.created_at).toLocaleDateString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
