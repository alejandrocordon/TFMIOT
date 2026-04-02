import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Tag, Plus, Trash2, Pencil, Check, X } from 'lucide-react'

interface ModelVersion {
  id: number
  project_id: number
  training_run_id: number
  version: string
  tag: string
  description: string
  architecture: string
  dataset: string
  accuracy: number
  loss: number
  created_at: string
}

interface TrainingRun {
  id: number
  project_id: number
  status: string
  architecture: string
  dataset: string
  best_accuracy: number
  best_loss: number
}

interface Project {
  id: number
  name: string
}

const TAG_COLORS: Record<string, string> = {
  production: 'bg-green-100 text-green-700 border-green-300',
  staging: 'bg-yellow-100 text-yellow-700 border-yellow-300',
  best: 'bg-blue-100 text-blue-700 border-blue-300',
  candidate: 'bg-purple-100 text-purple-700 border-purple-300',
  '': 'bg-gray-100 text-gray-500 border-gray-300',
}

function TagBadge({ tag }: { tag: string }) {
  const color = TAG_COLORS[tag] || TAG_COLORS['']
  return (
    <span className={`px-2 py-0.5 rounded border text-xs font-medium ${color}`}>
      {tag || 'untagged'}
    </span>
  )
}

function CreateVersionForm({ onClose }: { onClose: () => void }) {
  const queryClient = useQueryClient()
  const [runId, setRunId] = useState<number>(0)
  const [tag, setTag] = useState('')
  const [description, setDescription] = useState('')

  const { data: runs = [] } = useQuery<TrainingRun[]>({
    queryKey: ['training-runs'],
    queryFn: () => fetch('/api/training/').then(r => r.json()),
  })

  const completedRuns = runs.filter(r => r.status === 'completed')

  const createMutation = useMutation({
    mutationFn: (data: { training_run_id: number; tag: string; description: string }) =>
      fetch('/api/versions/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      }).then(r => {
        if (!r.ok) throw new Error('Failed to create version')
        return r.json()
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['versions'] })
      onClose()
    },
  })

  return (
    <div className="bg-white rounded-xl shadow-sm border p-6 mb-6">
      <h3 className="font-semibold text-lg mb-4">Create Model Version</h3>

      <div className="grid grid-cols-2 gap-4">
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

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Tag</label>
          <select
            value={tag}
            onChange={e => setTag(e.target.value)}
            className="w-full border rounded-lg px-3 py-2"
          >
            <option value="">No tag</option>
            <option value="candidate">Candidate</option>
            <option value="staging">Staging</option>
            <option value="production">Production</option>
            <option value="best">Best</option>
          </select>
        </div>

        <div className="col-span-2">
          <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
          <input
            value={description}
            onChange={e => setDescription(e.target.value)}
            placeholder="Optional description for this version"
            className="w-full border rounded-lg px-3 py-2"
          />
        </div>
      </div>

      <div className="flex gap-3 mt-4">
        <button
          onClick={() => createMutation.mutate({ training_run_id: runId, tag, description })}
          disabled={!runId || createMutation.isPending}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 font-medium"
        >
          {createMutation.isPending ? 'Creating...' : 'Create Version'}
        </button>
        <button onClick={onClose} className="px-4 py-2 text-gray-600 hover:text-gray-800">
          Cancel
        </button>
      </div>

      {createMutation.isError && (
        <p className="text-red-600 text-sm mt-3">Failed to create version. Only completed runs can be versioned.</p>
      )}
    </div>
  )
}

function EditableTag({ version }: { version: ModelVersion }) {
  const queryClient = useQueryClient()
  const [editing, setEditing] = useState(false)
  const [newTag, setNewTag] = useState(version.tag)

  const updateMutation = useMutation({
    mutationFn: (tag: string) =>
      fetch(`/api/versions/${version.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tag }),
      }).then(r => r.json()),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['versions'] })
      setEditing(false)
    },
  })

  if (!editing) {
    return (
      <button onClick={() => setEditing(true)} className="flex items-center gap-1 group">
        <TagBadge tag={version.tag} />
        <Pencil className="w-3 h-3 text-gray-400 opacity-0 group-hover:opacity-100" />
      </button>
    )
  }

  return (
    <div className="flex items-center gap-1">
      <select
        value={newTag}
        onChange={e => setNewTag(e.target.value)}
        className="border rounded px-2 py-0.5 text-xs"
        autoFocus
      >
        <option value="">No tag</option>
        <option value="candidate">Candidate</option>
        <option value="staging">Staging</option>
        <option value="production">Production</option>
        <option value="best">Best</option>
      </select>
      <button
        onClick={() => updateMutation.mutate(newTag)}
        className="text-green-600 hover:text-green-800"
      >
        <Check className="w-4 h-4" />
      </button>
      <button
        onClick={() => { setEditing(false); setNewTag(version.tag) }}
        className="text-gray-400 hover:text-gray-600"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  )
}

export default function Versions() {
  const queryClient = useQueryClient()
  const [showForm, setShowForm] = useState(false)

  const { data: versions = [] } = useQuery<ModelVersion[]>({
    queryKey: ['versions'],
    queryFn: () => fetch('/api/versions/').then(r => r.json()),
  })

  const { data: projects = [] } = useQuery<Project[]>({
    queryKey: ['projects'],
    queryFn: () => fetch('/api/projects/').then(r => r.json()),
  })

  const deleteMutation = useMutation({
    mutationFn: (id: number) =>
      fetch(`/api/versions/${id}`, { method: 'DELETE' }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['versions'] }),
  })

  const projectName = (id: number) => projects.find(p => p.id === id)?.name || `Project #${id}`

  return (
    <div className="p-8">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Model Versions</h2>
        <button
          onClick={() => setShowForm(!showForm)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          <Plus className="w-4 h-4" /> New Version
        </button>
      </div>

      {showForm && <CreateVersionForm onClose={() => setShowForm(false)} />}

      {versions.length === 0 && !showForm && (
        <div className="text-center py-12 text-gray-500">
          <Tag className="w-12 h-12 mx-auto mb-3 text-gray-300" />
          <p className="mb-2">No model versions yet.</p>
          <p className="text-sm">Complete a training run and create a version to track it.</p>
        </div>
      )}

      {versions.length > 0 && (
        <div className="bg-white rounded-xl shadow-sm border overflow-hidden">
          <table className="w-full">
            <thead className="bg-gray-50 border-b">
              <tr>
                <th className="text-left px-6 py-3 text-xs font-medium text-gray-500 uppercase">Version</th>
                <th className="text-left px-6 py-3 text-xs font-medium text-gray-500 uppercase">Project</th>
                <th className="text-left px-6 py-3 text-xs font-medium text-gray-500 uppercase">Architecture</th>
                <th className="text-left px-6 py-3 text-xs font-medium text-gray-500 uppercase">Dataset</th>
                <th className="text-left px-6 py-3 text-xs font-medium text-gray-500 uppercase">Accuracy</th>
                <th className="text-left px-6 py-3 text-xs font-medium text-gray-500 uppercase">Tag</th>
                <th className="text-left px-6 py-3 text-xs font-medium text-gray-500 uppercase">Date</th>
                <th className="px-6 py-3"></th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {versions.map(v => (
                <tr key={v.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 font-mono font-semibold text-blue-600">{v.version}</td>
                  <td className="px-6 py-4 text-sm">{projectName(v.project_id)}</td>
                  <td className="px-6 py-4 text-sm font-mono">{v.architecture}</td>
                  <td className="px-6 py-4 text-sm">{v.dataset || '-'}</td>
                  <td className="px-6 py-4 text-sm font-medium text-green-600">
                    {v.accuracy > 0 ? `${(v.accuracy * 100).toFixed(1)}%` : '-'}
                  </td>
                  <td className="px-6 py-4">
                    <EditableTag version={v} />
                  </td>
                  <td className="px-6 py-4 text-xs text-gray-400">
                    {new Date(v.created_at).toLocaleDateString()}
                  </td>
                  <td className="px-6 py-4">
                    <button
                      onClick={() => deleteMutation.mutate(v.id)}
                      className="text-gray-400 hover:text-red-500"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {versions.length > 0 && (
        <p className="text-xs text-gray-400 mt-4">
          {versions.length} version{versions.length !== 1 ? 's' : ''} total.
          Click a tag to edit it.
        </p>
      )}
    </div>
  )
}
