import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Plus, Trash2 } from 'lucide-react'

interface Project {
  id: number
  name: string
  task: string
  description: string
  config_path: string
  created_at: string
}

export default function Projects() {
  const queryClient = useQueryClient()
  const [showCreate, setShowCreate] = useState(false)
  const [name, setName] = useState('')
  const [task, setTask] = useState('classification')
  const [description, setDescription] = useState('')

  const { data: projects = [] } = useQuery<Project[]>({
    queryKey: ['projects'],
    queryFn: () => fetch('/api/projects/').then(r => r.json()),
  })

  const createMutation = useMutation({
    mutationFn: (data: { name: string; task: string; description: string }) =>
      fetch('/api/projects/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      }).then(r => r.json()),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
      setShowCreate(false)
      setName('')
      setDescription('')
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (id: number) =>
      fetch(`/api/projects/${id}`, { method: 'DELETE' }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['projects'] }),
  })

  return (
    <div className="p-8">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Projects</h2>
        <button
          onClick={() => setShowCreate(!showCreate)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          <Plus className="w-4 h-4" /> New Project
        </button>
      </div>

      {showCreate && (
        <div className="bg-white rounded-xl shadow-sm border p-6 mb-6">
          <h3 className="font-semibold mb-4">Create Project</h3>
          <div className="grid grid-cols-2 gap-4">
            <input
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="Project name"
              className="border rounded-lg px-3 py-2"
            />
            <select
              value={task}
              onChange={e => setTask(e.target.value)}
              className="border rounded-lg px-3 py-2"
            >
              <option value="classification">Classification</option>
              <option value="detection">Object Detection</option>
              <option value="segmentation">Segmentation</option>
            </select>
            <input
              value={description}
              onChange={e => setDescription(e.target.value)}
              placeholder="Description (optional)"
              className="border rounded-lg px-3 py-2 col-span-2"
            />
          </div>
          <button
            onClick={() => createMutation.mutate({ name, task, description })}
            disabled={!name}
            className="mt-4 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
          >
            Create
          </button>
        </div>
      )}

      <div className="grid gap-4">
        {projects.map(project => (
          <div key={project.id} className="bg-white rounded-xl shadow-sm border p-6 flex justify-between items-center">
            <div>
              <h3 className="font-semibold text-lg">{project.name}</h3>
              <div className="flex gap-3 mt-1 text-sm text-gray-500">
                <span className="bg-blue-100 text-blue-700 px-2 py-0.5 rounded">{project.task}</span>
                {project.description && <span>{project.description}</span>}
              </div>
            </div>
            <button
              onClick={() => deleteMutation.mutate(project.id)}
              className="text-gray-400 hover:text-red-500"
            >
              <Trash2 className="w-5 h-5" />
            </button>
          </div>
        ))}
        {projects.length === 0 && (
          <p className="text-gray-500 text-center py-12">No projects yet. Create one to get started!</p>
        )}
      </div>
    </div>
  )
}
