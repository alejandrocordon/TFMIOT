import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Download, Search, Trash2, Smartphone, Cpu, Zap, CheckCircle2, ExternalLink } from 'lucide-react'

interface HFModelInfo {
  timm_name: string
  display_name: string
  category: string
  params_m: number
  input_size: number
  imagenet_acc: number
  description: string
}

interface DownloadedModel {
  id: number
  timm_name: string
  display_name: string
  registry_name: string
  num_params: number
  default_input_size: number
  downloaded_at: string
}

const catConfig: Record<string, { color: string; bg: string; icon: typeof Smartphone; label: string }> = {
  Lightweight: { color: 'text-green-700', bg: 'bg-green-50 border-green-200', icon: Smartphone, label: 'Mobile / Edge' },
  Balanced: { color: 'text-blue-700', bg: 'bg-blue-50 border-blue-200', icon: Zap, label: 'Balanced' },
  'High Accuracy': { color: 'text-purple-700', bg: 'bg-purple-50 border-purple-200', icon: Cpu, label: 'High Accuracy' },
  Other: { color: 'text-gray-700', bg: 'bg-gray-50 border-gray-200', icon: Cpu, label: 'Other' },
}

function ModelCard({ model, isDownloaded, onDownload, downloading }: {
  model: HFModelInfo
  isDownloaded: boolean
  onDownload: () => void
  downloading: boolean
}) {
  const cat = catConfig[model.category] || catConfig.Other
  const Icon = cat.icon

  return (
    <div className={`rounded-xl border p-5 transition-all ${isDownloaded ? 'ring-2 ring-green-300' : 'hover:shadow-md'}`}>
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <Icon className={`w-5 h-5 ${cat.color}`} />
          <h3 className="font-semibold text-sm text-gray-900">{model.display_name}</h3>
        </div>
        <span className={`text-xs px-2 py-0.5 rounded border ${cat.bg} ${cat.color}`}>
          {cat.label}
        </span>
      </div>

      <p className="text-xs text-gray-500 mb-3">{model.description}</p>

      <div className="flex items-center gap-4 text-xs text-gray-500 mb-3">
        {model.params_m > 0 && (
          <span className="font-mono">{model.params_m}M params</span>
        )}
        <span>{model.input_size}x{model.input_size}</span>
        {model.imagenet_acc > 0 && (
          <span className="text-green-600 font-medium">
            {model.imagenet_acc}% ImageNet
          </span>
        )}
      </div>

      <div className="flex items-center justify-between">
        <code className="text-xs text-gray-400 font-mono">{model.timm_name}</code>
        {isDownloaded ? (
          <span className="flex items-center gap-1 text-xs text-green-600 font-medium">
            <CheckCircle2 className="w-3.5 h-3.5" />
            Ready
          </span>
        ) : (
          <button
            onClick={onDownload}
            disabled={downloading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 text-white rounded-lg text-xs font-medium hover:bg-blue-700 disabled:opacity-50"
          >
            <Download className="w-3.5 h-3.5" />
            {downloading ? 'Downloading...' : 'Download'}
          </button>
        )}
      </div>
    </div>
  )
}

export default function HFModels() {
  const queryClient = useQueryClient()
  const [searchQuery, setSearchQuery] = useState('')
  const [activeSearch, setActiveSearch] = useState('')

  const { data: popular = [] } = useQuery<HFModelInfo[]>({
    queryKey: ['hf-popular'],
    queryFn: () => fetch('/api/hf-models/popular').then(r => r.json()),
  })

  const { data: searchResults = [], isFetching: searching } = useQuery<HFModelInfo[]>({
    queryKey: ['hf-search', activeSearch],
    queryFn: () => fetch(`/api/hf-models/search?q=${encodeURIComponent(activeSearch)}`).then(r => r.json()),
    enabled: activeSearch.length >= 2,
  })

  const { data: downloaded = [] } = useQuery<DownloadedModel[]>({
    queryKey: ['hf-downloaded'],
    queryFn: () => fetch('/api/hf-models/downloaded').then(r => r.json()),
  })

  const downloadedNames = new Set(downloaded.map(m => m.timm_name))

  const downloadMut = useMutation({
    mutationFn: (timm_name: string) =>
      fetch('/api/hf-models/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ timm_name, num_classes: 10 }),
      }).then(r => {
        if (!r.ok) return r.json().then(e => { throw new Error(e.detail) })
        return r.json()
      }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['hf-downloaded'] }),
  })

  const deleteMut = useMutation({
    mutationFn: (id: number) => fetch(`/api/hf-models/downloaded/${id}`, { method: 'DELETE' }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['hf-downloaded'] }),
  })

  const handleSearch = () => {
    if (searchQuery.length >= 2) setActiveSearch(searchQuery)
  }

  const displayModels = activeSearch ? searchResults : popular

  // Group by category
  const categories = ['Lightweight', 'Balanced', 'High Accuracy', 'Other']
  const grouped: Record<string, HFModelInfo[]> = {}
  for (const m of displayModels) {
    const cat = m.category || 'Other'
    if (!grouped[cat]) grouped[cat] = []
    grouped[cat].push(m)
  }

  return (
    <div className="p-8">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900">HuggingFace Model Hub</h2>
        <p className="text-gray-500 text-sm mt-1">
          Browse and download 900+ pretrained vision models. Train on your data, export to any platform.
        </p>
      </div>

      {/* Search */}
      <div className="flex gap-2 mb-6">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSearch()}
            placeholder="Search models (e.g. vit, efficientnet, resnet, convnext...)"
            className="w-full border rounded-lg pl-9 pr-3 py-2 text-sm"
          />
        </div>
        <button
          onClick={handleSearch}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700"
        >
          Search
        </button>
        {activeSearch && (
          <button
            onClick={() => { setActiveSearch(''); setSearchQuery('') }}
            className="px-3 py-2 text-sm text-gray-600 hover:text-gray-800"
          >
            Clear
          </button>
        )}
      </div>

      {/* Downloaded models */}
      {downloaded.length > 0 && (
        <div className="mb-8">
          <h3 className="text-sm font-semibold text-gray-600 mb-3">
            Your Downloaded Models ({downloaded.length})
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {downloaded.map(m => (
              <div key={m.id} className="bg-white rounded-lg border p-3 flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-green-500" />
                    <p className="font-semibold text-sm text-gray-900">{m.display_name}</p>
                  </div>
                  <p className="text-xs text-gray-500 mt-0.5">
                    <code className="text-gray-400">{m.registry_name}</code>
                    {' '}&middot;{' '}
                    {m.num_params >= 1e6 ? `${(m.num_params / 1e6).toFixed(1)}M` : `${(m.num_params / 1e3).toFixed(0)}K`} params
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <a
                    href="/training"
                    className="text-xs text-blue-600 hover:underline"
                  >
                    Train
                  </a>
                  <button
                    onClick={() => deleteMut.mutate(m.id)}
                    className="text-gray-400 hover:text-red-500"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Model grid by category */}
      {activeSearch && searchResults.length === 0 && !searching && (
        <div className="text-center py-12 text-gray-500">
          <p>No models found for "{activeSearch}"</p>
          <p className="text-sm mt-1">Try a different search term</p>
        </div>
      )}

      {searching && (
        <div className="text-center py-12 text-gray-500">
          <p className="text-sm">Searching timm models...</p>
        </div>
      )}

      {categories.map(cat => {
        const models = grouped[cat]
        if (!models || models.length === 0) return null
        const config = catConfig[cat] || catConfig.Other
        return (
          <div key={cat} className="mb-8">
            <div className="flex items-center gap-2 mb-3">
              <config.icon className={`w-5 h-5 ${config.color}`} />
              <h3 className={`text-sm font-semibold ${config.color}`}>{cat}</h3>
              <span className="text-xs text-gray-400">({models.length} models)</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {models.map(m => (
                <ModelCard
                  key={m.timm_name}
                  model={m}
                  isDownloaded={downloadedNames.has(m.timm_name)}
                  onDownload={() => downloadMut.mutate(m.timm_name)}
                  downloading={downloadMut.isPending && downloadMut.variables === m.timm_name}
                />
              ))}
            </div>
          </div>
        )
      })}

      {/* Info footer */}
      <div className="mt-8 p-4 bg-gray-50 rounded-xl border text-xs text-gray-500">
        <p className="font-medium text-gray-600 mb-1">How it works</p>
        <ol className="list-decimal ml-4 space-y-1">
          <li><strong>Download</strong> a pretrained model from the catalog above</li>
          <li>Go to <strong>Training</strong> and create a new run — the model appears in the Architecture dropdown as <code>hf_modelname</code></li>
          <li>Fine-tune on your dataset (the pretrained weights give you a massive head start)</li>
          <li>Go to <strong>Export</strong> to convert to CoreML (iOS), TFLite (Android), ONNX (Server), or TF.js (Web)</li>
          <li>Go to <strong>Deploy Apps</strong> to generate a ready-to-build project for your platform</li>
        </ol>
        <p className="mt-2">
          Models are from the <a href="https://huggingface.co/docs/timm" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline inline-flex items-center gap-0.5">timm library <ExternalLink className="w-3 h-3" /></a> — 900+ pretrained vision models.
        </p>
      </div>
    </div>
  )
}
