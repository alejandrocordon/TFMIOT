import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Plus, Trash2, ChevronUp, ChevronDown, Lightbulb, Code, Layers, LayoutTemplate, Save, Info, Cpu, ArrowUp, ArrowDown } from 'lucide-react'

// ─── Types ──────────────────────────────────────────────────────────────

interface LayerParam {
  type: string
  default: number | string
  desc: string
  options?: (number | string)[]
}

interface LayerCatalogEntry {
  category: string
  params: Record<string, LayerParam>
  explanation: string
  tip: string
}

interface LayerDef {
  id: string
  type: string
  params: Record<string, any>
}

interface Template {
  id: string
  name: string
  description: string
  difficulty: string
  params: Record<string, LayerParam & { options?: any[] }>
}

interface CustomModel {
  id: number
  name: string
  description: string
  num_params: number
  created_at: string
}

interface GenerateResult {
  code: string
  num_params: number
  num_params_human: string
  num_layers: number
  explanations: { type: string; explanation: string; tip: string }[]
}

let layerIdCounter = 0
function newLayerId() { return `layer_${++layerIdCounter}_${Date.now()}` }

// ─── Category colors ────────────────────────────────────────────────────

const catColors: Record<string, string> = {
  Convolution: 'bg-blue-100 text-blue-700',
  Normalization: 'bg-green-100 text-green-700',
  Activation: 'bg-orange-100 text-orange-700',
  Pooling: 'bg-purple-100 text-purple-700',
  Linear: 'bg-indigo-100 text-indigo-700',
  Regularization: 'bg-red-100 text-red-700',
}

// ─── Sub-components ─────────────────────────────────────────────────────

function LayerCard({
  layer, catalog, index, total, onUpdate, onRemove, onMove, selected, onSelect,
}: {
  layer: LayerDef
  catalog: Record<string, LayerCatalogEntry>
  index: number
  total: number
  onUpdate: (params: Record<string, any>) => void
  onRemove: () => void
  onMove: (dir: -1 | 1) => void
  selected: boolean
  onSelect: () => void
}) {
  const info = catalog[layer.type]
  const cat = info?.category || 'Unknown'
  const color = catColors[cat] || 'bg-gray-100 text-gray-700'

  return (
    <div
      className={`border rounded-lg p-3 cursor-pointer transition-all ${selected ? 'ring-2 ring-blue-500 border-blue-300' : 'hover:border-gray-400'}`}
      onClick={onSelect}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400 w-5">{index + 1}</span>
          <span className={`px-2 py-0.5 rounded text-xs font-medium ${color}`}>{cat}</span>
          <span className="font-mono text-sm font-semibold">{layer.type}</span>
        </div>
        <div className="flex items-center gap-1">
          <button onClick={e => { e.stopPropagation(); onMove(-1) }} disabled={index === 0} className="p-1 text-gray-400 hover:text-gray-600 disabled:opacity-30"><ArrowUp className="w-3 h-3" /></button>
          <button onClick={e => { e.stopPropagation(); onMove(1) }} disabled={index === total - 1} className="p-1 text-gray-400 hover:text-gray-600 disabled:opacity-30"><ArrowDown className="w-3 h-3" /></button>
          <button onClick={e => { e.stopPropagation(); onRemove() }} className="p-1 text-gray-400 hover:text-red-500"><Trash2 className="w-3 h-3" /></button>
        </div>
      </div>

      {/* Inline params */}
      {info && Object.keys(info.params).length > 0 && (
        <div className="flex flex-wrap gap-2 mt-2" onClick={e => e.stopPropagation()}>
          {Object.entries(info.params).map(([key, pinfo]) => (
            <label key={key} className="flex items-center gap-1 text-xs">
              <span className="text-gray-500">{key}:</span>
              <input
                type={pinfo.type === 'float' ? 'number' : 'number'}
                step={pinfo.type === 'float' ? '0.01' : '1'}
                value={layer.params[key] ?? pinfo.default}
                onChange={e => {
                  const v = pinfo.type === 'float' ? parseFloat(e.target.value) : parseInt(e.target.value)
                  onUpdate({ ...layer.params, [key]: isNaN(v) ? pinfo.default : v })
                }}
                className="w-16 border rounded px-1 py-0.5 text-xs font-mono"
              />
            </label>
          ))}
        </div>
      )}
    </div>
  )
}

function ExplanationPanel({ layer, catalog }: { layer: LayerDef | null; catalog: Record<string, LayerCatalogEntry> }) {
  if (!layer) {
    return (
      <div className="text-center py-8 text-gray-400">
        <Lightbulb className="w-8 h-8 mx-auto mb-2" />
        <p className="text-sm">Select a layer to see its explanation</p>
      </div>
    )
  }

  const info = catalog[layer.type]
  if (!info) return null

  return (
    <div className="space-y-3">
      <h4 className="font-semibold text-sm flex items-center gap-2">
        <Info className="w-4 h-4 text-blue-500" />
        {layer.type}
      </h4>
      <p className="text-sm text-gray-600 leading-relaxed">{info.explanation}</p>
      {info.tip && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
          <p className="text-xs text-yellow-800"><strong>Tip:</strong> {info.tip}</p>
        </div>
      )}
      {Object.keys(info.params).length > 0 && (
        <div>
          <p className="text-xs font-medium text-gray-500 mb-1">Parameters:</p>
          {Object.entries(info.params).map(([k, v]) => (
            <p key={k} className="text-xs text-gray-500 ml-2">
              <code className="text-gray-700">{k}</code> — {v.desc}
            </p>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Main Component ─────────────────────────────────────────────────────

export default function ModelBuilder() {
  const queryClient = useQueryClient()
  const [tab, setTab] = useState<'visual' | 'code' | 'templates'>('templates')
  const [layers, setLayers] = useState<LayerDef[]>([])
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null)
  const [modelName, setModelName] = useState('')
  const [modelDesc, setModelDesc] = useState('')
  const [numClasses, setNumClasses] = useState(10)
  const [codeOverride, setCodeOverride] = useState<string | null>(null)

  // Fetch catalog
  const { data: catalog = {} } = useQuery<Record<string, LayerCatalogEntry>>({
    queryKey: ['layer-catalog'],
    queryFn: () => fetch('/api/model-builder/layers').then(r => r.json()),
  })

  // Fetch templates
  const { data: templates = [] } = useQuery<Template[]>({
    queryKey: ['templates'],
    queryFn: () => fetch('/api/model-builder/templates').then(r => r.json()),
  })

  // Fetch saved models
  const { data: savedModels = [] } = useQuery<CustomModel[]>({
    queryKey: ['custom-models'],
    queryFn: () => fetch('/api/model-builder/models').then(r => r.json()),
  })

  // Generate code when layers change
  const { data: generated } = useQuery<GenerateResult>({
    queryKey: ['generate-code', layers, numClasses],
    queryFn: () => fetch('/api/model-builder/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ layers: layers.map(l => ({ type: l.type, params: l.params })), model_name: 'CustomModel', num_classes: numClasses }),
    }).then(r => r.json()),
    enabled: layers.length > 0,
  })

  // Reset code override when layers change
  useEffect(() => { setCodeOverride(null) }, [layers])

  const displayCode = codeOverride ?? generated?.code ?? '# Add layers to see generated code'

  // ─── Actions ────────────────────────────────────────────────────

  const addLayer = (type: string) => {
    const info = catalog[type]
    const params: Record<string, any> = {}
    if (info) {
      for (const [k, v] of Object.entries(info.params)) {
        params[k] = v.default
      }
    }
    setLayers([...layers, { id: newLayerId(), type, params }])
  }

  const removeLayer = (idx: number) => {
    setLayers(layers.filter((_, i) => i !== idx))
    if (selectedIdx === idx) setSelectedIdx(null)
  }

  const moveLayer = (idx: number, dir: -1 | 1) => {
    const newIdx = idx + dir
    if (newIdx < 0 || newIdx >= layers.length) return
    const copy = [...layers]
    ;[copy[idx], copy[newIdx]] = [copy[newIdx], copy[idx]]
    setLayers(copy)
    setSelectedIdx(newIdx)
  }

  const updateLayerParams = (idx: number, params: Record<string, any>) => {
    const copy = [...layers]
    copy[idx] = { ...copy[idx], params }
    setLayers(copy)
  }

  // Apply template
  const applyTemplateMut = useMutation({
    mutationFn: (data: { template_id: string; params: Record<string, any> }) =>
      fetch('/api/model-builder/templates/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      }).then(r => r.json()),
    onSuccess: (data) => {
      setLayers(data.layers.map((l: any) => ({ id: newLayerId(), ...l })))
      setTab('visual')
    },
  })

  // Save model
  const saveMut = useMutation({
    mutationFn: () =>
      fetch('/api/model-builder/models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: modelName,
          description: modelDesc,
          layers: layers.map(l => ({ type: l.type, params: l.params })),
          num_classes: numClasses,
        }),
      }).then(r => {
        if (!r.ok) return r.json().then(e => { throw new Error(e.detail) })
        return r.json()
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['custom-models'] })
      setModelName('')
      setModelDesc('')
    },
  })

  const deleteMut = useMutation({
    mutationFn: (id: number) => fetch(`/api/model-builder/models/${id}`, { method: 'DELETE' }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['custom-models'] }),
  })

  // Group layers by category
  const categories = Object.entries(catalog).reduce((acc, [name, info]) => {
    const cat = info.category
    if (!acc[cat]) acc[cat] = []
    acc[cat].push(name)
    return acc
  }, {} as Record<string, string[]>)

  return (
    <div className="p-6 h-full flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Model Builder</h2>
          <p className="text-gray-500 text-sm">Design your own neural network architecture</p>
        </div>
        {generated && layers.length > 0 && (
          <div className="text-right">
            <p className="text-sm font-mono text-gray-600">{generated.num_params_human} params</p>
            <p className="text-xs text-gray-400">{generated.num_layers} layers</p>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-4 bg-gray-100 rounded-lg p-1 w-fit">
        {[
          { id: 'templates' as const, icon: LayoutTemplate, label: 'Templates' },
          { id: 'visual' as const, icon: Layers, label: 'Visual Builder' },
          { id: 'code' as const, icon: Code, label: 'Code' },
        ].map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              tab === t.id ? 'bg-white shadow text-blue-700' : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <t.icon className="w-4 h-4" />
            {t.label}
          </button>
        ))}
      </div>

      {/* ─── Templates Tab ─────────────────────────────────────── */}
      {tab === 'templates' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {templates.map(t => (
            <div key={t.id} className="bg-white rounded-xl border p-5 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="font-semibold">{t.name}</h3>
                  <p className="text-sm text-gray-500 mt-1">{t.description}</p>
                </div>
                <span className={`text-xs px-2 py-0.5 rounded ${
                  t.difficulty === 'Beginner' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'
                }`}>{t.difficulty}</span>
              </div>

              {/* Template params */}
              <div className="flex flex-wrap gap-3 mt-3">
                {Object.entries(t.params).map(([k, p]) => (
                  <label key={k} className="text-xs">
                    <span className="text-gray-500 block mb-0.5">{k}</span>
                    <select
                      className="border rounded px-2 py-1 text-xs"
                      defaultValue={p.default as string}
                      id={`tpl_${t.id}_${k}`}
                    >
                      {p.options?.map((o: any) => <option key={o} value={o}>{o}</option>)}
                    </select>
                  </label>
                ))}
              </div>

              <button
                onClick={() => {
                  const params: Record<string, any> = {}
                  for (const k of Object.keys(t.params)) {
                    const el = document.getElementById(`tpl_${t.id}_${k}`) as HTMLSelectElement
                    const v = el?.value
                    params[k] = t.params[k].type === 'float' ? parseFloat(v) : parseInt(v)
                  }
                  applyTemplateMut.mutate({ template_id: t.id, params })
                }}
                className="mt-3 px-4 py-1.5 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700"
              >
                Use this template
              </button>
            </div>
          ))}
        </div>
      )}

      {/* ─── Visual Builder Tab ────────────────────────────────── */}
      {tab === 'visual' && (
        <div className="flex gap-4 flex-1 min-h-0">
          {/* Left: Add layers */}
          <div className="w-48 flex-shrink-0 overflow-y-auto">
            <p className="text-xs font-medium text-gray-500 mb-2">Add Layer</p>
            {Object.entries(categories).map(([cat, names]) => (
              <div key={cat} className="mb-3">
                <p className={`text-xs font-medium mb-1 ${catColors[cat]?.split(' ')[1] || 'text-gray-600'}`}>{cat}</p>
                {names.map(name => (
                  <button
                    key={name}
                    onClick={() => addLayer(name)}
                    className="w-full text-left text-xs px-2 py-1.5 rounded hover:bg-gray-100 font-mono flex items-center gap-1"
                  >
                    <Plus className="w-3 h-3 text-gray-400" />
                    {name}
                  </button>
                ))}
              </div>
            ))}
          </div>

          {/* Center: Layer stack */}
          <div className="flex-1 overflow-y-auto space-y-2">
            {layers.length === 0 ? (
              <div className="text-center py-12 text-gray-400">
                <Layers className="w-10 h-10 mx-auto mb-2" />
                <p className="text-sm">Add layers from the left panel or start with a template</p>
              </div>
            ) : (
              layers.map((layer, idx) => (
                <LayerCard
                  key={layer.id}
                  layer={layer}
                  catalog={catalog}
                  index={idx}
                  total={layers.length}
                  onUpdate={p => updateLayerParams(idx, p)}
                  onRemove={() => removeLayer(idx)}
                  onMove={dir => moveLayer(idx, dir)}
                  selected={selectedIdx === idx}
                  onSelect={() => setSelectedIdx(idx)}
                />
              ))
            )}
          </div>

          {/* Right: Explanation */}
          <div className="w-64 flex-shrink-0 bg-white rounded-xl border p-4 overflow-y-auto">
            <ExplanationPanel
              layer={selectedIdx !== null ? layers[selectedIdx] : null}
              catalog={catalog}
            />
          </div>
        </div>
      )}

      {/* ─── Code Tab ──────────────────────────────────────────── */}
      {tab === 'code' && (
        <div className="flex-1 flex flex-col min-h-0">
          <textarea
            value={displayCode}
            onChange={e => setCodeOverride(e.target.value)}
            className="flex-1 bg-gray-900 text-green-400 font-mono text-sm p-4 rounded-xl border-0 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
            spellCheck={false}
          />
          {generated && (
            <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
              <span>{generated.num_params_human} parameters</span>
              <span>{generated.num_layers} layers</span>
              {codeOverride && <span className="text-yellow-600">Code manually edited</span>}
            </div>
          )}
        </div>
      )}

      {/* ─── Save Bar ──────────────────────────────────────────── */}
      {layers.length > 0 && (
        <div className="mt-4 bg-white rounded-xl border p-4">
          <div className="flex gap-3 items-end">
            <div className="flex-1">
              <label className="block text-xs font-medium text-gray-500 mb-1">Model Name</label>
              <input
                value={modelName}
                onChange={e => setModelName(e.target.value)}
                placeholder="my_custom_cnn"
                className="w-full border rounded-lg px-3 py-2 text-sm font-mono"
              />
            </div>
            <div className="flex-1">
              <label className="block text-xs font-medium text-gray-500 mb-1">Description</label>
              <input
                value={modelDesc}
                onChange={e => setModelDesc(e.target.value)}
                placeholder="3-block CNN for flowers"
                className="w-full border rounded-lg px-3 py-2 text-sm"
              />
            </div>
            <div className="w-28">
              <label className="block text-xs font-medium text-gray-500 mb-1">Classes</label>
              <input
                type="number"
                value={numClasses}
                onChange={e => setNumClasses(parseInt(e.target.value) || 10)}
                className="w-full border rounded-lg px-3 py-2 text-sm"
              />
            </div>
            <button
              onClick={() => saveMut.mutate()}
              disabled={!modelName || saveMut.isPending}
              className="px-5 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 font-medium text-sm flex items-center gap-2"
            >
              <Save className="w-4 h-4" />
              {saveMut.isPending ? 'Saving...' : 'Save & Register'}
            </button>
          </div>
          {saveMut.isError && (
            <p className="text-red-600 text-xs mt-2">{(saveMut.error as Error)?.message}</p>
          )}
          {saveMut.isSuccess && (
            <p className="text-green-600 text-xs mt-2">Model saved and registered! It's now available in Training.</p>
          )}
        </div>
      )}

      {/* ─── Saved Models ──────────────────────────────────────── */}
      {savedModels.length > 0 && (
        <div className="mt-4">
          <h3 className="text-sm font-semibold text-gray-600 mb-2">Your Custom Models</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {savedModels.map(m => (
              <div key={m.id} className="bg-white rounded-lg border p-3 flex items-center justify-between">
                <div>
                  <p className="font-mono text-sm font-semibold">{m.name}</p>
                  <p className="text-xs text-gray-500">
                    {m.num_params >= 1e6 ? `${(m.num_params / 1e6).toFixed(2)}M` : `${(m.num_params / 1e3).toFixed(1)}K`} params
                    {m.description && ` — ${m.description}`}
                  </p>
                </div>
                <button onClick={() => deleteMut.mutate(m.id)} className="text-gray-400 hover:text-red-500">
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
