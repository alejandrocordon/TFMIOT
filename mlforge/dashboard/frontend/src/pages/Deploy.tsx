import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Smartphone, Globe, Cpu, Plus, Download, Package, BookOpen, ChevronDown, ChevronUp, Terminal, FolderOpen, Play, CheckCircle2 } from 'lucide-react'

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

interface StepProps {
  number: number
  title: string
  children: React.ReactNode
}

function Step({ number, title, children }: StepProps) {
  return (
    <div className="flex gap-3">
      <div className="flex-shrink-0 w-7 h-7 rounded-full bg-blue-600 text-white flex items-center justify-center text-xs font-bold">
        {number}
      </div>
      <div className="flex-1 pb-4">
        <p className="font-medium text-sm mb-1">{title}</p>
        <div className="text-sm text-gray-600">{children}</div>
      </div>
    </div>
  )
}

function CodeBlock({ children }: { children: string }) {
  return (
    <pre className="bg-gray-900 text-gray-100 rounded-lg px-3 py-2 text-xs font-mono mt-1 overflow-x-auto">
      {children}
    </pre>
  )
}

function NextStepsGuide({ target, app }: { target: string; app: DeployedApp }) {
  const [expanded, setExpanded] = useState(false)

  const guides: Record<string, React.ReactNode> = {
    ios: (
      <div className="space-y-1">
        <Step number={1} title="Download and unzip the project">
          <p>Click the <strong>.zip</strong> button above to download, then unzip it.</p>
        </Step>
        <Step number={2} title="Open in Xcode">
          <p>Double-click the <code>.xcodeproj</code> file, or open Xcode and select the project folder.</p>
          <CodeBlock>{"open deploy_ios/"}</CodeBlock>
        </Step>
        <Step number={3} title="Verify the model is included">
          <p>In Xcode's Project Navigator, check that <code>model.mlpackage</code> appears under <strong>MLForgeApp</strong>. If not, drag it into the project.</p>
        </Step>
        <Step number={4} title="Update class labels (if needed)">
          <p>Open <code>ContentView.swift</code> and verify the <code>labels</code> array matches your model's classes.</p>
        </Step>
        <Step number={5} title="Select a target device">
          <p>Connect your iPhone/iPad or select a Simulator. Set the target to <strong>iOS 17+</strong>.</p>
        </Step>
        <Step number={6} title="Build and run">
          <p>Press <strong>Cmd + R</strong> or click the Play button. The app will launch with a photo picker to classify images using your model.</p>
        </Step>
        <div className="mt-3 p-3 bg-purple-50 border border-purple-200 rounded-lg">
          <p className="text-xs text-purple-700">
            <strong>Requirements:</strong> Xcode 15+, iOS 17+, Swift 5.9+. The app uses Vision framework + CoreML for on-device inference.
          </p>
        </div>
      </div>
    ),
    android: (
      <div className="space-y-1">
        <Step number={1} title="Download and unzip the project">
          <p>Click the <strong>.zip</strong> button above to download, then unzip it.</p>
        </Step>
        <Step number={2} title="Open in Android Studio">
          <p>Launch Android Studio, select <strong>Open</strong>, and navigate to the unzipped folder.</p>
          <CodeBlock>{"# Or from terminal:\nstudio deploy_android/"}</CodeBlock>
        </Step>
        <Step number={3} title="Wait for Gradle sync">
          <p>Android Studio will automatically download dependencies. Wait for the sync to complete (bottom status bar).</p>
        </Step>
        <Step number={4} title="Verify the model is in assets">
          <p>Check <code>app/src/main/assets/model.tflite</code> exists. If not, copy your exported TFLite model there.</p>
        </Step>
        <Step number={5} title="Update labels in MainActivity">
          <p>Open <code>MainActivity.kt</code> and verify the <code>LABELS</code> list matches your model's classes.</p>
        </Step>
        <Step number={6} title="Run on device or emulator">
          <p>Connect an Android device (USB debugging enabled) or start an emulator. Click the <strong>Run</strong> button (green play). The app opens the camera and classifies objects in real-time.</p>
        </Step>
        <div className="mt-3 p-3 bg-green-50 border border-green-200 rounded-lg">
          <p className="text-xs text-green-700">
            <strong>Requirements:</strong> Android Studio, Min SDK 24 (Android 7.0), TensorFlow Lite 2.16+, CameraX 1.3+, Kotlin 1.9+.
          </p>
        </div>
      </div>
    ),
    web: (
      <div className="space-y-1">
        <Step number={1} title="Download and unzip the project">
          <p>Click the <strong>.zip</strong> button above to download, then unzip it.</p>
        </Step>
        <Step number={2} title="Verify the model file">
          <p>Check that <code>model.onnx</code> is in the project folder. If not, copy your exported ONNX model there.</p>
        </Step>
        <Step number={3} title="Update class labels">
          <p>Open <code>app.js</code> and verify the <code>CLASS_LABELS</code> array matches your model's classes.</p>
        </Step>
        <Step number={4} title="Start a local server">
          <CodeBlock>{"cd deploy_web/\npython -m http.server 8080"}</CodeBlock>
        </Step>
        <Step number={5} title="Open in browser">
          <p>Navigate to <code>http://localhost:8080</code>. Drag & drop an image or use the file picker to classify.</p>
        </Step>
        <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-xs text-blue-700">
            <strong>How it works:</strong> The model runs entirely in the browser via ONNX Runtime Web (WebAssembly). No data leaves the user's device. Works in Chrome, Firefox, Safari, and Edge.
          </p>
        </div>
      </div>
    ),
    edge: (
      <div className="space-y-1">
        <Step number={1} title="Download and unzip the project">
          <p>Click the <strong>.zip</strong> button above to download, then transfer to your Raspberry Pi.</p>
          <CodeBlock>{"scp mlforge_edge_run1.zip pi@raspberrypi:~/"}</CodeBlock>
        </Step>
        <Step number={2} title="Install dependencies">
          <CodeBlock>{"cd deploy_edge/\npip install -r requirements.txt\n\n# Optional: for Coral TPU acceleration\npip install pycoral"}</CodeBlock>
        </Step>
        <Step number={3} title="Verify the model file">
          <p>Check that <code>model.tflite</code> is in the folder. For Edge TPU, use the <code>_edgetpu.tflite</code> variant.</p>
        </Step>
        <Step number={4} title="Run inference">
          <CodeBlock>{"# Live camera feed\npython infer.py --camera\n\n# Single image\npython infer.py --image test.jpg\n\n# CPU only (no Coral TPU)\npython infer.py --camera --no-tpu"}</CodeBlock>
        </Step>
        <div className="mt-3 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
          <p className="text-xs text-yellow-700">
            <strong>Hardware:</strong> Raspberry Pi 3B+/4/5 with USB camera or Pi Camera Module. Optional: Google Coral USB Accelerator for 5-10x faster inference.
          </p>
        </div>
      </div>
    ),
  }

  return (
    <div className="mt-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-sm text-blue-600 hover:text-blue-800 font-medium"
      >
        <BookOpen className="w-4 h-4" />
        {expanded ? 'Hide setup guide' : 'Next steps: How to open & run'}
        {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
      </button>

      {expanded && (
        <div className="mt-3 bg-white rounded-xl border p-5">
          <h4 className="font-semibold text-sm mb-4 flex items-center gap-2">
            <Terminal className="w-4 h-4" />
            Setup Guide: {targetInfo[target]?.label || target}
          </h4>
          {guides[target] || <p className="text-sm text-gray-500">No guide available for this platform.</p>}
        </div>
      )}
    </div>
  )
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
        <div className="space-y-4">
          {deploys.map(app => {
            const info = targetInfo[app.target]
            const Icon = targetIcons[app.target] || Package
            return (
              <div key={app.id} className="bg-white rounded-xl border p-5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className={`p-2.5 rounded-lg ${info?.bgColor || 'bg-gray-50'}`}>
                      <Icon className={`w-5 h-5 ${info?.color || 'text-gray-500'}`} />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className={`font-semibold ${info?.color || ''}`}>{info?.label || app.target}</span>
                        <span className="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded">Run #{app.training_run_id}</span>
                      </div>
                      <p className="text-xs text-gray-500 font-mono mt-0.5">{app.output_path}</p>
                      {app.labels && (
                        <p className="text-xs text-gray-400 mt-0.5 truncate max-w-md">{app.labels}</p>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-gray-400">
                      {new Date(app.created_at).toLocaleDateString()}
                    </span>
                    <a
                      href={`/api/deploy/${app.id}/download`}
                      className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium"
                    >
                      <Download className="w-4 h-4" />
                      .zip
                    </a>
                  </div>
                </div>

                {/* Next steps guide */}
                <NextStepsGuide target={app.target} app={app} />
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
