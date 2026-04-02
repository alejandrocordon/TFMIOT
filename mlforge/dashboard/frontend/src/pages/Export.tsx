import { useQuery } from '@tanstack/react-query'
import { Download } from 'lucide-react'

interface ExportedModel {
  id: number
  format: string
  file_path: string
  file_size_mb: number
  quantization: string
  created_at: string
}

const formatColors: Record<string, string> = {
  onnx: 'bg-blue-100 text-blue-700',
  tflite: 'bg-green-100 text-green-700',
  edgetpu: 'bg-yellow-100 text-yellow-700',
  coreml: 'bg-purple-100 text-purple-700',
  tfjs: 'bg-orange-100 text-orange-700',
}

export default function Export() {
  const { data: exports = [] } = useQuery<ExportedModel[]>({
    queryKey: ['exports'],
    queryFn: () => fetch('/api/exports/').then(r => r.json()),
  })

  return (
    <div className="p-8">
      <h2 className="text-2xl font-bold text-gray-900 mb-2">Exported Models</h2>
      <p className="text-gray-500 mb-6">Models converted for deployment on different platforms</p>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        {[
          { format: 'ONNX', desc: 'Universal CPU/GPU inference', target: 'Server, Desktop' },
          { format: 'TFLite', desc: 'Mobile and embedded', target: 'Android, RPi' },
          { format: 'Edge TPU', desc: 'Google Coral accelerator', target: 'Coral USB/PCIe' },
          { format: 'CoreML', desc: 'Apple Neural Engine', target: 'iPhone, iPad, Mac' },
          { format: 'TF.js', desc: 'Browser inference', target: 'Web apps' },
        ].map(item => (
          <div key={item.format} className="bg-white rounded-xl border p-4">
            <h4 className="font-semibold">{item.format}</h4>
            <p className="text-sm text-gray-500">{item.desc}</p>
            <p className="text-xs text-gray-400 mt-1">{item.target}</p>
          </div>
        ))}
      </div>

      {exports.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <p className="mb-2">No exported models yet.</p>
          <p className="text-sm font-mono">mlforge export --config config.yaml --formats onnx,tflite,coreml</p>
        </div>
      ) : (
        <div className="space-y-3">
          {exports.map(exp => (
            <div key={exp.id} className="bg-white rounded-xl border p-4 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <span className={`px-3 py-1 rounded-lg text-sm font-medium ${formatColors[exp.format] || 'bg-gray-100'}`}>
                  {exp.format.toUpperCase()}
                </span>
                <div>
                  <p className="text-sm font-mono text-gray-700">{exp.file_path}</p>
                  <p className="text-xs text-gray-400">
                    {exp.file_size_mb > 0 ? `${exp.file_size_mb} MB` : ''}
                    {exp.quantization !== 'none' && ` | ${exp.quantization}`}
                  </p>
                </div>
              </div>
              <Download className="w-5 h-5 text-gray-400" />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
