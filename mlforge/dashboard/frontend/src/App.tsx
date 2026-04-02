import { Routes, Route, NavLink } from 'react-router-dom'
import { Cpu, BarChart3, Upload, Play, FlaskConical, Layout } from 'lucide-react'
import Dashboard from './pages/Dashboard'
import Projects from './pages/Projects'
import Training from './pages/Training'
import Experiments from './pages/Experiments'
import Export from './pages/Export'
import Playground from './pages/Playground'

const navItems = [
  { to: '/', icon: Layout, label: 'Dashboard' },
  { to: '/projects', icon: FlaskConical, label: 'Projects' },
  { to: '/training', icon: Play, label: 'Training' },
  { to: '/experiments', icon: BarChart3, label: 'Experiments' },
  { to: '/export', icon: Upload, label: 'Export' },
  { to: '/playground', icon: Cpu, label: 'Playground' },
]

export default function App() {
  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className="w-64 bg-gray-900 text-white flex flex-col">
        <div className="p-6">
          <h1 className="text-xl font-bold flex items-center gap-2">
            <Cpu className="w-6 h-6 text-blue-400" />
            MLForge
          </h1>
          <p className="text-gray-400 text-sm mt-1">ML Model Factory</p>
        </div>

        <nav className="flex-1 px-4">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-lg mb-1 transition-colors ${
                  isActive
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-300 hover:bg-gray-800'
                }`
              }
            >
              <Icon className="w-5 h-5" />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="p-4 text-gray-500 text-xs">
          v0.1.0
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/projects" element={<Projects />} />
          <Route path="/training" element={<Training />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/export" element={<Export />} />
          <Route path="/playground" element={<Playground />} />
        </Routes>
      </main>
    </div>
  )
}
