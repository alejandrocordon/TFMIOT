import { useState, useEffect } from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import { Cpu, BarChart3, Upload, Play, FlaskConical, Layout, BookOpen, Tag, Rocket, Moon, Sun } from 'lucide-react'
import Dashboard from './pages/Dashboard'
import Projects from './pages/Projects'
import Training from './pages/Training'
import Experiments from './pages/Experiments'
import Export from './pages/Export'
import Deploy from './pages/Deploy'
import Playground from './pages/Playground'
import Versions from './pages/Versions'

const navItems = [
  { to: '/', icon: Layout, label: 'Dashboard' },
  { to: '/projects', icon: FlaskConical, label: 'Projects' },
  { to: '/training', icon: Play, label: 'Training' },
  { to: '/versions', icon: Tag, label: 'Versions' },
  { to: '/experiments', icon: BarChart3, label: 'Experiments' },
  { to: '/export', icon: Upload, label: 'Export' },
  { to: '/deploy', icon: Rocket, label: 'Deploy Apps' },
  { to: '/playground', icon: Cpu, label: 'Playground' },
]

function useTheme() {
  const [dark, setDark] = useState(() => {
    if (typeof window === 'undefined') return false
    const saved = localStorage.getItem('mlforge-theme')
    if (saved) return saved === 'dark'
    return window.matchMedia('(prefers-color-scheme: dark)').matches
  })

  useEffect(() => {
    const root = document.documentElement
    if (dark) {
      root.classList.add('dark')
    } else {
      root.classList.remove('dark')
    }
    localStorage.setItem('mlforge-theme', dark ? 'dark' : 'light')
  }, [dark])

  return { dark, toggle: () => setDark(d => !d) }
}

export default function App() {
  const { dark, toggle } = useTheme()

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className="w-64 bg-gray-900 text-white flex flex-col flex-shrink-0">
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

        <div className="px-4 pb-1">
          <a
            href="/docs-site/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 px-4 py-3 rounded-lg text-gray-300 hover:bg-gray-800 transition-colors"
          >
            <BookOpen className="w-5 h-5" />
            Docs
          </a>
        </div>

        <div className="px-4 pb-2">
          <button
            onClick={toggle}
            className="flex items-center gap-3 px-4 py-3 rounded-lg text-gray-300 hover:bg-gray-800 transition-colors w-full"
          >
            {dark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            {dark ? 'Light Mode' : 'Dark Mode'}
          </button>
        </div>

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
          <Route path="/versions" element={<Versions />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/export" element={<Export />} />
          <Route path="/deploy" element={<Deploy />} />
          <Route path="/playground" element={<Playground />} />
        </Routes>
      </main>
    </div>
  )
}
