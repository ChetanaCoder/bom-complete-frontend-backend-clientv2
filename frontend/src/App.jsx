import React from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { TranslationProvider } from './hooks/useTranslation'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Upload from './pages/Upload'
import Processing from './pages/Processing'
import Results from './pages/Results'
import KnowledgeBase from './pages/KnowledgeBase'
import Settings from './pages/Settings'

function App() {
  return (
    <TranslationProvider>
      <Router>
        <div className="App">
          <Layout>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/upload" element={<Upload />} />
              <Route path="/processing/:workflowId" element={<Processing />} />
              <Route path="/results/:workflowId" element={<Results />} />
              <Route path="/knowledge-base" element={<KnowledgeBase />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </Layout>
          <Toaster 
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#363636',
                color: '#fff',
              },
            }}
          />
        </div>
      </Router>
    </TranslationProvider>
  )
}

export default App