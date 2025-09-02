import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Database, ArrowLeft } from 'lucide-react'
import { useTranslation } from '../hooks/useTranslation'
import { Card } from '../components/UI'

export default function KnowledgeBase() {
  const navigate = useNavigate()
  const { t } = useTranslation()
  const [stats, setStats] = useState({
    total_items: 0,
    total_workflows: 0,
    total_matches: 0,
    match_rate: 0
  })

  useEffect(() => {
    // Mock data for demonstration
    setStats({
      total_items: 0,
      total_workflows: 0,
      total_matches: 0,
      match_rate: 0
    })
  }, [])

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto p-6">
        {/* Header */}
        <div className="mb-8">
          <button
            onClick={() => navigate('/dashboard')}
            className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700 mb-4"
          >
            <ArrowLeft className="h-4 w-4 mr-1" />
            Back to Dashboard
          </button>
          <h1 className="text-3xl font-bold text-gray-900">{t('knowledgeBase.title')}</h1>
          <p className="text-gray-600 mt-1">{t('knowledgeBase.subtitle')}</p>
        </div>

        {/* Statistics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <Card className="p-6">
            <div className="flex items-center">
              <Database className="h-8 w-8 text-blue-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">{t('knowledgeBase.totalItems')}</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.total_items}</p>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-center">
              <div className="h-8 w-8 bg-green-100 rounded-lg flex items-center justify-center">
                <span className="text-green-600 font-semibold">📋</span>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">{t('knowledgeBase.totalWorkflows')}</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.total_workflows}</p>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-center">
              <div className="h-8 w-8 bg-purple-100 rounded-lg flex items-center justify-center">
                <span className="text-purple-600 font-semibold">🔗</span>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">{t('knowledgeBase.totalMatches')}</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.total_matches}</p>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-center">
              <div className="h-8 w-8 bg-yellow-100 rounded-lg flex items-center justify-center">
                <span className="text-yellow-600 font-semibold">📊</span>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">{t('knowledgeBase.matchRate')}</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.match_rate}%</p>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}