import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, Play, CheckCircle, TrendingUp, Database } from 'lucide-react'
import { useTranslation } from '../hooks/useTranslation'
import { Card, Button } from '../components/UI'

export default function Dashboard() {
  const navigate = useNavigate()
  const { t } = useTranslation()
  const [workflows, setWorkflows] = useState([])
  const [stats, setStats] = useState({
    total: 0,
    completed: 0,
    processing: 0,
    success_rate: 0
  })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadDashboardData()
  }, [])

  const loadDashboardData = async () => {
    try {
      setLoading(true)
      // Mock data for demonstration
      setWorkflows([
        {
          workflow_id: 'wf-001',
          status: 'completed',
          created_at: new Date().toISOString(),
          message: 'Processing completed successfully'
        }
      ])
      setStats({
        total: 1,
        completed: 1,
        processing: 0,
        success_rate: 100
      })
    } catch (error) {
      console.error('Failed to load dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-gray-600">{t('common.loading')}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto p-6">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">{t('dashboard.title')}</h1>
          <p className="text-gray-600 mt-1">{t('dashboard.subtitle')}</p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card className="p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                  <Database className="h-5 w-5 text-blue-600" />
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Total Workflows</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.total}</p>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                  <CheckCircle className="h-5 w-5 text-green-600" />
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Completed</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.completed}</p>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-yellow-100 rounded-lg flex items-center justify-center">
                  <Play className="h-5 w-5 text-yellow-600" />
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Processing</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.processing}</p>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center">
                  <TrendingUp className="h-5 w-5 text-purple-600" />
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Success Rate</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.success_rate}%</p>
              </div>
            </div>
          </Card>
        </div>

        {/* Action Button */}
        <div className="mb-8">
          <Button 
            onClick={() => navigate('/upload')}
            className="btn-primary"
          >
            <Upload className="h-5 w-5 mr-2" />
            {t('dashboard.startProcessing')}
          </Button>
        </div>

        {/* Recent Workflows */}
        <Card>
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">Recent Workflows</h3>
          </div>
          <div className="p-6">
            {workflows.length > 0 ? (
              <div className="space-y-4">
                {workflows.map((workflow) => (
                  <div key={workflow.workflow_id} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
                    <div>
                      <h4 className="font-medium text-gray-900">{workflow.workflow_id}</h4>
                      <p className="text-sm text-gray-500">{workflow.message}</p>
                    </div>
                    <div className="flex items-center space-x-4">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        workflow.status === 'completed' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                      }`}>
                        {workflow.status}
                      </span>
                      {workflow.status === 'completed' && (
                        <Button 
                          size="sm" 
                          variant="outline"
                          onClick={() => navigate(`/results/${workflow.workflow_id}`)}
                        >
                          View Results
                        </Button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <p className="text-gray-500">No workflows yet. Start by uploading your first document.</p>
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  )
}