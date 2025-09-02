import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, Download, Database } from 'lucide-react'
import { useTranslation } from '../hooks/useTranslation'
import { TranslationService } from '../services/translation'
import { Card } from '../components/UI'
import toast from 'react-hot-toast'

export default function Results() {
  const { workflowId } = useParams()
  const navigate = useNavigate()
  const { t, currentLanguage } = useTranslation()

  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchResults = async () => {
      try {
        setLoading(true)
        const response = await fetch(`/api/autonomous/workflow/${workflowId}/results`)
        if (!response.ok) {
          throw new Error('Failed to fetch results')
        }
        const data = await response.json()
        setResults(data)
      } catch (err) {
        console.error('Error fetching results:', err)
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }

    if (workflowId) {
      fetchResults()
    }
  }, [workflowId])

  const handleExportResults = async () => {
    try {
      const dataToExport = {
        workflow_id: workflowId,
        results: results,
        export_date: new Date().toISOString(),
        language: currentLanguage
      }

      const blob = new Blob([JSON.stringify(dataToExport, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `bom-results-${workflowId}-${currentLanguage}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)

      toast.success(t('results.resultsExported'))
    } catch (error) {
      console.error('Export failed:', error)
      toast.error('Export failed')
    }
  }

  const formatNumber = (num) => {
    if (num === null || num === undefined || isNaN(num)) return '-'
    const locale = currentLanguage === 'ja' ? 'ja-JP' : 'en-US'
    return new Intl.NumberFormat(locale).format(num)
  }

  const formatPercentage = (value) => {
    if (value === null || value === undefined || isNaN(value)) return '-'
    const locale = currentLanguage === 'ja' ? 'ja-JP' : 'en-US'
    return new Intl.NumberFormat(locale, {
      style: 'percent',
      minimumFractionDigits: 1,
      maximumFractionDigits: 1
    }).format(value / 100)
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

  if (error || !results) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8">
            <button
              onClick={() => navigate('/dashboard')}
              className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700"
            >
              <ArrowLeft className="h-4 w-4 mr-1" />
              {t('results.backToDashboard')}
            </button>
          </div>

          <div className="bg-white rounded-lg shadow p-6 text-center">
            <div className="text-red-600 mb-4">
              <div className="w-16 h-16 mx-auto mb-4 bg-red-100 rounded-full flex items-center justify-center">
                <span className="text-2xl">⚠️</span>
              </div>
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">{t('common.error')}</h3>
            <p className="text-gray-600">{t('results.failedToLoadResults')}</p>
          </div>
        </div>
      </div>
    )
  }

  const matches = results.matches || []
  const summary = results.summary || {}
  const knowledgeBaseStats = results.knowledge_base_stats || {}

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto p-6">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <button
                onClick={() => navigate('/dashboard')}
                className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700 mb-4"
              >
                <ArrowLeft className="h-4 w-4 mr-1" />
                {t('results.backToDashboard')}
              </button>
              <h1 className="text-3xl font-bold text-gray-900">{t('results.title')}</h1>
              <p className="text-gray-600 mt-1">
                {t('results.workflowId')}: {workflowId} • {t('results.withItemClassificationReasons')}
              </p>
            </div>
            <div className="flex items-center space-x-3">
              <button
                onClick={handleExportResults}
                className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
              >
                <Download className="h-4 w-4 mr-2" />
                {t('results.exportResults')}
              </button>
            </div>
          </div>
        </div>

        {/* Statistics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card className="p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                  <span className="text-blue-600 font-semibold text-sm">📊</span>
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">{t('results.materialsProcessed')}</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {formatNumber(summary.total_materials || 0)}
                </p>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                  <span className="text-green-600 font-semibold text-sm">✅</span>
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">{t('results.successfulMatches')}</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {formatNumber(summary.successful_matches || 0)}
                </p>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-yellow-100 rounded-lg flex items-center justify-center">
                  <span className="text-yellow-600 font-semibold text-sm">📈</span>
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">{t('results.averageConfidence')}</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {formatPercentage(
                    matches.reduce((sum, m) => sum + (m.confidence_score || 0), 0) / matches.length || 0
                  )}
                </p>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center">
                  <Database className="h-4 w-4 text-purple-600" />
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">{t('results.knowledgeBaseMatches')}</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {formatNumber(summary.knowledge_base_matches || 0)}
                </p>
              </div>
            </div>
          </Card>
        </div>

        {/* Results Table */}
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    {t('results.columns.sno')}
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    {t('results.columns.materialName')}
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    {t('results.columns.qcProcess')}
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    {t('results.columns.partNumber')}
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    {t('results.columns.classification')}
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    {t('results.columns.confidence')}
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    {t('results.columns.supplierMatch')}
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    {t('results.columns.reason')}
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {matches.length > 0 ? (
                  matches.map((match, index) => {
                    const actionPath = TranslationService.translateActionPath(match.qa_classification_label || 5, currentLanguage)

                    return (
                      <tr key={index} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {index + 1}
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-900">
                          <div className="font-medium">{match.qa_material_name}</div>
                          {match.qa_excerpt && (
                            <div className="text-xs text-gray-500 mt-1">"{match.qa_excerpt}"</div>
                          )}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {match.qc_process_step || '-'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {match.part_number || '-'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          <div className="flex items-center">
                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                              {t('results.columns.classification')} {match.qa_classification_label || 5}
                            </span>
                          </div>
                          <div className="text-xs text-gray-500 mt-1">
                            {TranslationService.translateClassificationLabel(match.qa_classification_label || 5, currentLanguage)}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {TranslationService.translateConfidenceLevel(match.qa_confidence_level, currentLanguage)}
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-900">
                          {match.confidence_score > 0 ? (
                            <div>
                              <div className="font-medium">{match.supplier_description}</div>
                              <div className="text-xs text-gray-500">
                                {match.supplier_part_number} • {Math.round(match.confidence_score * 100)}%
                              </div>
                              <div className="text-xs text-blue-600 mt-1">
                                {TranslationService.translateMatchSource(match.match_source, currentLanguage)}
                              </div>
                            </div>
                          ) : (
                            <span className="text-gray-400">-</span>
                          )}
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-600">
                          {match.reasoning || t('results.noReasonProvided')}
                        </td>
                      </tr>
                    )
                  })
                ) : (
                  <tr>
                    <td colSpan="8" className="px-6 py-8 text-center text-gray-500">
                      {t('results.noMaterialsMatch')}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>

        {/* Knowledge Base Stats */}
        {knowledgeBaseStats && Object.keys(knowledgeBaseStats).length > 0 && (
          <Card className="mt-8 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              {t('knowledgeBase.title')} {t('dashboard.showing')}
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-900">
                  {formatNumber(knowledgeBaseStats.total_items || 0)}
                </div>
                <div className="text-sm text-gray-500">{t('knowledgeBase.totalItems')}</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-900">
                  {formatNumber(knowledgeBaseStats.total_workflows || 0)}
                </div>
                <div className="text-sm text-gray-500">{t('knowledgeBase.totalWorkflows')}</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-900">
                  {formatNumber(knowledgeBaseStats.total_matches || 0)}
                </div>
                <div className="text-sm text-gray-500">{t('knowledgeBase.totalMatches')}</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-900">
                  {formatPercentage(knowledgeBaseStats.match_rate || 0)}
                </div>
                <div className="text-sm text-gray-500">{t('knowledgeBase.matchRate')}</div>
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  )
}