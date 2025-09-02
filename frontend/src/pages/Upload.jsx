import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload as UploadIcon, File, ArrowLeft } from 'lucide-react'
import { useTranslation } from '../hooks/useTranslation'
import { Card, Button } from '../components/UI'
import toast from 'react-hot-toast'

export default function Upload() {
  const navigate = useNavigate()
  const { t } = useTranslation()
  const [wiDocument, setWiDocument] = useState(null)
  const [itemMaster, setItemMaster] = useState(null)
  const [uploading, setUploading] = useState(false)

  const handleFileChange = (event, type) => {
    const file = event.target.files[0]
    if (type === 'wi') {
      setWiDocument(file)
    } else {
      setItemMaster(file)
    }
  }

  const handleUpload = async () => {
    if (!wiDocument || !itemMaster) {
      toast.error('Please select both files')
      return
    }

    try {
      setUploading(true)

      const formData = new FormData()
      formData.append('wi_document', wiDocument)
      formData.append('item_master', itemMaster)

      const response = await fetch('/api/autonomous/upload', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error('Upload failed')
      }

      const result = await response.json()

      if (result.success) {
        toast.success('Upload successful!')
        navigate(`/processing/${result.workflow_id}`)
      } else {
        throw new Error(result.message || 'Upload failed')
      }
    } catch (error) {
      console.error('Upload error:', error)
      toast.error('Upload failed: ' + error.message)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto p-6">
        {/* Header */}
        <div className="mb-8">
          <button
            onClick={() => navigate('/dashboard')}
            className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700 mb-4"
          >
            <ArrowLeft className="h-4 w-4 mr-1" />
            Back to Dashboard
          </button>
          <h1 className="text-3xl font-bold text-gray-900">{t('navigation.upload')}</h1>
          <p className="text-gray-600 mt-1">
            Upload your Japanese WI/QC document and Item Master to start autonomous processing
          </p>
        </div>

        {/* Upload Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* WI Document Upload */}
          <Card className="p-6">
            <div className="text-center">
              <File className="mx-auto h-12 w-12 text-gray-400 mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                WI/QC Document
              </h3>
              <p className="text-sm text-gray-500 mb-4">
                Supports PDF, DOCX, DOC, TXT formats
              </p>
              <input
                type="file"
                accept=".pdf,.docx,.doc,.txt"
                onChange={(e) => handleFileChange(e, 'wi')}
                className="hidden"
                id="wi-upload"
              />
              <label
                htmlFor="wi-upload"
                className="cursor-pointer inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
              >
                <UploadIcon className="h-4 w-4 mr-2" />
                Choose File
              </label>
              {wiDocument && (
                <p className="mt-2 text-sm text-green-600">
                  Selected: {wiDocument.name}
                </p>
              )}
            </div>
          </Card>

          {/* Item Master Upload */}
          <Card className="p-6">
            <div className="text-center">
              <File className="mx-auto h-12 w-12 text-gray-400 mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                Item Master
              </h3>
              <p className="text-sm text-gray-500 mb-4">
                Supports Excel (XLSX, XLS) and CSV formats
              </p>
              <input
                type="file"
                accept=".xlsx,.xls,.csv"
                onChange={(e) => handleFileChange(e, 'item')}
                className="hidden"
                id="item-upload"
              />
              <label
                htmlFor="item-upload"
                className="cursor-pointer inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
              >
                <UploadIcon className="h-4 w-4 mr-2" />
                Choose File
              </label>
              {itemMaster && (
                <p className="mt-2 text-sm text-green-600">
                  Selected: {itemMaster.name}
                </p>
              )}
            </div>
          </Card>
        </div>

        {/* Upload Button */}
        <div className="text-center">
          <Button
            onClick={handleUpload}
            disabled={!wiDocument || !itemMaster || uploading}
            className="btn-primary"
          >
            {uploading ? 'Uploading...' : 'Start Processing'}
          </Button>
          <p className="text-sm text-gray-500 mt-4">
            Our autonomous agents will process your documents through translation, 
            extraction with WI/QC Item classification, and intelligent comparison stages. 
            You can monitor progress in real-time.
          </p>
        </div>
      </div>
    </div>
  )
}