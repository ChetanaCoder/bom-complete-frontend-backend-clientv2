// Translation utility functions for result data
export class TranslationService {

  // Translate classification labels
  static translateClassificationLabel(label, language = 'en') {
    const translations = {
      en: {
        1: 'Direct Material - Production',
        2: 'Indirect Material - Support', 
        3: 'Tools & Equipment',
        4: 'Consumable Items',
        5: 'Other/Miscellaneous'
      },
      ja: {
        1: '直接材料 - 製造',
        2: '間接材料 - サポート',
        3: '工具・設備', 
        4: '消耗品',
        5: 'その他・雑項目'
      }
    }

    return translations[language]?.[label] || translations.en[label] || `Label ${label}`
  }

  // Translate confidence levels
  static translateConfidenceLevel(level, language = 'en') {
    const translations = {
      en: {
        'high': 'High',
        'medium': 'Medium',
        'low': 'Low'
      },
      ja: {
        'high': '高',
        'medium': '中', 
        'low': '低'
      }
    }

    return translations[language]?.[level?.toLowerCase()] || level
  }

  // Translate match sources
  static translateMatchSource(source, language = 'en') {
    const translations = {
      en: {
        'knowledge_base': 'From Knowledge Base',
        'supplier_bom': 'From Supplier BOM',
        'hybrid': 'Verified Match',
        'no_match': 'No Match'
      },
      ja: {
        'knowledge_base': '知識ベースから',
        'supplier_bom': 'サプライヤーBOMから', 
        'hybrid': '検証済みマッチ',
        'no_match': 'マッチなし'
      }
    }

    return translations[language]?.[source] || source
  }

  // Translate action paths based on classification
  static translateActionPath(classificationLabel, language = 'en') {
    const paths = {
      en: {
        1: { icon: '🟢', text: 'Production Use' },
        2: { icon: '🟡', text: 'Support Review' },
        3: { icon: '🔧', text: 'Equipment Check' },
        4: { icon: '📦', text: 'Inventory Track' },
        5: { icon: '❓', text: 'Manual Review' }
      },
      ja: {
        1: { icon: '🟢', text: '生産使用' },
        2: { icon: '🟡', text: 'サポート確認' },
        3: { icon: '🔧', text: '設備チェック' },
        4: { icon: '📦', text: '在庫追跡' },
        5: { icon: '❓', text: '手動確認' }
      }
    }

    return paths[language]?.[classificationLabel] || paths.en[classificationLabel] || { icon: '❓', text: 'Review' }
  }

  // Translate boolean values
  static translateBoolean(value, language = 'en') {
    if (value === null || value === undefined) return '-'

    const translations = {
      en: {
        true: 'Yes',
        false: 'No'
      },
      ja: {
        true: 'はい',
        false: 'いいえ'
      }
    }

    return translations[language]?.[value.toString()] || value.toString()
  }

  // Translate entire result item for display
  static translateResultItem(item, language = 'en') {
    if (!item) return item

    return {
      ...item,
      classification_description: this.translateClassificationLabel(item.qa_classification_label, language),
      confidence_level_text: this.translateConfidenceLevel(item.qa_confidence_level, language),
      match_source_text: this.translateMatchSource(item.match_source, language),
      action_path: this.translateActionPath(item.qa_classification_label, language),
      consumable_text: this.translateBoolean(item.consumable_jigs_tools, language)
    }
  }
}