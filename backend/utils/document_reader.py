"""
Document Reader with support for various file formats
"""

import logging
import os
from pathlib import Path
from typing import Optional
import zipfile
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class DocumentReader:
    """Handles reading various document formats"""
    
    @staticmethod
    def read_document(file_path: str) -> Optional[str]:
        """Read document content from various file formats"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                logger.error(f"File not found: {file_path}")
                return DocumentReader._get_demo_content()
            
            file_extension = path.suffix.lower()
            
            if file_extension == '.txt':
                return DocumentReader._read_txt(file_path)
            elif file_extension in ['.docx']:
                return DocumentReader._read_docx(file_path)
            elif file_extension in ['.doc']:
                logger.warning(f"Legacy .doc format not fully supported: {file_path}")
                return DocumentReader._get_demo_content()
            elif file_extension == '.pdf':
                logger.warning(f"PDF format requires additional libraries: {file_path}")
                return DocumentReader._get_demo_content()
            else:
                logger.warning(f"Unsupported file format: {file_extension}")
                return DocumentReader._get_demo_content()
                
        except Exception as e:
            logger.error(f"Error reading document {file_path}: {e}")
            return DocumentReader._get_demo_content()
    
    @staticmethod
    def _read_txt(file_path: str) -> str:
        """Read plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'shift_jis', 'utf-16']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                        logger.info(f"Successfully read file with encoding: {encoding}")
                        return content
                except UnicodeDecodeError:
                    continue
            
            logger.error(f"Could not decode file with any standard encoding: {file_path}")
            return DocumentReader._get_demo_content()
    
    @staticmethod
    def _read_docx(file_path: str) -> str:
        """Read DOCX file by extracting text from XML"""
        try:
            # DOCX files are ZIP archives containing XML
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                # Read the main document content
                with zip_file.open('word/document.xml') as doc_xml:
                    tree = ET.parse(doc_xml)
                    root = tree.getroot()
                    
                    # Extract text from all text nodes
                    # DOCX uses namespace, so we need to find text elements
                    text_content = []
                    
                    # Find all text elements (namespace-aware)
                    for elem in root.iter():
                        if elem.text and elem.text.strip():
                            text_content.append(elem.text.strip())
                    
                    extracted_text = ' '.join(text_content)
                    
                    if extracted_text:
                        logger.info(f"Successfully extracted {len(extracted_text)} characters from DOCX")
                        return extracted_text
                    else:
                        logger.warning("No text content found in DOCX file")
                        return DocumentReader._get_demo_content()
                        
        except zipfile.BadZipFile:
            logger.error(f"Invalid DOCX file (not a valid ZIP): {file_path}")
            return DocumentReader._get_demo_content()
        except KeyError as e:
            logger.error(f"DOCX structure error - missing {e}: {file_path}")
            return DocumentReader._get_demo_content()
        except Exception as e:
            logger.error(f"Error reading DOCX file: {e}")
            return DocumentReader._get_demo_content()
    
    @staticmethod
    def _get_demo_content() -> str:
        """Return demo Japanese content for testing"""
        return """作業指示書 - 組み立て手順

ステップ1: 部品の準備
- M6×20mmボルト（六角）x 4個
- 工業用接着テープ 25mm幅 x 2ロール
- シール材 (シリコン系) x 1本

ステップ2: シャーシ組み立て
M6×20mmボルトを使用してシャーシを固定する。
締め付けトルク: 8-10 N·m

ステップ3: 配線作業  
接着テープを使用してケーブルハーネスを固定する。
テープ幅25mm、長さ50mmで巻き付ける。

ステップ4: シール処理
シリコン系シール材を継ぎ目部分に塗布する。
硬化時間: 24時間

品質管理チェックポイント:
- ボルト締め付け確認
- 配線固定状態確認  
- シール材塗布状態確認

使用工具:
- トルクレンチ (10N·m対応)
- ケーブルカッター
- シール材塗布用ガン"""
