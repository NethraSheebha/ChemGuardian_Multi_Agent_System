import pdfplumber
import re
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Extracts and processes text from MSDS and safety PDFs"""
    
    # Chemical name patterns (common in MSDS headers)
    CHEMICAL_PATTERNS = [
        r"(?:Product Name|Chemical Name|Substance)[:\s]+([A-Z][a-zA-Z0-9\s\-]+)",
        r"^([A-Z][A-Za-z\s]+)(?:\s+MSDS|\s+Safety Data Sheet)",
    ]
    
    # Hazard level keywords
    HAZARD_KEYWORDS = {
        "CRITICAL": ["fatal", "deadly", "extremely toxic", "severe"],
        "HIGH": ["toxic", "corrosive", "flammable", "explosive", "danger"],
        "MEDIUM": ["harmful", "irritant", "warning", "caution"],
        "LOW": ["mild", "notice", "information"]
    }
    
    # Document type detection
    DOC_TYPE_PATTERNS = {
        "MSDS": ["material safety data sheet", "msds", "safety data sheet", "sds"],
        "SOP": ["standard operating procedure", "sop", "operating procedure"],
        "SAFETY_MANUAL": ["safety manual", "safety guide", "emergency procedures"]
    }
    
    def __init__(self, chunk_size: int = 400):
        """
        Initialize PDF processor
        
        Args:
            chunk_size: Target token count per chunk (words as proxy)
        """
        self.chunk_size = chunk_size
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract all text from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            logger.info(f"Extracted {len(text)} characters from {pdf_path}")
            return text
            
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\:\;\-\(\)\[\]\/]', '', text)
        
        return text.strip()
    
    def detect_chemical_name(self, text: str) -> Optional[str]:
        """
        Extract chemical name from document
        
        Args:
            text: Document text
            
        Returns:
            Chemical name if found, else None
        """
        # Check first 500 characters (usually in header)
        header = text[:500]
        
        for pattern in self.CHEMICAL_PATTERNS:
            match = re.search(pattern, header, re.IGNORECASE | re.MULTILINE)
            if match:
                chemical = match.group(1).strip()
                # Clean up
                chemical = re.sub(r'\s+', ' ', chemical)
                return chemical
        
        return None
    
    def detect_document_type(self, text: str) -> str:
        """
        Detect document type from content
        
        Args:
            text: Document text
            
        Returns:
            Document type string
        """
        text_lower = text.lower()
        
        for doc_type, keywords in self.DOC_TYPE_PATTERNS.items():
            if any(kw in text_lower for kw in keywords):
                return doc_type
        
        return "UNKNOWN"
    
    def detect_hazard_level(self, text: str) -> str:
        """
        Detect hazard level from content
        
        Args:
            text: Document text
            
        Returns:
            Hazard level string
        """
        text_lower = text.lower()
        
        # Check in order of severity
        for level, keywords in self.HAZARD_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return level
        
        return "UNKNOWN"
    
    def extract_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Split text into sections based on headers
        
        Args:
            text: Document text
            
        Returns:
            List of section dictionaries with title and content
        """
        sections = []
        
        # Common MSDS section headers
        section_patterns = [
            r'\n(\d+\.?\s+[A-Z][A-Za-z\s]+)\n',
            r'\n([A-Z][A-Z\s]{10,})\n',  # ALL CAPS headers
        ]
        
        current_section = {"title": "Introduction", "content": ""}
        
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text))
            if matches:
                # Split by matched headers
                last_pos = 0
                for match in matches:
                    # Save previous section
                    if current_section["content"]:
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = {
                        "title": match.group(1).strip(),
                        "content": text[last_pos:match.start()].strip()
                    }
                    last_pos = match.end()
                
                # Add final section
                current_section["content"] = text[last_pos:].strip()
                sections.append(current_section)
                break
        
        # If no sections found, treat entire doc as one section
        if not sections:
            sections.append({"title": "Full Document", "content": text})
        
        return sections
    
    def chunk_text(self, text: str, section_title: str = "") -> List[Dict[str, str]]:
        """
        Split text into semantic chunks
        
        Args:
            text: Text to chunk
            section_title: Optional section title for context
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        # Split by sentences (approximate)
        sentences = re.split(r'(Section\s+\d+[:\.\-]\s+[A-Za-z\s]+)', text)
        
        current_chunk = ""
        current_word_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            word_count = len(sentence.split())
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_word_count + word_count > self.chunk_size and current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "section": section_title,
                    "word_count": current_word_count
                })
                current_chunk = sentence
                current_word_count = word_count
            else:
                current_chunk += " " + sentence
                current_word_count += word_count
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "section": section_title,
                "word_count": current_word_count
            })
        
        return chunks
    
    def process_pdf(self, pdf_path: str, filename: str) -> List[Dict]:
        """
        Complete processing pipeline for a PDF
        
        Args:
            pdf_path: Path to PDF file
            filename: Original filename
            
        Returns:
            List of processed chunks with metadata
        """
        # Extract text
        raw_text = self.extract_text(pdf_path)
        if not raw_text:
            return []
        
        # Clean text
        clean_text = self.clean_text(raw_text)
        
        # Extract metadata
        chemical = self.detect_chemical_name(clean_text)
        doc_type = self.detect_document_type(clean_text)
        hazard_level = self.detect_hazard_level(clean_text)
        
        # Extract sections
        sections = self.extract_sections(clean_text)
        
        # Chunk each section
        all_chunks = []
        for section in sections:
            chunks = self.chunk_text(section["content"], section["title"])
            
            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk["text"],
                    "chemical": chemical or "UNKNOWN",
                    "doc_type": doc_type,
                    "hazard_level": hazard_level,
                    "source": filename,
                    "section": chunk["section"],
                    "chunk_id": f"{filename}_section_{section['title']}_chunk_{idx}"
                })
        
        logger.info(f"Processed {filename}: {len(all_chunks)} chunks, chemical={chemical}")
        return all_chunks