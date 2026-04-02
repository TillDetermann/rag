from llama_index.readers.file import PDFReader
from pathlib import Path
from typing import List, Optional
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

class OneDocumentPDFReader(BaseReader):
    """
    Custom PDF Reader that combines all pages into a SINGLE document.
    Wrapper around the standard PDFReader.
    
    This is useful when you want to:
    - Preserve cross-page context during chunking
    - Avoid breaking semantic units across page boundaries
    - Simplify document ingestion pipeline
    
    Note: For very large PDFs (>100 pages), consider memory implications.
    """
    
    def __init__(self):
        """Initialize the combined PDF reader with base PDFReader."""
        self.base_reader = PDFReader()
    
    def load_data(
        self,
        file: Path,
        extra_info: Optional[dict] = None,
    ) -> List[Document]:
        """
        Load PDF and combine all pages into a single document.
        
        Args:
            file: Path to the PDF file
            extra_info: Optional metadata to attach to the document
            
        Returns:
            List containing a single Document with combined text from all pages
            
        Raises:
            Same exceptions as base PDFReader (file not found, corrupt PDF, etc.)
        """
        
        # Load using standard PDFReader (returns individual page documents)
        page_documents = self.base_reader.load_data(file=file, extra_info=extra_info)
        
        if not page_documents:
            return []
        
        # Combine all page texts with double newline separator
        # This preserves paragraph breaks between pages
        combined_text = "\n\n".join([doc.text for doc in page_documents])
        
        # Create a single combined document with enriched metadata
        combined_document = Document(
            text=combined_text,
            metadata={
                # Preserve metadata from first page
                **(page_documents[0].metadata if page_documents[0].metadata else {}),
                # Add any custom metadata passed in
                **(extra_info or {}),
                # Add combination-specific metadata
                "total_pages": len(page_documents),
                "file_name": file.name if isinstance(file, Path) else str(file),
                "source_type": "pdf",
            },
            ref_doc_id=file.name if isinstance(file, Path) else str(file),
        )
        
        return [combined_document]
