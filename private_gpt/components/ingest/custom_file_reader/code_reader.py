from pathlib import Path
from typing import List, Optional
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

class CodeReader(BaseReader):
    """
    Custom Reader that creates TWO documents from a single code file:
    one with doc_type='text' and one with doc_type='code'.
    
    This allows the same content to be ingested into two separate
    indices with different embedding models and chunking strategies.
    """
    
    def __init__(self):
        pass
    
    def load_data(
        self,
        file: Path,
        extra_info: Optional[dict] = None,
    ) -> List[Document]:
        """
        Load a code file and return two identical documents with different doc_types.
        
        Args:
            file: Path to the code file
            extra_info: Optional metadata to attach to both documents
            
        Returns:
            List containing two Documents: one with doc_type='text', one with doc_type='code'
        """
        file_path = Path(file) if not isinstance(file, Path) else file
        content = file_path.read_text(encoding="utf-8")
        
        if not content.strip():
            return []
        
        base_metadata = {
            **(extra_info or {}),
            "file_name": file_path.name,
        }
        
        code_document_for_summary = Document(
            text=content,
            metadata={
                **base_metadata,
                "doc_type": "text",
            },
        )
        
        code_document = Document(
            text=content,
            metadata={
                **base_metadata,
                "doc_type": "code",
            },
        )
        
        return [code_document_for_summary, code_document]