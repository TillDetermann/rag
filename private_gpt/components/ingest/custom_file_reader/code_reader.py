"""
Enhanced Code Reader with Qwen2.5-Coder 3B Integration

This module reads code files and generates high-level conceptual summaries
using Qwen2.5-Coder 3B LLM, focusing on system-level understanding of
processes and workflows implemented in the code.

Designed for PrivateGPT ingestion with emphasis on semantic understanding
rather than syntactic analysis.
"""

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from typing import Any, List, Optional
from pathlib import Path
import logging
from private_gpt.settings.settings import settings
from llama_index.llms.ollama import Ollama
from llama_index.core import Document

logger = logging.getLogger(__name__)

class CodeIngestionMode:
    """Enum for code ingestion modes"""
    CODE = "code"
    SUMMARIZED = "summarized"
    ALL = "all"


class CodeReader(BaseReader):
    """
    Enhanced code reader that generates conceptual summaries using Qwen2.5-Coder.
    
    Supports multiple ingestion modes:
    - full: Returns complete code
    - summarized: LLM generates high-level conceptual summary
    """
    
    LANGUAGE_MAP = {
        ".py": "python",
        ".c": "c",
        ".h": "h",
        ".cpp": "cpp",
        ".cc": "cc",
        ".hpp": "hpp",
        ".cs": "csharp",
    }
    
    def __init__(
        self
    ):
        """
        Initialize the Enhanced Code Reader.
        """
        self.settings = settings()
        self.code_llm = Ollama(
            model=self.settings.ollama.code_llm,
            api_base=self.settings.ollama.api_base,
            request_timeout=self.settings.ollama.request_timeout,
            temperature=0.3,
        )
    
    def load_data(
        self,
        file: Path,
        extra_info: Optional[dict[str, Any]] = None
    ) -> List[Document]:
        """
        Loads code file and generates summary based on selected mode.
        
        Args:
            file: Path to the code file
            extra_info: Additional metadata to include
            
        Returns:
            List of Document objects with conceptual content
        """
        extension = file.suffixes[-1]
        language = self.LANGUAGE_MAP.get(extension, "python")
        
        # Read file content
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if self.settings.ollama.code_summary_mode == CodeIngestionMode.SUMMARIZED:
            return self._generate_conceptual_summary(
                content, file, language, extra_info
            )
        elif self.settings.ollama.code_summary_mode == CodeIngestionMode.CODE:
            return self._read_code(content, file, language, extra_info)
        elif self.settings.ollama.code_summary_mode == CodeIngestionMode.ALL:
            return self._generate_conceptual_summary(
                content, file, language, extra_info
            ) + self._read_code(content, file, language, extra_info)
    
    def _read_code(
        self,
        content: str,
        file: Path,
        language: str,
        extra_info: Optional[dict[str, Any]] = None
    ) -> List[Document]:
        """
        Returns the complete code file.
        
        Args:
            content: File content as string
            file: Path to the code file
            language: Programming language
            extra_info: Additional metadata
            
        Returns:
            List containing a single Document with full code
        """
        logger.info("read full code and turn into document.")
        metadata = extra_info or {}
        metadata.update({
            "file_name": file.name,
            "file_type": language,
            "lines_of_code": len(content.split('\n')),
            "doc_type": "source_code",
            "content_type": "code",
        })
        
        return [Document(text=content, metadata=metadata)]
    
    def _generate_conceptual_summary(
        self,
        content: str,
        file: Path,
        language: str,
        extra_info: Optional[dict[str, Any]] = None
    ) -> List[Document]:
        """
        Generates a high-level conceptual summary using Qwen2.5-Coder.
        
        The summary focuses on:
        - Overall purpose and system-level functionality
        - Main processes and workflows
        - Key components and their interactions
        - Data flow and control flow at abstract level
        - System behaviors observable from external perspective
        
        Args:
            content: File content as string
            file: Path to the code file
            language: Programming language
            extra_info: Additional metadata
            
        Returns:
            List containing Document with summary
        """

        logger.info("summarize code and transfer into document")
        
        # Prepare the prompt for Qwen2.5-Coder
        prompt = self._build_summarization_prompt(content, language, file.name)
        try:
            # Call Ollama with Qwen2.5-Coder
            response = self.code_llm.complete(prompt)
            code_explanation = response.text.strip()
                
        except Exception as e:
            return []

        # Build metadata
        metadata = extra_info or {}
        metadata.update({
            "file_name": file.name,
            "file_type": language,
            "lines_of_code": len(content.split('\n')),
        })

        summary_metadata = metadata.copy()
        summary_metadata.update({
            "doc_type": "summary_of_code",
            "content_type": "explanation",
        })
        summary_doc = Document(
            text=code_explanation,
            metadata=summary_metadata,
            ref_doc_id=file.name if isinstance(file, Path) else str(file),
        )

        return [summary_doc]
    
    def _build_summarization_prompt(
        self,
        code: str,
        language: str,
        file_name: str
    ) -> str:
        """
        Builds the prompt for Qwen2.5-Coder to generate conceptual summary.
        
        Focus: System-level understanding, not code-level details.
        """

        # Truncate code if too long to fit in context
        max_code_length = 10000  # Reasonable limit
        if len(code) > max_code_length:
            code = code[:max_code_length] + f"\n\n[... truncated, total lines: {len(code.split(chr(10)))} ...]"
        
        prompt = f"""Analyze this code file. Be precise and factual. Only describe what you see in the code.

File: {file_name}
Language: {language}
```{language}
{code}
```

---

Write your analysis in exactly this format:

## FILE OVERVIEW
What this file does in 2-3 sentences. What is its role in the project.

## IMPORTS AND DEPENDENCIES
List each import and what it is used for.

## FUNCTIONS AND METHODS
For EVERY function/method in this file, write:

### function_name(parameters)
- **Purpose:** What it does in one sentence.
- **Parameters:** List each parameter and its type/meaning.
- **Returns:** What it returns.
- **Logic:** Step-by-step what happens inside (3-5 bullet points).
- **Calls:** Other functions it calls.
- **Side effects:** Any file writes, API calls, state changes, prints.

## CLASSES
For each class:
- **Name and purpose**
- **Attributes** with types
- **Methods** (described as above)

## CONSTANTS AND CONFIGURATION
List all constants, config values, and global variables with their purpose.

## DATA FLOW
How data moves through the file: input -> processing -> output. One short paragraph.

---

Rules:
- Describe EVERY function, no exceptions.
- Be short and direct per item.
- Do not invent things not in the code.
- Do not explain general programming concepts.
- Use the exact function and variable names from the code."""

        return prompt