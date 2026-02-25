from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from typing import Any, List
from pathlib import Path
import ast
import re


class CodeIngestionMode:
    """Enum for code ingestion modes"""
    FULL = "full"
    COMMENTS_ONLY = "comments_only"


class CodeReader(BaseReader):
    """Code reader with different modes for multiple programming languages"""
    
    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".c": "c",
        ".cpp": "cpp",
        ".cs": "csharp",
    }
    
    def __init__(self):
        """
        Initialize the CodeReader
        """
        self.mode = "full"
    
    @staticmethod
    def _get_base_extension(file_path: Path) -> tuple[str, bool]:
        """
        Extracts the base extension and checks if it's a .comment file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (base_extension, is_comment_file)
        """
        suffixes = file_path.suffixes
        
        # Check if file ends with .comment
        is_comment_file = (len(suffixes) > 0 and suffixes[-1] == '.comments')
        
        # If .comment file, get the base extension
        if is_comment_file and len(suffixes) > 1:
            base_ext = suffixes[-2]
            if base_ext in CodeReader.LANGUAGE_MAP:
                return base_ext, True
        
        # Return the last suffix if single extension
        return file_path.suffix.lower(), False
    
    def load_data(
        self, 
        file: Path, 
        extra_info: dict[str, Any] | None = None
    ) -> List[Document]:
        """
        Loads code file based on the selected mode.
        Files ending in .comment automatically use comments_only mode.
        
        Args:
            file: Path to the code file
            extra_info: Additional metadata to include
            
        Returns:
            List of Document objects
        """
        extension, is_comment_file = self._get_base_extension(file)
        language = self.LANGUAGE_MAP.get(extension, "python")
        
        # Auto-switch to comments_only mode for .comment files
        effective_mode = (
            CodeIngestionMode.COMMENTS_ONLY 
            if is_comment_file 
            else self.mode
        )
        
        if effective_mode == CodeIngestionMode.COMMENTS_ONLY:
            return self._extract_comments_only(file, language, extra_info)
        else:
            # Read full code file
            return self._read_full_code(file, language, extra_info)
        
    def _read_full_code(
        self,
        file: Path,
        language: str,
        extra_info: dict[str, Any] | None = None
    ) -> List[Document]:
        """
        Reads the complete code file.
        
        Args:
            file: Path to the code file
            language: Programming language
            extra_info: Additional metadata
            
        Returns:
            List containing a single Document with full code
        """
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        metadata = extra_info or {}
        metadata.update({
            "file_name": file.name,
            "file_path": str(file),
            "file_type": language,
            "extraction_mode": "full",
            "file_size": file.stat().st_size
        })
        
        return [Document(text=content, metadata=metadata)]
    
    def _extract_comments_only(
        self, 
        file: Path, 
        language: str,
        extra_info: dict[str, Any] | None = None
    ) -> List[Document]:
        """
        Extracts only comments from the code file.
        
        Args:
            file: Path to the code file
            language: Programming language
            extra_info: Additional metadata
            
        Returns:
            List containing a single Document with extracted comments
        """
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Language-specific extraction
        if language == "python":
            comments = self._extract_python_comments(content)
        elif language in ["javascript", "c", "cpp", "csharp"]:
            comments = self._extract_c_style_comments(content)
        else:
            comments = self._extract_generic_comments(content)
        
        # Assemble metadata
        metadata = extra_info or {}
        metadata.update({
            "file_name": file.name,
            "file_path": str(file),
            "file_type": language,
            "extraction_mode": "comments_only",
            "file_size": file.stat().st_size
        })
        return [Document(text=comments, metadata=metadata)]
    
    def _extract_python_comments(self, content: str) -> str:
        """
        Extracts Python comments and docstrings.
        
        Args:
            content: File content as string
            file: Path to the file
            
        Returns:
            Formatted string containing all extracted comments
        """
        comments = []
        
        try:
            # AST-based extraction for docstrings
            tree = ast.parse(content)
            
            # Module docstring
            module_doc = ast.get_docstring(tree)
            if module_doc:
                comments.append(f"\nModule\n{module_doc}\n")
            
            # Iterate through functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        comments.append(f"\nFunction {node.name}\n{docstring}\n")
                
                elif isinstance(node, ast.ClassDef):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        comments.append(f"\nClass {node.name}\n{docstring}\n")
                        
                        # Class methods
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                method_doc = ast.get_docstring(item)
                                if method_doc:
                                    comments.append(
                                        f"\nMethod: {node.name}.{item.name}\n  {method_doc}\n"
                                    )
        
        except SyntaxError as e:
            comments.append(f"\n# Warning: Could not parse file completely: {e}\n")
        
        # Inline comments using regex
        inline_comments = re.findall(r'^\s*#\s*(.+)$', content, re.MULTILINE)
        if inline_comments:
            for i, comment in enumerate(inline_comments, 1):
                # Filter empty or separator comments
                if comment.strip() and not all(c in '#=-_*' for c in comment.strip()):
                    comments.append(f"{i}. {comment}\n")
        
        result = "\n".join(comments)
        return result if len(comments) > 1 else "# No comments found in this file"
    
    def _extract_c_style_comments(
        self, 
        content: str
    ) -> str:
        """
        Extracts C-style comments (// and /* */) for C, C++, C#, JavaScript.
        
        Args:
            content: File content as string
            file: Path to the file
            language: Programming language
            
        Returns:
            Formatted string containing all extracted comments
        """
        comments = []
        
        # Multi-line comments /** ... */ and /* ... */
        # Regex for Javadoc/JSDoc style
        javadoc_pattern = r'/\*\*\s*(.*?)\*/'
        javadoc_comments = re.findall(javadoc_pattern, content, re.DOTALL)
        
        if javadoc_comments:
            for i, comment in enumerate(javadoc_comments, 1):
                # Clean up: remove asterisks at line beginnings
                cleaned = re.sub(r'^\s*\*\s?', '', comment, flags=re.MULTILINE)
                comments.append(f"{i}. {cleaned.strip()}\n\n")
        
        # Regular multi-line comments /* ... */
        multiline_pattern = r'/\*(?!\*)\s*(.*?)\*/'
        multiline_comments = re.findall(multiline_pattern, content, re.DOTALL)
        
        if multiline_comments:
            for i, comment in enumerate(multiline_comments, 1):
                cleaned = re.sub(r'^\s*\*\s?', '', comment, flags=re.MULTILINE)
                comments.append(f"{i}. {cleaned.strip()}\n\n")
        
        # Single-line comments //
        singleline_pattern = r'//\s*(.+)$'
        singleline_comments = re.findall(singleline_pattern, content, re.MULTILINE)
        
        if singleline_comments:
            for i, comment in enumerate(singleline_comments, 1):
                # Filter separator lines like //========
                if not all(c in '/-=_*' for c in comment.strip()):
                    comments.append(f"{i}. {comment.strip()}\n")
        
        result = "\n".join(comments)
        return result if len(comments) > 1 else "// No comments found in this file"
    
    def _extract_generic_comments(self, content: str) -> str:
        """
        Fallback: Generic comment extraction for unknown languages.
        
        Args:
            content: File content as string
            file: Path to the file
            
        Returns:
            Formatted string containing all extracted comments
        """
        comments = []
        
        # Try all common comment styles
        patterns = [
            (r'#\s*(.+)$', '#'),           # Python, Bash, Ruby
            (r'//\s*(.+)$', '//'),         # C-style single-line
            (r'/\*\*(.*?)\*/', '/** */'),  # Javadoc
            (r'/\*(?!\*)(.*?)\*/', '/* */'), # C-style multi-line
        ]
        
        for pattern, style in patterns:
            if style.startswith('/*'):
                matches = re.findall(pattern, content, re.DOTALL)
            else:
                matches = re.findall(pattern, content, re.MULTILINE)
            
            if matches:
                for match in matches:
                    cleaned = match.strip()
                    if cleaned:
                        comments.append(f"{cleaned}\n")
        
        result = "\n".join(comments)
        return result if len(comments) > 1 else "# No comments found in this file"