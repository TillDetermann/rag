from typing import List, Sequence, Optional
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.llms import LLM
from llama_index.core.node_parser import NodeParser
from llama_index.core.bridge.pydantic import Field, PrivateAttr
import logging

logger = logging.getLogger(__name__)


class LLMSummaryTransformation(NodeParser):
    """
    Transformation that adds LLM-generated summaries at the beginning of each chunk.
    Should be inserted AFTER chunking in the transformation pipeline.
    """
    
    _summary_llm: LLM = PrivateAttr()
    
    # Regular Fields for serializable config
    add_summary: bool = Field(default=True, description="Whether to add summaries")
    summary_prompt: str = Field(
        default="Summarize the following text in EXACTLY 3 sentences. Do not write more than 3 sentences:\n\n {text}\n\nSummary:",
        description="Prompt template for generating summaries"
    )
    max_text_length_for_summary: int = Field(
        default=1000,
        description="Maximum text length for summary input (will be truncated if longer)"
    )

    summary_format: str = Field(
        default="natural",
        description="Format style: 'natural', 'contextual', 'metadata', or 'inline'"
    )
    
    def __init__(
        self,
        summary_llm: LLM,
        add_summary: bool = True,
        summary_prompt: Optional[str] = None,
        max_text_length_for_summary: int = 2000,
        summary_format: str = "contextual",
        **kwargs
    ):
        """
        Initialize the LLM Summary Transformation.
        
        Args:
            summary_llm: LLM for generating summaries (e.g., Ollama llama3.2:1b)
            add_summary: Whether to add summaries to chunks
            summary_prompt: Custom prompt template for summary generation
            max_text_length_for_summary: Maximum text length for summary input
            summary_format: Format style - 'contextual', 'metadata', or 'inline'
        """
        # Initialize parent without passing summary_llm to avoid validation error
        super().__init__(
            add_summary=add_summary,
            summary_prompt=summary_prompt or self.__fields__['summary_prompt'].default,
            max_text_length_for_summary=max_text_length_for_summary,
            summary_format=summary_format,
            **kwargs
        )
        
        # Set LLM as private attribute (not validated by Pydantic)
        self._summary_llm = summary_llm
        
        logger.info(f"LLMSummaryTransformation initialized with format: {summary_format}")
    
    @classmethod
    def class_name(cls) -> str:
        return "LLMSummaryTransformation"
    
    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs,
    ) -> List[BaseNode]:
        """
        Add summaries to existing nodes.
        This method is automatically called by LlamaIndex.
        
        Args:
            nodes: Input nodes to enhance with summaries
            show_progress: Whether to show progress logs
            
        Returns:
            List of enhanced nodes with summaries
        """
        if not self.add_summary:
            return list(nodes)
        
        enhanced_nodes = []
        
        for idx, node in enumerate(nodes):
            if show_progress:
                logger.info(f"Processing node {idx + 1}/{len(nodes)} for summary generation")
            
            # Create new node with summary
            enhanced_node = self._add_summary_to_node(node, idx)
            enhanced_nodes.append(enhanced_node)
        
        logger.info(f"Added summaries to {len(enhanced_nodes)} nodes")
        return enhanced_nodes
    
    def _add_summary_to_node(self, node: BaseNode, index: int) -> BaseNode:
        """
        Add summary to a single node.
        
        Args:
            node: Node to enhance
            index: Node index for logging
            
        Returns:
            Enhanced node with summary
        """
        # Extract text content
        original_text = node.get_content()
        
        # Generate summary
        try:
            summary = self._generate_summary(original_text)
            logger.debug(f"Generated summary for node {index}: {summary[:100]}...")
        except Exception as e:
            logger.warning(f"Failed to generate summary for node {index}: {e}")
            summary = ""
        
        # Combine summary with original text
        enhanced_text = self._format_text_with_summary(summary, original_text)
        
        # Create new TextNode with enhanced text
        if isinstance(node, TextNode):
            enhanced_node = TextNode(
                text=enhanced_text,
                metadata={
                    **node.metadata,
                    "has_summary": bool(summary),
                    "summary": summary,
                    "original_length": len(original_text),
                    "enhanced_length": len(enhanced_text),
                    "summary_format": self.summary_format,
                },
                excluded_llm_metadata_keys=[
                    "original_length", 
                    "enhanced_length",
                    *(node.excluded_llm_metadata_keys or [])
                ],
                excluded_embed_metadata_keys=[
                    "summary",  # Don't embed summary separately
                    *(node.excluded_embed_metadata_keys or [])
                ],
                relationships=node.relationships,
            )
        else:
            # Fallback for other node types
            enhanced_node = node
            enhanced_node.text = enhanced_text
            if hasattr(enhanced_node, 'metadata'):
                enhanced_node.metadata["has_summary"] = bool(summary)
                enhanced_node.metadata["summary"] = summary
                enhanced_node.metadata["summary_format"] = self.summary_format
        
        return enhanced_node
    
    def _generate_summary(self, text: str) -> str:
        """
        Generate summary using LLM.
        
        Args:
            text: Text to summarize
            
        Returns:
            Generated summary
        """
        # Truncate text if too long
        if len(text) > self.max_text_length_for_summary:
            text = text[:self.max_text_length_for_summary] + "..."
        
        prompt = self.summary_prompt.format(text=text)
        
        # Generate summary with LLM (use private attribute)
        response = self._summary_llm.complete(prompt)
        summary = response.text.strip()
        
        # Clean up summary
        summary = summary.replace("**", "")
        summary = summary.replace("Summary:", "").strip()
        summary = summary.replace("Zusammenfassung:", "").strip()
        
        return summary
    
    def _format_text_with_summary(self, summary: str, content: str) -> str:
        """
        Format text with summary at the beginning.
        
        Supports multiple formatting styles:
        - 'natural': Clean format, summary directly before content (RECOMMENDED)
        - 'contextual': Adds context markers and structure
        - 'metadata': Structured metadata format
        - 'inline': Simple inline format with label
        
        Args:
            summary: Generated summary
            content: Original content
            
        Returns:
            Formatted text with summary
        """
        if not summary:
            return content
        
        if self.summary_format == "natural":
            # RECOMMENDED: Most natural format - just summary then content
            return f"{summary}\n\n{content}"
        
        elif self.summary_format == "contextual":
            # Contextual format with XML-like structure
            return (
                f"<context>\n"
                f"<summary>{summary}</summary>\n"
                f"</context>\n\n"
                f"{content}"
            )
        
        elif self.summary_format == "metadata":
            # Structured metadata format
            return (
                f"[METADATA]\n"
                f"Summary: {summary}\n"
                f"[/METADATA]\n\n"
                f"[CONTENT]\n"
                f"{content}\n"
                f"[/CONTENT]"
            )
        
        elif self.summary_format == "inline":
            # Simple inline format with label
            return f"Summary: {summary}\n\n{content}"
        
        else:
            # Default fallback
            return f"SUMMARY: {summary}\n\n---\n\n{content}"