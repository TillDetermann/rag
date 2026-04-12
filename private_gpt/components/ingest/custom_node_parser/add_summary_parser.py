from typing import List, Sequence
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.node_parser import NodeParser
from llama_index.llms.ollama import Ollama
from llama_index.core.bridge.pydantic import PrivateAttr

from private_gpt.settings.settings import settings

import logging

logger = logging.getLogger(__name__)

class AddSummaryParser(NodeParser):
    """
    Transformation that adds LLM-generated summaries at the beginning of each chunk.
    Should be inserted AFTER chunking in the transformation pipeline.
    """
    def __init__(
        self,
        add_summary: bool = True,
        **kwargs
    ):
        """
        Initialize the LLM Summary Transformation.
        """
        _summary_llm: Ollama = PrivateAttr()
        _summary_prompt: str = PrivateAttr()
        _add_summary: bool = PrivateAttr()

        # Initialize parent
        super().__init__(
            add_summary=add_summary,
            **kwargs
        )
        _settings = settings()
        # Init LLM
        self._summary_llm = Ollama(
            model=_settings.ollama.worker_llm,
            api_base=_settings.ollama.api_base,
            request_timeout=_settings.ollama.request_timeout,
            temperature=0.3
        )
        self._add_summary = add_summary
        self._summary_prompt = "Summarize the following text in EXACTLY 3 sentences. Do not write more than 3 sentences:\n\n {text}\n\nSummary:"
        
        logger.info(f"AddSummaryParser initialized")
    
    @classmethod
    def class_name(cls) -> str:
        return "AddSummaryParser"
    
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
        if not self._add_summary:
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
        enhanced_text = f"{summary}\n\n{original_text}"
        
        # Create new TextNode with enhanced text
        if isinstance(node, TextNode):
            enhanced_node = TextNode(
                text=enhanced_text,
                metadata={
                    **node.metadata,
                    "has_summary": bool(summary),
                    "used_llm_for_summary": self._summary_llm.model
                },
                excluded_llm_metadata_keys=[
                    *(node.excluded_llm_metadata_keys or [])
                ],
                excluded_embed_metadata_keys=[
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
        
        return enhanced_node
    
    def _generate_summary(self, text: str) -> str:
        """
        Generate summary using LLM.
        
        Args:
            text: Text to summarize
            
        Returns:
            Generated summary
        """
        
        prompt = self._summary_prompt.format(text=text)
        
        # Generate summary with LLM
        response = self._summary_llm.complete(prompt)
        summary = response.text.strip()
        
        # Clean up summary
        summary = summary.replace("**", "")
        summary = summary.replace("Summary:", "").strip()
        
        return summary