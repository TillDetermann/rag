import logging
from typing import List, Sequence, Optional
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.node_parser import NodeParser
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from private_gpt.components.metadata_retrivial.metadata_retrivial_component import MetadataRetrivialComponent

logger = logging.getLogger(__name__)


class LLMMetadataTransformation(NodeParser):
    """
    Transformation that adds LLM-generated metadata to node metadata.
    Should be inserted AFTER chunking in the transformation pipeline.
    """
    
    _metadata_retrivial_component:  MetadataRetrivialComponent= PrivateAttr()
    
    # Regular Fields for serializable config
    max_metadata: int = Field(
        default=3,
        description="Maximum number of metadata to generate per node"
    )
    
    def __init__(
        self,
        metadata_retrivial_component: MetadataRetrivialComponent,
        max_metadata: int = 2,
        **kwargs
    ):
        """
        Initialize the LLM metadata Transformation.
        
        Args:
            metadata_component: metadataComponent instance for tag generation
            add_metadata: Whether to add metadata to nodes
            max_metadata: Maximum number of metadata per node
        """
        super().__init__(
            max_metadata=max_metadata,
            **kwargs
        )
        
        # Set metadataComponent as private attribute
        self._metadata_retrivial_component = metadata_retrivial_component
        
        logger.info(f"LLMMetadataTransformation initialized with max_metadata: {max_metadata}")
    
    @classmethod
    def class_name(cls) -> str:
        return "LLMMetadataTransformation"
    
    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs,
    ) -> List[BaseNode]:
        """
        Add metadata to existing nodes.
        This method is automatically called by LlamaIndex.
        
        Args:
            nodes: Input nodes to enhance with metadata
            show_progress: Whether to show progress logs
            
        Returns:
            List of enhanced nodes with metadata in metadata
        """
        enhanced_nodes = []
        
        for idx, node in enumerate(nodes):
            if show_progress:
                logger.info(f"Processing node {idx + 1}/{len(nodes)} for tag generation")
            
            # Create new node with metadata
            enhanced_node = self._add_metadata_to_node(node, idx)
            enhanced_nodes.append(enhanced_node)
        
        logger.info(f"Added metadata to {len(enhanced_nodes)} nodes")
        return enhanced_nodes
    
    def _add_metadata_to_node(self, node: BaseNode, index: int) -> BaseNode:
        """
        Add metadata to a single node based on its content.
        
        Args:
            node: Node to enhance
            index: Node index for logging
            
        Returns:
            Enhanced node with metadata in metadata
        """
        # Extract text content as prompt for tag generation
        original_text = node.get_content()
        
        # Generate metadata
        try:
            file_name = node.metadata.get("file_name") if node.metadata else None
            metadata = self._generate_metadata_for_content(original_text, file_name, self.max_metadata )
            logger.debug(f"Generated metadata for node {index}: {metadata}")
        except Exception as e:
            logger.warning(f"Failed to generate metadata for node {index}: {e}")
            metadata = {}
        
        # Create new TextNode with metadata in metadata
        if isinstance(node, TextNode):
            enhanced_node = TextNode(
                text=original_text,
                metadata={
                    **node.metadata,
                    **metadata,
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
            if hasattr(enhanced_node, 'metadata') and enhanced_node.metadata is not None:
                enhanced_node.metadata.update(metadata)
            elif hasattr(enhanced_node, 'metadata'):
                enhanced_node.metadata = {**metadata}
        
        return enhanced_node
    
    def _generate_metadata_for_content(self, content: str, file_name:str, max_metadata:int) -> dict:
        """
        Generate metadata for content.
        
        Args:
            content: Content to generate metadata for
            file_name: File name of node,
            max_metadata: Maximum number of metadata generated to node
            
        Returns:
            List of generated metadata (limited by max_metadata)
        """
        # Use the content as prompt for tag generation
        metadata = self._metadata_retrivial_component.generate_chunk_metadata(content, file_name, max_metadata)
        
        return metadata