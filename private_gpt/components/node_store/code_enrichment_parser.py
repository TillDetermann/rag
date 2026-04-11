from typing import List, Sequence
from llama_index.core.schema import BaseNode, TextNode, NodeWithScore
from llama_index.core.node_parser import NodeParser
from llama_index.core.embeddings.utils import EmbedType
from llama_index.llms.ollama import Ollama
from llama_index.core.storage import StorageContext
from llama_index.core.indices import load_index_from_storage
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.data_structs import IndexDict
from llama_index.core.indices.base import BaseIndex
import logging

from private_gpt.settings.settings import settings

logger = logging.getLogger(__name__)


class CodeEnrichmentParser(NodeParser):
    """
    Transformation for code chunks that:
    1. Uses a small LLM to convert code into natural language description
    2. Performs similarity search with that description against the existing vector store
    3. Provides a hook for a larger LLM to generate the final enriched text
       using: original code + retrieved chunks + full source file

    Should be inserted AFTER code chunking (e.g. CodeSplitter) in the pipeline.
    """
    
    _small_code_llm: Ollama = PrivateAttr()
    _larger_code_llm: Ollama = PrivateAttr()
    _index: BaseIndex[IndexDict] = PrivateAttr()
    _similarity_top_k: int = PrivateAttr()

    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        **kwargs,
    ):
        """
        Initialize the Code Enrichment Transformation.
        """
        super().__init__(
            **kwargs,
        ) 

        _settings = settings()

        self._small_code_llm = Ollama(
            model=_settings.ollama.code_llm,
            api_base=_settings.ollama.api_base,
            request_timeout=_settings.ollama.request_timeout,
            temperature=0.3,
        )

        self._larger_code_llm = Ollama(
            model=_settings.ollama.code_llm,
            api_base=_settings.ollama.api_base,
            request_timeout=_settings.ollama.request_timeout,
            temperature=0.3,
        )
        self._index = load_index_from_storage(
                storage_context=storage_context,
                embed_model=embed_model,
            )

        self._similarity_top_k = 5

        logger.info("CodeEnrichmentParser initialized ")

    @classmethod
    def class_name(cls) -> str:
        return "CodeEnrichmentParser"

    # ------------------------------------------------------------------ #
    #  Main pipeline entry point (called by LlamaIndex automatically)
    # ------------------------------------------------------------------ #

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs,
    ) -> List[BaseNode]:
        """Process code nodes: describe → retrieve → enrich."""
        if not nodes:
            return []

        enriched_nodes: List[BaseNode] = []

        for idx, node in enumerate(nodes):
            if show_progress:
                logger.info(
                    f"Code enrichment: processing node {idx + 1}/{len(nodes)}"
                )
            enriched = self._enrich_code_node(node)
            enriched_nodes.append(enriched)

        logger.info(f"Code enrichment completed for {len(enriched_nodes)} nodes")
        return enriched_nodes

    # ------------------------------------------------------------------ #
    #  Step 1: Code → Natural Language  (_small_code_llm)
    # ------------------------------------------------------------------ #

    def code_to_natural_language(self, code: str) -> str:
        """
        Use the small LLM to convert a code chunk into a natural language
        description that can be used for similarity search.
        """
    
        code_to_text_prompt = (
    "Describe the following code line by line as continuous prose. "
    "Cover every meaningful line: what it does, which functions are called, and what "
    "the parameters mean in real-world terms. Resolve boolean parameters and method "
    "names to their actual intent. Describe control flow explicitly. "
    "Include all identifier names alongside their resolved meaning for searchability. "
    "Do not reproduce the code.\n\n"
    "Code:\n```\n{code}\n```\n\n"
    "Description:"
)
        prompt = code_to_text_prompt.format(code=code)

        try:
            response = self._small_code_llm.complete(prompt)
            description = response.text.strip()

            # Clean common LLM artifacts
            for prefix in ("Description:", "**Description**:"):
                if description.startswith(prefix):
                    description = description[len(prefix):].strip()
            logger.debug(f"Code description: {description[:50]}...")
            return description
        except Exception as e:
            logger.warning(f"Failed to generate code description: {e}")
            return ""

    # ------------------------------------------------------------------ #
    #  Step 2: Similarity Search with generated description
    # ------------------------------------------------------------------ #

    def retrieve_similar_chunks(self, description: str) -> List[NodeWithScore]:
        """
        Use the natural language description to perform a similarity search
        against the existing vector store _index.

        Returns a list of NodeWithScore objects from the vector store.
        """
        if not description:
            return []

        try:
            retriever = self._index.as_retriever(_similarity_top_k=5)
            results = retriever.retrieve(description)
            logger.debug(
                f"Retrieved {len(results)} similar chunks "
                f"(scores: {[round(r.score or 0, 3) for r in results]})"
            )
            return results
        except Exception as e:
            logger.warning(f"Similarity search failed: {e}")
            return []

    # ------------------------------------------------------------------ #
    #  Step 3: Code LLM call
    # ------------------------------------------------------------------ #
    def call_large_code_llm(
        self,
        original_code: str,
        retrieved_chunks_text: str
    ) -> str:
        """
        Call a larger LLM with the full context for final enrichment..
        """

        prompt_template = (
            "You are a senior software engineer writing documentation.\n\n"
            "You are given:\n"
            "1. A CODE SNIPPET from a larger file\n"
            "2. RELATED CONTEXT for reference only\n\n"
            "Your task: Write a natural-language description of the CODE SNIPPET ONLY. "
            "Use the related context to resolve variable names, understand types, "
            "and clarify dependencies, but do NOT summarize or describe the context itself. "
            "Focus exclusively on what the code snippet does.\n\n"
            "--- CODE SNIPPET ---\n"
            "{original_code}\n\n"
            "--- RELATED CONTEXT (use for reference, do NOT describe) ---\n"
            "{retrieved_chunks}\n\n"
            "--- DESCRIPTION OF THE CODE SNIPPET ---\n"
        )
        prompt = prompt_template.format(
            original_code=original_code,
            retrieved_chunks=retrieved_chunks_text,
        )
        try:
            response = self._larger_code_llm.complete(prompt)
            description = response.text.strip()
            # Clean common LLM artifacts
            for prefix in ("Description:","**Description**:"):
                if description.startswith(prefix):
                    description = description[len(prefix):].strip()
            logger.debug(f"Code description: {description[:120]}...")
            return description
        except Exception as e:
            logger.warning(f"Failed to generate code description: {e}")
            return ""
    # ------------------------------------------------------------------ #
    #  Orchestration: combine steps 1-3 per node
    # ---------------------------------^------------------------------- #

    def _enrich_code_node(self, node: BaseNode) -> BaseNode:
        """Enrich a single code node through the full pipeline."""
        original_code = node.get_content()

        # --- Step 1: code → natural language ---
        description = self.code_to_natural_language(original_code)

        # --- Step 2: similarity search ---
        retrieved_nodes = self.retrieve_similar_chunks(description)
        retrieved_texts = [
            r.node.get_content() for r in retrieved_nodes if r.node
        ]
        retrieved_chunks_text = "\n\n---\n\n".join(retrieved_texts)

        # --- Step 3: large LLM ---
        enriched_description = self.call_large_code_llm(original_code, retrieved_chunks_text)

        # --- Build enriched node ---
        enriched_text = f"{enriched_description}\n\n{original_code}"

        if isinstance(node, TextNode):
            enriched_node = TextNode(
                text=enriched_text,
                metadata={
                    **node.metadata,
                    "has_code_enrichment": True,
                },
                excluded_embed_metadata_keys=[
                    *(node.excluded_embed_metadata_keys or []),
                ],
                relationships=node.relationships,
            )
        else:
            enriched_node = node
            enriched_node.text = enriched_text  # type: ignore[attr-defined]
            if hasattr(enriched_node, "metadata"):
                enriched_node.metadata["has_code_enrichment"] = True

        return enriched_node