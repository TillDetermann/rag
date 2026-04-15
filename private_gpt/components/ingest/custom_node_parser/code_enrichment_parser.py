from typing import List, Sequence
from llama_index.core.schema import BaseNode, TextNode, NodeWithScore
from llama_index.core.node_parser import NodeParser
from llama_index.core.embeddings.utils import EmbedType
from llama_index.llms.ollama import Ollama
from llama_index.core.indices import VectorStoreIndex, load_index_from_storage
from llama_index.core.storage import StorageContext
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
            temperature=0.3
        )

        self._larger_code_llm = Ollama(
            model=_settings.ollama.code_llm,
            api_base=_settings.ollama.api_base,
            request_timeout=_settings.ollama.request_timeout,
            temperature=0.3,
        )
        try:
            self._index = load_index_from_storage(
                storage_context=storage_context,
                index_id="code_enrichment",
                store_nodes_override=True,
                embed_model=embed_model,
            )
        except ValueError:
            self._index = VectorStoreIndex.from_documents(
                [],
                storage_context=storage_context,
                store_nodes_override=True,
                embed_model=embed_model,
            )
            self._index.set_index_id("code_enrichment")

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
            "You are writing a highly detailed, searchable technical description of the following code.\n\n"
            "Instructions:\n"
            "- Describe EVERY single line of code in order, as continuous prose. Do not skip any line.\n"
            "- Always mention the exact variable names, function names, class names, and parameter names "
            "as they appear in the code. For example: 'The variable user_count is assigned the return value "
            "of get_active_users(), which retrieves all currently active users from the database.'\n"
            "- When the code contains comments, incorporate them into your description and use them to explain "
            "the developer's intent. For example, if a comment says '# retry up to 3 times', your description "
            "should say 'The following block retries the operation up to 3 times, as noted by the developer.'\n"
            "- Resolve all boolean parameters, magic numbers, and abbreviated names to their real-world meaning. "
            "For example: 'store_nodes_override=True means that nodes will be forcefully stored even if they "
            "already exist' or 'the timeout of 30 refers to 30 seconds.'\n"
            "- Describe every control flow structure in full: what each if-condition checks and why, "
            "what each else-branch handles, what each loop iterates over and what happens in each iteration, "
            "what each try block attempts and what each except block catches.\n"
            "- For function and method calls, describe what the function does, what each argument passed to it "
            "means, and what the return value represents.\n"
            "- For assignments, describe what value is being stored, where it comes from, and what it will be "
            "used for if apparent from the surrounding code.\n"
            "- For imports, describe what module or class is being imported and what role it plays.\n"
            "- Be exhaustive and verbose. Every detail matters. A 50-line code block should produce a description "
            "of at least several paragraphs.\n"
            "- Write as continuous prose, not as a numbered list or bullet points.\n"
            "- Do not reproduce the code itself.\n\n"
            "Code:\n```\n{code}\n```\n\n"
            "Detailed description:\n"
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
            "You are a senior software engineer writing detailed technical documentation.\n\n"
            "You are given:\n"
            "1. A CODE SNIPPET from a larger file\n"
            "2. RELATED CONTEXT containing descriptions of variables, functions, classes, "
            "and types that appear in the code snippet\n\n"
            "Your task: Write a thorough, detailed natural-language description of the CODE SNIPPET ONLY.\n\n"
            "Instructions:\n"
            "- Use the vocabulary, terminology, and descriptions from the RELATED CONTEXT to explain "
            "what the code does. If the context describes a function as 'validates user authentication tokens', "
            "use that exact phrasing when explaining the code's call to that function.\n"
            "- Explain EVERY line and every operation in the code snippet. Do not skip or gloss over anything.\n"
            "- When the code calls a function or uses a variable that is described in the context, "
            "incorporate that description to explain what is happening at that point. "
            "For example, instead of saying 'calls process_data()', say 'calls process_data(), which "
            "transforms raw sensor readings into normalized temperature values as described above'.\n"
            "- Describe the control flow in detail: what conditions are checked, what happens in each branch, "
            "what loops iterate over, and what the expected outcomes are.\n"
            "- Explain WHY the code does what it does, not just WHAT it does, wherever the context provides enough information.\n"
            "- Be verbose and comprehensive. A longer, more detailed description is always preferred over a short one.\n"
            "- Do NOT summarize or describe the related context itself — only use it as a reference to enrich "
            "your description of the code snippet.\n\n"
            "--- CODE SNIPPET ---\n"
            "{original_code}\n\n"
            "--- RELATED CONTEXT (descriptions of variables, functions, and types — use as reference, do NOT describe) ---\n"
            "{retrieved_chunks}\n\n"
            "--- DETAILED DESCRIPTION OF THE CODE SNIPPET ---\n"
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
    # ----------------------------------------------------------------- #

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

        if isinstance(node, TextNode):
            enriched_node = TextNode(
                text=enriched_description,
                metadata={
                    **node.metadata,
                    "code": original_code,
                    "used_llm_for_code_to_text": self._small_code_llm.model, 
                    "used_llm_for_final_describition": self._larger_code_llm.model
                },
                excluded_embed_metadata_keys=[
                    "code",
                    *(node.excluded_embed_metadata_keys or [])
                ],
                relationships=node.relationships,
            )
        else:
            enriched_node = node
            enriched_node.text = enriched_description
            if hasattr(enriched_node, "metadata"):
                enriched_node.metadata["code"] = original_code
            if hasattr(enriched_node, "excluded_embed_metadata_keys"):
                enriched_node.excluded_embed_metadata_keys.push("code")

        return enriched_node