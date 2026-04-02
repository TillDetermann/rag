from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.core.schema import TransformComponent
from llama_index.llms.ollama import Ollama
from llama_index.core.readers.base import BaseReader

from llama_index.core.embeddings import BaseEmbedding
from private_gpt.components.ingest.custom_file_reader.code_reader import CodeReader
from private_gpt.components.ingest.ingest_strategy import IngestionStrategy
from private_gpt.components.metadata_retrivial.metadata_retrivial_component import MetadataRetrivialComponent
from private_gpt.components.metadata_retrivial.metadata_retrivial_parser import LLMMetadataTransformation
from private_gpt.components.node_store.context_retrivial_parser import LLMSummaryTransformation
from private_gpt.settings.settings import Settings
from tree_sitter import Language, Parser
import tree_sitter_language_pack as tslp


class CodeStrategy(IngestionStrategy):
    """Ingestion strategy for source code files.

    Uses Tree-Sitter based CodeSplitter to split by functions/classes,
    then applies the same summary + metadata enrichment pipeline.

    Pipeline:
        1. Code splitting (Tree-Sitter, per function/class)
        2. Size limiting (SentenceSplitter as guardrail)
        3. Contextual summary per chunk (LLM)
        4. Metadata enrichment (LLM)
        5. Final embedding for vector store
    """

    # Extension -> Tree-Sitter language identifier
    LANG_MAP: dict[str, str] = {
        ".py": "python",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".cc": "cpp",
        ".cs": "c_sharp",
    }

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        self._size_limiter = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50,
        )

        # --- 3. Contextual Summary ---
        summary_llm = Ollama(
            model=self._settings.ollama.worker_llm,
            api_base=self._settings.ollama.api_base,
            request_timeout=self._settings.ollama.request_timeout,
            temperature=0.3,
        )
        self._summary_transform = LLMSummaryTransformation(
            summary_llm=summary_llm,
            summary_format="natural",
        )

        # --- 4. Metadata Enrichment ---
        metadata_component = MetadataRetrivialComponent(settings=self._settings)
        self._metadata_transform = LLMMetadataTransformation(
            metadata_retrivial_component=metadata_component,
            max_metadata=self._settings.metadata_generation.max_entry_per_category,
        )

    def supported_extensions(self) -> set[str]:
        return set(self.LANG_MAP.keys())

    def get_transformations_per_doc_type(self, extension: str | None = None) -> dict[str, list[TransformComponent]]:
        language = self.LANG_MAP.get(extension or "", "python")

        def _build_parser(language_name: str) -> Parser:
            lang = tslp.get_language(language_name)
            parser = Parser(lang)
            return parser

        code_splitter = CodeSplitter(
            language=language,
            parser=_build_parser(language),
            chunk_lines=40,
            chunk_lines_overlap=15,
            max_chars=1500,
        )

        return {
            "source_code": [
                code_splitter,
                self._size_limiter,
                self._metadata_transform,
            ],
            "summary_of_code": [
                self._size_limiter,
                self._summary_transform,
                self._metadata_transform,
            ],
        }

    def get_reader(self)-> BaseReader:
        return CodeReader()