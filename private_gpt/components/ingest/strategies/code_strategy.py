from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TransformComponent
from llama_index.core.readers.base import BaseReader

from private_gpt.components.ingest.custom_splitter.function_splitter import FunctionSplitter
from private_gpt.components.ingest.ingest_strategy import IngestionStrategy
from private_gpt.components.ingest.custom_node_parser.add_summary_parser import AddSummaryParser
from private_gpt.components.ingest.custom_node_parser.code_enrichment_parser import CodeEnrichmentParser
from private_gpt.settings.settings import Settings
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.storage import StorageContext
from llama_index.readers.file import FlatReader


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

    # Extension to Tree-Sitter language identifier
    LANG_MAP: dict[str, str] = {
        ".py": "python",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".cc": "cpp",
        ".cs": "c_sharp",
    }

    def __init__(self, settings: Settings, embed_model: EmbedType,  storage_context: StorageContext) -> None:
        self.settings = settings

        self.code_enrichment_parser = CodeEnrichmentParser(storage_context=storage_context, embed_model=embed_model)

        self.summary_transform = AddSummaryParser()

        self.size_limiter = SentenceSplitter(
            chunk_size=1000,
            chunk_overlap=50,
        )

    def supported_extensions(self) -> set[str]:
        return set(self.LANG_MAP.keys())

    def get_transformations_per_doc_type(self, extension: str | None = None) -> dict[str, list[TransformComponent]]:
        language = self.LANG_MAP.get(extension or "", "python")
        code_splitter = FunctionSplitter(
            language=language
        )
        return {
            "code": [
                code_splitter,
                self.code_enrichment_parser,
                self.size_limiter,
                self.summary_transform
            ],
        }
    
    def get_reader(self)-> BaseReader:
        return FlatReader()