from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.schema import TransformComponent
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.readers.base import BaseReader

from private_gpt.components.ingest.custom_file_reader.img_reader import ImageReader
from private_gpt.components.ingest.ingest_strategy import IngestionStrategy
from private_gpt.components.metadata_retrivial.metadata_retrivial_component import MetadataRetrivialComponent
from private_gpt.components.metadata_retrivial.metadata_retrivial_parser import LLMMetadataTransformation
from private_gpt.components.ingest.custom_node_parser.add_summary_parser import AddSummaryParser
from private_gpt.settings.settings import Settings

class ImageStrategy(IngestionStrategy):
    """Ingestion strategy for cad documents.
    """

    EXTENSIONS: set[str] = {".png", ".jpg", "jpeg"}

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        # --- 1. Semantic Splitter ---
        chunking_embed = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            cache_folder="./models/embedding_cache",
        )
    
        self.semantic_splitter = SemanticSplitterNodeParser.from_defaults(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=chunking_embed,
        )

        # --- 2. Size Limiter ---
        self.size_limiter = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50,
        )

        # --- 3. Contextual Summary ---
        self.summary_transform = AddSummaryParser()

        # --- 4. Metadata Enrichment ---
        metadata_component = MetadataRetrivialComponent(settings=self.settings)
        self.metadata_transform = LLMMetadataTransformation(
            metadata_retrivial_component=metadata_component,
            max_metadata=self.settings.metadata_generation.max_entry_per_category,
        )
    def supported_extensions(self) -> set[str]:
        return self.EXTENSIONS

    def get_transformations_per_doc_type(self, extension: str | None = None) -> dict[str, list[TransformComponent]]:
        return {
            "image-summary": [
            self.semantic_splitter,
            self.size_limiter,
            self.summary_transform,
            self.metadata_transform
            ],
        }
    
    def get_reader(self)-> BaseReader:
        return ImageReader()