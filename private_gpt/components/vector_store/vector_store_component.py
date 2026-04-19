import logging
import typing

from injector import inject, singleton
from llama_index.core.indices.vector_store import VectorIndexRetriever, VectorStoreIndex
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
)

from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.paths import local_data_path
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

def _extended_metadata_filter(
    context_filter: ContextFilter | None,
    additional_filters: dict[str, typing.Any] | None = None,
) -> MetadataFilters:
    
    filters = MetadataFilters(filters=[], condition=FilterCondition.OR)

    if context_filter is not None and context_filter.docs_ids is not None:
        doc_id_filters = MetadataFilters(filters=[], condition=FilterCondition.OR)
        for doc_id in context_filter.docs_ids:
            doc_id_filters.filters.append(MetadataFilter(key="doc_id", value=doc_id))
        filters.filters.append(doc_id_filters)

    if additional_filters:
        for key, value in additional_filters.items():
            if value is not None:
                # Handle arrays with OR logic
                if isinstance(value, list) and len(value) > 0:
                    tag_filters = MetadataFilters(filters=[], condition=FilterCondition.OR)
                    for tag in value:
                        tag_filters.filters.append(MetadataFilter(key=key, value=tag))
                    filters.filters.append(tag_filters)
                elif not isinstance(value, list):
                    filters.filters.append(MetadataFilter(key=key, value=value))
    return filters
@singleton
class VectorStoreComponent:
    settings: Settings
    vector_store_text: BasePydanticVectorStore
    vector_store_code: BasePydanticVectorStore

    @inject
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        match settings.vectorstore.database:
            case "postgres":
                try:
                    from llama_index.vector_stores.postgres import (  # type: ignore
                        PGVectorStore,
                    )
                except ImportError as e:
                    raise ImportError(
                        "Postgres dependencies not found, install with `poetry install --extras vector-stores-postgres`"
                    ) from e

                if settings.postgres is None:
                    raise ValueError(
                        "Postgres settings not found. Please provide settings."
                    )

                self.vector_store_text = typing.cast(
                    BasePydanticVectorStore,
                    PGVectorStore.from_params(
                        **settings.postgres.model_dump(exclude_none=True),
                        table_name="embeddings_text",
                        embed_dim=settings.embedding.embed_dim_text,
                    ),
                )


                self.vector_store_code = typing.cast(
                    BasePydanticVectorStore,
                    PGVectorStore.from_params(
                        **settings.postgres.model_dump(exclude_none=True),
                        table_name="embeddings_code",
                        embed_dim=settings.embedding.embed_dim_code,
                    ),
                )

    def get_retriever(
        self,
        index: VectorStoreIndex,
        context_filter: ContextFilter | None = None,
        similarity_top_k: int = 2,
        additional_filters: dict[str, typing.Any] | None = None,
    ) -> VectorIndexRetriever:
        
        return VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
            filters=_extended_metadata_filter(context_filter, additional_filters),
        )

    def close(self) -> None:
        if hasattr(self.vector_store_text.client, "close"):
            self.vector_store_text.client.close()
        if hasattr(self.vector_store_code.client, "close"):
            self.vector_store_code.client.close()