from typing import TYPE_CHECKING

from injector import inject, singleton
from llama_index.core.chat_engine import ContextChatEngine, SimpleChatEngine
from llama_index.core.chat_engine.types import (
    BaseChatEngine,
)
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.llms import ChatMessage
from llama_index.core.postprocessor import (
    SentenceTransformerRerank,
    SimilarityPostprocessor,
)
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.storage import StorageContext

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chat.chat_utils import ChatEngineInput, Completion, CompletionGen
from private_gpt.server.utils.chunk import Chunk
from private_gpt.settings.settings import Settings

import logging

if TYPE_CHECKING:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor

logger = logging.getLogger(__name__)


@singleton
class ChatService:
     settings: Settings

     @inject
     def __init__(
          self,
          settings: Settings,
          llm_component: LLMComponent,
          vector_store_component: VectorStoreComponent,
          embedding_component: EmbeddingComponent,
          node_store_component: NodeStoreComponent,
     ) -> None:
          self.settings = settings
          self.llm_component = llm_component
          self.embedding_component = embedding_component
          self.vector_store_component = vector_store_component

          # --- Text Index ---
          self.storage_context_text = StorageContext.from_defaults(
               vector_store=vector_store_component.vector_store_text,
               docstore=node_store_component.doc_store,
               index_store=node_store_component.index_store,
          )
          self.vector_index_text = VectorStoreIndex.from_vector_store(
               vector_store_component.vector_store_text,
               storage_context=self.storage_context_text,
               llm=llm_component.llm,
               embed_model=embedding_component.embedding_model_text,
               show_progress=True,
          )

          # --- Code Index ---
          self.storage_context_code = StorageContext.from_defaults(
               vector_store=vector_store_component.vector_store_code,
               docstore=node_store_component.doc_store,
               index_store=node_store_component.index_store,
          )
          self.vector_index_code = VectorStoreIndex.from_vector_store(
               vector_store_component.vector_store_code,
               storage_context=self.storage_context_code,
               llm=llm_component.llm,
               embed_model=embedding_component.embedding_model_code,
               show_progress=True,
          )

     def _build_node_postprocessors(self) -> list["BaseNodePostprocessor"]:
          settings = self.settings
          node_postprocessors: list[BaseNodePostprocessor] = [
               MetadataReplacementPostProcessor(target_metadata_key="window"),
          ]
          if settings.rag.similarity_value:
               node_postprocessors.append(
                    SimilarityPostprocessor(
                         similarity_cutoff=settings.rag.similarity_value
                    )
               )
          if settings.rag.rerank.enabled:
               node_postprocessors.append(
                    SentenceTransformerRerank(
                         model=settings.rag.rerank.model,
                         top_n=settings.rag.rerank.top_n,
                    )
               )
          return node_postprocessors

     def _chat_engine(
          self,
          system_prompt: str | None = None,
          use_context: bool = False,
          context_filter: ContextFilter | None = None,
     ) -> BaseChatEngine:

          if use_context:
               # Build a retriever for the text index
               text_retriever = self.vector_store_component.get_retriever(
                    index=self.vector_index_text,
                    context_filter=context_filter,
                    similarity_top_k=self.settings.rag.similarity_top_k,
               )

               # Build a retriever for the code index
               code_retriever = self.vector_store_component.get_retriever(
                    index=self.vector_index_code,
                    context_filter=context_filter,
                    similarity_top_k=self.settings.rag.similarity_top_k,
               )

               # Combine both retrievers into a hybrid retriever using reciprocal
               # rank fusion to merge and deduplicate results from both indexes.
               hybrid_retriever = QueryFusionRetriever(
                    retrievers=[text_retriever, code_retriever],
                    llm=self.llm_component.llm,
                    similarity_top_k=self.settings.rag.similarity_top_k,
                    num_queries=1,  # Don't generate additional queries, just fuse results
                    mode="reciprocal_rerank",
                    use_async=False,
               )

               return ContextChatEngine.from_defaults(
                    system_prompt=system_prompt,
                    retriever=hybrid_retriever,
                    llm=self.llm_component.llm,
                    node_postprocessors=self._build_node_postprocessors(),
               )
          else:
               return SimpleChatEngine.from_defaults(
                    system_prompt=system_prompt,
                    llm=self.llm_component.llm,
               )

     def stream_chat(
          self,
          messages: list[ChatMessage],
          use_context: bool = False,
          context_filter: ContextFilter | None = None,
     ) -> CompletionGen:
          chat_engine_input = ChatEngineInput.from_messages(messages)
          last_message = (
               chat_engine_input.last_message.content
               if chat_engine_input.last_message
               else None
          )
          system_prompt = (
               chat_engine_input.system_message.content
               if chat_engine_input.system_message
               else None
          )
          chat_history = (
               chat_engine_input.chat_history if chat_engine_input.chat_history else None
          )

          chat_engine = self._chat_engine(
               system_prompt=system_prompt,
               use_context=use_context,
               context_filter=context_filter,
          )
          streaming_response = chat_engine.stream_chat(
               message=last_message if last_message is not None else "",
               chat_history=chat_history,
          )
          sources = [Chunk.from_node(node) for node in streaming_response.source_nodes]
          completion_gen = CompletionGen(
               response=streaming_response.response_gen, sources=sources
          )
          return completion_gen

     def chat(
          self,
          messages: list[ChatMessage],
          use_context: bool = False,
          context_filter: ContextFilter | None = None,
     ) -> Completion:
          chat_engine_input = ChatEngineInput.from_messages(messages)
          last_message = (
               chat_engine_input.last_message.content
               if chat_engine_input.last_message
               else None
          )
          system_prompt = (
               chat_engine_input.system_message.content
               if chat_engine_input.system_message
               else None
          )
          chat_history = (
               chat_engine_input.chat_history if chat_engine_input.chat_history else None
          )

          chat_engine = self._chat_engine(
               system_prompt=system_prompt,
               use_context=use_context,
               context_filter=context_filter,
          )
          wrapped_response = chat_engine.chat(
               message=last_message if last_message is not None else "",
               chat_history=chat_history,
          )
          sources = [Chunk.from_node(node) for node in wrapped_response.source_nodes]
          completion = Completion(response=wrapped_response.response, sources=sources)
          return completion