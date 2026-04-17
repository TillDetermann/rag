from dataclasses import dataclass
from typing import TYPE_CHECKING

from injector import inject, singleton
from llama_index.core.chat_engine import SimpleChatEngine
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
from llama_index.core.storage import StorageContext
from private_gpt.server.chat.chat_service import ChatEngineInput, Completion, CompletionGen
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chunks.chunks_service import Chunk
from private_gpt.settings.settings import Settings

from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

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
          self.storage_context = StorageContext.from_defaults(
               vector_store=vector_store_component.vector_store,
               docstore=node_store_component.doc_store,
               index_store=node_store_component.index_store,
          )
          self.vector_index = VectorStoreIndex.from_vector_store(
               vector_store_component.vector_store,
               storage_context=self.storage_context,
               llm=llm_component.llm,
               embed_model=embedding_component.embedding_model,
               show_progress=True,
          )

     def _build_tools(self) -> list:
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

          vector_retriever = self.vector_store_component.get_retriever(
               index=self.vector_index,
               context_filter=None,
               similarity_top_k=self.settings.rag.similarity_top_k,
          )
          bm25_retriever = BM25Retriever.from_defaults(
               docstore=self.storage_context.docstore,
               similarity_top_k=1,
          )
          hybrid_retriever = QueryFusionRetriever(
               retrievers=[vector_retriever, bm25_retriever],
               mode="reciprocal_rerank",
               num_queries=1,
               similarity_top_k=self.settings.rag.similarity_top_k,
               llm=self.llm_component.llm,
          )

          from llama_index.core.query_engine import RetrieverQueryEngine
          from llama_index.core.response_synthesizers import get_response_synthesizer

          response_synthesizer = get_response_synthesizer(
               response_mode="no_text"
          )

          query_engine = RetrieverQueryEngine(
               retriever=hybrid_retriever,
               node_postprocessors=node_postprocessors,
               response_synthesizer=response_synthesizer,
          )

          doc_search_tool = QueryEngineTool(
               query_engine=query_engine,
               metadata=ToolMetadata(
                    name="document_search",
                    description=(
                         "Durchsucht die hochgeladenen Dokumente. "
                         "Nutze dieses Tool für alle Fragen zu den "
                         "Dokumenten des Users."
                    ),
               ),
          )

          return [doc_search_tool]

     def _chat_engine(
          self,
          system_prompt: str | None = None,
          use_context: bool = False,
          context_filter: ContextFilter | None = None,
     ) -> BaseChatEngine:
          
          if use_context:
               # ---- AGENT statt ContextChatEngine ----
               tools = self._build_tools()
               
               agent = ReActAgent.from_tools(
                    tools=tools,
                    llm=self.llm_component.llm,
                    verbose=True,
                    system_prompt=system_prompt or (""),
                    max_iterations=10,
               )
               return agent
          else:
               return SimpleChatEngine.from_defaults(
                    system_prompt=system_prompt,
                    llm=self.llm_component.llm,
               )
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
          
          sources = []
          if hasattr(wrapped_response, 'source_nodes'):
               sources = [Chunk.from_node(node) for node in wrapped_response.source_nodes]
          elif hasattr(wrapped_response, 'sources'):
          
               for tool_output in wrapped_response.sources:
                    if hasattr(tool_output, 'raw_output') and hasattr(tool_output.raw_output, 'source_nodes'):
                         sources.extend(
                              Chunk.from_node(node) 
                              for node in tool_output.raw_output.source_nodes
                         )

          return Completion(response=str(wrapped_response), sources=sources)