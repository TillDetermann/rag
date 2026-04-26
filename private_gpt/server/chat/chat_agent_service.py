
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
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.schema import QueryBundle

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

from llama_index.retrievers.bm25 import BM25Retriever
import logging

if TYPE_CHECKING:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor

logger = logging.getLogger(__name__)



@singleton
class ChatAgentService:
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

     def _build_tools(self) -> list:
          node_postprocessors = self._build_node_postprocessors()

          # --- BM25 Keyword Search ---
          bm25_retriever = BM25Retriever.from_defaults(
               docstore=self.storage_context_text.docstore,
               similarity_top_k=self.settings.rag.similarity_top_k,
          )

          def _postprocess(nodes, query_str):
               for pp in node_postprocessors:
                    nodes = pp.postprocess_nodes(
                         nodes, query_bundle=QueryBundle(query_str=query_str)
                    )
               return nodes

          def _format_nodes(nodes):
               if not nodes:
                    return "No relevant documents found."
               parts = []
               for i, node in enumerate(nodes, 1):
                    text = node.get_content()
                    file_name = node.metadata.get("file_name", "unknown")
                    score = f" (score: {node.score:.2f})" if node.score else ""
                    parts.append(f"[Source {i} - {file_name}{score}]: {text}")
               return "\n\n".join(parts)

          def document_keyword_search(query: str) -> str:
               """Keyword search in text documents."""
               nodes = bm25_retriever.retrieve(query)
               nodes = _postprocess(nodes, query)
               return _format_nodes(nodes)

          # --- Vector Semantic Search ---
          vector_retriever_text = self.vector_store_component.get_retriever(
               index=self.vector_index_text,
               context_filter=None,
               similarity_top_k=self.settings.rag.similarity_top_k,
          )

          def document_semantic_search(query: str) -> str:
               """Semantic search in text documents."""
               nodes = vector_retriever_text.retrieve(query)
               nodes = _postprocess(nodes, query)
               return _format_nodes(nodes)

          # --- Code Search ---
          vector_retriever_code = self.vector_store_component.get_retriever(
               index=self.vector_index_code,
               context_filter=None,
               similarity_top_k=self.settings.rag.similarity_top_k,
          )

          def code_search(query: str) -> str:
               """Search in code files."""
               nodes = vector_retriever_code.retrieve(query)
               nodes = _postprocess(nodes, query)
               return _format_nodes(nodes)

          return [
               FunctionTool.from_defaults(
                    fn=document_keyword_search,
                    name="document_keyword_search",
                    description=(
                         "Keyword and exact-term search over the user's uploaded documents. "
                         "Best for exact phrases, specific terminology, names, identifiers, "
                         "error messages, section titles, and literal text matches."
                    ),
               ),
               FunctionTool.from_defaults(
                    fn=document_semantic_search,
                    name="document_semantic_search",
                    description=(
                         "Semantic search over the user's uploaded documents. "
                         "Best for meaning-based questions, summaries, explanations, "
                         "and finding relevant passages even when the exact wording differs."
                    ),
               ),
               FunctionTool.from_defaults(
                    fn=code_search,
                    name="code_search",
                    description=(
                         "Searches the user's uploaded source code files. "
                         "Use for questions about functions, classes, methods, APIs, "
                         "implementations, logic, bugs, and software engineering details."
                    ),
               ),
          ]
     def _agent_engine(
     self,
     system_prompt: str | None = None,
     use_context: bool = False,
     context_filter: ContextFilter | None = None,
     user_prompt: str | None = None,
     ) -> BaseChatEngine:
     
          if use_context:
               # 1. Retrieval
               retriever = self.vector_store_component.get_retriever(
                    index=self.vector_index_text,
                    context_filter=context_filter,
                    similarity_top_k=2,
               )
               nodes = retriever.retrieve(user_prompt or "")
               
               # Postprocessing (Rerank ...)
               for pp in self._build_node_postprocessors():
                    nodes = pp.postprocess_nodes(
                         nodes, query_bundle=QueryBundle(query_str=user_prompt or "")
                    )
               
               # 2. Merge conext and system prompt
               context_str = self._format_nodes(nodes)
               augmented_prompt = (
                    f"{system_prompt or ''}\n\n"
                    f"## Retrieved Context\n{context_str}\n\n"
                    "Use the above context to answer the user's question. "
                    "If the context is not relevant, say so."
               )
               
               # 3. Build agent
               tools = self._build_tools()
               agent = FunctionCallingAgent.from_tools(
                    tools=tools,
                    llm=self.llm_component.llm,
                    system_prompt=augmented_prompt,
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
          agent_engine = self._agent_engine(
               system_prompt=system_prompt,
               use_context=use_context,
               context_filter=context_filter,
               user_prompt=last_message
          )
          
          wrapped_response = agent_engine.chat(
               message=last_message if last_message is not None else "",
               chat_history=chat_history,
          )
          
          sources = []
          if hasattr(wrapped_response, 'source_nodes'):
               sources = [Chunk.from_node(node) for node in wrapped_response.source_nodes]
          elif hasattr(wrapped_response, 'sources'):
               for tool_output in wrapped_response.sources:
                    if hasattr(tool_output, 'raw_output'):
                         raw = tool_output.raw_output
                         if hasattr(raw, 'source_nodes'):
                              sources.extend(
                                   Chunk.from_node(node)
                                   for node in raw.source_nodes
                              )
                         elif isinstance(raw, str):
                              pass

          return Completion(response=str(wrapped_response), sources=sources)
     

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
          agent_engine = self._agent_engine(
               system_prompt=system_prompt,
               use_context=use_context,
               context_filter=context_filter,
               user_prompt=last_message,
          )

          response = agent_engine.chat(
               message=last_message if last_message is not None else "",
               chat_history=chat_history,
          )

          def _token_gen():
               yield response.response

          return CompletionGen(
               response=_token_gen(),
               sources=[],
          )