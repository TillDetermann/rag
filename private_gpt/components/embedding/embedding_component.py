import logging

from injector import inject, singleton
from llama_index.core.embeddings import BaseEmbedding

from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class EmbeddingComponent:
    embedding_model_text: BaseEmbedding
    embedding_model_code: BaseEmbedding

    @inject
    def __init__(self, settings: Settings) -> None:
        embedding_mode = settings.embedding.mode
        logger.info("Initializing the embedding model in mode=%s", embedding_mode)
        match embedding_mode:
            case "ollama":
                try:
                    from llama_index.embeddings.ollama import (  # type: ignore
                        OllamaEmbedding,
                    )
                    from ollama import Client  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "Local dependencies not found, install with `poetry install --extras embeddings-ollama`"
                    ) from e

                ollama_settings = settings.ollama

                # Calculate embedding model. If not provided tag, it will be use latest
                model_name_text = (
                    ollama_settings.embedding_model_text + ":latest"
                    if ":" not in  ollama_settings.embedding_model_text
                    else  ollama_settings.embedding_model_text
                )

                self.embedding_model_text = OllamaEmbedding(
                    model_name=model_name_text,
                    base_url=ollama_settings.embedding_api_base,
                )

                # Calculate embedding model. If not provided tag, it will be use latest
                model_name_code = (
                    ollama_settings.embedding_model_code + ":latest"
                    if ":" not in  ollama_settings.embedding_model_code
                    else  ollama_settings.embedding_model_code
                )

                self.embedding_model_code = OllamaEmbedding(
                    model_name=model_name_code,
                    base_url=ollama_settings.embedding_api_base,
                )

                if ollama_settings.autopull_models:
                    if ollama_settings.autopull_models:
                        from private_gpt.utils.ollama import (
                            check_connection,
                            pull_model,
                        )

                        # TODO: Reuse llama-index client when llama-index is updated
                        client = Client(
                            host=ollama_settings.embedding_api_base,
                            timeout=ollama_settings.request_timeout,
                        )

                        if not check_connection(client):
                            raise ValueError(
                                f"Failed to connect to Ollama, "
                                f"check if Ollama server is running on {ollama_settings.api_base}"
                            )
                        pull_model(client, model_name_text)
                        pull_model(client, model_name_code)
