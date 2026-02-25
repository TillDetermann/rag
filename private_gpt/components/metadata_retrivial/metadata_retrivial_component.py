import json
import logging
from typing import Optional
from injector import inject, singleton
from llama_index.llms.ollama import Ollama
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class MetadataRetrivialComponent:
    """Manages predefined and custom tags with LLM generation."""
    
    settings: Settings
    predefined_tags: list[str]
    custom_tags: set[str]
    
    @inject
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.predefined_tags = settings.metadata_generation.predefined if settings.metadata_generation else []
    
    def get_all_tags(self) -> list[str]:
        """Returns all available tags (predefined + custom)"""
        return self.predefined_tags
    def generate_chunk_metadata(self, chunk_text: str, document_title: str = "", max_tags: int = 2) -> dict:
        """
        Generate comprehensive metadata for a given chunk.
        Args:
            chunk_text: The text content of the chunk
            document_title: Optional document title for context
            max_tags: Maximum number of tags per category
            
        Returns:
            Dictionary with metadata tags
        """
        if not self.settings.metadata_generation.enable:
            return {}

        tag_llm = Ollama(
            model="qwen2.5:0.5b",
            request_timeout=30.0,
            temperature=0.3,
        )
        
        available_tags = self.get_all_tags()
        
        categories = "\n".join(f"   {key}: {', '.join(available_tags[key])}" for key in available_tags.keys())
        json_structure = ",\n".join(f'  "{key}": []' for key in available_tags.keys())

        metadata_prompt = f"""Analyze the document chunk and extract metadata.

        RULES:
        1. ONLY use tags from these lists. Do NOT create or invent new tags:
        {categories}
        2. ONLY return valid JSON.
        3. Maximum {max_tags} tags per category.

        Document: {document_title}
        Text: {chunk_text[:1000]}

        Return ONLY this JSON:
        {{
        {json_structure}
        }}"""
        try:
            response = tag_llm.complete(metadata_prompt)
            content = response.text.strip().replace("```json", "").replace("```", "").strip()
            metadata = json.loads(content)
            
            validated_metadata = {}
            for key in available_tags.keys():
                tags = metadata.get(key, [])
                validated_metadata[key] = self._validate_tags(
                    tags, 
                    available_tags[key], 
                    max_tags
                )
            
            logger.debug(f"Generated chunk metadata: {validated_metadata}")
            return validated_metadata
            
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            logger.warning(f"Failed to generate chunk metadata: {e}")
            return {key: [] for key in available_tags.keys()}

    def _validate_tags(self, tags: list, available: list, max_count: int = None) -> list:
            """Validate tags against available list"""
            if not isinstance(tags, list):
                return []
            validated = [t for t in tags if isinstance(t, str) and t in available]
            return validated[:max_count] if max_count else validated