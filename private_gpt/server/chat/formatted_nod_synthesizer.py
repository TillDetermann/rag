
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.types import RESPONSE_TEXT_TYPE

class FormattedNodeSynthesizer(BaseSynthesizer):
    """Gibt die Nodes als formatierten Text zurück, ohne LLM-Call."""

    def _get_prompts(self) -> dict:
        return {}

    def _update_prompts(self, prompts: dict) -> None:
        pass

    def get_response(
        self,
        query_str: str,
        text_chunks: list[str],
        **kwargs,
    ) -> RESPONSE_TEXT_TYPE:
        if not text_chunks:
            return "No relevant documents found."

        formatted = []
        for i, chunk in enumerate(text_chunks, 1):
            formatted.append(f"[Source {i}]: {chunk}")

        return "\n\n".join(formatted)

    async def aget_response(self, query_str: str, text_chunks: list[str], **kwargs):
        return self.get_response(query_str, text_chunks, **kwargs)