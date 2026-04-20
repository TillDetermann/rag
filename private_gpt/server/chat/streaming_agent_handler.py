from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from typing import Any
import queue


class StreamingAgentHandler(BaseCallbackHandler):
    """Streaming thinking process"""

    def __init__(self) -> None:
        super().__init__([], [])
        self.queue: queue.Queue[str | None] = queue.Queue()

    def on_event_start(self, event_type: str, payload: dict | None = None, **kwargs: Any) -> None:
        if event_type == "llm" and payload:
            pass

    def on_event_end(self, event_type: str, payload: dict | None = None, **kwargs: Any) -> None:
        if event_type == "function_call" and payload:
            tool_name = payload.get("tool", {}).get("name", "unknown")
            self.queue.put(f"\n🔧 Tool: {tool_name}\n")
        elif event_type == "function_call_response" and payload:
            self.queue.put("✅ Got result\n")

    def start_trace(self, trace_id: str | None = None) -> None:
        pass

    def end_trace(self, trace_id: str | None = None, trace_map: dict | None = None) -> None:
        self.queue.put(None) 