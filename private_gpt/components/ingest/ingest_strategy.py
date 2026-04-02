from abc import ABC, abstractmethod
from pathlib import Path

from llama_index.core.schema import TransformComponent
from llama_index.core.readers.base import BaseReader


class IngestionStrategy(ABC):
     """Base class for file-type-specific ingestion strategies."""

     @abstractmethod
     def supported_extensions(self) -> set[str]:
          """Return the set of file extensions this strategy handles (e.g. {'.pdf', '.docx'})."""
          ...

     @abstractmethod
     def get_transformations_per_doc_type(self, extension: str | None = None) -> dict[str, list[TransformComponent]]:
          """Return the ordered transformation pipeline for the given file type per docuent type."""
          ...

     @abstractmethod
     def get_reader(self) -> BaseReader:
          """Return the reader for the given file type."""
          ...


class IngestionStrategyRegistry:
     """Registry that maps file extensions to their ingestion strategy."""

     def __init__(self) -> None:
          self._strategies: list[IngestionStrategy] = []
          self._default: IngestionStrategy | None = None

     def register(self, strategy: IngestionStrategy, *, default: bool = False) -> None:
          """Register a strategy. Optionally mark it as the fallback default."""
          self._strategies.append(strategy)
          if default:
               self._default = strategy

     def get_strategy(self, file_name: str) -> IngestionStrategy:
          """Look up the strategy for a given filename.

          :raises ValueError: if no strategy matches and no default is set.
          """
          ext = Path(file_name).suffix.lower()
          for strategy in self._strategies:
               if ext in strategy.supported_extensions():
                    return strategy

          if self._default is not None:
               return self._default

          registered = {ext for s in self._strategies for ext in s.supported_extensions()}
          raise ValueError(
               f"No ingestion strategy registered for extension '{ext}'. "
               f"Registered extensions: {sorted(registered)}"
          )