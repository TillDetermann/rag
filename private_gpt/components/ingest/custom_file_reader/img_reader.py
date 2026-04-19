import logging
import base64
import requests
from pathlib import Path
from typing import List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
import logging
from private_gpt.settings.settings import settings
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import ImageDocument
logger = logging.getLogger(__name__)


class ImageReader(BaseReader):
     def __init__(
          self
     ) -> None:
          self.settings = settings()
          self.vision_llm = Ollama(
            model=self.settings.ollama.vision_llm,
            api_base=self.settings.ollama.api_base,
            request_timeout=self.settings.ollama.request_timeout,
            temperature=0.3,
          )

     def load_data(
        self,
        file: Path,
        extra_info: Optional[dict] = None,
     ) -> List[Document]:
          file = Path(file)

          if not file.exists():
               logger.error(f"File not found: {file}")
               return []

          logger.info(f"Processing image: {file.name}")

          try:
               image_doc = ImageDocument(image_path=str(file.resolve()))
               promp = """You are a mechanical engineering assistant analyzing a 2D rendering of a 3D CAD model.

Describe this image systematically:

1. OVERALL SHAPE: What type of component is this? (e.g., bracket, housing, shaft, gear, connector, frame, enclosure). Estimate overall proportions and approximate dimensions relative to each other.

2. GEOMETRIC FEATURES: Identify all visible machining and design features such as holes, slots, pockets, chamfers, fillets, ribs, bosses, threads, flanges, steps, grooves, and curved surfaces.

3. STRUCTURE: Describe how the parts relate spatially. Is it a single solid body or an assembly of multiple components? Note any symmetry, patterns, or repeated elements.

4. FUNCTIONAL ANALYSIS: Based on the geometry, what is the likely mechanical function? How might it connect to other parts? Identify mounting points, load-bearing surfaces, or motion interfaces.

5. MANUFACTURING: What manufacturing processes would likely produce this part? (e.g., CNC milling, turning, casting, sheet metal bending, 3D printing, injection molding).

Be precise and technical. Use standard engineering terminology. Do not speculate beyond what is visible."""
               response = self.vision_llm.complete(
                    prompt=promp,
                    image_documents=[image_doc],
               )
               description = response.text
          except Exception as e:
               logger.error(f"Error processing {file.name}: {e}")
               return []

          if not description.strip():
               return []

          description_document = Document(
               text=description,
               metadata={
                    "file_name": file.name,
                    "file_path": str(file.resolve()),
                    "source_type": "image",
                    "doc_type": "text",
               },
          )
          if extra_info:
               description_document.metadata.update(extra_info)
          return [description_document]
