from typing import List, Optional, Set

import tree_sitter_language_pack as tslp

from llama_index.core.node_parser.interface import TextSplitter
from llama_index.core.bridge.pydantic import PrivateAttr

CONCEPT_MAP = {
    "function": {
        "function_definition",
        "function_declaration",
        "method_definition",
        "method_declaration",
        "function_item",
        "function_expression",
        "arrow_function",
        "lambda",
    },
    "class": {
        "class_definition",
        "class_specifier",
        "struct_specifier",
        "class_declaration",
        "struct_item",
    },
    "enum": {
        "enum_specifier",
        "enum_declaration",
        "enum_item",
    },
    "namespace": {
        "namespace_definition",
        "module",
        "package_declaration",
    },
}

class FunctionSplitter(TextSplitter):
    """Splits code into individual function/method chunks.

    Recursively walks the tree-sitter AST and extracts each function,
    method, enum, etc. as a separate chunk. Language-agnostic via CONCEPT_MAP.

    Args:
        language: tree-sitter language ("python", "cpp", "c", "javascript", ...)
        extract: Which concepts to extract ("function", "enum", "class", "namespace").
                 Default: functions only.
    """

    _language: str = PrivateAttr()
    _comment: str = PrivateAttr()
    _extract_types: Set[str] = PrivateAttr()
    _class_types: Set[str] = PrivateAttr()

    def __init__(
        self,
        language: str,
        extract: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._language = language
        self._comment = "#" if language == "python" else "//"

        concepts = extract or ["function"]
        self._extract_types: Set[str] = set()
        for concept in concepts:
            self._extract_types |= CONCEPT_MAP.get(concept, set())

        self._class_types = CONCEPT_MAP["class"]

    def split_text(self, code: str) -> List[str]:
        """Split a code string into chunks, one per function/method."""
        parser = tslp.get_parser(self._language)
        tree = parser.parse(bytes(code, "utf-8"))
        chunks: List[str] = []
        self.collect(tree.root_node, chunks)
        return chunks

    def collect(self, node, chunks: List[str], class_name: Optional[str] = None):
        # Target concept found -> collect as chunk, don't recurse deeper
        if node.type in self._extract_types:
            prefix = f"{self._comment} class {class_name}\n" if class_name else ""
            chunks.append(prefix + node.text.decode())
            return

        # Header files: method declarations without body (.h / .hpp)
        if node.type == "field_declaration" and self.is_func_decl(node):
            prefix = f"{self._comment} class {class_name}\n" if class_name else ""
            chunks.append(prefix + node.text.decode())
            return

        # Free-standing function declarations in headers (e.g. void foo(int x);)
        if (
            node.type == "declaration"
            and node.parent
            and node.parent.type == "translation_unit"
            and self.is_func_decl(node)
        ):
            chunks.append(node.text.decode())
            return

        # Track class name for method context prefix
        name = class_name
        if node.type in self._class_types:
            name_node = node.child_by_field_name("name")
            if name_node:
                name = name_node.text.decode()

        # Recurse into children
        for child in node.children:
            self.collect(child, chunks, class_name=name)

    @staticmethod
    def is_func_decl(node) -> bool:
        """Check whether a node contains a function declaration (for .h/.hpp)."""
        for child in node.children:
            if child.type == "function_declarator":
                return True
            if FunctionSplitter.is_func_decl(child):
                return True
        return False