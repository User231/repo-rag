"""AST-aware code chunking (tree-sitter) with text fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Chunk dataclass ──────────────────────────────────────────────────────


@dataclass
class Chunk:
    content: str
    chunk_type: str  # "function", "class", "method", "interface", "block", "text"
    symbol_name: str | None = None  # function/class name or None
    start_line: int = 0
    end_line: int = 0
    language: str | None = None


# ── Language mappings ────────────────────────────────────────────────────

# File extension -> tree-sitter language name
EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".kt": "kotlin",
    ".cs": "c_sharp",
    ".fs": "c_sharp",  # F# not well-supported; fallback to text
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".ex": "elixir",
    ".exs": "elixir",
    ".md": "markdown",
    ".mdx": "markdown",
    ".sh": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".sql": "sql",
}

# Language -> node types to extract as top-level semantic chunks
EXTRACTABLE_NODES: dict[str, dict[str, str]] = {
    # language: {node_type: chunk_type_label}
    "python": {
        "function_definition": "function",
        "class_definition": "class",
    },
    "javascript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "export_statement": "function",  # often wraps function/class
    },
    "typescript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "interface_declaration": "interface",
        "type_alias_declaration": "interface",
        "export_statement": "function",
    },
    "java": {
        "class_declaration": "class",
        "method_declaration": "method",
        "interface_declaration": "interface",
    },
    "kotlin": {
        "class_declaration": "class",
        "function_declaration": "function",
    },
    "c_sharp": {
        "class_declaration": "class",
        "method_declaration": "method",
        "interface_declaration": "interface",
        "record_declaration": "class",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "interface",
    },
    "rust": {
        "function_item": "function",
        "impl_item": "class",
        "struct_item": "class",
        "enum_item": "class",
        "trait_item": "interface",
    },
    "elixir": {
        "call": "function",  # defmodule, def, defp are all "call" nodes
    },
    "php": {
        "function_definition": "function",
        "class_declaration": "class",
        "method_declaration": "method",
    },
    "ruby": {
        "method": "method",
        "class": "class",
        "module": "class",
    },
}

# Languages where we don't do AST chunking (just text fallback)
_TEXT_ONLY_LANGUAGES = {"markdown", "yaml", "toml", "json", "sql", "bash"}


# ── Public API ───────────────────────────────────────────────────────────


def detect_language(file_path: str) -> str | None:
    """Detect language from file extension."""
    from pathlib import Path

    suffix = Path(file_path).suffix.lower()
    return EXTENSION_MAP.get(suffix)


def is_code_file(file_path: str) -> bool:
    """Return True if the file is a code file (not docs/config)."""
    lang = detect_language(file_path)
    if lang is None:
        return False
    return lang not in _TEXT_ONLY_LANGUAGES


def chunk_file(
    content: str,
    file_path: str,
    language: str | None = None,
    max_chunk_size: int = 1500,
    overlap: int = 200,
) -> list[Chunk]:
    """Chunk a file using tree-sitter AST when possible, else text fallback.

    Args:
        content: File content as string.
        file_path: Relative file path (used for context header and language detection).
        language: Override language detection.
        max_chunk_size: Max characters per chunk.
        overlap: Character overlap between text chunks (only for text fallback).

    Returns:
        List of Chunk objects.
    """
    if not content.strip():
        return []

    lang = language or detect_language(file_path)

    # For text-only formats or unknown languages, use text chunking
    if lang is None or lang in _TEXT_ONLY_LANGUAGES:
        return chunk_text(content, chunk_size=max_chunk_size, overlap=overlap, language=lang)

    # Try AST-aware chunking
    if lang in EXTRACTABLE_NODES:
        try:
            chunks = _ast_chunk(content, file_path, lang, max_chunk_size)
            if chunks:
                return chunks
        except Exception as e:
            logger.debug("Tree-sitter failed for %s (%s): %s. Falling back.", file_path, lang, e)

    # Fallback: code-aware text chunking
    return _code_text_chunk(content, file_path, lang, max_chunk_size, overlap)


# ── AST chunking ─────────────────────────────────────────────────────────


def _ast_chunk(
    content: str,
    file_path: str,
    language: str,
    max_chunk_size: int,
) -> list[Chunk]:
    """Walk tree-sitter AST and extract semantic chunks."""
    from tree_sitter_languages import get_parser

    parser = get_parser(language)
    source_bytes = content.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    node_types = EXTRACTABLE_NODES[language]
    chunks: list[Chunk] = []
    block_buffer: list[str] = []
    block_start_line: int = 0

    file_header = f"// File: {file_path}\n"

    def flush_block() -> None:
        """Flush accumulated non-extractable code as a 'block' chunk."""
        nonlocal block_buffer, block_start_line
        if not block_buffer:
            return
        block_text = "\n".join(block_buffer).strip()
        if len(block_text) > 50:  # Skip trivially small gaps
            chunks.append(Chunk(
                content=file_header + block_text,
                chunk_type="block",
                symbol_name=None,
                start_line=block_start_line,
                end_line=block_start_line + len(block_buffer),
                language=language,
            ))
        block_buffer = []

    for child in root.children:
        child_text = source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")

        if child.type in node_types:
            flush_block()
            chunk_type = node_types[child.type]
            name = _extract_name(child, source_bytes)

            if len(child_text) <= max_chunk_size:
                chunks.append(Chunk(
                    content=file_header + child_text,
                    chunk_type=chunk_type,
                    symbol_name=name,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    language=language,
                ))
            else:
                # Split oversized node at nested extractable children or lines
                sub_chunks = _split_large_node(
                    child, source_bytes, file_path, language, node_types, max_chunk_size,
                )
                chunks.extend(sub_chunks)
        else:
            if not block_buffer:
                block_start_line = child.start_point[0] + 1
            block_text_lines = child_text.split("\n")
            block_buffer.extend(block_text_lines)

            # Flush block if it gets too large
            joined = "\n".join(block_buffer)
            if len(joined) > max_chunk_size:
                flush_block()

    flush_block()
    return chunks


def _extract_name(node, source_bytes: bytes) -> str | None:
    """Extract the symbol name from a tree-sitter node."""
    # Most grammars use a 'name' field
    name_node = node.child_by_field_name("name")
    if name_node:
        return source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")

    # Kotlin uses simple_identifier as child
    for child in node.children:
        if child.type in ("identifier", "simple_identifier", "property_identifier"):
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")

    return None


def _split_large_node(
    node,
    source_bytes: bytes,
    file_path: str,
    language: str,
    node_types: dict[str, str],
    max_chunk_size: int,
) -> list[Chunk]:
    """Split a node exceeding max_chunk_size into sub-chunks."""
    file_header = f"// File: {file_path}\n"
    parent_name = _extract_name(node, source_bytes)
    parent_type = node_types.get(node.type, "block")

    # Try splitting at nested extractable children (e.g., methods inside a class)
    sub_chunks: list[Chunk] = []
    has_extractable_children = False

    for child in node.children:
        if child.type in node_types:
            has_extractable_children = True
            child_text = source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
            name = _extract_name(child, source_bytes)
            sub_chunks.append(Chunk(
                content=file_header + child_text,
                chunk_type=node_types[child.type],
                symbol_name=name,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                language=language,
            ))

    if has_extractable_children:
        return sub_chunks

    # Last resort: split the node text by lines
    full_text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    lines = full_text.split("\n")
    current_lines: list[str] = []
    chunks: list[Chunk] = []
    chunk_start_line = node.start_point[0] + 1

    for i, line in enumerate(lines):
        current_lines.append(line)
        if len("\n".join(current_lines)) >= max_chunk_size:
            chunks.append(Chunk(
                content=file_header + "\n".join(current_lines),
                chunk_type=parent_type,
                symbol_name=parent_name,
                start_line=chunk_start_line,
                end_line=node.start_point[0] + 1 + i,
                language=language,
            ))
            current_lines = []
            chunk_start_line = node.start_point[0] + 2 + i

    if current_lines and len("\n".join(current_lines).strip()) > 50:
        chunks.append(Chunk(
            content=file_header + "\n".join(current_lines),
            chunk_type=parent_type,
            symbol_name=parent_name,
            start_line=chunk_start_line,
            end_line=node.end_point[0] + 1,
            language=language,
        ))

    return chunks


# ── Text-based chunking (fallback) ──────────────────────────────────────


def _code_text_chunk(
    content: str,
    file_path: str,
    language: str,
    max_chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Code-aware text chunking: split on double blank lines, then single."""
    file_header = f"// File: {file_path}\n"

    # Try splitting on double blank lines first (class/function boundaries)
    parts = content.split("\n\n\n")
    if len(parts) < 2:
        parts = content.split("\n\n")

    chunks: list[Chunk] = []
    current = ""
    line_offset = 0

    for part in parts:
        if current and len(current) + len(part) + 2 > max_chunk_size:
            chunks.append(Chunk(
                content=file_header + current.strip(),
                chunk_type="block",
                symbol_name=None,
                start_line=line_offset + 1,
                end_line=line_offset + current.count("\n") + 1,
                language=language,
            ))
            # Keep overlap from end of previous chunk
            overlap_text = current[-overlap:] if overlap else ""
            line_offset += current.count("\n") - overlap_text.count("\n")
            current = overlap_text + "\n\n" + part
        else:
            if current:
                current += "\n\n" + part
            else:
                current = part

    if current.strip():
        chunks.append(Chunk(
            content=file_header + current.strip(),
            chunk_type="block",
            symbol_name=None,
            start_line=line_offset + 1,
            end_line=line_offset + current.count("\n") + 1,
            language=language,
        ))

    return chunks


def chunk_text(
    content: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    language: str | None = None,
) -> list[Chunk]:
    """Paragraph-aware text chunking for markdown/docs."""
    paragraphs = content.split("\n\n")
    chunks: list[Chunk] = []
    current = ""
    line_offset = 0

    for para in paragraphs:
        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(Chunk(
                content=current.strip(),
                chunk_type="text",
                symbol_name=None,
                start_line=line_offset + 1,
                end_line=line_offset + current.count("\n") + 1,
                language=language,
            ))
            overlap_text = current[-overlap:] if overlap else ""
            line_offset += current.count("\n") - overlap_text.count("\n")
            current = overlap_text + "\n\n" + para
        else:
            if current:
                current += "\n\n" + para
            else:
                current = para

    if current.strip():
        chunks.append(Chunk(
            content=current.strip(),
            chunk_type="text",
            symbol_name=None,
            start_line=line_offset + 1,
            end_line=line_offset + current.count("\n") + 1,
            language=language,
        ))

    return chunks
