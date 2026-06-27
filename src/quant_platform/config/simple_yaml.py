"""Small YAML subset loader for local configuration files.

The project intentionally avoids adding PyYAML in this iteration. This parser
supports the limited config style used here: nested mappings, scalar values, and
lists of scalars with two-space indentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class SimpleYAMLError(ValueError):
    """Raised when a local YAML config is outside the supported subset."""


def _strip_comment(line: str) -> str:
    in_quote: str | None = None
    for position, character in enumerate(line):
        if character in {"'", '"'}:
            if in_quote is None:
                in_quote = character
            elif in_quote == character:
                in_quote = None
        if character == "#" and in_quote is None:
            return line[:position]
    return line


def _parse_scalar(value: str) -> Any:
    clean_value = value.strip()
    if clean_value.startswith("[") and clean_value.endswith("]"):
        inner = clean_value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]
    if clean_value in {"", "null", "Null", "NULL"}:
        return None
    if clean_value.lower() == "true":
        return True
    if clean_value.lower() == "false":
        return False
    if (
        len(clean_value) >= 2
        and clean_value[0] in {"'", '"'}
        and clean_value[-1] == clean_value[0]
    ):
        return clean_value[1:-1]
    try:
        if any(marker in clean_value for marker in (".", "e", "E")):
            return float(clean_value)
        return int(clean_value)
    except ValueError:
        return clean_value


def _tokenize(text: str) -> list[tuple[int, str]]:
    tokens = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise SimpleYAMLError(f"Unsupported indentation at line {line_number}.")
        tokens.append((indent, line.strip()))
    return tokens


def _parse_block(tokens: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(tokens):
        return {}, index
    current_indent, content = tokens[index]
    if current_indent != indent:
        raise SimpleYAMLError("Invalid indentation structure.")
    if content.startswith("- "):
        return _parse_list(tokens, index, indent)
    return _parse_mapping(tokens, index, indent)


def _parse_list(tokens: list[tuple[int, str]], index: int, indent: int) -> tuple[list[Any], int]:
    values = []
    while index < len(tokens):
        current_indent, content = tokens[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise SimpleYAMLError("Unexpected nested list indentation.")
        if not content.startswith("- "):
            break
        item = content[2:].strip()
        if not item:
            nested, index = _parse_block(tokens, index + 1, indent + 2)
            values.append(nested)
        elif ":" in item:
            raise SimpleYAMLError("Inline mapping list items are not supported.")
        else:
            values.append(_parse_scalar(item))
            index += 1
    return values, index


def _parse_mapping(
    tokens: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[dict[str, Any], int]:
    values: dict[str, Any] = {}
    while index < len(tokens):
        current_indent, content = tokens[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise SimpleYAMLError("Unexpected nested mapping indentation.")
        key, separator, remainder = content.partition(":")
        if not separator:
            raise SimpleYAMLError(f"Expected mapping entry, got: {content}")
        clean_key = key.strip()
        if not clean_key:
            raise SimpleYAMLError("Mapping keys must not be empty.")
        clean_remainder = remainder.strip()
        if clean_remainder:
            values[clean_key] = _parse_scalar(clean_remainder)
            index += 1
            continue
        next_index = index + 1
        if next_index >= len(tokens) or tokens[next_index][0] <= indent:
            values[clean_key] = {}
            index = next_index
            continue
        values[clean_key], index = _parse_block(tokens, next_index, indent + 2)
    return values, index


def loads_simple_yaml(text: str) -> dict[str, Any]:
    """Load a local YAML subset string into a dictionary."""

    tokens = _tokenize(text)
    if not tokens:
        return {}
    loaded, index = _parse_block(tokens, 0, tokens[0][0])
    if index != len(tokens):
        raise SimpleYAMLError("Could not parse complete YAML document.")
    if not isinstance(loaded, dict):
        raise SimpleYAMLError("Top-level YAML document must be a mapping.")
    return loaded


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Load a supported local YAML config file."""

    return loads_simple_yaml(Path(path).read_text(encoding="utf-8"))
