"""LaTeX chapters, sections and subsections, via tree-sitter.

A heading is not a definition, so what this produces belongs in a chunk's
``heading_path`` rather than its ``symbol``.
"""

import re

from tree_sitter_language_pack import get_parser

from juena_rag.parsers.definition import Definition

#: Sectioning commands, and the nesting they imply. Starred forms (``\section*``)
#: are separate node types in the grammar and are matched by the same names.
DEFINITIONS = frozenset(
    {"part", "chapter", "section", "subsection", "subsubsection", "paragraph"}
)

#: `\emph{Notation}` -> `Notation`. Titles carry markup, and a heading path is
#: read by a person, not compiled.
_COMMAND = re.compile(r"\\[a-zA-Z]+\*?\s*")


def _title(node) -> str | None:
    """The heading's text, with its braces and markup commands taken off."""
    group = node.child_by_field_name("text")
    if group is None:
        return None
    text = group.text.decode("utf-8", errors="replace").strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    text = _COMMAND.sub("", text).replace("{", "").replace("}", "")
    return " ".join(text.split()) or None


def outline(source: str) -> list[Definition]:
    """Every heading, in source order, deepest last.

    A section's range runs to the start of the next one at its level, which is
    what the grammar's own nesting already gives.
    """
    tree = get_parser("latex").parse(source.encode("utf-8", errors="replace"))

    found: list[Definition] = []

    def walk(node, prefix: tuple[str, ...]) -> None:
        for child in node.children:
            title = _title(child) if child.type in DEFINITIONS else None
            if title is not None:
                path = (*prefix, title)
                found.append(
                    Definition(
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        path=path,
                    )
                )
                walk(child, path)
            else:
                walk(child, prefix)

    walk(tree.root_node, ())
    return found
