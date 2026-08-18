"""Fortran subroutines, functions, modules and programs, via tree-sitter.

Fortran has no parser in the standard library and its two source layouts differ
enough that pattern matching gets it wrong; tree-sitter ships a real grammar.
"""

from tree_sitter_language_pack import get_parser

from juena_rag.parsers.definition import Definition

#: The node types that name something worth citing. ``interface`` and
#: ``submodule`` are here because they nest like the rest, not because datreat
#: uses them.
DEFINITIONS = frozenset(
    {"subroutine", "function", "module", "program", "interface", "submodule"}
)


def _declared_name(node) -> str | None:
    """The name a definition node declares, or ``None`` if it declares none.

    The grammar puts it one level down: a ``subroutine`` holds a
    ``subroutine_statement``, and that holds the ``name``.
    """
    for child in node.children:
        if child.type.endswith("_statement"):
            for grandchild in child.children:
                if grandchild.type == "name":
                    return grandchild.text.decode("utf-8", errors="replace")
            return None
    return None


def outline(source: str) -> list[Definition]:
    """Every named Fortran definition, in source order, innermost last.

    Fixed-form source -- the punch-card layout ``.f`` files still use -- parses
    about two thirds of the time, and an unparsed file yields nothing rather than
    raising.
    """
    tree = get_parser("fortran").parse(source.encode("utf-8", errors="replace"))

    found: list[Definition] = []

    def walk(node, prefix: tuple[str, ...]) -> None:
        for child in node.children:
            name = _declared_name(child) if child.type in DEFINITIONS else None
            if name is not None:
                path = (*prefix, name)
                found.append(
                    Definition(
                        # tree-sitter counts rows from 0; every line number that
                        # reaches a person counts from 1.
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
