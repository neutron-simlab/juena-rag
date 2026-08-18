"""Python definitions, from the standard library's own parser.

The only parser here that needs no third-party grammar -- ``ast`` ships with
Python and knows the language exactly.
"""

import ast

from juena_rag.parsers.definition import Definition


def outline(source: str) -> list[Definition]:
    """Every ``def`` and ``class``, in source order, innermost last.

    A file that does not parse yields nothing rather than raising: a syntax error
    in one file of two hundred should cost that file its symbols, not the ingest.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []

    found: list[Definition] = []

    def walk(node: ast.AST, prefix: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                path = (*prefix, child.name)
                found.append(
                    Definition(
                        start_line=child.lineno,
                        end_line=child.end_lineno or child.lineno,
                        path=path,
                    )
                )
                walk(child, path)
            else:
                walk(child, prefix)

    walk(tree, ())
    return found
