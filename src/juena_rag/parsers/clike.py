"""C and C++ functions, classes and namespaces, via tree-sitter.

One module for both, because C++ is C's grammar with more node types in it -- the
walk below is identical and only the set of definitions differs.
"""

from tree_sitter_language_pack import get_parser

from juena_rag.parsers.definition import Definition

#: What names something worth citing, per language. C has functions and nothing
#: that nests; C++ adds the three things functions nest inside.
DEFINITIONS = {
    "c": frozenset({"function_definition"}),
    "cpp": frozenset(
        {
            "function_definition",
            "class_specifier",
            "struct_specifier",
            "namespace_definition",
        }
    ),
}


def _declared_name(node) -> str | None:
    """The name a definition declares, or ``None`` for an anonymous one.

    A function's name sits under its ``declarator``, wrapped in as many pointer
    and array declarators as the return type needs, so this walks down to the
    first identifier rather than reading a fixed field.
    """
    named = node.child_by_field_name("name")  # class, struct, namespace
    if named is not None:
        return named.text.decode("utf-8", errors="replace")

    declarator = node.child_by_field_name("declarator")
    while declarator is not None:
        if declarator.type in ("identifier", "field_identifier", "type_identifier"):
            return declarator.text.decode("utf-8", errors="replace")
        # `qualified_identifier` is C++'s `Class::method`; its last part is the name.
        if declarator.type == "qualified_identifier":
            return declarator.text.decode("utf-8", errors="replace")
        declarator = declarator.child_by_field_name("declarator")
    return None


def outline(source: str, language: str = "c") -> list[Definition]:
    """Every named definition, in source order, innermost last.

    A file that does not parse yields whatever the grammar recovered rather than
    raising -- tree-sitter always returns a tree, with the broken parts marked.
    """
    definitions = DEFINITIONS.get(language, DEFINITIONS["c"])
    tree = get_parser(language).parse(source.encode("utf-8", errors="replace"))

    found: list[Definition] = []

    def walk(node, prefix: tuple[str, ...]) -> None:
        for child in node.children:
            name = _declared_name(child) if child.type in definitions else None
            if name is not None:
                path = (*prefix, name)
                found.append(
                    Definition(
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        path=path,
                    )
                )
                # Neither language nests a function inside another function, so
                # stopping here is not an optimisation -- it is the rule. Without
                # it, an unexpanded macro that the grammar mistakes for a
                # definition harvests the next keyword as a name, and citations
                # start reading `main.while`. There is no preprocessor here and
                # there should not be one.
                if child.type != "function_definition":
                    walk(child, path)
            else:
                walk(child, prefix)

    walk(tree.root_node, ())
    return found


def outline_c(source: str) -> list[Definition]:
    """C, for the mapping in :mod:`juena_rag.parsers`."""
    return outline(source, "c")


def outline_cpp(source: str) -> list[Definition]:
    """C++, for the mapping in :mod:`juena_rag.parsers`."""
    return outline(source, "cpp")
