"""The one shape every parser returns, and the lookup that reads it."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Definition:
    """One named region of a file: a function, a subroutine, a section.

    Attributes:
        start_line: The first line, counting from 1.
        end_line: The last line, included.
        path: The chain of enclosing names ending in this one, so a method
            inside a class is ``("FormFactor", "sphere")`` and a subsection
            inside a chapter is ``("Data Treatment", "Fitting")``.
    """

    start_line: int
    end_line: int
    path: tuple[str, ...]

    @property
    def name(self) -> str:
        """The innermost name on its own, without its ancestors."""
        return self.path[-1]

    @property
    def dotted(self) -> str:
        """The whole chain as one name -- ``"FormFactor.sphere"``."""
        return ".".join(self.path)


def enclosing(
    definitions: list[Definition], start_line: int, end_line: int | None = None
) -> Definition | None:
    """Find the innermost definition that contains a whole line range.

    A chunk running across two subroutines sits inside neither; naming the one
    it starts in would send the reader to a place that does not hold the quoted
    text. So the answer is their shared parent, or nothing.

    Args:
        definitions: An outline, in source order.
        start_line: The range's first line.
        end_line: The range's last line. None means the range is one line.

    Returns:
        The innermost definition containing the whole range, or None.
    """
    if end_line is None:
        end_line = start_line

    # The last match wins, which is the innermost one: every parser here returns
    # source order, and a nested definition always follows the one it sits inside.
    found = None
    for definition in definitions:
        if definition.start_line <= start_line and definition.end_line >= end_line:
            found = definition
    return found
