"""What a file declares, per language.

The grammars belong to tree-sitter and the standard library; these assert the
shape this package puts around them -- 1-based lines, source order, nesting, and
that an unreadable file costs its own symbols and nothing else.
"""

from juena_rag.parsers import Definition, enclosing, outline

PYTHON = '''"""A module."""


def scattering_length(element):
    return TABLE[element]


class FormFactor:
    def sphere(self, q, radius):
        return 3 * (sin(q * radius) - q * radius * cos(q * radius))
'''

FORTRAN = """module scattering
contains
      subroutine formfactor(q, r, f)
      real :: q, r, f
      f = 3.0 * q * r
      end subroutine formfactor

      function sld(rho) result(s)
      real :: rho, s
      s = rho * 1.0e-6
      end function sld
end module scattering
"""

LATEX = r"""\chapter{Data Treatment}
Words.
\section{Fitting}
More words.
\subsection{Least squares}
Details.
\section*{Unnumbered}
Tail.
"""


# --------------------------------------------------------------------------- #
# The shape every parser returns
# --------------------------------------------------------------------------- #


def test_a_definition_renders_its_chain_two_ways():
    definition = Definition(start_line=1, end_line=9, path=("FormFactor", "sphere"))
    assert definition.name == "sphere"
    assert definition.dotted == "FormFactor.sphere"


def test_an_unparsed_language_yields_nothing_rather_than_raising():
    """A pipeline offers whatever files it finds; only some have a grammar here."""
    assert outline("cobol", "IDENTIFICATION DIVISION.") == []


# --------------------------------------------------------------------------- #
# Python
# --------------------------------------------------------------------------- #


def test_python_nests_a_method_under_its_class():
    found = outline("python", PYTHON)
    assert [d.dotted for d in found] == [
        "scattering_length",
        "FormFactor",
        "FormFactor.sphere",
    ]


def test_python_lines_are_one_based_and_cover_the_definition():
    first = outline("python", PYTHON)[0]
    assert first.start_line == 4
    assert first.end_line == 5


def test_a_python_file_that_does_not_parse_yields_nothing():
    """One unparseable file of two hundred should cost that file, not the ingest."""
    assert outline("python", "def broken(:\n    pass\n") == []


# --------------------------------------------------------------------------- #
# Fortran
# --------------------------------------------------------------------------- #


def test_fortran_finds_subroutines_and_functions_inside_their_module():
    found = outline("fortran", FORTRAN)
    assert [d.dotted for d in found] == [
        "scattering",
        "scattering.formfactor",
        "scattering.sld",
    ]


def test_fortran_lines_are_one_based():
    """tree-sitter counts rows from zero; every line a person reads counts from one."""
    assert outline("fortran", FORTRAN)[0].start_line == 1


def test_a_bare_fortran_program_is_found():
    found = outline("fortran", "program main\n      end program main\n")
    assert [d.dotted for d in found] == ["main"]


def test_fortran_that_declares_nothing_yields_nothing():
    assert outline("fortran", "! just a comment\n") == []


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #


def test_latex_headings_nest_by_level():
    found = outline("latex", LATEX)
    assert [list(d.path) for d in found][:3] == [
        ["Data Treatment"],
        ["Data Treatment", "Fitting"],
        ["Data Treatment", "Fitting", "Least squares"],
    ]


def test_a_starred_heading_counts_as_a_heading():
    """`\\section*` is a separate node type in the grammar and easy to miss."""
    assert "Unnumbered" in {d.name for d in outline("latex", LATEX)}


def test_markup_inside_a_title_is_stripped():
    """A heading path is read by a person, not compiled."""
    found = outline("latex", r"\section{The \emph{fast} path}" + "\n")
    assert found[0].name == "The fast path"


# --------------------------------------------------------------------------- #
# Looking one up
# --------------------------------------------------------------------------- #


def test_the_innermost_definition_wins():
    """A method's lines are also its class's lines; the method is the useful answer."""
    found = outline("python", PYTHON)
    assert enclosing(found, 10).dotted == "FormFactor.sphere"
    assert enclosing(found, 8).dotted == "FormFactor"


def test_a_line_outside_every_definition_has_none():
    assert enclosing(outline("python", PYTHON), 1) is None


def test_looking_into_an_empty_outline_is_not_an_error():
    assert enclosing([], 42) is None


def test_a_range_inside_one_definition_still_names_it():
    found = outline("fortran", FORTRAN)
    assert enclosing(found, 4, 5).dotted == "scattering.formfactor"


def test_a_range_spanning_two_definitions_names_their_parent():
    """The whole reason this takes a range: a chunk can run across a boundary.

    Naming the subroutine it started in would send a reader to a line that does
    not contain what was quoted, so the shared parent is the honest answer.
    """
    found = outline("fortran", FORTRAN)
    assert enclosing(found, 4, 10).dotted == "scattering"


def test_a_range_spanning_everything_names_nothing():
    found = outline("fortran", FORTRAN)
    assert enclosing(found, 1, 999) is None


def test_latex_headings_collapse_to_a_common_ancestor_across_sections():
    found = outline("latex", LATEX)
    assert list(enclosing(found, 3, 6).path) == ["Data Treatment", "Fitting"]
    assert list(enclosing(found, 1, 6).path) == ["Data Treatment"]


# --------------------------------------------------------------------------- #
# C and C++
# --------------------------------------------------------------------------- #

C_SOURCE = """
#include <math.h>

double form_volume(double radius) {
    return 4.0 / 3.0 * M_PI * radius * radius * radius;
}

static double Iq(double q, double sld) {
    return sld * q;
}
"""

CPP_SOURCE = """
namespace scattering {

class FormFactor {
public:
    double sphere(double q, double r);
};

double FormFactor::sphere(double q, double r) {
    return 3.0 * (sin(q * r) - q * r * cos(q * r));
}

}
"""


def test_c_functions_are_found_by_name():
    assert [d.dotted for d in outline("c", C_SOURCE)] == ["form_volume", "Iq"]


def test_a_cpp_method_keeps_the_class_it_belongs_to():
    """Note the mixed separator, which is what a citation will actually show.

    The chain is joined with ``.`` while C++ carries its own ``::`` inside a
    qualified name, so an out-of-line method reads ``scattering.FormFactor::sphere``.
    """
    found = {d.dotted for d in outline("cpp", CPP_SOURCE)}
    assert "scattering.FormFactor::sphere" in found


def test_a_macro_the_grammar_cannot_expand_does_not_invent_a_function():
    """There is no preprocessor here, so an unknown macro misparses as a definition.

    Neither language nests a function inside a function, so refusing to look
    inside one is what keeps `main.while` out of the citations.
    """
    source = """
#define CHECK if (error) return;

int main(void) {
    CHECK
    while (running) { step(); }
    return 0;
}
"""
    assert [d.dotted for d in outline("c", source)] == ["main"]


def test_a_c_file_of_only_declarations_yields_nothing():
    """A header of prototypes declares no bodies, so there is nothing to cite."""
    assert outline("c", "int foo(int a);\ndouble bar(void);\n") == []
