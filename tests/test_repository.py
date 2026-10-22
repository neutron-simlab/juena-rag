"""Which files get indexed, and which of them count as documentation.

No git and no network: every test builds a directory tree in a temporary folder and
asks the same questions of it that an ingest asks of a clone.
"""

import pytest

pytest.importorskip("yaml", reason="needs the `[ingest]` extra")

from juena_rag.ingest.code.repository import (  # imported below the skip, deliberately
    RepoConfig,
    RepoSource,
    is_doc,
    list_files,
    load_repositories,
    read_file,
    wanted,
)


def a_config(**overrides) -> RepoConfig:
    """A config with the jscatter-shaped defaults, so a test names only its point."""
    settings = {
        "id": "example",
        "name": "Example",
        "source": RepoSource(url="https://example.invalid/x.git"),
        "chunk_size": 3000,
        "chunk_overlap": 400,
        "include_globs": ["**/*.py", "**/*.rst"],
        "exclude_globs": ["**/build/**", "**/.git/**"],
        "docs_paths": ["README.rst", "src/example/doc/"],
    }
    settings.update(overrides)
    return RepoConfig(**settings)


def a_tree(root, files: dict[str, str]) -> None:
    """Write ``{relative path: contents}`` under ``root``."""
    for rel_path, text in files.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


# --------------------------------------------------------------------------- #
# Globs
# --------------------------------------------------------------------------- #


def test_a_double_star_prefix_also_matches_the_top_level():
    """The bug the plain `fnmatch` has, which is why `_matches` retries.

    `fnmatch` reads `**/*.py` as requiring at least one directory, so a top-level
    `setup.py` silently would not be indexed.
    """
    assert wanted(a_config(), "setup.py")
    assert wanted(a_config(), "src/example/formel.py")


def test_an_excluded_path_loses_even_when_it_is_also_included():
    assert not wanted(a_config(), "build/generated.py")


def test_an_extension_nobody_asked_for_is_not_indexed():
    assert not wanted(a_config(), "src/example/data.csv")


# --------------------------------------------------------------------------- #
# Documentation
# --------------------------------------------------------------------------- #


def test_a_docs_path_entry_covers_a_whole_directory():
    assert is_doc(a_config(), "src/example/doc/source/sas.rst")


def test_a_docs_path_entry_naming_a_file_covers_only_that_file():
    assert is_doc(a_config(), "README.rst")
    assert not is_doc(a_config(), "src/example/tipsreadme.rst")


def test_the_jscatter_docs_paths_that_shipped_match_nothing():
    """The bug this use case found, kept as a test so it cannot come back.

    The chatbot's config names `README.md` and `docs/`; jscatter has `README.rst`
    and `src/jscatter/doc/`. Nothing failed -- every chunk simply came out
    `is_doc=False`, documentation included.
    """
    stale = a_config(docs_paths=["README.md", "docs/"])
    assert not is_doc(stale, "README.rst")
    assert not is_doc(stale, "src/jscatter/doc/source/sas.rst")


# --------------------------------------------------------------------------- #
# Walking a tree
# --------------------------------------------------------------------------- #


def test_the_file_list_is_sorted_so_two_runs_number_chunks_the_same(tmp_path):
    """`chunk_index` is position in the batch, and `os.walk` has no fixed order."""
    a_tree(tmp_path, {"z.py": "z = 1", "a.py": "a = 1", "m/n.py": "n = 1"})
    assert list_files(a_config(), tmp_path) == ["a.py", "m/n.py", "z.py"]


def test_a_file_over_the_size_limit_is_skipped(tmp_path):
    a_tree(tmp_path, {"small.py": "x = 1", "big.py": "x = 1" * 500})
    config = a_config(max_file_bytes=100)
    assert list_files(config, tmp_path) == ["small.py"]


def test_reading_outside_the_repository_is_refused(tmp_path):
    """A config is trusted; a path built from one is still checked."""
    a_tree(tmp_path, {"inside.py": "x = 1"})
    with pytest.raises(ValueError, match="outside the repository"):
        read_file(tmp_path, "../escaped.py")


# --------------------------------------------------------------------------- #
# The YAML
# --------------------------------------------------------------------------- #


def test_the_shipped_config_loads_and_names_jscatter():
    from pathlib import Path

    shipped = Path(__file__).parent.parent / "use-cases" / "code-repository"
    configs = load_repositories(shipped / "repositories.yaml")
    assert "jscatter" in configs
    assert configs["jscatter"].source.branch == "main"
    assert configs["jscatter"].docs_paths == ["README.rst", "src/jscatter/doc/"]
    assert (configs["jscatter"].chunk_size, configs["jscatter"].chunk_overlap) == (3000, 400)


@pytest.mark.parametrize("missing", ["chunk_size", "chunk_overlap"])
def test_an_entry_that_does_not_say_how_to_split_is_refused(tmp_path, missing):
    """The caller's config decides the split; juena-rag has no default to use."""
    sizes = {"chunk_size": "3000", "chunk_overlap": "400"}
    del sizes[missing]
    path = tmp_path / "repositories.yaml"
    path.write_text(
        "repositories:\n  - id: unsplit\n    source: {url: 'https://a.invalid/x.git'}\n"
        + "".join(f"    {name}: {value}\n" for name, value in sizes.items())
    )
    with pytest.raises(ValueError, match=f"sets no {missing}"):
        load_repositories(path)


def test_an_entry_with_no_url_is_refused(tmp_path):
    path = tmp_path / "repositories.yaml"
    path.write_text("repositories:\n  - id: nowhere\n    name: Nowhere\n")
    with pytest.raises(ValueError, match="names no source.url"):
        load_repositories(path)


def test_two_entries_with_one_id_are_refused(tmp_path):
    """They would be one source in the database, each overwriting the other."""
    path = tmp_path / "repositories.yaml"
    path.write_text(
        "repositories:\n"
        "  - id: twice\n    source: {url: 'https://a.invalid/x.git'}\n"
        "    chunk_size: 3000\n    chunk_overlap: 400\n"
        "  - id: twice\n    source: {url: 'https://b.invalid/y.git'}\n"
        "    chunk_size: 3000\n    chunk_overlap: 400\n"
    )
    with pytest.raises(ValueError, match="appears twice"):
        load_repositories(path)
