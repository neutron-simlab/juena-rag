"""The `juena-rag` command: what each subcommand does before it needs a service.

No network and no database. `ingest-repo --dry-run` clones a two-file repository
built in a temporary folder, so it runs the real clone, list, chunk and count; the
rest check that a bad invocation ends in a sentence and an exit code, not a
traceback.
"""

import shutil
import subprocess

import pytest

from juena_rag.service.cli import main

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

PYTHON_FILE = '''\
def fit(data, model):
    """Fit a model to the data."""
    return model(data)
'''


@pytest.fixture
def upstream(tmp_path):
    """A repository with one Python file and a README, on branch `main`."""
    pytest.importorskip("chonkie", reason="needs the `[ingest]` extra")
    root = tmp_path / "upstream"
    root.mkdir()
    (root / "fit.py").write_text(PYTHON_FILE)
    (root / "README.md").write_text("# Tiny\n\nA repository with one function.\n")

    def git(*arguments):
        subprocess.run(
            ["git", "-C", str(root), *arguments], check=True, capture_output=True
        )

    git("init", "--initial-branch", "main")
    git("add", ".")
    git(
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "--message",
        "two files",
    )
    return root


def a_config(tmp_path, url, **entry):
    """A `repositories.yaml` with one repository called `tiny`."""
    settings = {"chunk_size": 1500, "chunk_overlap": 200} | entry
    lines = [
        "repositories:",
        "  - id: tiny",
        "    source:",
        f"      url: {url}",
        "      branch: main",
        "    include_globs: ['**/*.py', '**/*.md']",
        *(f"    {key}: {value}" for key, value in settings.items()),
    ]
    path = tmp_path / "repositories.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def ingest(*arguments):
    return main(["ingest-repo", *map(str, arguments)])


@needs_git
def test_a_dry_run_clones_chunks_and_counts_without_storing(tmp_path, upstream, capsys):
    config = a_config(tmp_path, upstream)

    code = ingest(
        "tiny", "--config", config, "--cache-dir", tmp_path / "cache", "--dry-run"
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "2 files -> " in out
    assert "1 with a symbol" in out
    assert "dry run -- nothing was embedded or stored" in out
    assert (tmp_path / "cache" / "tiny" / "fit.py").is_file()


@needs_git
def test_no_sync_without_a_clone_is_refused(tmp_path, upstream, capsys):
    config = a_config(tmp_path, upstream)

    code = ingest(
        "tiny",
        "--config",
        config,
        "--cache-dir",
        tmp_path / "cache",
        "--no-sync",
        "--dry-run",
    )

    assert code == 2
    assert "there is no clone" in capsys.readouterr().err


def test_an_id_the_config_does_not_have_is_refused_with_the_ones_it_does(
    tmp_path, capsys
):
    pytest.importorskip("chonkie", reason="needs the `[ingest]` extra")
    config = a_config(tmp_path, "https://example.org/tiny.git")

    code = ingest("other", "--config", config, "--dry-run")

    assert code == 2
    assert "Configured: tiny" in capsys.readouterr().err


def test_a_config_without_a_chunk_size_is_refused_with_a_sentence(tmp_path, capsys):
    pytest.importorskip("chonkie", reason="needs the `[ingest]` extra")
    config = tmp_path / "repositories.yaml"
    config.write_text(
        "repositories:\n"
        "  - id: tiny\n"
        "    source: {url: 'https://example.org/tiny.git'}\n"
        "    chunk_overlap: 200\n"
    )

    code = ingest("tiny", "--config", config, "--dry-run")

    assert code == 2
    assert "sets no chunk_size" in capsys.readouterr().err


def test_serve_without_a_database_is_refused_before_it_starts(monkeypatch, capsys):
    pytest.importorskip("uvicorn", reason="needs the `[serve]` extra")
    monkeypatch.delenv("JUENA_RAG_DSN", raising=False)

    code = main(["serve"])

    assert code == 2
    assert "JUENA_RAG_DSN is not set" in capsys.readouterr().err
