"""Ask the corpus a question and see which files answer it.

    uv run --group use-cases python use-cases/code-repository/search.py \\
        "how do I fit a form factor to SANS data?"

The counterpart to ``ingest.py``: that one writes a repository in, this one reads
it back. Nothing here is specific to code repositories except what it prints.
"""

import argparse
import os
import sys
import textwrap

from dotenv import load_dotenv


def describe(hit) -> str:
    """One line naming where a hit came from: file, line, and what encloses it.

    Code names the definition it sits in; prose names the headings above it. Both
    live in ``extra`` rather than in columns -- they belong to some categories of
    chunk, not to every chunk.
    """
    where = f"{hit.source_uri}:{hit.extra.get('start_line', '?')}"
    context = hit.extra.get("symbol") or " > ".join(hit.extra.get("heading_path") or [])
    return (
        f"{hit.distance:.3f}  {where}"
        + (f"  [{context}]" if context else "")
        + ("  (doc)" if hit.extra.get("is_doc") else "")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", help="a question, in any language the model knows")
    parser.add_argument(
        "-n", type=int, default=5, help="how many hits to show (default 5)"
    )
    parser.add_argument(
        "-k",
        type=int,
        help="how many candidates the index retrieves before filtering. Raise it "
        "when a filter below is narrow, or the filter can quietly starve the result",
    )
    parser.add_argument(
        "--source", action="append", help="restrict to a source id; repeatable"
    )
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="skip prose chunks -- documentation, comments, README text",
    )
    parser.add_argument(
        "--full", action="store_true", help="print each chunk's text, not just where"
    )
    arguments = parser.parse_args()

    load_dotenv()

    dsn = os.environ.get("JUENA_RAG_DSN") or os.environ.get("JUENA_RAG_TEST_DSN")
    if not dsn:
        print("set JUENA_RAG_DSN (see .env.example)", file=sys.stderr)
        return 1

    from juena_rag.embedding.blablador import BlabladorEmbedder
    from juena_rag.store.pgvector import Store

    with Store(dsn, embedder=BlabladorEmbedder.from_env()) as store:
        result = store.search(
            arguments.query,
            sources=arguments.source,
            chunk_kinds=["code_block"] if arguments.code_only else None,
            n=arguments.n,
            k=arguments.k,
        )

    if not result.hits:
        print("nothing matched -- is anything ingested? (see ingest.py)")
        return 1

    # Cosine distance: smaller is closer. Not turned into a percentage anywhere,
    # because the number the database computed is the honest one.
    print(f'"{arguments.query}"\n')
    for hit in result.hits:
        print(f"  {describe(hit)}")
        if arguments.full:
            print(textwrap.indent(hit.text.strip(), "      "), end="\n\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
