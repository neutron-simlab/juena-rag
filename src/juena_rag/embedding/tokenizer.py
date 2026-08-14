"""Counting tokens with the model's own vocabulary -- for whoever writes chunks.

**This is a tool for callers, not a check this package runs.** The embedder does
not count anything and does not police the model's ceiling; it takes text ready
to embed and sends it. But "ready to embed" means "under
``EmbeddingMetadata.max_input_tokens``", and a chunker cannot honour a limit it
has no way to measure. So the measurement ships here, against the same vocabulary
the server uses::

    from juena_rag.embedding.blablador import BlabladorEmbedder
    from juena_rag.embedding.tokenizer import ModelTokenCounter

    embedder = BlabladorEmbedder.from_env()
    tokens = ModelTokenCounter()
    ceiling = embedder.metadata.max_input_tokens          # 8192

    counts = tokens.count_many([chunk.text for chunk in chunks])
    too_big = [chunk for chunk, n in zip(chunks, counts) if n > ceiling]

What happens to ``too_big`` is the caller's decision, because it is a chunking
decision -- split it, drop it, rewrite it. This package validates rather than
transforms and could not split it anyway.

**Why measuring beats estimating.** The cheap answer is to count characters and
call it a token count, on the theory that English runs about four characters per
token so the estimate over-shoots and over-shooting is safe. Measured against
this vocabulary, it does neither reliably::

    "small-angle neutron scattering"   30 chars ->   5 tokens   (6.0x over)
    "Neutronenkleinwinkelstreuung …"   37 chars ->  16 tokens   (2.3x over)
    "def f(x):\\n    return x * 2\\n"     27 chars ->  12 tokens   (2.3x over)
    "中性子小角散乱"                        7 chars ->   8 tokens   *under*
    "🧪🔬⚛"                              3 chars ->   4 tokens   *under*

The last two rows are the ones that matter. Tokens are built out of *bytes*, so
any character needing more than one byte in UTF-8 can cost more than one token --
and a corpus with German, Japanese or a stray emoji in it walks straight past a
ceiling that was supposed to be conservative. Past the ceiling the server
truncates and returns a normal-looking vector, so the passage is indexed by its
opening lines and the rest becomes unsearchable, silently.

Needs the ``[embed]`` extra, which brings ``tokenizers`` -- Hugging Face's Rust
tokenizer. It is the small one: a few megabytes, no ``transformers``, no PyTorch,
and nothing that loads a model.
"""

from juena_rag.embedding.base import EmbeddingError

try:
    from tokenizers import Tokenizer
except ImportError:  # pragma: no cover - depends on the install, not the code
    raise ImportError(
        "counting tokens needs the `tokenizers` package, which is not "
        "installed. `pip install 'juena-rag[embed]'`."
    ) from None

#: Where the vocabulary file is published.
#:
#: This is a *download location*, not the model's identity. The model is called
#: ``alias-qwen3-8b-embeddings`` everywhere it matters -- in configuration, in the
#: request, and in the ``embedding_model`` column -- and this string never reaches
#: the database. It appears here for the same reason a URL would.
DEFAULT_TOKENIZER_REPO = "Qwen/Qwen3-Embedding-8B"

#: The file to point ``tokenizer_path`` at, if you are shipping one.
TOKENIZER_FILENAME = "tokenizer.json"


class TokenizerUnavailable(EmbeddingError):
    """The vocabulary could not be loaded, so nothing can be counted."""


class ModelTokenCounter:
    """Counts tokens exactly as the embedding model does.

    Loading takes a second or two and happens once. Counting afterwards is fast
    enough to be uninteresting -- 500 paragraph-sized texts in about 30
    milliseconds -- because the work happens in Rust rather than in Python.
    """

    def __init__(
        self,
        tokenizer_path: str | None = None,
        repo_id: str = DEFAULT_TOKENIZER_REPO,
    ) -> None:
        """
        :param tokenizer_path: a local ``tokenizer.json``. Use this in a
            container image or anywhere without a route to Hugging Face; bake the
            file in at build time and nothing has to be fetched at startup.
        :param repo_id: where to download the vocabulary from when no local file
            is given. Cached after the first call.
        """
        self.name = tokenizer_path or repo_id
        try:
            if tokenizer_path:
                self._tokenizer = Tokenizer.from_file(tokenizer_path)
            else:
                self._tokenizer = Tokenizer.from_pretrained(repo_id)
        except Exception as exc:
            raise TokenizerUnavailable(
                f"could not load the tokenizer from {self.name!r}: {exc}. "
                "There is no estimate to fall back to -- character counts "
                "under-count multi-byte text, which is the direction that loses "
                f"data. Download {TOKENIZER_FILENAME} from "
                f"https://huggingface.co/{repo_id} and pass its path as "
                "`tokenizer_path`."
            ) from exc

    def count_many(self, texts: list[str]) -> list[int]:
        """How many tokens each text costs, in the order given.

        Special tokens are included, because the server counts them too.
        """
        if not texts:
            return []
        return [len(encoding.ids) for encoding in self._tokenizer.encode_batch(texts)]
