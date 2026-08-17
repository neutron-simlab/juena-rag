"""Counting tokens with the model's own vocabulary -- for whoever writes chunks.

**A tool for callers, not a check this package runs**: the embedder sends text
ready to embed, but "ready" means "under ``EmbeddingMetadata.max_input_tokens``",
and a chunker cannot honour a limit it has no way to measure. Counting characters
instead is wrong in *both* directions -- tokens are built out of bytes, so German,
Japanese or a stray emoji cost more tokens than characters and walk straight past
a ceiling that was supposed to be conservative.
"""

from juena_rag.embedding.base import EmbeddingError

try:
    from tokenizers import Tokenizer
except ImportError:  # pragma: no cover - depends on the install, not the code
    raise ImportError(
        "counting tokens needs the `tokenizers` package, which is not "
        "installed. `pip install 'juena-rag[embed]'`."
    ) from None

#: Where the vocabulary file is published -- a *download location*, not the
#: model's identity. This string never reaches the database; the model is called
#: ``alias-qwen3-8b-embeddings`` everywhere it matters.
DEFAULT_TOKENIZER_REPO = "Qwen/Qwen3-Embedding-8B"

#: The file to point ``tokenizer_path`` at, if you are shipping one.
TOKENIZER_FILENAME = "tokenizer.json"


class TokenizerUnavailable(EmbeddingError):
    """The vocabulary could not be loaded, so nothing can be counted."""


class ModelTokenCounter:
    """Counts tokens exactly as the embedding model does.

    Loading takes a second or two and happens once; counting afterwards is fast
    enough to be uninteresting, because the work happens in Rust rather than in
    Python.
    """

    def __init__(
        self,
        tokenizer_path: str | None = None,
        repo_id: str = DEFAULT_TOKENIZER_REPO,
    ) -> None:
        """
        :param tokenizer_path: a local ``tokenizer.json``. Use this in a container
            image or anywhere without a route to Hugging Face, baked in at build
            time so nothing has to be fetched at startup.
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
