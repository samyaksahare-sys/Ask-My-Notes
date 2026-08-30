# Retrieval augmented generation

Working notes on how this project is put together, used as the demo corpus.

## Why retrieval at all

A language model knows nothing about a private document. Retrieval solves this
by finding passages relevant to a question and placing them in the prompt, so
the answer is grounded in a source that can be cited and checked rather than
recalled from training data.

## Chunking

Chunk size should never exceed the embedding model's maximum sequence length.
Anything past that limit is silently truncated at embedding time: the text is
still stored, but contributes nothing to whether the chunk is ever retrieved.
This project uses 300 token chunks with 40 tokens of overlap, capped at 248
because all-MiniLM-L6-v2 stops reading at 256 tokens.

## Boilerplate

Slide decks repeat a banner on every page. On one lecture deck measured here it
was 5,586 characters, or 18.8 percent of all extracted text. Because it is
identical everywhere it pulls every chunk toward every other chunk, flattening
the ranking. Stripping it moved a representative query from a distance of 1.226
to 0.985.
