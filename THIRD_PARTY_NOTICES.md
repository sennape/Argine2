# Third-party notices and provenance

Argine2 is released under the repository's `LICENSE`. This file records third-party material and provenance that should remain visible in public distributions.

## llama.cpp / ggml

Argine2 contains patches and build metadata targeting `llama.cpp` commit
`254b177307b14d055bbe4c050c9c441a1e06d8a4`.

The patch files under `patches/` contain diff context from upstream `llama.cpp` / `ggml` source files in addition to Argine-specific additions. Upstream `llama.cpp` is distributed under the MIT License.

The upstream license text for the pinned commit is preserved verbatim in:

- `LICENSES/llama.cpp-MIT.txt`

Argine-specific source files and additions remain covered by the repository's own MIT license.

## Literary calibration/reference texts

Two tracked text fixtures predate the publication cleanup and were imported in the Argine2 baseline commit. Their original local download path was not preserved, so provenance is recorded conservatively here without claiming a stronger chain of custody than the repository contains.

### `testo-calibrazione.txt`

The content is from *Dracula* by Bram Stoker (1897). The opening passage in the tracked file matches the Project Gutenberg transcription of *Dracula*, eBook #345.

Source reference:
- Project Gutenberg, *Dracula*: https://www.gutenberg.org/ebooks/345

The underlying literary work is in the public domain in the United States; copyright status can vary by jurisdiction.

### `testo-riferimento.txt`

The content is from *Moby-Dick; or, The Whale* by Herman Melville (1851). The tracked passage beginning in Chapter 9/10 matches the Project Gutenberg transcription of *Moby-Dick*, eBook #2701.

Source reference:
- Project Gutenberg, *Moby-Dick; or, The Whale*: https://www.gutenberg.org/ebooks/2701

The underlying literary work is in the public domain in the United States; copyright status can vary by jurisdiction.

These files are retained byte-for-byte because historical experiment signatures and reproducibility depend on their exact contents. They are data fixtures, not model weights and not bundled third-party software.
