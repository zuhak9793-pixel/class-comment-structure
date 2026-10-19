# Data and model permissions

**CONFIRMED — checked 2026-10-19.** This source package contains synthetic test text and aggregate numerical results only. No original target/comment text is included.

| Required source | Evidence read | Redistribution conclusion | Release handling |
|---|---|---|---|
| Archived NLBSE 2024 Python/Pharo `project_classes` CSV | Pinned upstream README and root `LICENSE`, commit `0a1d728e7058bd572c35e97a88efff1f5b92b1ae` | Root license declares CC0 1.0 Universal. That declaration does not independently resolve rights in every originating software project's comment text. | CSVs and reconstructed comments excluded; explicit pinned local acquisition only. |
| NLBSE 2026 Python/Pharo Train parquet | Pinned dataset card/README and recorded repository metadata, revision `d0dc51c11dbb660ba08f60bd714f4733e9560f3b` | No explicit dataset license declaration in the available card/metadata. Redistribution permission not established. | All parquet and text excluded. Obtain from upstream for permitted use or seek clarification from maintainers. |
| `sentence-transformers/paraphrase-MiniLM-L6-v2` | Pinned model README, revision `c9a2bfebc254878aee8c3aca9e6844d5bbb102d1` | Model card declares Apache-2.0. | No weights/tokenizer files redistributed; users acquire the pinned checkpoint and comply with its terms. |

Pinned URLs and file hashes appear in `configs/`. Acquisition fetches only the four named data files; it does not fetch official Test. Archived enclosing comments may contain overlapping text with separately released Test. Do not describe reconstruction as Test-unexposed.

Upstream records:

- [NLBSE 2024 README](https://github.com/nlbse2024/code-comment-classification/blob/0a1d728e7058bd572c35e97a88efff1f5b92b1ae/README.md) and [CC0 license](https://github.com/nlbse2024/code-comment-classification/blob/0a1d728e7058bd572c35e97a88efff1f5b92b1ae/LICENSE).
- [NLBSE 2026 dataset card](https://huggingface.co/datasets/NLBSE/nlbse26-code-comment-classification/blob/d0dc51c11dbb660ba08f60bd714f4733e9560f3b/README.md).
- [Pinned MiniLM model card](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2/blob/c9a2bfebc254878aee8c3aca9e6844d5bbb102d1/README.md).

**TODO:** obtain any necessary clarification before redistributing data or original software comments. The proposed code license does not license these sources. No upstream baseline notebook is bundled.
