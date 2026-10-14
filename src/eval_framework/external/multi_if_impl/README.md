Multi-IF grading follows [facebookresearch/Multi-IF](https://github.com/facebookresearch/Multi-IF) as of commit
`1cdb53e` (Oct 28 2024). Its checkers are the IFEval ones in `../ifeval_impl`, except for three it makes
language-aware, which `instructions.py` overrides:

- `length_constraints:number_sentences` splits Hindi on `।!?` and Chinese/Japanese on their sentence punctuation.
- `length_constraints:number_words` counts each Chinese/Japanese/Korean character as a word.
- `startend:quotation` also accepts `“…”` and `「…」`.

Deviations from upstream:

- The Thai branches are dropped: there are no Thai rows in the dataset, and they need `pythainlp`.
- The CJK word count does not use the optional `emoji` package, so consecutive emojis count as one word rather than
  one each.
- Language detection for the two counters uses a seeded `langdetect` detector. Upstream is unseeded, so a short
  Chinese answer is occasionally detected as Korean and counted with the whitespace tokenizer.
