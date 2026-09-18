# Jailbreak Prompt Classifier

## Why I built this

I've always learned hands-on, since I was a kid — I need to find the strong
and weak points myself, investigate, poke at things. Reading about something
and taking it for granted has never been enough for me. This project was a
way to find out, for real, whether I actually cared about ML and AI
development, even if the process itself was guided with AI assistance — I
learned a lot about the field along the way. It wasn't built in one sitting:
I worked through it in stages over several weeks, alongside coursework, with
real gaps between files and some inconsistency in how often I pushed to Git
— which is reflected in the commit history itself. What I enjoyed most was
working through the analysis myself: noticing where I was actually good at
it, and learning from the mistakes I made instead of just moving past them.
The moment that stuck with me most was seeing the accuracy number print out
for the first time, and then sitting with it — understanding why it was
what it was, after hours of learning, was more satisfying than the number
itself.

## Data

The source is a public research dataset (Shen et al., CCS 2024) of real
prompts scraped from places like Reddit and Discord, already labeled as
jailbreak or regular. The raw data was heavily imbalanced: about 13,000
regular prompts against only ~1,363 jailbreak prompts. Training on that as-is
would let a lazy model just predict "regular" every time and still hit ~90%
accuracy without learning anything real. To fix this, I cleaned duplicates
and near-empty prompts, then downsampled the majority class to match the
minority one — ending up with 1,363 examples of each class, 2,726 total. I
split that balanced set into train/val/test (70/15/15) using stratification,
so each split kept the same class proportions as the original.

## Method

Each prompt is converted into numbers with TF-IDF, combining two feature
types at once: word n-grams (1-2 words, to capture jailbreak vocabulary and
phrasing) and character n-grams (3-5 characters, to catch obfuscation like
"ign0re" instead of "ignore", which a word-only model would miss entirely).
I trained two linear models on top of these features — Logistic Regression
and Linear SVM — and compared them on the validation set (never on test)
using F1 as the selection criterion. Linear SVM doesn't output probabilities
natively, so I wrapped it in `CalibratedClassifierCV` to get calibrated
probabilities, needed for ROC-AUC and for a confidence score on each
prediction.

## Results

A clean run with the dependency versions available on 18 September 2026 selected
Linear SVM on validation F1 and produced the following held-out test results:

| Metric | Score |
|---|---|
| Accuracy | 0.824 |
| Precision (jailbreak) | 0.859 |
| Recall (jailbreak) | 0.775 |
| F1 (jailbreak) | 0.814 |
| ROC-AUC | 0.911 |

The exact machine-readable results are committed in `reports/metrics.json`.
`requirements.txt` pins the versions used to generate them, and the training and split
seeds are fixed for repeatable results on Python 3.10.

## Limitations and error analysis

The held-out run misclassified 72 of 409 prompts: 46 false negatives and 26 false
positives. Aggregate statistics are in [`reports/error_analysis.md`](reports/error_analysis.md).
Prompt text is intentionally excluded from the repository because the source data may
contain unsafe or sensitive content.

This baseline uses n-gram surface features from isolated prompts. It cannot reliably
infer intent or use conversation context. The random stratified split removes exact
duplicates before splitting, but it does not yet group templates or near-duplicates;
closely related variants could therefore cross splits and make generalization look
better than it is. The balanced held-out set is useful for model comparison, but its
precision and accuracy do not represent deployment prevalence. A grouped split and a
naturally distributed external test set are the next evaluation improvements.

## Reproduce the published evaluation

Use Python 3.10 in a clean virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/download_data.py
python src/data_prep.py
python src/train.py
pytest -q
```

Training regenerates:

- `reports/metrics.json`: machine-readable validation and held-out test metrics.
- `reports/error_analysis.md`: aggregate error statistics without raw prompt text.
- `reports/figures/`: confusion matrices for each evaluated model.

Raw data, fitted models, and prompt-level errors are intentionally not committed. The data
can contain unsafe or sensitive text; regenerate and inspect it locally if needed. GitHub
Actions runs the deterministic unit tests on every push and pull request. Full training is
kept as an explicit local step because it downloads the research dataset.

## Repository layout

- `src/download_data.py`: fetch the source dataset.
- `src/data_prep.py`: clean, balance, and create deterministic splits.
- `src/train.py`: train, select, evaluate, and write reports.
- `src/predict.py`: classify one prompt with a locally trained model.
- `tests/`: data-preparation and privacy-preserving reporting checks.
- `reports/`: committed, reproducible evaluation outputs.
