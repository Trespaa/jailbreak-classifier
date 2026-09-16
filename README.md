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

The selected model (Linear SVM) evaluated on the held-out test set — touched
exactly once, after every other decision was already made — scored:

| Metric | Score |
|---|---|
| Accuracy | 0.81 |
| Precision (jailbreak) | 0.84 |
| Recall (jailbreak) | 0.76 |
| F1 (jailbreak) | 0.80 |
| ROC-AUC | 0.906 |

In practice: when the model flags something as a jailbreak, it's right 84%
of the time, but it still misses about 1 in 4 real jailbreak attempts.

## Limitations and error analysis

The error analysis (see `reports/error_analysis.md`) showed two clear
patterns. False negatives — jailbreaks the model misses — tend to be
obfuscated prompts or unfilled template scaffolding. False positives —
benign prompts flagged as dangerous — tend to be legitimate roleplay
requests that share surface structure with DAN-style jailbreaks.

The underlying issue: a classifier built on n-grams only sees the surface
form of a prompt, not its actual intent, so it can't reliably tell a benign
roleplay request from a malicious one — they can look nearly identical at
the text level. This isn't something more training data fixes on its own,
because the problem isn't data volume, it's what kind of information the
model has access to. A more realistic fix would be giving the model access
to conversation context rather than a single isolated prompt — with enough
turns of context, it becomes possible to tell whether someone is actually
building toward a harmful request or just running an innocent roleplay
scenario, something a single-prompt classifier structurally cannot do.
