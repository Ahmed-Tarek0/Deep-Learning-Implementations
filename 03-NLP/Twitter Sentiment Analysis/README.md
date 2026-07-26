# Twitter Sentiment Analysis — Logistic Regression from Scratch

A sentiment classifier that predicts whether a tweet is **positive** or **negative**, built with logistic regression implemented from scratch (no `sklearn` model) using NumPy — including manual gradient descent and feature extraction.

> 📘 This notebook is based on the **Week 1 assignment of Course 1 ("Natural Language Processing with Classification and Vector Spaces") from the NLP Specialization by DeepLearning.AI (Andrew Ng), on Coursera.**

---

## Dataset

The built-in `twitter_samples` corpus from NLTK — 5,000 positive and 5,000 negative tweets.

| Split | # Tweets |
|---|---|
| Training | 8,000 |
| Test | 2,000 |

---

## Workflow

**1. Data preparation**
Positive and negative tweets are split 80/20 into train/test sets, and labels are combined into `train_y` / `test_y` arrays (1 = positive, 0 = negative).

**2. Text preprocessing** (`utils.py → process_tweet`)
Each tweet is cleaned and tokenized:
- Remove stock tickers (`$GE`), old-style retweet markers (`RT`), hyperlinks, and hashtag symbols
- Tokenize with NLTK's `TweetTokenizer` (lowercased, handles stripped, repeated letters reduced)
- Remove English stopwords and punctuation
- Stem each remaining word with `PorterStemmer`

**3. Building word frequencies** (`utils.py → build_freqs`)
A dictionary mapping every `(word, label)` pair to how often it appears in the training set — this becomes the basis for feature extraction.

**4. Feature extraction** (`extract_features`)
Each tweet is converted into a 3-element vector: `[bias=1, sum of positive-word frequencies, sum of negative-word frequencies]`.

**5. Model — Logistic Regression from scratch**
- `sigmoid(z)` — the activation function
- `gradientDescent(x, y, theta, alpha, num_iters)` — manually implemented batch gradient descent, minimizing binary cross-entropy loss
- Trained for 1,500 iterations with a learning rate of `1e-9`

**6. Prediction & evaluation**
- `predict_tweet` — runs a tweet through feature extraction + sigmoid to get a probability
- `test_logistic_regression` — computes overall accuracy by thresholding predictions at 0.5
- A short error analysis section prints out the misclassified tweets for manual inspection

**7. Trying it on a custom tweet**
The notebook ends with a free-form test on a new, hand-written tweet to sanity-check the model.

---

## Results

| Metric | Value |
|---|---|
| Final training cost | 0.2252 |
| **Test accuracy** | **99.65%** |

The model reaches very high accuracy quickly, since tweet sentiment here correlates strongly with just word frequency counts — this is a relatively easy dataset for the technique, not necessarily a sign the model would generalize this well to more nuanced or sarcastic text (the error analysis section shows a few such misclassified examples).

---

## Files

| File | Purpose |
|---|---|
| `notebook.ipynb` | Main notebook — data prep, model training, evaluation |
| `utils.py` | Helper functions: `process_tweet()`, `build_freqs()` |

---

## Tech Stack

- NumPy, Pandas
- NLTK (`twitter_samples`, `stopwords`, `TweetTokenizer`, `PorterStemmer`)
