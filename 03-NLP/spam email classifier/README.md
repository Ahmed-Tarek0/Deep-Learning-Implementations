# 📧 Spam Email Classifier

A machine learning project that classifies SMS messages as **spam** or **ham (not spam)** using NLP preprocessing and Logistic Regression.

---

## 📁 Dataset

- **Source:** [Spam Emails Dataset — Kaggle](https://www.kaggle.com/datasets/abdallahwagih/spam-emails)
- **Size:** 5,572 messages
- **Labels:** `ham` (legitimate) / `spam`
- **Class distribution:** ~87% ham, ~13% spam

---

## ⚙️ Pipeline

### 1. Text Preprocessing (`TextPreprocessor`)

A custom, fully configurable preprocessing class that applies the following steps in order:

| Step | Description |
|------|-------------|
| HTML cleaning | Strips HTML tags and decodes entities |
| URL / email / mention removal | Removes links, emails, @mentions, #hashtags |
| Emoji removal | Uses the `emoji` library for full Unicode coverage |
| Contraction expansion | "can't" → "cannot", "it's" → "it is" |
| Lowercasing | Normalises casing |
| Slang normalisation | "gr8" → "great", "omg" → "oh my god" |
| Repeated character removal | "loooove" → "loove" |
| Punctuation removal | Strips non-word characters |
| Tokenisation | NLTK `word_tokenize` — splits text into individual word tokens |
| Stop word removal | Configurable — **disabled** for this task to preserve negation signals |
| Lemmatisation | POS-aware lemmatisation using WordNet — reduces words to their base dictionary form (e.g. "running" → "run", "better" → "good" as adjective) |

> **Why stop word removal is disabled:** Words like "not", "no", and "never" are classified as stop words in NLTK's default list. Removing them would flip meaning — "not spam" becomes "spam". For spam detection, keeping these words is critical.

> **Why POS-aware lemmatisation matters:** The same word lemmatises differently depending on its grammatical role. "meeting" as a noun stays "meeting", but as a verb becomes "meet". Without POS tags, the lemmatiser defaults to noun and may give wrong results.

### 2. Feature Extraction — TF-IDF

**TF-IDF (Term Frequency–Inverse Document Frequency)** converts cleaned text into numerical vectors the model can learn from.

- **TF (Term Frequency):** how often a word appears in a message
- **IDF (Inverse Document Frequency):** how rare the word is across all messages — rare words get higher weight, common words get lower weight
- `max_features=5000` — keeps only the 5,000 most informative words

Example: "FREE" and "WIN" appear often in spam but rarely in normal messages → high TF-IDF score → strong spam signal.

### 3. Model — Logistic Regression

A linear classifier that learns the relationship between TF-IDF features and the spam/ham label.

- Train/test split: 80% / 20% (`random_state=42`)
- Two versions trained: baseline and class-weight-balanced (see Results)

---

## 📊 Results

### Baseline Model

```
              precision    recall  f1-score
ham (0)          0.97      1.00      0.98
spam (1)         0.99      0.79      0.88
accuracy                             0.97
```

**Problem:** Recall of 0.79 means the model missed 21% of spam messages — it classified them as legitimate.

---

### Improved Model — `class_weight='balanced'`

```
              precision    recall  f1-score
ham (0)          0.99      0.98      0.98
spam (1)         0.89      0.91      0.90
accuracy                             0.97
```

**Why `class_weight='balanced'`?** The dataset is imbalanced (~87% ham, ~13% spam). By default, the model is biased toward the majority class (ham) and plays it safe — it avoids flagging messages as spam unless very confident, causing it to miss real spam.

Setting `class_weight='balanced'` tells the model to penalise mistakes on the minority class (spam) more heavily during training, making it more sensitive to spam patterns.

**Result:** Spam recall improved from **0.79 → 0.91** — the model now catches 91% of spam.

**Trade-off:** Spam precision dropped from 0.99 → 0.89, meaning a small increase in false positives (legitimate messages flagged as spam). This is an acceptable trade-off — missing spam is more harmful than occasionally filtering a real message.

---

## 🧪 Sample Predictions (Final Model)

| Message | Prediction |
|---------|-----------|
| "Congratulations! You won a free iPhone. Click here now!" | 🚨 SPAM |
| "Hey, are we still meeting tomorrow at 5?" | ✅ HAM |
| "URGENT: Your account will be suspended. Verify now!" | 🚨 SPAM |
| "Can you pick up some milk on your way home?" | ✅ HAM |