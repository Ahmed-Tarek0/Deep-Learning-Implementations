# Chest X-Ray Pneumonia Classification — CNN (Practice Project)

A CNN trained from scratch to classify chest X-ray images as **Normal** or **Pneumonia**, using the [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data) from Kaggle.

⚠️ This is a learning/practice notebook meant to reinforce CNN fundamentals (Conv blocks, batch normalization, augmentation, class imbalance handling) — **not** a diagnostic tool.

---

## Dataset

Binary classification: `NORMAL` vs `PNEUMONIA`.

| Split | # Images | Used? |
|---|---|---|
| Training | 5,216 | ✅ |
| Test | 624 | ✅ |
| Validation (Kaggle's official split) | 16 | ❌ Not used |

The dataset's official `val` folder only contains **16 images**, far too few to give a meaningful validation signal — so it was skipped, and the model was trained without validation monitoring (see Limitations below).

The classes are also imbalanced (far more `PNEUMONIA` images than `NORMAL`), so class weights were computed and passed into training to compensate.

---

## Workflow

**1. Loading the data**
Images are loaded with `image_dataset_from_directory`, resized to 224×224, and batched in groups of 32.

**2. Pipeline optimization**
The train/test pipelines are chained with `.cache()`, `.shuffle()`, and `.prefetch(buffer_size=AUTOTUNE)` to avoid I/O bottlenecks during training.

**3. Data augmentation**
Applied on the fly (training only): random horizontal flip, small rotation, zoom, and contrast adjustments — a lightweight defense against overfitting.

**4. Model architecture**
A custom CNN built from scratch:
- 4 convolutional blocks (32 → 64 → 128 → 256 filters), each: `Conv2D → BatchNorm → ReLU → MaxPooling`
- `GlobalAveragePooling2D` to flatten feature maps
- `Dense(128, relu)` → `Dropout(0.5)` for regularization
- Final `Dense(softmax)` output layer for the 2 classes

**5. Handling class imbalance**
Since `PNEUMONIA` images significantly outnumber `NORMAL` ones, class weights are computed with `sklearn.utils.class_weight.compute_class_weight("balanced", ...)` and passed to `model.fit()` so the minority class isn't underweighted during training.

**6. Training**
- Optimizer: Adam, `learning_rate=1e-4`
- Loss: `sparse_categorical_crossentropy`
- 15 epochs, no validation set monitored (see [Limitations](#notes--limitations))

**7. Evaluation**
- Test-set accuracy and loss
- Confusion matrix
- Full classification report (precision / recall / F1 per class)
- A visual grid of sample test predictions, color-coded green (correct) / red (incorrect)

---

## Results

| Metric | Value |
|---|---|
| Final training accuracy (epoch 15) | ~95% |
| **Test accuracy** | **~88%** |
| Test loss | ~0.32 |

**Classification report (test set):**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| NORMAL | 0.86 | 0.82 | 0.84 |
| PNEUMONIA | 0.89 | 0.92 | 0.91 |

---

## Notes / Limitations

- **No real validation during training.** Kaggle's official validation folder has only 16 images — not enough to trust as a validation signal — so it wasn't used. As a side effect, training runs "blind" epoch-to-epoch (only training accuracy is visible), with no early-stopping check against overfitting; evaluation only happens once, after training, directly on the test set. An `EarlyStopping` callback is even imported in the notebook but never actually used, since there's no validation loss to monitor it with.
- **Gap between train and test accuracy** (~95% vs ~88%) suggests mild overfitting, which lines up with the lack of validation-based monitoring during training.
- **Recall on `NORMAL` (0.82) is lower than on `PNEUMONIA` (0.92)** — the model is somewhat more likely to miss a genuinely normal case (false PNEUMONIA prediction) than to miss real pneumonia. That's the safer direction of error for a screening context, but still worth improving.
- As with any medical imaging task, this dataset/model combo is for **practice and learning purposes only** and should not be used for real diagnostic decisions.

---

## Tech Stack

- TensorFlow / Keras
- Scikit-learn
- Matplotlib, NumPy