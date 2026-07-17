# Flowers Recognition — Transfer Learning with EfficientNetB0

An image classifier that identifies 5 flower species from photos, built with transfer learning (EfficientNetB0) and fine-tuning, using the [Flowers Recognition dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) from Kaggle.

---

## Dataset

5 classes: `daisy`, `dandelion`, `rose`, `sunflower`, `tulip` — 4,317 images total, split 70/15/15 with `split-folders`.

| Split | # Images |
|---|---|
| Training | 3,019 |
| Validation | 644 |
| Test | 654 |

Unlike some smaller practice datasets, this one has a healthy number of images per class (roughly 600–900 each), which is reflected in the stronger, more stable results below.

---

## Workflow

**1. Loading the data**
Images are loaded with `image_dataset_from_directory` from the pre-split `train`/`val`/`test` folders, resized to 224×224, batch size 32.

**2. Pipeline optimization**
`.cache()`, `.shuffle()` (training only), and `.prefetch(buffer_size=AUTOTUNE)` on all three splits to speed up training.

**3. Data augmentation**
Random horizontal flip, rotation, zoom, and contrast — applied as a preprocessing layer inside the model itself, so it only takes effect during training.

**4. Transfer learning setup**
- Base: `EfficientNetB0` pretrained on ImageNet, `include_top=False`, **frozen** (`base_model.trainable = False`)
- Head: `GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.5) → Dense(5, softmax)`

**5. Initial training (frozen base)**
- Optimizer: Adam, loss: `sparse_categorical_crossentropy`
- Up to 30 epochs with `EarlyStopping` (`monitor="val_loss"`, `patience=5`, `restore_best_weights=True`)

**6. Evaluation (before fine-tuning)**
Test accuracy/loss, confusion matrix, full classification report, and a visual grid of sample predictions (green = correct, red = incorrect).

**7. Fine-tuning**
- Base unfrozen, but only the **last 20 layers** made trainable — the rest of EfficientNetB0 stays frozen
- Recompiled with the same Adam/loss/metrics setup and trained again with the same `EarlyStopping` callback

**8. Final evaluation**
Test accuracy/loss recomputed after fine-tuning, for direct comparison against the frozen-base model.

---

## Results

| Stage | Test Accuracy | Test Loss |
|---|:---:|:---:|
| Frozen base (transfer learning) | **93.3%** | 0.196 |
| After fine-tuning (last 20 layers) | **93.6%** | 0.222 |

**Classification report — frozen base model (test set):**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| daisy | 0.98 | 0.91 | 0.95 |
| dandelion | 0.97 | 0.96 | 0.97 |
| rose | 0.92 | 0.91 | 0.91 |
| sunflower | 0.91 | 0.96 | 0.94 |
| tulip | 0.89 | 0.91 | 0.90 |

---

## Notes / Observations

- **Fine-tuning barely moved accuracy but increased loss.** Test accuracy improved only marginally (93.3% → 93.6%), while test loss got noticeably worse (0.196 → 0.222). During fine-tuning, validation loss climbed steadily from epoch 1 onward (0.183 → 0.29) even as validation accuracy ticked up slightly — a sign the model was starting to overfit on the training set once more layers became trainable.
- **`EarlyStopping` did its job here**: with `restore_best_weights=True` and `patience=5`, fine-tuning stopped after 6 epochs and reverted to the best checkpoint (epoch 1, the lowest val_loss), which is why the final model isn't wildly overfit — but it also means fine-tuning added little real value over the frozen-base model for this dataset/architecture combination.
- **Rose and tulip are the weakest classes** (precision 0.92/0.89), most likely due to visual similarity between these flower types, which is a natural source of confusion rather than a data or training issue.
- Overall, the frozen-base transfer learning model is arguably the better trade-off here: nearly identical accuracy to the fine-tuned version, but with lower loss and less overfitting risk.

---

## Tech Stack

- TensorFlow / Keras
- EfficientNetB0 (`tensorflow.keras.applications`)
- Scikit-learn (`confusion_matrix`, `classification_report`)
- split-folders, Matplotlib, NumPy