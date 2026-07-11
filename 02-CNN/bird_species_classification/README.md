# Bird Species Classification — CNN (Practice Project)

⚠️ **This is a learning/practice notebook**, built to explore and compare a from-scratch CNN against transfer learning (EfficientNetB0) on an image classification task — **not** a production-ready model. The results below are limited primarily by the size of the dataset.

## Dataset

[Bird Species Classification](https://www.kaggle.com/datasets/akash2907/bird-species-classification) (Kaggle) — 16 bird species.

| Split | # Images |
|---|---|
| Training | 125 |
| Validation | 25 |
| Test | 157 |

**The core problem: this dataset is extremely small** — only 150 images total across 16 classes for training/validation (roughly ~9 images per class). A dataset this size is nowhere near enough to train a CNN from scratch, and even transfer learning is pushed to its limit.

## Workflow

1. **Preprocessing:** images loaded via `image_dataset_from_directory`, resized to 224×224, batch size 16, 17% split off for validation.
2. **Augmentation:** random flip, rotation, zoom, contrast, and translation — an attempt to artificially compensate for the small dataset size.
3. **Baseline CNN:** a small custom CNN (one Conv2D block + GlobalAveragePooling + Dense) trained from scratch.
4. **Transfer Learning:** EfficientNetB0 (ImageNet weights, frozen base) + a small classification head.
5. **Fine-Tuning:** unfreezing the last 20 layers of EfficientNetB0 and continuing training with a very low learning rate (1e-5).

## Results

| Model | Test Accuracy |
|---|---|
| Baseline CNN (from scratch) | ~28.7% |
| EfficientNetB0 (transfer learning, frozen base) | ~49.7% |
| EfficientNetB0 (after fine-tuning) | ~49.0% |

### What these numbers show

- **The baseline CNN barely beats random guessing** (random guess for 16 classes ≈ 6.25%, so 28.7% is *some* signal, but far from usable). A CNN trained from zero simply has nowhere near enough images to learn meaningful features on its own.
- **Transfer learning roughly doubled the accuracy** (~49.7%), which makes sense: EfficientNetB0's ImageNet-pretrained features already encode general visual patterns (edges, textures, shapes) that transfer reasonably well even with very few training images per class.
- **Fine-tuning did not help — and clearly overfit.** During fine-tuning, training accuracy climbed to ~83–86% while validation accuracy reached ~76–80%, but **test accuracy stayed flat at ~49%**. This gap between validation and test performance is a classic overfitting symptom: with only 125 training images, unfreezing more layers gives the model enough capacity to start memorizing the training/validation set rather than learning generalizable features, so performance on unseen test data doesn't actually improve.

## Main Takeaways / Limitations

- **Dataset size is the bottleneck**, not model architecture. No amount of augmentation or tuning fully compensates for ~9 images per class.
- A from-scratch CNN is not a realistic approach for a dataset this small.
- Transfer learning (frozen base) is the most reliable approach here — it generalizes better than fine-tuning does, precisely because it changes fewer parameters and is less prone to overfitting on so little data.
- Fine-tuning pretrained layers needs a meaningfully larger dataset to pay off; here it just increases the train/test gap.
- To meaningfully improve results, the dataset itself would need to grow substantially (more images per class), rather than tweaking the model further.

## Tech Stack

- TensorFlow / Keras
- EfficientNetB0 (`tensorflow.keras.applications`)
- Matplotlib, NumPy