# Face the Future: Deepfake ML Challenge (Synergy'25) - Catalyst Submission

**Team:** Catalyst (Saikat Munshib)
**Final Validation Accuracy:** **94.25%** (Achieved with ConvNeXt-Base)

---

## 📖 Project Overview

This repository documents the solution developed for the Synergy'25 Deepfake ML Challenge. The core task presented a unique challenge: rather than building a standard binary deepfake detector, the objective was to accurately **predict the output distribution of a provided proprietary, black-box deepfake model**. This required a nuanced approach focused on **replicating the target model's behavior**, including its potential biases and decision boundaries, based solely on its outputs for a given training set.

Our final solution leverages the **ConvNeXt-Base** architecture, incorporating **Transfer Learning** and **aggressive regularization techniques** to combat overfitting. This approach successfully learned to mimic the proprietary model, achieving a **peak validation accuracy of 94.25%**.

---

## 🎯 Core Strategy: Neural Network Cloning

Given the "black-box" nature of the target model and the requirement to replicate its specific outputs, we adopted a **Neural Network Cloning** strategy.

* **Problem Reframing:** The task was transformed from direct deepfake detection into a supervised learning problem where the goal was to learn a mapping `f(image) -> proprietary_label`.
* **Ground Truth Definition:** The `real`/`fake` labels provided in the `real_cifake_preds.json` and `fake_cifake_preds.json` files were treated as the definitive ground truth for training our clone model.
* **Objective Function:** The model was trained using standard Binary Cross-Entropy loss to minimize the difference between its predictions and the proprietary model's labels provided in the JSON files.
* **Success Metric:** High validation accuracy indicates the model's ability to successfully generalize the *proprietary model's decision function* to unseen data from the same distribution.

This strategy acknowledges that perfect real/fake detection was secondary to achieving high fidelity in mimicking the target system's outputs.

---

## 📊 Final Result & Significance

* **Peak Validation Accuracy:** **94.25%**
* **Model:** **ConvNeXt-Base** (fine-tuned)
* **Interpretation:** This high accuracy suggests our model effectively learned the underlying patterns and decision boundaries employed by the proprietary model on the provided dataset distribution.

---

## ⚙️ Methodology & Implementation Details

Our development process involved several key phases, implemented within a Google Colab notebook leveraging PyTorch and the `timm` library.

### 1. Data Preprocessing & Pipeline Construction

Robust data handling was critical for both accuracy and performance.

* **Master Dataset Creation:**
    * Combined `real_cifake_images` (1000 images) and `fake_cifake_images` (1000 images) folders.
    * Parsed the corresponding `*.json` files to extract the `index` and `prediction` (`real`/`fake`) for each image.
    * Generated a **Master CSV (`training_master.csv`)** mapping each image file path to its "clone" label. This served as the unified input for our data loaders.
        * *Example CSV Row:* `/path/to/real_cifake_images/2.png`, `fake` (Illustrating learning the proprietary model's "mistake").
    * **File Format Handling:** Ensured correct handling of `.png` image extensions and numerical filenames (`1.png`, `2.png`, ...).
* **Performance Optimization (Local Copy):**
    * Reading ~2000 small image files repeatedly from Google Drive is a major bottleneck.
    * Implemented a crucial step (**Cell 6**) to copy the entire training image dataset (`real` + `fake` folders) to the fast local Colab disk (`/content/local_dataset/`) at the start of each session.
    * Paths in the DataFrame were then updated (**Cell 7**) to point to these local copies, drastically improving data loading speed and stability during training.
* **Data Splitting Strategy:**
    * Employed a **stratified 80% Training / 20% Validation split** (1600 training, 400 validation images).
    * **Reasoning:** An 80/20 split provides a large enough training set for the complex model to learn effectively, while the 400-image validation set offers a more statistically stable and reliable measure of generalization compared to a smaller 10% split. Stratification ensures consistent class ratios between sets.
    ```python
    # Cell 7 Snippet: Data Splitting
    train_df, val_df = train_test_split(
        df,
        test_size=0.2, # 20% validation
        random_state=42,
        stratify=df['label_id'] # Ensure balanced classes
    )
    ```

### 2. Model Selection & Architecture

An iterative approach was used to find the optimal architecture.

* **Transfer Learning:** Leveraged pre-trained weights from ImageNet using the `timm` library for all models. This provided a strong feature extraction foundation.
* **Experimental Journey:**
    1.  `EfficientNet-B3`: Initial baseline, achieved ~87% validation accuracy but quickly overfit.
    2.  `EfficientNet-B4`: Slightly larger model, improved slightly to ~89% but still showed significant overfitting.
    3.  `ConvNeXt-Tiny`: Different architecture family, showed promise by reaching 91.5% but exhibited extreme overfitting (Train Acc ~99%).
    4.  **`ConvNeXt-Base` (Final Model):** More powerful variant. Combined with aggressive regularization, this model achieved the peak **94.25%** validation accuracy, demonstrating the best capacity to learn the complex cloning task without excessive memorization.
* **Model Loading (Example - ConvNeXt-Base):**
    ```python
    # Cell 10 Snippet: Loading the Champion Model
    import timm
    model = timm.create_model(
        'convnext_base',
        pretrained=True, # Load ImageNet weights
        num_classes=2 # Adapt head for binary classification
    )
    model.to(device)
    ```

### 3. Training Strategy & Regularization

Combating overfitting in the powerful `ConvNeXt` models was the primary focus during training.

* **Optimizer:** `AdamW` (Adam with decoupled weight decay).
* **Learning Rate:** Initial `lr=1e-4`.
* **Scheduler:** `ReduceLROnPlateau` monitored validation loss and reduced LR by a factor of 0.1 after 2 epochs with no improvement, allowing for finer convergence.
* **Weight Decay (L2 Regularization):** Applied a relatively high `weight_decay=1e-2`. This penalized large model weights, strongly discouraging overfitting.
    ```python
    # Cell 10 Snippet: Optimizer and Scheduler Setup
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    ```
* **Aggressive Data Augmentation:** Applied only during training to artificially increase data diversity and make memorization harder.
    ```python
    # Cell 8 Snippet: Training Augmentations
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        # Strong policy-based augmentation:
        transforms.TrivialAugmentWide(interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # Randomly remove patches:
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
    ])
    ```
* **Epochs:** Trained for 30 epochs, saving the best model based on validation accuracy directly to Google Drive. The peak score was achieved around Epoch 7 for `ConvNeXt-Base`.
* **Batch Size:** Reduced to **8** for `ConvNeXt-Base` to fit within Colab's free tier GPU memory limits.

---

## 📁 Repository Structure & Contents

* `Catalyst_DeepFake_Model.ipynb`: The complete Google Colab notebook containing all code (data prep, training, prediction).
* `README.md`: This detailed explanation file.
* `Catalyst_presentation.ppt`: The required PowerPoint documentation summarizing the methodology and results.
* **Stored Separately (Google Drive - Due to Size):** https://drive.google.com/drive/folders/1Ij1KKcVEkTbuXiOermS-OLxyNto_uy8E?usp=sharing
    * `best_model_v7.pth`: The final trained PyTorch model weights (~350MB) for the 94.25% `ConvNeXt-Base` model. *(Provide link or instructions on access)*
    * `Catalyst_prediction_v7.json`: The final prediction JSON file generated by the V7 model for the test set. *(Provide link or instructions on access)*

---

## 🚀 How to Reproduce Results (Run the Code)

1.  **Environment:** Google Colab with a **T4 GPU** runtime is required.
2.  **Data Setup:**
    * Create a folder named `DATASET` in the root of your Google Drive.
    * Upload the challenge data (`real_cifake_images`, `fake_cifake_images`, `test`, `real_cifake_preds.json`, `fake_cifake_preds.json`) into this `DATASET` folder.
    * If the `DATASET` folder is in "Shared with me", add a shortcut to your "My Drive".
3.  **Notebook Execution:**
    * Open `Catalyst_DeepFake_Model.ipynb` in Google Colab.
    * Ensure the Runtime type is set to **T4 GPU**.
    * Run cells **sequentially** from Cell 1 through Cell 15.
        * **Cells 1-9:** Perform setup, data preparation, copying images locally (takes a few minutes), and creating DataLoaders. Verify paths and outputs.
        * **Cells 10-12:** These cells were used for *training* the model. You can **skip** running these if you only want to generate predictions using the provided `best_clone_model_v7.pth`.
        * **Cell 13:** Copies the `test` images locally (quick).
        * **Cell 14:** Loads the `best_model_v7.pth` file from your Google Drive (ensure the path matches). Requires the `.pth` file to be present.
        * **Cell 15:** Performs inference on the test set (takes ~5-10 minutes with `BATCH_SIZE=8`) and saves the final `Catalyst_prediction_v7.json` to your Google Drive root.
4.  **Dependencies:** `torch`, `torchvision`, `timm`, `pandas`, `scikit-learn`, `opencv-python`. Cell 5 handles installation via `pip`.

---

## ⚠️ Key Challenges Faced & Solutions

* **Challenge 1: Severe Overfitting:** Powerful models (especially `ConvNeXt`) quickly memorized the training data (Train Acc >> Valid Acc).
    * **Solution:** Implemented **aggressive data augmentation** (`TrivialAugmentWide`, `RandomErasing`) and strong **L2 regularization** (`weight_decay=1e-2`). This successfully closed the gap between training and validation performance.
* **Challenge 2: Colab RAM Limitations:** The large `ConvNeXt-Base` model exceeded the memory capacity of the free Colab T4 GPU, causing crashes.
    * **Solution:** Reduced `BATCH_SIZE` from 16 to **8** and set `num_workers` from 2 to **0** in the DataLoaders. This allowed training to complete successfully, albeit more slowly.
* **Challenge 3: Slow Google Drive I/O:** Reading thousands of image files directly from Google Drive during training was extremely slow and unstable.
    * **Solution:** Implemented a **local copying mechanism** (Cell 6 for train/val, Cell 13 for test) to move all images to the faster Colab local disk before processing.

---

## ✅ Conclusion

This project successfully addressed the unique "Neural Network Cloning" challenge by employing a robust data pipeline, iterative model experimentation leading to **ConvNeXt-Base**, and effective regularization techniques. The final model achieves a strong **94.25% validation accuracy**, demonstrating its capability to accurately replicate the behavior of the target proprietary system.

---
