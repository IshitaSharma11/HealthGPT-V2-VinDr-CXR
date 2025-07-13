
# 🩺 HealthGPT V2: Enhancing Medical AI Fairness & Interpretability with VinDr-CXR

> **Towards Fair and Explainable Medical AI**  
> _Augmenting HealthGPT with Vietnamese Chest X-ray Data (VinDr-CXR)_

![Banner](A_comprehensive_project_overview_of_"HealthGPT_V2:.png)

---

## 📘 Overview

HealthGPT V2 enhances the foundational HealthGPT model by integrating **VinDr-CXR**, a large chest X-ray dataset from Vietnam, to improve **fairness**, **clinical interpretability**, and **spatial grounding** of predictions. Targeted for use in **Lower-Middle-Income Countries (LMICs)**, this model addresses geographic bias and improves disease localization using bounding box annotations.

---

## 🧠 Key Features

- 🔄 Multi-Task Learning: Presence detection, spatial localization, and diagnosis prediction.
- 🧭 Region Alignment Module: Concept-to-region mapping for explainable outputs.
- 🌍 Domain Adaptation: Adversarial learning + histogram matching for cross-region generalization.
- ⚖️ Fairness Metrics: Evaluated using Demographic Parity and Equal Opportunity Difference.
- 🧠 Interpretability: Grad-CAM & Attention Rollout for radiologist-interpretable insights.
- ⚙️ H-LoRA Fine-Tuning: Efficient adaptation using Heterogeneous Low-Rank Adaptation.

---

## 🗂 Project Structure

```
📂HealthGPT-V2/
├── configs/             # Training and model configs
├── models/              # Vision-language model components
├── scripts/             # Preprocessing, training, and evaluation scripts
├── datasets/            # Data loading and transformation utilities
├── results/             # Outputs, predictions, and metrics
├── README.md            # Project documentation (this file)
```

---

## 🩻 Dataset: VinDr-CXR

- 18,000+ annotated chest X-rays from Vietnam
- 14 clinical findings with bounding boxes
- Official dataset: [VinDr-CXR Dataset](https://vindr.ai/datasets/cxr)

---

## ⚙️ Pipeline Highlights

### 🧬 Enhanced Architecture

```text
Input X-ray
    ↓
Image Encoder (Frozen CLIP-ViT)
    ↓                            →→→→→→→→→→
Text Encoder (Disease Concepts)         ↓
    ↓                            Bounding Box Head → Location
    ↓                            Answer Head       → Diagnosis
    ↓                            Presence Head     → Normal/Abnormal
```

### 🧪 Multi-Task Loss Function

```python
Loss = L_presence + λ_bbox * L_bbox + L_answer
```

- **L_presence**: Binary cross-entropy
- **L_bbox**: Generalized IoU loss (for abnormal cases)
- **L_answer**: Multi-class cross-entropy

---

## 🚦 Inference Workflow

1. Load DICOM → Convert to RGB Tensor → CLIP Normalize  
2. Extract Features (Image + Text)
3. Predict:
    - Normal: Return “No Finding”
    - Abnormal: Return Disease + Bounding Box
4. Export results as JSON

---

## 📊 Evaluation Metrics

| Task             | Metric                               |
|------------------|----------------------------------------|
| Classification   | ROC-AUC, F1-score                     |
| Localization     | IoU, mAP, Recall                      |
| Fairness         | Demographic Parity, Equal Opportunity |
| Interpretability | Grad-CAM, Attention Rollout           |

---

## 💡 Use Cases

- 👩‍⚕️ **Radiologists** – Visualize and verify region-based findings.
- 🌎 **Public Health** – Validate fairness across global populations.
- 👨‍💻 **Engineers** – Build upon a robust and explainable VL-Medical model.

---

## ⚠️ Known Limitations

- Limited training data usage (<100 GB) due to compute limits
- Geographic data mostly from Vietnam (limited LMIC diversity)
- High GPU demand (A100 preferred)
- 6–7 month dataset access approval timeline

---

## 🚀 Future Work

- Expand geographic dataset diversity
- Incorporate multilingual medical questions
- Improve model sustainability and efficiency

---

## 📚 References

1. [HealthGPT Paper](https://doi.org/10.48550/arXiv.2502.09838)  
2. [VinDr-CXR Dataset](https://doi.org/10.1038/s41597-022-01498-w)  
3. [LoRA: Low-rank Adaptation](https://arxiv.org/abs/2106.09685)  
4. [DETR Object Detection](https://arxiv.org/abs/2005.12872)  
5. [Bias Metrics](https://doi.org/10.1145/3308558.3313444)

---

## 🔗 Repository

> [🔬 GitHub: HealthGPT-V2 VinDr-CXR](https://github.com/Salmansaleem007/HealthGPT-V2-VinDr-CXR)
