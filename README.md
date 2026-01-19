# 🏥 Winning the MedGemma Impact Challenge with FiftyOne

**A complete workflow for exploring, evaluating, and fine-tuning MedGemma for medical imaging tasks.**

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-MedGemma%20Impact%20Challenge-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
[![HuggingFace](https://img.shields.io/badge/🤗%20Model-MedGemma--1.5--4B--IT-yellow)](https://huggingface.co/google/medgemma-1.5-4b-it)

---

## 🎯 About the Competition

The **[MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)** is a Kaggle competition hosted by Google Research, challenging participants to build human-centered AI applications using MedGemma models for healthcare.

| | |
|---|---|
| **Prize Pool** | $100,000 |
| **Deadline** | February 24, 2026 |
| **Focus** | Privacy-first, deployable AI solutions for healthcare |
| **Models** | MedGemma (part of Google's Health AI Developer Foundations) |

The competition encourages innovative applications across medical imaging, clinical reasoning, and healthcare accessibility—while respecting patient privacy and safety.

---

## 🧠 What is MedGemma?

**MedGemma** is Google's collection of state-of-the-art open models designed to understand and process both medical text and images, built on the Gemma 3 architecture.

**Supported tasks include:**
- Medical image interpretation (radiology, pathology, dermatology, ophthalmology)
- Visual question answering (VQA)
- Anatomical localization with bounding boxes
- Clinical reasoning and text summarization
- EHR-based question answering

## 🚀 What You'll Learn

The main notebook (`example_nb.ipynb`) walks you through a complete workflow:

| Step | What You'll Do | Why It Matters |
|------|----------------|----------------|
| **1. Load & Explore** | Understand data distribution before modeling | Catch potential issues early |
| **2. Embeddings** | Visualize MedSigLIP clusters | Diagnose whether classes are separable |
| **3. Inference** | Run MedGemma, store predictions with data | Everything in one place for analysis |
| **4. Evaluation** | Slice accuracy by modality, body part, etc. | Find *where* the model fails |
| **5. Error Analysis** | Visualize failures, tag patterns | Understand *why* it fails |
| **6. Fine-Tuning** | Use GetItem + SFTTrainer for localization | Improve model on specific failure modes |

---

## 📊 Dataset

We use the **[SLAKE dataset](https://huggingface.co/datasets/Voxel51/SLAKE)**, a medical VQA benchmark with:

- **Multiple modalities**: CT, MRI, X-ray
- **Rich annotations**: Bounding boxes, segmentation masks
- **Question types**: Anatomy, abnormalities, position, size
- **Body parts**: Brain, chest, abdomen, and more

The dataset is pre-formatted for FiftyOne and loads with a single line:

```python
from fiftyone.utils.huggingface import load_from_hub

dataset = load_from_hub("Voxel51/SLAKE", name="SLAKE")
```

---

## 🎓 Fine-Tuning Approach

### Key Implementation Choices

| Aspect | Approach |
|--------|----------|
| **Base Model** | `google/medgemma-1.5-4b-it` (4B multimodal) |
| **Quantization** | 4-bit QLoRA for memory efficiency |
| **Training** | TRL's `SFTTrainer` with gradient checkpointing |
| **Data Loading** | FiftyOne's `GetItem` + `to_patches()` + `to_torch()` |
| **Output Format** | JSON with `[y0, x0, y1, x1]` coords normalized to [0, 1000] |

### Bounding Box Format

The model is trained to output structured JSON:

```json
[{"box_2d": [150, 200, 450, 600], "label": "lung"}]
```

Where coordinates are `[y0, x0, y1, x1]` normalized to the range [0, 1000].

---

## 🏃 Quick Start

### 1. Run the Notebook

Open `medgemma_impact_starter.ipynb` in Jupyter or VS Code and run cells sequentially.

### 2. Or Use Individual Components

```python
# Load dataset
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

dataset = load_from_hub("Voxel51/SLAKE", name="SLAKE")

# Run MedGemma inference
import fiftyone.zoo as foz

foz.register_zoo_model_source("https://github.com/harpreetsahota204/medgemma_1_5", overwrite=True)
medgemma = foz.load_zoo_model("google/medgemma-1.5-4b-it")

medgemma.operation = "vqa"
dataset.apply_model(medgemma, label_field="predictions", prompt_field="question")

# Evaluate
session = fo.launch_app(dataset)
```

---

## 📈 Evaluation

After fine-tuning, use the notebook's earlier sections to assess performance:

1. **Accuracy by modality** — Does the model perform better on CT vs MRI vs X-ray?
2. **Accuracy by body part** — Which anatomical regions are hardest?
3. **Error analysis** — What patterns appear in failures?
4. **Similarity search** — Are similar images also failing?


---

## 📚 Resources

### Documentation
- [FiftyOne Documentation](https://docs.voxel51.com/)
- [FiftyOne PyTorch Integration](https://docs.voxel51.com/api/fiftyone.utils.torch.html)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [PEFT LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)

### Models
- [MedGemma 1.5 4B IT](https://huggingface.co/google/medgemma-1.5-4b-it)
- [MedSigLIP 448](https://huggingface.co/google/medsiglip-448)
- [Google Health AI Developer Foundations](https://developers.google.com/health-ai-developer-foundations/medgemma)

### Competition
- [MedGemma Impact Challenge on Kaggle](https://www.kaggle.com/competitions/med-gemma-impact-challenge)

---

## ⚠️ Important Notes

- **Not for clinical use**: MedGemma models are research tools and have not been validated for clinical diagnosis or treatment decisions.
- **Privacy**: Ensure all data used respects patient privacy and HIPAA/GDPR requirements.
- **Gated access**: You must accept the model license on Hugging Face before downloading.

---


## 📄 License

This project is for educational and competition purposes. Model usage is governed by Google's Health AI Developer Foundations terms of use.

---

**Good luck with the challenge! 🏆**
