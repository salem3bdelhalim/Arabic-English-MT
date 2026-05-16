# 🌐 Arabic → English Neural Machine Translation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

**Fine-tuning Helsinki-NLP/opus-mt-ar-en on 250,000 Arabic–English parallel sentences**  
with full EDA · Preprocessing · Training · Evaluation · BERTScore · Error Analysis

[Overview](#-overview) · [Results](#-results) · [Installation](#-installation) · [Usage](#-usage) · [Project Structure](#-project-structure) · [Methodology](#-methodology)

</div>

---

## 📖 Overview

This project presents a production-ready **Arabic-to-English Neural Machine Translation (NMT)** pipeline built on top of the pre-trained [`Helsinki-NLP/opus-mt-ar-en`](https://huggingface.co/Helsinki-NLP/opus-mt-ar-en) MarianMT model, fine-tuned on 250,000 samples from the [opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) dataset.

The pipeline covers the entire ML lifecycle: from raw data exploration and Arabic-specific text normalization, through model training with mixed-precision and cosine scheduling, to multi-metric evaluation including BLEU, ChrF, ROUGE, and BERTScore — with a dedicated error analysis section.

---

## 📝 Description

Arabic is one of the world's most morphologically complex languages — with rich inflection, cliticization, optional diacritics, and a script that reads right-to-left. Building a reliable Arabic→English translation system demands more than just plugging data into a model; it requires careful linguistic preprocessing, domain-aware training, and rigorous evaluation.

This project fine-tunes **Helsinki-NLP's `opus-mt-ar-en`** — a MarianMT Transformer trained on the OPUS multilingual corpus — on **250,000 curated Arabic–English sentence pairs** drawn from the `opus-100` dataset. The result is a high-performance translation model that generalizes across domains including news, legal, medical, and conversational text.

### What makes this project stand out

- **Arabic-aware preprocessing** — diacritics (tashkeel), tatweel/kashida elongation, and Alef variant normalization are handled before tokenization, reducing vocabulary noise and improving alignment quality.
- **Full evaluation suite** — the model is benchmarked with BLEU (n-gram precision), ChrF (character-level F-score, particularly well-suited for Arabic's morphological richness), ROUGE-1/2/L, and BERTScore (deep contextual similarity), giving a 360° view of translation quality.
- **Transparent training** — loss curves, per-epoch metric progression, and early stopping ensure the model is not over- or under-trained.
- **Error analysis** — a dedicated analysis section identifies failure modes: short vs. long sentences, length deviation, and per-sentence BLEU vs. BERTScore scatter, making the model's weaknesses interpretable.
- **Production-ready inference** — the trained model is saved and can be loaded in two lines of code for real-time translation.

### Sample translations from the fine-tuned model

| Arabic | English (predicted) |
|--------|-------------------|
| مرحبا بكم في مؤتمر الذكاء الاصطناعي السنوي. | Welcome to the annual AI conference. |
| تحتل مصر مكانة بارزة في تاريخ الحضارة الإنسانية. | Egypt occupies a prominent place in the history of human civilization. |
| يعتبر التعلم الآلي أحد أبرز مجالات تقنية المعلومات في العصر الحديث. | Machine learning is one of the most prominent fields of information technology in the modern era. |
| الأسرة هي اللبنة الأساسية في بناء المجتمع. | The family is the basic building block of society. |
| يسعدنا الإعلان عن إطلاق منتجنا الجديد في الأسواق العالمية. | We are pleased to announce the launch of our new product in global markets. |

---

## 🏆 Results

Fine-tuning yields strong performance across all standard MT benchmarks:

| Metric | Score | Description |
|--------|-------|-------------|
| **BLEU** | **43.02** | n-gram precision (SacreBLEU) |
| **ChrF** | **63.43** | Character n-gram F-score |
| **ROUGE-1** | **0.6642** | Unigram overlap |
| **ROUGE-2** | **0.4781** | Bigram overlap |
| **ROUGE-L** | **0.6474** | Longest common subsequence |

> A BLEU score of **43** is considered *high-quality* translation output — approaching human-level fluency on general-domain text, and well above the ≥30 threshold typically cited as "understandable, good translation" in the research literature.

---

## 🏗 Architecture

```
Input (Arabic)
     │
     ▼
┌─────────────────────────────────┐
│   Helsinki-NLP/opus-mt-ar-en    │
│         (MarianMT)              │
│                                 │
│  Encoder (6 layers, d=512)      │
│       ↓                         │
│  Cross-Attention                │
│       ↓                         │
│  Decoder (6 layers, d=512)      │
└─────────────────────────────────┘
     │
     ▼
Output (English)
```

| Component | Detail |
|-----------|--------|
| Base model | `Helsinki-NLP/opus-mt-ar-en` |
| Architecture | MarianMT (Transformer Seq2Seq) |
| Encoder / Decoder layers | 6 / 6 |
| Hidden size | 512 |
| Attention heads | 8 |
| Vocabulary size | 65,839 |
| Total parameters | ~77M |

---

## 📊 Methodology

### 1. Exploratory Data Analysis
- Sentence length distributions (Arabic & English)
- AR vs EN length scatter with regression line
- EN/AR translation length ratio histogram
- Top-30 most frequent token frequency charts
- Data quality report (nulls, duplicates, empty pairs)

### 2. Preprocessing
Arabic-specific normalization applied before tokenization:

| Step | What it does |
|------|-------------|
| Diacritics removal | Strips tashkeel (ً ٌ ٍ َ ُ ِ ّ ْ) |
| Tatweel removal | Removes kashida elongation character (ـ) |
| Alef normalization | Unifies أ إ آ → ا |
| Teh Marbuta | Normalizes ة → ه |
| URL / HTML stripping | Removes noise from web-sourced data |
| Length filtering | Keeps pairs with 3–128 words |

### 3. Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 5 (early stopping, patience=2) |
| Batch size | 32 (train) / 64 (eval) |
| Gradient accumulation | 2 steps (effective batch = 64) |
| Learning rate | 3e-5 |
| LR scheduler | Cosine with warmup (5%) |
| Weight decay | 0.01 |
| Precision | FP16 (mixed precision) |
| Max sequence length | 128 tokens |
| Best model criterion | BLEU ↑ |

### 4. Evaluation Suite
- **SacreBLEU** — standardized n-gram precision MT metric
- **ChrF** — character-level F-score, robust for morphologically rich Arabic
- **ROUGE-1 / 2 / L** — recall-oriented overlap metrics
- **BERTScore** — contextual embedding similarity (P / R / F1) via `bert-base-uncased`

### 5. Error Analysis
- Per-sentence BLEU vs BERTScore scatter plot
- Translation length deviation histogram (predicted − reference)
- BLEU bucket distribution across the test set
- Qualitative inspection of best and worst translations

---

## 📁 Project Structure

```
arabic-english-nmt/
│
├── arabic_english_nmt.ipynb      # Main notebook (full pipeline)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── outputs/
│   ├── ar_en_finetuned/          # Saved model & tokenizer
│   └── plots/
│       ├── eda_length_dist.png
│       ├── eda_scatter.png
│       ├── eda_ratio.png
│       ├── eda_top_tokens.png
│       ├── prep_token_lengths.png
│       ├── loss_curves.png
│       ├── metric_curves.png
│       ├── bertscore_distribution.png
│       ├── error_bleu_vs_bert.png
│       ├── error_len_deviation.png
│       ├── error_bleu_buckets.png
│       └── final_summary_dashboard.png
```

---

## ⚙️ Installation

### Requirements
- Python 3.10+
- CUDA-capable GPU recommended (Tesla T4 or better)

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/arabic-english-nmt.git
cd arabic-english-nmt

# Create a virtual environment
python -m venv venv
source venv/bin/activate       # Linux/macOS
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
transformers==4.44.2
datasets
sacrebleu
evaluate
bert_score
rouge_score
sentencepiece
accelerate==0.34.0
peft==0.12.0
seaborn
matplotlib
tqdm
torch>=2.0
```

---

## 🚀 Usage

### Run the full pipeline

Open and run `arabic_english_nmt.ipynb` end-to-end in Jupyter or Google Colab.

### Inference — translate your own text

```python
from transformers import MarianMTModel, MarianTokenizer

MODEL_PATH = "./ar_en_finetuned"   # or "Helsinki-NLP/opus-mt-ar-en" for base
tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH)
model     = MarianMTModel.from_pretrained(MODEL_PATH)

def translate(text: str) -> str:
    inputs = tokenizer([text], return_tensors="pt",
                       padding=True, truncation=True, max_length=128)
    output = model.generate(**inputs, num_beams=5, max_length=128)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(translate("يعمل الباحثون على تطوير نماذج لغوية كبيرة قادرة على فهم اللغة العربية."))
# → "Researchers are working on developing large language models capable of understanding Arabic."
```

### Run on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/arabic-english-nmt/blob/main/arabic_english_nmt.ipynb)

> Make sure to select **Runtime → Change runtime type → T4 GPU** for full-speed training.

---

## 📈 Training Curves

The model converges smoothly with no overfitting, as shown by the tight gap between training and validation loss across all 5 epochs. Early stopping triggered after the best BLEU checkpoint was confirmed.

---

## 🔬 Dataset

| Split | Samples | Source |
|-------|---------|--------|
| Train | 225,000 | `Helsinki-NLP/opus-100` (ar-en) |
| Validation | 12,500 | `Helsinki-NLP/opus-100` (ar-en) |
| Test | 12,500 | `Helsinki-NLP/opus-100` (ar-en) |
| **Total** | **250,000** | |

The opus-100 dataset is a multilingual corpus covering 100 languages sampled from the OPUS collection of parallel corpora, covering domains such as news, legal, medical, and conversational text.

---

## 🔭 Future Work

- [ ] Experiment with larger base models (`Helsinki-NLP/opus-mt-tc-big-ar-en`)
- [ ] Domain adaptation on medical / legal Arabic corpora
- [ ] Back-translation data augmentation
- [ ] Beam search hyperparameter tuning (beam width, length penalty)
- [ ] Deployment via FastAPI + Docker container
- [ ] Integrate with a Gradio web demo

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) for the pre-trained `opus-mt-ar-en` model
- [HuggingFace](https://huggingface.co) for the Transformers, Datasets, and Evaluate libraries
- [OPUS](https://opus.nlpl.eu/) for the parallel corpora powering the opus-100 dataset

---

<div align="center">
Made with ❤️ | If you find this useful, please ⭐ the repo
</div>
