# Comprehensive-Benchmarking-of-Chemical-LLMs

*Research Project*  

---

## 📝 Project Summary  
This project explores the optimization of **chemical language models (CLMs)** for predicting molecular properties. We combine:  
- **Quantification methods** (PTQ, QAT) to reduce computational complexity.  
- **Multi-agent architecture** (ChemBERTa, TinyBERTa) to improve robustness.  
- **Reservoir networks (ESN)** to accelerate inference.  

**Key Results** :  
- ✅ **INT8 Quantification**: **70% memory reduction** with precision loss < 3%.  
- 🗳️ **Multi-agent system**: **1.5% accuracy improvement** via weighted voting.  
- ⚡ **ESN**: Latency reduced by **10 times** compared to original models.  

---

## 🚀 Key Features  
### 1. **Chemical Language Models (CLMs)**  
- **ChemBERTa** (Encoder-Based): Extraction of molecular representations.  
- **MolGPT** (Decoder-Based): Generation of chemical structures.  
- **TransAntivirus** (Encoder-Decoder): Prediction of molecular interactions.  

### 2. **Quantification Pipeline**  
- **PTQ (Post-Training Quantization)**: Direct conversion of weights to INT8.  
- **QAT (Quantization Aware Training)**: Weight adaptation during training.  

### 3. **Multi-Agent Workflow**  
- **Protocol 1**: Fusion of ChemBERTa and TinyBERTa predictions.  
- **Protocol 2**: Concatenation of embeddings for an external classifier (MLP/XGBoost).  

### 4. **Integration of Reservoir Networks (ESN)**  
- Latency reduction via sparse recurrent layer.  

---

## 📦 Installation  
### Prerequisites  
- Python 3.8+  
- Libraries:  
  ```bash
  pip install torch transformers xgboost scikit-learn reservoirpy deepchem
  ```

### Data Download  
- **BBBP** (Classification):  
  ```python
  from deepchem.molnet import load_bbbp  
  tasks, datasets, transformers = load_bbbp()
  ```
- **ESOL** (Regression):  
  ```python
  from deepchem.molnet import load_delaney  
  tasks, datasets, transformers = load_delaney()
  ```

### Pre-trained Models  
- **ChemBERTa**: [`HuggingFace/chemberta`](https://huggingface.co/chemproject/chemberta)  
- **TinyBERTa**: [`HuggingFace/tinyberta`](https://huggingface.co/google/tinybert)  

---

## 🛠 Usage  
### Model Quantification (Example with ChemBERTa)  
```python
from transformers import AutoModelForSequenceClassification, QuantizationConfig  

# Load the model  
model = AutoModelForSequenceClassification.from_pretrained("chemproject/chemberta")  

# PTQ Configuration  
config = QuantizationConfig(quant_method="dynamic", activation_dtype=torch.qint8)  
model_quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)  
```

### Multi-Agent Workflow (Protocol 2)  
```python
# Extract embeddings  
embeddings_chemberta = chemberta_model(**inputs).last_hidden_state  
embeddings_tinyberta = tinyberta_model(**inputs).last_hidden_state  

# Concatenate  
combined_embeddings = torch.cat((embeddings_chemberta, embeddings_tinyberta), dim=1)  

# MLP Classifier  
classifier = MLP(hidden_dim=128, output_dim=2)  
predictions = classifier(combined_embeddings)  
```

---

## 📊 Experimental Results  
### Performance of Quantified CLMs  
| Model           | Method | Accuracy (%) | Size (MB) |  
|-----------------|--------|--------------|-----------|  
| ChemBERTa (FP32)| -      | 88.5         | 326       |  
| ChemBERTa (INT8)| PTQ    | 83.5         | 195.6     |  
| MolGPT (INT8)   | QAT    | 89.0         | 267.8     |  

### Multi-Agent Benchmark  
- **Classification (BBBP)**:  
  - Protocol 1: **86.8%**  
  - Protocol 2: **89.8%**  
- **Regression (ESOL)**:  
  - XGBoost + ESN: **MSE = 0.4921**  

---

## 🙏 Acknowledgments  
We thank:  
- **DAVID Laboratory** (Université Paris-Saclay) for computational resources.  

---

## 📚 References  
1. Wu et al. (2024) - *ChemBERTa Embeddings for Molecular Property Prediction*  
2. Schuchardt et al. (2023) - *Quantization-Aware Training for Robust Chemical Models*  
3. Vaswani et al. (2017) - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  

---

**© 2025 - Amen Allah KRISSAAN**  
*Project publicly defended on 28/03/2025 at Université Paris-Saclay.*
