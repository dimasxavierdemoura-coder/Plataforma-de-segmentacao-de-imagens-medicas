# MedVision Assist

Plataforma web avançada de segmentação médica com suporte a NIfTI e DICOM, métricas clínicas, treinamento reprodutível e interface Streamlit. Implementação em nível de produção com testes automatizados e Docker.

![Status](https://img.shields.io/badge/status-Active-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/license-MIT-green)

## 📋 Características

- ✅ **Segmentação U-Net**: Arquitetura moderna para segmentação médica de múltiplos órgãos
- ✅ **Suporte a Formatos**: NIfTI (.nii, .nii.gz) e DICOM (.dcm)
- ✅ **Métricas Clínicas**: Dice, IoU, Sensibilidade, Especificidade, F1-Score
- ✅ **Interface Interativa**: Streamlit para visualização 2D/3D e análise comparativa
- ✅ **Processamento em Lote**: Inferência paralela para datasets grandes
- ✅ **Reprodutibilidade**: Seed control, logging automático, experiment tracking
- ✅ **Containerização**: Dockerfile pronto para produção
- ✅ **Testes Automatizados**: pytest com cobertura para utilities e inference
- ✅ **Relatórios JSON**: Exportação estruturada de resultados de inferência

## 🚀 Quick Start

### Pré-requisitos

- Python 3.10+
- GPU CUDA 11.8+ (opcional, mas recomendado)
- Windows 10+ ou Linux

### Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/MedVision-Assist.git
cd MedVision-Assist

# Crie um ambiente virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Instale dependências
pip install -r requirements.txt
```

### Setup de Dados

Veja [DATASET_SETUP.md](DATASET_SETUP.md) para configurar datasets públicos como BraTS ou Medical Segmentation Decathlon.

### Executar a Interface Web

```bash
streamlit run src/app.py
```

Acesse em `http://localhost:8501`

### Treinar um Modelo

```bash
# Treino com seed reprodutível
python src/train.py --seed 42 --experiment-name my_experiment

# Visualize logs em data/experiments/
```

### Executar Inferência

```bash
python src/inference.py --image data/test/image.nii.gz --model models/unet_medvision.pth
```

## 📁 Estrutura do Projeto

```
MedVision-Assist/
├── src/
│   ├── app.py                  # Interface Streamlit
│   ├── train.py                # Pipeline de treinamento
│   ├── inference.py            # Inference pipeline
│   ├── utils.py                # Utilidades (métricas, visualização)
│   ├── model/
│   │   └── unet.py            # Arquitetura U-Net
│   ├── data/
│   │   └── preprocessing.py    # Pré-processamento de imagens
│   └── scripts/
│       └── download_datasets.py
├── tests/
│   ├── test_utils.py           # Testes de utilidades
│   └── test_inference.py       # Testes de inference
├── data/
│   ├── train/                  # Dataset de treinamento
│   ├── val/                    # Dataset de validação
│   └── test/                   # Dataset de teste
├── models/                     # Modelos treinados
├── results/                    # Resultados de inferência
├── requirements.txt            # Dependências Python
├── Dockerfile                  # Container para produção
├── start.bat                   # Script de setup (Windows)
└── README.md                   # Este arquivo
```

## 🧪 Testes

```bash
# Executar todos os testes
python -m pytest tests/ -v

# Testes com cobertura
python -m pytest tests/ --cov=src --cov-report=html
```

Status: ✅ **6/6 testes passando**

## 🐳 Docker

```bash
# Build da imagem
docker build -t medvision-assist .

# Executar container
docker run -p 8501:8501 medvision-assist
```

## 📊 Métricas de Segmentação

O projeto calcula as seguintes métricas automaticamente:

- **Dice Coefficient**: Sobreposição de segmentação
- **IoU (Intersection over Union)**: Métrica de comparação
- **Sensibilidade**: Verdadeiros Positivos / (VP + FN)
- **Especificidade**: Verdadeiros Negativos / (VN + FP)
- **F1-Score**: Média harmônica de Precisão e Recall

## 🔍 Inferência

Exemplo de código:

```python
from src.inference import load_nifti_image, normalize_image, predict_volume
import torch

# Carregar imagem
image_volume = load_nifti_image("data/test/image.nii.gz")
image_normalized = normalize_image(image_volume)

# Prever
model = torch.load("models/unet_medvision.pth")
predictions = predict_volume(model, image_normalized, device="cuda")
```

## 📝 Configuração e Logging

O projeto gera logs automáticos:

```
logs/
├── experiment_metadata.json    # Informações do experimento
├── training_log.csv            # Histórico de treinamento
└── training_metrics.json       # Métricas consolidadas
```

## 🛠️ Tecnologias

- **PyTorch**: Deep learning framework
- **Streamlit**: Web UI interativa
- **Nibabel**: Leitura de NIfTI
- **pydicom**: Suporte a DICOM
- **NumPy/SciPy**: Processamento numérico
- **pytest**: Testes automatizados

## 🚀 Próximas Melhorias (v2.0)

- [ ] Integração com Azure AI Foundry
- [ ] Visualização 3D interativa (Plotly/VTK)
- [ ] Autenticação OAuth2 (Azure AD)
- [ ] Dashboard de métricas em tempo real
- [ ] Suporte a multi-GPU (DistributedDataParallel)
- [ ] API FastAPI para integração
- [ ] CI/CD com GitHub Actions

## 📖 Documentação

- [Dataset Setup Guide](DATASET_SETUP.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Architecture Overview](docs/ARCHITECTURE.md) *(em construção)*

## 📄 Licença

MIT License - veja LICENSE para detalhes

## 👤 Autor

Desenvolvido como uma solução de nível NASA para análise de imagens médicas com IA.

## 📧 Suporte

Para issues ou sugestões, abra uma issue no GitHub.

---

**Última atualização**: 20 de Abril de 2026
**Status do Projeto**: ✅ Pronto para produção
