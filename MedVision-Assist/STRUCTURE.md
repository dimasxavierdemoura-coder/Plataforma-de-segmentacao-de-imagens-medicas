# Estrutura do Projeto MedVision Assist

Visão geral da organização e arquivos do projeto.

## 📂 Árvore de Diretórios

```
MedVision-Assist/
│
├── 📄 README.md                    # Documentação principal
├── 📄 CHANGELOG.md                 # Histórico de versões
├── 📄 CONTRIBUTING.md              # Guia de contribuição
├── 📄 DATASET_SETUP.md             # Configuração de datasets públicos
├── 📄 LICENSE                      # Licença MIT
├── 📄 STRUCTURE.md                 # Este arquivo
│
├── 📋 requirements.txt             # Dependências Python
├── 📋 Dockerfile                   # Containerização
├── 📋 .dockerignore                # Exclusões Docker
├── 📋 .gitignore                   # Exclusões Git
├── 📋 .gitattributes               # Configurações de line endings
│
├── 🔧 start.bat                    # Script de setup (Windows)
│
├── 📁 src/                         # Código-fonte
│   ├── 📄 __init__.py             # Package init
│   ├── 📄 app.py                  # Interface Streamlit
│   ├── 📄 train.py                # Pipeline de treinamento
│   ├── 📄 inference.py            # Pipeline de inferência
│   ├── 📄 utils.py                # Utilidades e métricas
│   │
│   ├── 📁 model/                   # Arquitetura de ML
│   │   ├── 📄 __init__.py
│   │   └── 📄 unet.py             # Arquitetura U-Net
│   │
│   ├── 📁 data/                    # Processamento de dados
│   │   ├── 📄 __init__.py
│   │   └── 📄 preprocessing.py    # Pré-processamento
│   │
│   └── 📁 scripts/                 # Scripts utilitários
│       ├── 📄 download_mama_mia_complete.py
│       ├── 📄 download_mama_mia_synapse.py
│       ├── 📄 organize_mama_mia_data.py
│       └── 📄 generate_synthetic_dataset.py
│
├── 📁 tests/                       # Testes automatizados
│   ├── 📄 test_utils.py           # Testes de utilidades
│   └── 📄 test_inference.py       # Testes de inference
│
├── 📁 data/                        # Dados do projeto (não versionado)
│   ├── train/                      # Dataset de treinamento
│   ├── val/                        # Dataset de validação
│   ├── test/                       # Dataset de teste
│   └── 📄 .gitkeep                # Rastrear diretório vazio
│
├── 📁 models/                      # Modelos treinados (não versionado)
│   └── 📄 .gitkeep                # Rastrear diretório vazio
│
└── 📁 results/                     # Resultados (não versionado)
    └── 📄 .gitkeep                # Rastrear diretório vazio
```

## 📄 Arquivos Importantes

### Documentação
- **README.md**: Documentação principal, quick start e features
- **CHANGELOG.md**: Histórico de versões e roadmap
- **CONTRIBUTING.md**: Guia para contribuidores
- **DATASET_SETUP.md**: Como configurar datasets públicos
- **LICENSE**: Licença MIT do projeto

### Código-Fonte

#### Aplicação Principal
- **src/app.py**: Interface Streamlit para upload, visualização e inferência
- **src/train.py**: Pipeline de treinamento com logging e reproducibilidade
- **src/inference.py**: Carregamento de modelos e geração de predições
- **src/utils.py**: Funções de utilidade (métricas, visualização, relatórios)

#### Arquitetura de ML
- **src/model/unet.py**: Modelo U-Net para segmentação médica

#### Processamento de Dados
- **src/data/preprocessing.py**: Normalização, redimensionamento, transformações

#### Scripts Utilitários
- **src/scripts/download_mama_mia_complete.py**: Download automático MAMA-MIA
- **src/scripts/generate_synthetic_dataset.py**: Geração de dados sintéticos para teste

### Testes
- **tests/test_utils.py**: 3 testes para funções de utilidade
- **tests/test_inference.py**: 3 testes para pipeline de inference

### Configuração
- **requirements.txt**: Todas as dependências Python
- **Dockerfile**: Imagem Docker pronta para produção
- **.gitignore**: Exclusões para Git (data/, models/, cache, etc)
- **.dockerignore**: Exclusões para Docker
- **.gitattributes**: Configurações de line endings por tipo de arquivo
- **start.bat**: Script de setup automático para Windows

## 🗂️ Diretórios de Dados (Vazios no Repositório)

Estes diretórios são mantidos via `.gitkeep` mas não contêm arquivos no repositório:

```
data/                    # Datasets para treinamento
  ├── train/            # ~80% dos dados
  ├── val/              # ~10% dos dados
  └── test/             # ~10% dos dados

models/                  # Modelos treinados (.pth)
  └── unet_medvision.pth (não versionado)

results/                 # Resultados de inferência
  ├── masks/
  ├── reports/
  └── logs/
```

## 📦 Dependências Principais

Ver `requirements.txt` para lista completa:

- **PyTorch**: Deep learning framework
- **Streamlit**: Interface web interativa
- **NumPy/SciPy**: Processamento numérico
- **Nibabel**: Leitura de NIfTI
- **pydicom**: Suporte a DICOM
- **Matplotlib**: Visualização
- **pytest**: Framework de testes

## 🚀 Como Usar

### Desenvolvimento Local

```bash
# Setup
pip install -r requirements.txt

# Executar interface
streamlit run src/app.py

# Treinar modelo
python src/train.py --seed 42

# Testes
python -m pytest tests/ -v
```

### Docker

```bash
# Build
docker build -t medvision-assist .

# Run
docker run -p 8501:8501 medvision-assist
```

## 📊 Status do Projeto

- ✅ Código-fonte completo
- ✅ Testes passando (6/6)
- ✅ Documentação completa
- ✅ Docker pronto
- ⏳ Datasets: Use públicos (BraTS, Medical Segmentation Decathlon)
- ⏳ Models: Treinar localmente com seus dados

## 🔄 Próximas Etapas para Produção

1. **GitHub**: Fazer upload do repositório
2. **CI/CD**: Configurar GitHub Actions
3. **Azure**: Deploy em Azure App Service ou Azure Container Apps
4. **Datasets**: Integrar datasets públicos ou seus próprios
5. **Melhorias v2.0**: Implementar roadmap planejado

---

**Criado**: 20 de Abril de 2026
**Status**: Pronto para repositório público
