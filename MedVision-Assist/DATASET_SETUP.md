# Dataset Setup Guide

Este projeto foi limpo de dados originais por privacidade. Use datasets públicos para desenvolvimento e testes.

## Datasets Públicos Recomendados

### 1. BraTS (Brain Tumor Segmentation)
- **Descrição**: Brain Tumor Segmentation Challenge dataset
- **Link**: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation
- **Formato**: NIfTI (.nii.gz)
- **Casos**: ~369 pacientes

**Setup**:
```bash
cd data/
kaggle datasets download -d awsaf49/brats20-dataset-training-validation
unzip brats20-dataset-training-validation.zip
```

### 2. Medical Segmentation Decathlon
- **Descrição**: Multi-organ segmentation from CT/MRI
- **Link**: http://medicaldecathlon.com/
- **Formatos**: NIfTI
- **Casos**: ~600+ pacientes

**Setup**:
```bash
cd data/
# Download do site manualmente ou use script provided
```

### 3. ISLES (Ischemic Stroke Lesion Segmentation)
- **Link**: https://www.kaggle.com/datasets/rosikri/isles2018
- **Formato**: NIfTI
- **Descrição**: Stroke lesion segmentation

## Estrutura de Diretórios

```
data/
├── train/
│   ├── images/     # NIfTI ou DICOM files
│   └── masks/      # Segmentation masks
├── val/
│   ├── images/
│   └── masks/
└── test/
    └── images/
```

## Dados Sintéticos (Para Testes Rápidos)

Para desenvolvimento rápido sem dados reais:

```python
import nibabel as nib
import numpy as np

# Gerar imagem sintética
img_data = np.random.rand(128, 128, 64).astype(np.float32) * 255
img = nib.Nifti1Image(img_data, affine=np.eye(4))
nib.save(img, 'data/train/images/synthetic_001.nii.gz')

# Gerar máscara sintética
mask_data = (np.random.rand(128, 128, 64) > 0.7).astype(np.uint8)
mask = nib.Nifti1Image(mask_data, affine=np.eye(4))
nib.save(mask, 'data/train/masks/synthetic_001.nii.gz')
```

## Treinamento com Dados Públicos

```bash
python src/train.py --seed 42 --experiment-name brats_experiment
```

O script carregará dados de `data/train/` automaticamente.

## Notas

- Todos os datasets devem estar em formato NIfTI ou DICOM
- Respeite as licenças dos datasets utilizados
- Para produção, considere usar Azure Blob Storage para armazenar dados
- Backup local recomendado antes de subir ao GitHub
