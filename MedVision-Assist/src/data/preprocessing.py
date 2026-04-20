import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import nibabel as nib

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

class MedicalImageDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, transform=None, target_size=(256, 256)):
        # Filtrar apenas pares que têm tanto imagem quanto máscara
        if mask_paths:
            valid_pairs = []
            for img_path, mask_path in zip(image_paths, mask_paths):
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    valid_pairs.append((img_path, mask_path))
            
            self.image_paths = [pair[0] for pair in valid_pairs]
            self.mask_paths = [pair[1] for pair in valid_pairs]
        else:
            self.image_paths = [p for p in image_paths if os.path.exists(p)]
            self.mask_paths = None
            
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self._load_image(self.image_paths[idx])
        mask = None
        if self.mask_paths and idx < len(self.mask_paths):
            mask = self._load_mask(self.mask_paths[idx])

        sample = {"image": image, "mask": mask}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_image(self, path):
        if path.lower().endswith(('.nii', '.nii.gz')):
            # Carregar imagem NIfTI
            nii_img = nib.load(path)
            image = nii_img.get_fdata()
            
            if image.ndim == 3:
                image = image[:, :, image.shape[2] // 2]
            elif image.ndim == 4:
                image = image[:, :, image.shape[2] // 2, 0]
            
            image = ((image - np.nanmin(image)) / (np.nanmax(image) - np.nanmin(image)) * 255).astype(np.uint8)
            image = Image.fromarray(image).convert("RGB")
        elif path.lower().endswith('.dcm'):
            if not PYDICOM_AVAILABLE:
                raise ImportError("pydicom não está instalado, instale-o para carregar arquivos DICOM")
            dicom_data = pydicom.dcmread(path)
            image = dicom_data.pixel_array.astype(np.float32)
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            image = Image.fromarray(image).convert("RGB")
        else:
            image = Image.open(path).convert("RGB")
        
        image = image.resize(self.target_size)
        image = np.array(image).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        return image

    def _load_mask(self, path):
        if path.lower().endswith(('.nii', '.nii.gz')):
            nii_img = nib.load(path)
            mask = nii_img.get_fdata()
            
            if mask.ndim == 3:
                mask = mask[:, :, mask.shape[2] // 2]
            elif mask.ndim == 4:
                mask = mask[:, :, mask.shape[2] // 2, 0]
            
            mask = (mask > 0).astype(np.uint8)
            mask = Image.fromarray(mask * 255).convert("L")
        elif path.lower().endswith('.dcm'):
            if not PYDICOM_AVAILABLE:
                raise ImportError("pydicom não está instalado, instale-o para carregar máscaras DICOM")
            dicom_data = pydicom.dcmread(path)
            mask = dicom_data.pixel_array.astype(np.float32)
            mask = (mask > 0).astype(np.uint8)
            mask = Image.fromarray(mask * 255).convert("L")
        else:
            mask = Image.open(path).convert("L")
        
        mask = mask.resize(self.target_size)
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)
        return mask


def list_images(folder, extensions={"nii", "nii.gz", "png", "jpg", "jpeg", "dcm"}):
    entries = []
    for filename in os.listdir(folder):
        for ext in extensions:
            if filename.lower().endswith(ext.lower()):
                entries.append(os.path.join(folder, filename))
                break
    return sorted(entries)


def _find_data_dir(root_dir, candidates):
    for name in candidates:
        path = os.path.join(root_dir, name)
        if os.path.isdir(path):
            return path
    return None


def find_image_dir(root_dir):
    return _find_data_dir(root_dir, ["images", "imgs", "image", "scan", "scans", "data", "dicom", "dcm"])


def find_mask_dir(root_dir):
    return _find_data_dir(root_dir, ["masks", "labels", "seg", "segmentation", "segmentations", "annotation", "annotations", "label"])


def validate_dataset(image_paths, mask_paths=None, max_samples=20):
    def _check_path(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        if path.lower().endswith((".nii", ".nii.gz")):
            try:
                nib.load(path)
            except Exception as exc:
                raise ValueError(f"Erro ao carregar NIfTI: {path} ({exc})")
        if path.lower().endswith('.dcm') and not PYDICOM_AVAILABLE:
            raise ImportError("pydicom não está instalado para processar arquivos DICOM")

    print("Validando dataset...")
    for image_path in image_paths[:max_samples]:
        _check_path(image_path)
    if mask_paths:
        for mask_path in mask_paths[:max_samples]:
            _check_path(mask_path)

    if mask_paths:
        for idx, (img_path, mask_path) in enumerate(zip(image_paths[:max_samples], mask_paths[:max_samples])):
            try:
                if img_path.lower().endswith(('.nii', '.nii.gz')):
                    img = nib.load(img_path).get_fdata()
                elif img_path.lower().endswith('.dcm'):
                    img = pydicom.dcmread(img_path).pixel_array
                else:
                    img = np.array(Image.open(img_path))

                if mask_path.lower().endswith(('.nii', '.nii.gz')):
                    mask = nib.load(mask_path).get_fdata()
                elif mask_path.lower().endswith('.dcm'):
                    mask = pydicom.dcmread(mask_path).pixel_array
                else:
                    mask = np.array(Image.open(mask_path))

                if img.ndim >= 2 and mask.ndim >= 2 and img.shape[0:2] != mask.shape[0:2]:
                    print(f"Atenção: dimensões diferentes entre imagem e máscara no par {idx}: {img.shape} vs {mask.shape}")
            except Exception as exc:
                print(f"Atenção: erro ao validar par {idx} ({img_path}, {mask_path}): {exc}")

    print("Validação do dataset concluída.")


def build_dataset_from_root(root_dir, manifest_file=None):
    """
    Constrói listas de caminhos de imagens e máscaras a partir de um diretório raiz.
    Espera estrutura: root_dir/images/ e root_dir/masks/
    Para datasets médicos onde uma máscara 3D corresponde a múltiplas imagens 2D.
    """
    if manifest_file:
        manifest_path = manifest_file if os.path.isabs(manifest_file) else os.path.join(root_dir, manifest_file)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        image_paths = []
        mask_paths = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = line.strip()
                if not entry or entry.startswith("#"):
                    continue
                parts = [p.strip() for p in entry.split(",") if p.strip()]
                if len(parts) < 2:
                    continue
                image_paths.append(parts[0])
                mask_paths.append(parts[1])

        validate_dataset(image_paths, mask_paths)
        return image_paths, mask_paths

    images_dir = find_image_dir(root_dir)
    masks_dir = find_mask_dir(root_dir)

    if images_dir is None and masks_dir is None:
        print(f"Nenhuma pasta de imagens ou máscaras encontrada em {root_dir}. Tentando usar o diretório raiz diretamente.")
        images_dir = root_dir
        masks_dir = root_dir

    if images_dir is None:
        raise FileNotFoundError(f"Diretório de imagens não encontrado em {root_dir}")
    if masks_dir is None:
        raise FileNotFoundError(f"Diretório de máscaras/labels não encontrado em {root_dir}")

    image_extensions = {"nii", "nii.gz", "png", "jpg", "jpeg", "dcm"}
    mask_extensions = {"nii", "nii.gz", "png", "jpg", "jpeg", "dcm"}

    image_paths = list_images(images_dir, image_extensions)
    mask_paths = list_images(masks_dir, mask_extensions)

    # Criar dicionário de máscaras por prefixo
    mask_dict = {}
    for mask_path in mask_paths:
        mask_name = os.path.basename(mask_path)
        # Remover extensão para obter prefixo
        prefix = mask_name.replace('.nii.gz', '').replace('.nii', '')
        mask_dict[prefix] = mask_path

    # Parear imagens com máscaras baseadas no prefixo
    paired_images = []
    paired_masks = []
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        base_name = img_name.replace('.nii.gz', '').replace('.nii', '')
        if base_name in mask_dict:
            paired_images.append(img_path)
            paired_masks.append(mask_dict[base_name])
            continue

        parts = base_name.split('_')
        if len(parts) >= 2:
            prefix = '_'.join(parts[:2])
            if prefix in mask_dict:
                paired_images.append(img_path)
                paired_masks.append(mask_dict[prefix])
                continue

    if not paired_images:
        raise ValueError("Nenhuma imagem foi pareada com máscaras. Verifique a estrutura e os nomes dos arquivos.")

    validate_dataset(paired_images, paired_masks)

    print(f"Total de imagens: {len(image_paths)}")
    print(f"Total de máscaras: {len(mask_paths)}")
    print(f"Imagens pareadas: {len(paired_images)}")
    print(f"Máscaras pareadas: {len(paired_masks)}")

    return paired_images, paired_masks


def split_dataset(image_paths, mask_paths, val_fraction=0.2, test_fraction=0.0, seed=42):
    if len(image_paths) != len(mask_paths):
        raise ValueError("Número de imagens e máscaras deve ser igual para separar train/val/test")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("A soma de val_fraction e test_fraction deve ser menor que 1.0")

    indices = np.arange(len(image_paths))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    test_count = int(len(indices) * test_fraction)
    val_count = int(len(indices) * val_fraction)
    test_count = max(0, min(test_count, len(indices) - 2))
    val_count = max(1, min(val_count, len(indices) - test_count - 1))

    val_idx = indices[:val_count]
    test_idx = indices[val_count:val_count + test_count]
    train_idx = indices[val_count + test_count:]

    train_images = [image_paths[i] for i in train_idx]
    train_masks = [mask_paths[i] for i in train_idx]
    val_images = [image_paths[i] for i in val_idx]
    val_masks = [mask_paths[i] for i in val_idx]
    test_images = [image_paths[i] for i in test_idx]
    test_masks = [mask_paths[i] for i in test_idx]

    print(f"Treino: {len(train_images)} pares, Validação: {len(val_images)} pares, Teste: {len(test_images)} pares")
    return train_images, train_masks, val_images, val_masks, test_images, test_masks
