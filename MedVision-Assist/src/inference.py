import os
import sys
import json
import csv
import torch
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
import nibabel as nib

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

# Adicionar o diretório raiz ao path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.unet import UNet
from src.utils import dice_coefficient, iou_score, sensitivity_score, specificity_score, f1_score

def load_model(model_path: str = None):
    if model_path is None:
        model_path = os.path.join('models', 'unet_medvision.pth')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    min_val = np.nanmin(image)
    max_val = np.nanmax(image)
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    else:
        if max_val == 0.0:
            image = np.zeros_like(image)
        else:
            image = np.ones_like(image)
    return (image * 255).astype(np.uint8)


def load_dicom_image(image_path: str) -> np.ndarray:
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom não está instalado para processar arquivos DICOM")
    dicom_data = pydicom.dcmread(image_path)
    image = dicom_data.pixel_array.astype(np.float32)
    return normalize_image(image)


def load_image_file(image_path: str) -> np.ndarray:
    if image_path.lower().endswith(('.nii', '.nii.gz')):
        return load_nifti_image(image_path)
    if image_path.lower().endswith('.dcm'):
        return load_dicom_image(image_path)
    image = Image.open(image_path).convert('RGB')
    return np.array(image).astype(np.uint8)


def load_nifti_image(image_path: str) -> np.ndarray:
    nii_img = nib.load(image_path)
    image = nii_img.get_fdata()
    
    if image.ndim == 3:
        image = image[:, :, image.shape[2] // 2]
    elif image.ndim == 4:
        image = image[:, :, image.shape[2] // 2, 0]
    
    return normalize_image(image)


def load_nifti_volume(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    nii_img = nib.load(image_path)
    image = nii_img.get_fdata()
    if image.ndim == 4:
        image = image[:, :, :, 0]
    if image.ndim != 3:
        raise ValueError(f"Esperado volume 3D ou 4D em {image_path}, mas obtido ndim={image.ndim}")
    return image, nii_img.affine


def predict_prob_map(model, image: np.ndarray) -> np.ndarray:
    tensor = preprocess_image(image)
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor)
        prob_map = torch.sigmoid(output).squeeze().cpu().numpy()

    return np.clip(prob_map, 0.0, 1.0)


def predict_mask(model, image: np.ndarray) -> np.ndarray:
    prob_map = predict_prob_map(model, image)
    return (prob_map > 0.5).astype(np.uint8)


def predict_prob_volume(model, volume: np.ndarray) -> np.ndarray:
    prob_volume = np.zeros_like(volume, dtype=np.float32)
    for z in range(volume.shape[2]):
        slice_img = normalize_image(volume[:, :, z])
        if np.nanmax(slice_img) == np.nanmin(slice_img):
            continue
        prob_volume[:, :, z] = predict_prob_map(model, slice_img)
    return np.clip(prob_volume, 0.0, 1.0)


def save_nifti_mask(mask: np.ndarray, original_path: str, output_path: str):
    original_nii = nib.load(original_path)
    original_data = original_nii.get_fdata()

    if mask.ndim == 2:
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_resized = mask_pil.resize((original_data.shape[0], original_data.shape[1]), Image.NEAREST)
        mask_resized = np.array(mask_resized) > 127
        mask_3d = np.zeros_like(original_data)
        if original_data.ndim == 3:
            mask_3d[:, :, original_data.shape[2] // 2] = mask_resized
        elif original_data.ndim == 4:
            mask_3d[:, :, original_data.shape[2] // 2, 0] = mask_resized
    else:
        if original_data.ndim == 3:
            if mask.shape != original_data.shape:
                raise ValueError("A máscara volumétrica não confere com as dimensões da imagem original")
            mask_3d = mask.astype(np.uint8)
        elif original_data.ndim == 4:
            mask_3d = np.zeros_like(original_data)
            if mask.shape != original_data.shape[:3]:
                raise ValueError("A máscara volumétrica não confere com as dimensões da imagem original")
            mask_3d[:, :, :, 0] = mask.astype(np.uint8)
        else:
            raise ValueError("Formato de imagem original não suportado para salvar máscara volumétrica")

    mask_nii = nib.Nifti1Image(mask_3d.astype(np.uint8), original_nii.affine)
    nib.save(mask_nii, output_path)


def save_report_json(report: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def save_summary_csv(rows: list, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys() if rows else []))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def preprocess_image(image: np.ndarray, size=(256, 256)) -> torch.Tensor:
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif len(image.shape) == 3 and image.shape[-1] == 1:
        image = np.concatenate([image] * 3, axis=-1)
    
    image = Image.fromarray(image.astype(np.uint8))
    image = image.resize(size)
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image).unsqueeze(0)
    return image

def predict_report(model, image: np.ndarray) -> dict:
    mask = predict_mask(model, image)
    positive_ratio = mask.mean()
    return {
        'análise': 'segmentação gerada',
        'probabilidade_indicativa': f'{positive_ratio:.2%}',
        'observação': 'Esta é uma estimativa inicial; use com dados clínicos completos.'
    }

def create_heatmap(image: np.ndarray, prob_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    import matplotlib.cm as cm

    prob = np.clip(np.asarray(prob_map), 0.0, 1.0)
    heatmap_rgba = cm.get_cmap("jet")(prob)
    heatmap = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    image_rgb = np.asarray(image)
    if image_rgb.ndim == 2:
        image_rgb = np.stack([image_rgb] * 3, axis=-1)
    if image_rgb.ndim == 3 and image_rgb.shape[2] == 1:
        image_rgb = np.concatenate([image_rgb] * 3, axis=-1)
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    if image_rgb.shape[:2] != heatmap.shape[:2]:
        heatmap = np.array(Image.fromarray(heatmap).resize((image_rgb.shape[1], image_rgb.shape[0]), Image.BILINEAR))

    overlay = (image_rgb.astype(np.float32) * (1 - alpha) + heatmap.astype(np.float32) * alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def main():
    MODEL_PATH_DEFAULT = os.path.join('models', 'unet_medvision.pth')
    
    parser = argparse.ArgumentParser(description='Faz inferência com o modelo U-Net treinado')
    parser.add_argument('--model-path', type=str, default=MODEL_PATH_DEFAULT, help='Caminho para o modelo treinado')
    parser.add_argument('--input-image', type=str, help='Caminho para a imagem de entrada (.nii.gz, .nii, .dcm, .png)')
    parser.add_argument('--input-dir', type=str, default=None, help='Diretório para inferência em lote')
    parser.add_argument('--mask-path', type=str, default=None, help='Caminho para máscara de referência para avaliação')
    parser.add_argument('--mask-dir', type=str, default=None, help='Diretório com máscaras de referência para inferência em lote')
    parser.add_argument('--output-dir', type=str, default='results', help='Diretório para salvar resultados')
    parser.add_argument('--volume', action='store_true', help='Executar inferência volumétrica em todo o volume NIfTI')
    parser.add_argument('--heatmap-slice', type=int, default=None, help='Índice de slice para gerar heatmap quando inferência volumétrica for usada')
    parser.add_argument('--recursive', action='store_true', help='Percorrer subpastas ao processar um diretório de imagens')
    parser.add_argument('--report-format', type=str, choices=['txt', 'json', 'both'], default='both', help='Formato de relatório para salvar')

    args = parser.parse_args()
    
    if not args.input_image and not args.input_dir:
        parser.error('Informe --input-image ou --input-dir')

    print(f'Carregando modelo: {args.model_path}')
    model = load_model(args.model_path)

    os.makedirs(args.output_dir, exist_ok=True)
    summary_rows = []

    def summarize_image(image_path, mask_path=None, volume_mode=False):
        nonlocal summary_rows
        filename = Path(image_path).stem
        print(f'Processando: {image_path}')
        try:
            if volume_mode:
                volume, affine = load_nifti_volume(image_path)
                prob_volume = predict_prob_volume(model, volume)
                mask_volume = (prob_volume > 0.5).astype(np.uint8)
                mask_path_out = os.path.join(args.output_dir, f'{filename}_mask_volume.nii.gz')
                save_nifti_mask(mask_volume, image_path, mask_path_out)
                slice_index = args.heatmap_slice if args.heatmap_slice is not None else volume.shape[2] // 2
                slice_index = max(0, min(slice_index, volume.shape[2] - 1))
                heatmap = create_heatmap(volume[:, :, slice_index].astype(np.uint8), prob_volume[:, :, slice_index])
                heatmap_path = os.path.join(args.output_dir, f'{filename}_heatmap_slice_{slice_index}.png')
                Image.fromarray(heatmap).save(heatmap_path)
                report = predict_report(model, volume[:, :, slice_index])
                save_report_json(
                    {
                        'image': image_path,
                        'model': args.model_path,
                        'slice_index': slice_index,
                        'volume_shape': volume.shape,
                        'report': report,
                        'mask_path': mask_path_out,
                        'heatmap_path': heatmap_path,
                    },
                    os.path.join(args.output_dir, f'{filename}_report.json'),
                )
                if args.report_format in ['txt', 'both']:
                    with open(os.path.join(args.output_dir, f'{filename}_report.txt'), 'w', encoding='utf-8') as f:
                        f.write('RELATÓRIO DE SEGMENTAÇÃO MEDVISION ASSIST\n')
                        f.write('=' * 60 + '\n')
                        f.write(f'Imagem processada: {image_path}\n')
                        f.write(f'Modelo usado: {args.model_path}\n')
                        f.write(f'Máscara volumétrica salva em: {mask_path_out}\n')
                        f.write(f'Heatmap slice {slice_index} salvo em: {heatmap_path}\n')
                        f.write(f'Análise: {report["análise"]}\n')
                        f.write(f'Probabilidade indicativa: {report["probabilidade_indicativa"]}\n')
                        f.write(f'Área positiva estimada (pixels): {int(mask_volume[:, :, slice_index].sum())}\n')
                        f.write(f'Observação: {report["observação"]}\n')
                summary_rows.append({
                    'image': image_path,
                    'mode': 'volume',
                    'output_mask': mask_path_out,
                    'heatmap': heatmap_path,
                })
                return

            image = load_image_file(image_path)
            prob_map = predict_prob_map(model, image)
            mask = (prob_map > 0.5).astype(np.uint8)
            report = predict_report(model, image)
            mask_path_out = os.path.join(args.output_dir, f'{filename}_mask.nii.gz')
            if image_path.lower().endswith(('.nii', '.nii.gz')):
                save_nifti_mask(mask, image_path, mask_path_out)
            else:
                Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path_out.replace('.nii.gz', '.png'))
            heatmap = create_heatmap(image, prob_map)
            heatmap_path = os.path.join(args.output_dir, f'{filename}_heatmap.png')
            Image.fromarray(heatmap).save(heatmap_path)
            save_report_json(
                {
                    'image': image_path,
                    'model': args.model_path,
                    'report': report,
                    'mask_path': mask_path_out,
                    'heatmap_path': heatmap_path,
                },
                os.path.join(args.output_dir, f'{filename}_report.json'),
            )
            if args.report_format in ['txt', 'both']:
                with open(os.path.join(args.output_dir, f'{filename}_report.txt'), 'w', encoding='utf-8') as f:
                    f.write('RELATÓRIO DE SEGMENTAÇÃO MEDVISION ASSIST\n')
                    f.write('=' * 60 + '\n')
                    f.write(f'Imagem processada: {image_path}\n')
                    f.write(f'Modelo usado: {args.model_path}\n')
                    f.write(f'Máscara salva em: {mask_path_out}\n')
                    f.write(f'Heatmap salvo em: {heatmap_path}\n')
                    f.write(f'Análise: {report["análise"]}\n')
                    f.write(f'Probabilidade indicativa: {report["probabilidade_indicativa"]}\n')
                    f.write(f'Área positiva estimada (pixels): {int(mask.sum())}\n')
                    f.write(f'Observação: {report["observação"]}\n')
            summary_rows.append({
                'image': image_path,
                'mode': 'slice',
                'output_mask': mask_path_out,
                'heatmap': heatmap_path,
            })
        except Exception as e:
            print(f'Erro ao processar {image_path}: {e}')
            summary_rows.append({'image': image_path, 'error': str(e)})

    def gather_files(directory, recursive=False):
        extensions = ('.nii', '.nii.gz', '.png', '.jpg', '.jpeg', '.dcm')
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.lower().endswith(extensions):
                    files.append(os.path.join(root, filename))
            if not recursive:
                break
        return sorted(files)

    if args.input_dir:
        image_files = gather_files(args.input_dir, recursive=args.recursive)
        if not image_files:
            print(f'Nenhuma imagem encontrada em {args.input_dir}')
            return

        mask_paths = None
        if args.mask_dir:
            mask_files = gather_files(args.mask_dir, recursive=args.recursive)
            mask_map = {Path(p).stem: p for p in mask_files}
            mask_paths = mask_map

        for image_path in image_files:
            mask_path = None
            if mask_paths:
                mask_path = mask_paths.get(Path(image_path).stem)
            summarize_image(image_path, mask_path=mask_path, volume_mode=args.volume)

        if summary_rows:
            save_summary_csv(summary_rows, os.path.join(args.output_dir, 'inference_summary.csv'))
            print(f'Resumo da inferência salvo em: {os.path.join(args.output_dir, "inference_summary.csv")}')
        return

    if args.input_image:
        summarize_image(args.input_image, mask_path=args.mask_path, volume_mode=args.volume)
        if summary_rows:
            save_summary_csv(summary_rows, os.path.join(args.output_dir, 'inference_summary.csv'))
            print(f'Resumo da inferência salvo em: {os.path.join(args.output_dir, "inference_summary.csv")}')

if __name__ == '__main__':
    main()
