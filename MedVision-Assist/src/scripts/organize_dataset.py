import os
import shutil
from pathlib import Path


def organize_dataset(source_dir, target_dir="data/organized", image_pattern=None, mask_pattern=None):
    """
    Organiza um dataset médico desorganizado em estrutura padrão.
    
    Args:
        source_dir: Diretório contendo imagens e máscaras misturadas
        target_dir: Diretório de saída (padrão: data/organized)
        image_pattern: Padrão para identificar imagens (ex: '*_image.png')
        mask_pattern: Padrão para identificar máscaras (ex: '*_mask.png')
    """
    source = Path(source_dir)
    target = Path(target_dir)
    
    images_dir = target / "images"
    masks_dir = target / "masks"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    if image_pattern:
        for f in source.glob(image_pattern):
            shutil.copy(f, images_dir / f.name)
            print(f"Imagem copiada: {f.name}")
    
    if mask_pattern:
        for f in source.glob(mask_pattern):
            shutil.copy(f, masks_dir / f.name)
            print(f"Máscara copiada: {f.name}")
    
    print(f"Dataset organizado em {target_dir}")
    print(f"  - {len(list(images_dir.glob('*')))} imagens")
    print(f"  - {len(list(masks_dir.glob('*')))} máscaras")


def list_dataset_structure(root_dir):
    """
    Explora a estrutura de um dataset.
    """
    root = Path(root_dir)
    print(f"\n📁 Estrutura de {root_dir}:\n")
    
    for level, path in enumerate(root.rglob('*')):
        if path.is_file():
            indent = "  " * level
            size = path.stat().st_size / (1024 * 1024)
            print(f"{indent}📄 {path.name} ({size:.2f} MB)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Organiza e explora datasets médicos.")
    parser.add_argument("command", choices=["organize", "explore"], help="Comando")
    parser.add_argument("--source", type=str, help="Diretório de origem")
    parser.add_argument("--target", type=str, default="data/organized", help="Diretório de destino")
    parser.add_argument("--image-pattern", type=str, default="*", help="Padrão para imagens")
    parser.add_argument("--mask-pattern", type=str, default="*_mask*", help="Padrão para máscaras")
    
    args = parser.parse_args()
    
    if args.command == "organize":
        if not args.source:
            raise ValueError("--source é obrigatório para o comando 'organize'")
        organize_dataset(args.source, args.target, args.image_pattern, args.mask_pattern)
    
    elif args.command == "explore":
        if not args.source:
            raise ValueError("--source é obrigatório para o comando 'explore'")
        list_dataset_structure(args.source)
