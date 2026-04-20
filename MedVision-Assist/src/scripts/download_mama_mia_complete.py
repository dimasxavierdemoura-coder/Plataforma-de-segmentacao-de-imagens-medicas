#!/usr/bin/env python
"""
MAMA-MIA Dataset Downloader - Versão completa
Baixa todo o projeto MAMA-MIA do Synapse recursivamente.
"""

import os
import sys
import argparse
from pathlib import Path

try:
    import synapseclient
    import synapseutils
except ImportError:
    raise ImportError("Instale synapseclient e synapseutils: pip install synapseclient synapseutils")


def download_mama_mia_project(syn_id, output_dir, token):
    """Baixa todo o projeto MAMA-MIA recursivamente."""
    syn = synapseclient.Synapse()
    syn.login(authToken=token, silent=True)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🔐 Login bem-sucedido no Synapse")
    print(f"📦 Baixando projeto {syn_id}...")
    print(f"   Destino: {output_path.absolute()}")
    
    try:
        # Usar syncFromSynapse para download recursivo
        files = synapseutils.syncFromSynapse(
            syn,
            entity=syn_id,
            path=str(output_path),
            ifcollision='overwrite.local'
        )
        
        print(f"\n✓ Download concluído!")
        print(f"   Total de arquivos baixados: {len(files)}")
        
        # Listar arquivos baixados
        all_files = list(output_path.rglob('*'))
        nifti_files = [f for f in all_files if f.is_file() and (f.name.endswith('.nii') or f.name.endswith('.nii.gz'))]
        
        print(f"   Arquivos NIfTI encontrados: {len(nifti_files)}")
        
        if nifti_files:
            print(f"\n📄 Amostra de arquivos NIfTI:")
            for f in nifti_files[:5]:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"   - {f.name} ({size_mb:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro no download: {e}")
        return False


def organize_mama_mia_data(source_dir, target_dir="data/organized"):
    """Organiza dados MAMA-MIA em estrutura padrão."""
    source = Path(source_dir)
    target = Path(target_dir)
    
    images_dir = target / "images"
    masks_dir = target / "masks"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Procurar por arquivos NIfTI nas pastas baixadas
    nifti_files = list(source.rglob('*.nii.gz')) + list(source.rglob('*.nii'))
    
    print(f"\n🔄 Organizando dados MAMA-MIA...")
    print(f"   Encontrados {len(nifti_files)} arquivos NIfTI")
    
    images_count = 0
    masks_count = 0
    
    for f in nifti_files:
        # Verificar se está na pasta de imagens ou segmentações
        if 'images' in str(f).lower():
            import shutil
            shutil.copy(f, images_dir / f.name)
            images_count += 1
            print(f"   ✓ Imagem: {f.name}")
        elif 'segmentations' in str(f).lower():
            import shutil
            shutil.copy(f, masks_dir / f.name)
            masks_count += 1
            print(f"   ✓ Máscara: {f.name}")
    
    print(f"\n✓ Dados organizados em:")
    print(f"   - {images_dir} ({images_count} imagens)")
    print(f"   - {masks_dir} ({masks_count} máscaras)")
    
    return images_count, masks_count


def main():
    parser = argparse.ArgumentParser(
        description="MAMA-MIA Dataset Downloader Completo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Baixar todo o projeto MAMA-MIA
  python download_mama_mia_complete.py syn60868042 --token YOUR_TOKEN

  # Baixar e organizar automaticamente
  python download_mama_mia_complete.py syn60868042 --token YOUR_TOKEN --organize
        """
    )
    
    parser.add_argument("synapse_id", type=str, help="ID do Synapse (ex: syn60868042)")
    parser.add_argument("--token", type=str, help="Auth token Synapse", required=True)
    parser.add_argument("--output-dir", type=str, default="data/synapse", help="Diretório de saída")
    parser.add_argument("--organize", action="store_true", help="Organizar após download")
    
    args = parser.parse_args()
    
    print("🚀 Iniciando download completo do MAMA-MIA...")
    
    # Download
    success = download_mama_mia_project(args.synapse_id, args.output_dir, args.token)
    
    if success and args.organize:
        organize_mama_mia_data(args.output_dir)
    
    print("\n🎉 Processo concluído!")
    print("Agora você pode treinar o modelo:")
    print("python src/train.py --train-root data/organized --epochs 20 --batch-size 8")


if __name__ == "__main__":
    main()