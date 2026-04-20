#!/usr/bin/env python
"""
MAMA-MIA Dataset Downloader with diagnosis and organization.
Downloads from Synapse and organizes into images/masks structure.
"""

import os
import sys
import argparse
from pathlib import Path

try:
    import synapseclient
except ImportError:
    raise ImportError("Instale synapseclient: pip install synapseclient")


def test_synapse_login(token):
    """Testa se é possível fazer login no Synapse."""
    syn = synapseclient.Synapse()
    try:
        syn.login(authToken=token, silent=True)
        print("✓ Login bem-sucedido no Synapse")
        return True
    except Exception as e:
        print(f"✗ Falha ao fazer login: {e}")
        return False


def explore_synapse_entity(syn_id, token):
    """Explora o conteúdo de uma entidade Synapse."""
    syn = synapseclient.Synapse()
    syn.login(authToken=token, silent=True)
    
    try:
        entity = syn.get(syn_id, downloadFile=False)
        print(f"\n📦 Entidade encontrada: {entity.name}")
        print(f"   ID: {syn_id}")
        print(f"   Tipo: {entity.concreteType}")
        
        # Listar filhos se for um projeto/pasta
        if hasattr(entity, 'concreteType') and 'Folder' in entity.concreteType or 'Project' in entity.concreteType:
            print(f"\n📂 Listando conteúdo:")
            results = syn.getChildren(syn_id, includeTypes=['file', 'folder'])
            for item in results:
                print(f"   - {item['name']} ({item['type']})")
        
        return entity
    except Exception as e:
        print(f"✗ Erro ao explorar: {e}")
        return None


def download_synapse_folders(images_id, masks_id, output_dir, token):
    """Faz o download das pastas de imagens e máscaras."""
    syn = synapseclient.Synapse()
    syn.login(authToken=token, silent=True)
    
    output_path = Path(output_dir)
    images_path = output_path / "images"
    masks_path = output_path / "masks"
    
    images_path.mkdir(parents=True, exist_ok=True)
    masks_path.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"\n⬇️  Download de imagens (syn{images_id[-10:]})...")
        syn.get(images_id, downloadLocation=str(images_path), ifcollision='overwrite.local', recursive=True)
        print(f"✓ Imagens baixadas em {images_path}")
        
        print(f"\n⬇️  Download de segmentações (syn{masks_id[-10:]})...")
        syn.get(masks_id, downloadLocation=str(masks_path), ifcollision='overwrite.local', recursive=True)
        print(f"✓ Segmentações baixadas em {masks_path}")
        
        # Verificar o que foi baixado
        images = list(images_path.rglob('*.nii.gz')) + list(images_path.rglob('*.nii'))
        masks = list(masks_path.rglob('*.nii.gz')) + list(masks_path.rglob('*.nii'))
        
        print(f"\n📊 Resumo:")
        print(f"   - {len(images)} imagens encontradas")
        print(f"   - {len(masks)} segmentações encontradas")
        
        return True
    except Exception as e:
        print(f"✗ Erro: {e}")
        return False


def organize_downloaded_data(source_dir, target_dir="data/organized"):
    """Organiza dados baixados em estrutura padrão."""
    source = Path(source_dir)
    target = Path(target_dir)
    
    images_dir = target / "images"
    masks_dir = target / "masks"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Procurar por arquivos
    nifti_files = list(source.rglob('*.nii.gz')) + list(source.rglob('*.nii'))
    
    print(f"\n🔄 Organizando dados...")
    print(f"   Encontrados {len(nifti_files)} arquivos NIfTI")
    
    for f in nifti_files:
        if 'mask' in f.name.lower() or 'seg' in f.name.lower():
            import shutil
            shutil.copy(f, masks_dir / f.name)
            print(f"   ✓ Máscara: {f.name}")
        else:
            import shutil
            shutil.copy(f, images_dir / f.name)
            print(f"   ✓ Imagem: {f.name}")
    
    print(f"\n✓ Dados organizados em:")
    print(f"   - {images_dir} ({len(list(images_dir.glob('*')))} imagens)")
    print(f"   - {masks_dir} ({len(list(masks_dir.glob('*')))} máscaras)")


def main():
    parser = argparse.ArgumentParser(
        description="MAMA-MIA Dataset Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Explorar conteúdo (sem baixar)
  python download_mama_mia.py syn60868042 --token YOUR_TOKEN --explore

  # Fazer download
  python download_mama_mia.py syn60868042 --token YOUR_TOKEN --output-dir data/synapse

  # Download + organização automática
  python download_mama_mia.py syn60868042 --token YOUR_TOKEN --output-dir data/synapse --organize
        """
    )
    
    parser.add_argument("synapse_id", type=str, help="ID do Synapse (ex: syn60868042)")
    parser.add_argument("--token", type=str, help="Auth token Synapse", required=True)
    parser.add_argument("--output-dir", type=str, default="data/synapse", help="Diretório de saída")
    parser.add_argument("--explore", action="store_true", help="Apenas explorar, não baixar")
    parser.add_argument("--organize", action="store_true", help="Organizar após download")
    
    args = parser.parse_args()
    
    print(f"🔐 Testando credenciais...")
    if not test_synapse_login(args.token):
        sys.exit(1)
    
    if args.explore:
        explore_synapse_entity(args.synapse_id, args.token)
    else:
        success = download_synapse_entity(
            args.synapse_id,
            args.output_dir,
            args.token
        )
        
        if success and args.organize:
            organize_downloaded_data(args.output_dir)


if __name__ == "__main__":
    main()
