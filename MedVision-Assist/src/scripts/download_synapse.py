import os
import argparse

try:
    import synapseclient
    from synapseclient.operations import get
except ImportError:
    raise ImportError("Instale o synapseclient primeiro: pip install synapseclient")


def download_project(syn_id, output_dir, auth_token=None, username=None, password=None):
    syn = synapseclient.Synapse()
    
    try:
        if auth_token:
            print(f"[INFO] Autenticando com token...")
            syn.login(authToken=auth_token, silent=True)
        elif username and password:
            print(f"[INFO] Autenticando com usuário e senha...")
            syn.login(username, password, silent=True)
        else:
            raise ValueError("Forneça auth_token ou username e password para login no Synapse.")
        
        print(f"[INFO] Autenticação bem-sucedida!")
        print(f"[INFO] Explorando projeto {syn_id}...")
        
        # Tentar obter informações sobre a entidade
        try:
            entity = syn.get(syn_id, downloadLocation=output_dir, ifcollision='overwrite.local')
            print(f"[SUCCESS] Download concluído em: {output_dir}")
            print(f"[INFO] Entidade: {entity}")
            return entity
        except Exception as e:
            print(f"[ERROR] Falha ao baixar: {e}")
            raise
    
    except Exception as e:
        print(f"[ERROR] Erro geral: {e}")
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Baixa um projeto Synapse localmente.")
    parser.add_argument("synapse_id", type=str, help="ID do Synapse (por exemplo, syn60868042)")
    parser.add_argument("--output-dir", type=str, default="data/synapse", help="Diretório local de saída")
    parser.add_argument("--token", type=str, default=None, help="Auth token do Synapse")
    parser.add_argument("--username", type=str, default=None, help="Usuário Synapse")
    parser.add_argument("--password", type=str, default=None, help="Senha Synapse")
    return parser.parse_args()


def main():
    args = parse_args()
    token = args.token or os.environ.get("SYNAPSE_AUTH_TOKEN")
    if token:
        print("[INFO] Usando token do Synapse para login.")
    elif args.username and args.password:
        print("[INFO] Usando usuário e senha para login no Synapse.")
    else:
        raise RuntimeError("Nenhum token ou credenciais fornecidos. Use --token ou defina SYNAPSE_AUTH_TOKEN.")

    download_project(args.synapse_id, args.output_dir, auth_token=token, username=args.username, password=args.password)

if __name__ == "__main__":
    main()
