"""
classificar.py

Objetivo:
- Renomeia imagens externas ou arquivos fora da nomenclatura padrao do projeto.
- Adapta o nome original para o formato usado nos testes externos: MATERIAL_CP_CLASSE.
- Exemplo atual: scratches_1.jpg -> A36_X01_risco.jpg.

IMPORTANTE:
- Este e um script VOLUVEL: deve ser adaptado para cada novo dataset externo.
- Antes de executar, revisar obrigatoriamente a pasta, o padrao de entrada, material e rotulo.
- Nao faz parte da regra fixa de processamento do dataset principal.

Usado antes de:
- predict_imagens.py, quando o dataset externo precisa ser identificado/rotulado pelo nome.

Parametros ajustaveis:
- PASTA: diretorio contendo as imagens.
- MATERIAL: material conhecido das pecas.
- ROTULO: classe real conhecida.
- PADRAO: formato atual dos nomes dos arquivos.
- PREFIXO_CP e DIGITOS_CP: identificacao dos CPs externos.
- EXTENSOES: formatos considerados.

Observacoes:
- Mantem a extensao original do arquivo.
- Mostra todas as renomeacoes e pede confirmacao antes de alterar os arquivos.
- Arquivos fora do PADRAO sao ignorados.
"""

from pathlib import Path
import re


# =============================================================================
# PARAMETROS AJUSTAVEIS - REVISAR PARA CADA NOVO DATASET
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[3]

PASTA = BASE_DIR / "dataset_externo" / "scratches"   # Pasta a renomear
MATERIAL = "A36"                                     # AL, A36 ou FoFo
ROTULO = "risco"                                     # amassamento, corrosao, risco ou sem_falha

PADRAO = re.compile(r"^scratches_(\d+)$", re.IGNORECASE)  # Nome original esperado
PREFIXO_CP = "X"                                     # CP externo/fora da numeracao principal
DIGITOS_CP = 2                                       # Quantidade minima de digitos: X01, X02...

EXTENSOES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def listar_arquivos():
    if not PASTA.exists(): raise FileNotFoundError(f"Pasta nao encontrada:\n{PASTA}")

    arquivos, ignorados = [], []

    for caminho in PASTA.iterdir():
        if not caminho.is_file() or caminho.suffix.lower() not in EXTENSOES: continue

        correspondencia = PADRAO.fullmatch(caminho.stem)

        if correspondencia: arquivos.append((int(correspondencia.group(1)), caminho))
        else: ignorados.append(caminho)

    return sorted(arquivos), sorted(ignorados)


def montar_renomeacoes(arquivos):
    renomeacoes = []

    for numero, origem in arquivos:
        cp = f"{PREFIXO_CP}{numero:0{DIGITOS_CP}d}"
        destino = origem.with_name(f"{MATERIAL}_{cp}_{ROTULO}{origem.suffix.lower()}")

        if destino.exists() and destino != origem:
            raise FileExistsError(f"Arquivo de destino ja existe:\n{destino}")

        renomeacoes.append((origem, destino))

    return renomeacoes


def main():
    # Localiza arquivos e prepara os novos nomes
    arquivos, ignorados = listar_arquivos()
    renomeacoes = montar_renomeacoes(arquivos)

    print("\n" + "=" * 70)
    print("RENOMEAR DATASET EXTERNO")
    print("=" * 70)
    print(f"\nPasta: {PASTA}")
    print(f"Material: {MATERIAL}")
    print(f"Rotulo real: {ROTULO}")
    print(f"Arquivos para renomear: {len(renomeacoes)}")
    print(f"Arquivos ignorados: {len(ignorados)}")

    if not renomeacoes:
        print("\nNenhum arquivo correspondente ao PADRAO foi encontrado.")
        return

    # Mostra tudo antes de alterar os arquivos
    print("\nRENOMEACOES")
    print("-" * 70)

    for origem, destino in renomeacoes: print(f"{origem.name} -> {destino.name}")

    if input("\nDeseja continuar? [Y/N]: ").strip().upper() != "Y":
        print("\nOperacao cancelada.")
        return

    # Renomeia somente apos confirmacao
    for origem, destino in renomeacoes: origem.rename(destino)

    print("\n" + "=" * 70)
    print("CONCLUIDO")
    print("=" * 70)
    print(f"\nArquivos renomeados: {len(renomeacoes)}")
    print(f"Material: {MATERIAL}")
    print(f"CP: {PREFIXO_CP}NN")
    print(f"Rotulo real: {ROTULO}")


if __name__ == "__main__":
    main()