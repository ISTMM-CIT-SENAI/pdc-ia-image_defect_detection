"""
split.py

Objetivo:
- Divide dataset_processado em train, val e test separadamente por classe.
- Valida classes, nomenclatura RAC e formato dos arquivos NPY antes da divisao.
- Detecta automaticamente o shape das imagens e exige consistencia em todo o dataset.
- Gera README e grafico com a distribuicao final.

Usa:
- utils/nomenclatura.py: ROTULOS, CONDICOES e interpretacao dos nomes.

Usado antes de:
- augmentation_train.py: gera augmentation somente em train.
- main.py: utiliza train e val.
- predict.py: utiliza test.

Parametros ajustaveis:
- PROPORCAO_TREINO, PROPORCAO_VALIDACAO e PROPORCAO_TESTE.
- SEED: controla somente a reproducibilidade da divisao.

Observacoes:
- Executar depois de pre_processing_data.py.
- As proporcoes devem somar 1.0.
- A resolucao NAO e definida aqui; e detectada automaticamente nos NPY.
- dataset_processado_split e recriado a cada execucao.
- Nenhum arquivo e alterado; os NPY sao apenas copiados.
- Para manter comparacoes entre experimentos, evitar refazer o split sem necessidade.
"""

import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pdc_ia_image_defect_detection.utils.nomenclatura import CONDICOES, ROTULOS, interpretar_nome


# =============================================================================
# PARAMETROS AJUSTAVEIS
# =============================================================================

PROPORCAO_TREINO = 0.70       # Percentual destinado ao treino
PROPORCAO_VALIDACAO = 0.15    # Percentual destinado a validacao
PROPORCAO_TESTE = 0.15        # Percentual destinado ao teste
SEED = 42                     # Reprodutibilidade apenas do split


# Caminhos
BASE_DIR = Path(__file__).resolve().parents[3]
DATASET_ORIGINAL = BASE_DIR / "dataset_processado"
DATASET_SPLIT = BASE_DIR / "dataset_processado_split"


def obter_classes(root):
    return sorted([pasta for pasta in root.iterdir() if pasta.is_dir()])


def obter_arquivos(pasta):
    return sorted([arquivo for arquivo in pasta.iterdir() if arquivo.is_file() and arquivo.suffix.lower() == ".npy"])


def rotulo_pelo_rac(rac):
    rotulos = {rotulo for rotulo, rac_condicao in CONDICOES.values() if rac_condicao == rac}
    if len(rotulos) != 1: raise ValueError(f"RAC desconhecido ou ambiguo: {rac}")
    return rotulos.pop()


def validar_dataset():
    erros, corretos, shape_esperado = [], 0, None
    classes_encontradas = {pasta.name for pasta in obter_classes(DATASET_ORIGINAL)}
    classes_esperadas = set(ROTULOS)

    for classe in sorted(classes_esperadas - classes_encontradas):
        erros.append((DATASET_ORIGINAL / classe, f"Classe esperada nao encontrada: {classe}"))

    for classe in sorted(classes_encontradas - classes_esperadas):
        erros.append((DATASET_ORIGINAL / classe, f"Classe desconhecida encontrada: {classe}"))

    for rotulo in ROTULOS:
        pasta = DATASET_ORIGINAL / rotulo
        if not pasta.exists(): continue

        for arquivo in obter_arquivos(pasta):
            try:
                # Confere classe da pasta com o RAC do nome
                dados = interpretar_nome(arquivo)
                rotulo_nome = rotulo_pelo_rac(dados["rac"])

                if rotulo_nome != rotulo:
                    raise ValueError(f"Rotulo inconsistente: pasta={rotulo}, RAC={dados['rac']} -> {rotulo_nome}")

                # Detecta e valida o formato dos NPY
                array = np.load(arquivo, mmap_mode="r", allow_pickle=False)

                if array.ndim != 3 or array.shape[0] != 1: raise ValueError(f"Shape invalido: {array.shape}. Esperado: (1, H, W).")
                if array.shape[1] != array.shape[2]: raise ValueError(f"Imagem nao quadrada: {array.shape}.")

                if shape_esperado is None: shape_esperado = array.shape
                elif array.shape != shape_esperado: raise ValueError(f"Shape inconsistente: {array.shape}. Esperado: {shape_esperado}.")

                corretos += 1

            except Exception as erro:
                erros.append((arquivo, str(erro)))

    return corretos, erros, shape_esperado


def calcular_divisao(quantidade):
    n_treino = round(quantidade * PROPORCAO_TREINO)
    n_validacao = round(quantidade * PROPORCAO_VALIDACAO)
    n_teste = quantidade - n_treino - n_validacao
    return n_treino, n_validacao, n_teste


def analisar_dataset():
    resultados = {}

    for rotulo in ROTULOS:
        quantidade = len(obter_arquivos(DATASET_ORIGINAL / rotulo))
        n_treino, n_validacao, n_teste = calcular_divisao(quantidade)
        resultados[rotulo] = {"total": quantidade, "train": n_treino, "val": n_validacao, "test": n_teste}

    return resultados


def mostrar_plano_divisao(resultados, corretos, erros, shape_imagem):
    print("\n" + "=" * 78)
    print("ANALISE DO DATASET PROCESSADO")
    print("=" * 78)
    print(f"\nArquivos analisados : {corretos + len(erros)}")
    print(f"Arquivos corretos   : {corretos}")
    print(f"Arquivos com erro   : {len(erros)}")

    if shape_imagem is not None: print(f"Shape identificado  : {shape_imagem}")

    if erros:
        print("\nERROS ENCONTRADOS")
        for caminho, erro in erros:
            print(f"\n  Arquivo: {caminho}")
            print(f"  Erro   : {erro}")
        return

    print("\n" + "=" * 78)
    print("PLANO DE DIVISAO DO DATASET")
    print("=" * 78)
    print(f"\nOrigem : {DATASET_ORIGINAL}")
    print(f"Destino: {DATASET_SPLIT}")
    print(f"\nProporcao: {PROPORCAO_TREINO:.0%} treino | {PROPORCAO_VALIDACAO:.0%} validacao | {PROPORCAO_TESTE:.0%} teste")
    print(f"Semente aleatoria: {SEED}")

    print("\n" + "-" * 78)
    print(f"{'Classe':<20}{'Total':>10}{'Train':>10}{'Val':>10}{'Test':>10}")
    print("-" * 78)

    for classe, resultado in resultados.items():
        print(f"{classe:<20}{resultado['total']:>10}{resultado['train']:>10}{resultado['val']:>10}{resultado['test']:>10}")

    total = sum(r["total"] for r in resultados.values())
    treino = sum(r["train"] for r in resultados.values())
    validacao = sum(r["val"] for r in resultados.values())
    teste = sum(r["test"] for r in resultados.values())

    print("-" * 78)
    print(f"{'TOTAL':<20}{total:>10}{treino:>10}{validacao:>10}{teste:>10}")
    print("-" * 78)

    if DATASET_SPLIT.exists():
        print("\nATENCAO: dataset_processado_split ja existe.")
        print("A pasta sera recriada caso a divisao seja confirmada.")


def confirmar_divisao():
    while True:
        resposta = input("\nDeseja continuar com esta divisao? [Y/N]: ").strip().upper()
        if resposta == "Y": return True
        if resposta == "N": return False
        print("Opcao invalida. Digite Y ou N.")


def criar_estrutura():
    # Recria train, val e test
    if DATASET_SPLIT.exists(): shutil.rmtree(DATASET_SPLIT)

    for conjunto in ("train", "val", "test"):
        for rotulo in ROTULOS:
            (DATASET_SPLIT / conjunto / rotulo).mkdir(parents=True, exist_ok=True)


def dividir_dataset():
    # A SEED garante o mesmo split quando os dados nao mudam
    random.seed(SEED)
    distribuicao = {}

    for rotulo in ROTULOS:
        arquivos = obter_arquivos(DATASET_ORIGINAL / rotulo)
        random.shuffle(arquivos)

        n_treino, n_validacao, _ = calcular_divisao(len(arquivos))
        conjuntos = {
            "train": arquivos[:n_treino],
            "val": arquivos[n_treino:n_treino + n_validacao],
            "test": arquivos[n_treino + n_validacao:],
        }

        distribuicao[rotulo] = {conjunto: [arquivo.name for arquivo in lista] for conjunto, lista in conjuntos.items()}

        print(f"\n{rotulo}:")

        for conjunto, lista in conjuntos.items():
            pasta_saida = DATASET_SPLIT / conjunto / rotulo

            for indice, arquivo in enumerate(lista, start=1):
                shutil.copy2(arquivo, pasta_saida / arquivo.name)
                print(f"\r  {conjunto}: {indice}/{len(lista)}", end="", flush=True)

            print()

    return distribuicao


def verificar_dataset(distribuicao):
    problemas, total_original, total_final = [], 0, 0

    for rotulo in ROTULOS:
        originais = {arquivo.name for arquivo in obter_arquivos(DATASET_ORIGINAL / rotulo)}
        treino = {arquivo.name for arquivo in obter_arquivos(DATASET_SPLIT / "train" / rotulo)}
        validacao = {arquivo.name for arquivo in obter_arquivos(DATASET_SPLIT / "val" / rotulo)}
        teste = {arquivo.name for arquivo in obter_arquivos(DATASET_SPLIT / "test" / rotulo)}

        total_original += len(originais)
        total_final += len(treino) + len(validacao) + len(teste)

        if treino & validacao: problemas.append(f"{rotulo}: arquivos presentes em treino e validacao.")
        if treino & teste: problemas.append(f"{rotulo}: arquivos presentes em treino e teste.")
        if validacao & teste: problemas.append(f"{rotulo}: arquivos presentes em validacao e teste.")
        if originais != treino | validacao | teste: problemas.append(f"{rotulo}: arquivos finais nao correspondem aos originais.")

        if len(treino) != len(distribuicao[rotulo]["train"]): problemas.append(f"{rotulo}: quantidade incorreta no treino.")
        if len(validacao) != len(distribuicao[rotulo]["val"]): problemas.append(f"{rotulo}: quantidade incorreta na validacao.")
        if len(teste) != len(distribuicao[rotulo]["test"]): problemas.append(f"{rotulo}: quantidade incorreta no teste.")

    if total_original != total_final:
        problemas.append(f"Total original ({total_original}) diferente do total final ({total_final}).")

    return problemas


def criar_grafico_distribuicao(resultados):
    classes = list(resultados)
    x = np.arange(len(classes))
    largura = 0.25

    train = [resultados[classe]["train"] for classe in classes]
    val = [resultados[classe]["val"] for classe in classes]
    test = [resultados[classe]["test"] for classe in classes]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - largura, train, largura, label="Train")
    ax.bar(x, val, largura, label="Val")
    ax.bar(x + largura, test, largura, label="Test")

    ax.set_xlabel("Classes")
    ax.set_ylabel("Quantidade de arquivos")
    ax.set_title("Distribuicao das classes")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()

    plt.tight_layout()
    plt.savefig(DATASET_SPLIT / "distribuicao_classes.png", dpi=300)
    plt.close()


def criar_readme(resultados, shape_imagem):
    total = sum(r["total"] for r in resultados.values())
    total_treino = sum(r["train"] for r in resultados.values())
    total_validacao = sum(r["val"] for r in resultados.values())
    total_teste = sum(r["test"] for r in resultados.values())

    linhas = [
        "DATASET PROCESSADO - DIVISAO TREINO / VALIDACAO / TESTE",
        "=" * 60, "",
        "Este dataset foi criado a partir da pasta 'dataset_processado'.", "",
        "Os dados ja foram pre-processados antes da divisao:",
        "  - grayscale",
        "  - resize direto, sem crop",
        f"  - shape {shape_imagem}",
        "  - dtype float32",
        "  - valores entre 0.0 e 1.0",
        "  - formato NPY", "",
        "Rotulos utilizados pelo modelo:",
    ]

    linhas += [f"  - {rotulo}" for rotulo in ROTULOS]

    linhas += [
        "",
        "O material e a intensidade permanecem identificados no nome dos arquivos,",
        "mas nao sao utilizados como rotulos do modelo nesta etapa.", "",
        "A divisao foi realizada separadamente para cada rotulo:", "",
        f"Treino    : {PROPORCAO_TREINO:.0%}",
        f"Validacao : {PROPORCAO_VALIDACAO:.0%}",
        f"Teste     : {PROPORCAO_TESTE:.0%}", "",
        f"Semente aleatoria utilizada: {SEED}", "",
        "Os arquivos de cada classe foram ordenados, embaralhados de forma pseudoaleatoria",
        "e depois distribuidos entre treino, validacao e teste.", "",
        "O numero de arquivos de treino e validacao foi arredondado para o inteiro mais",
        "proximo. Os arquivos restantes foram destinados ao conjunto de teste.", "",
        "-" * 60,
        f"{'Classe':<20}{'Total':>8}{'Train':>8}{'Val':>8}{'Test':>8}",
        "-" * 60,
    ]

    for classe, resultado in resultados.items():
        linhas.append(f"{classe:<20}{resultado['total']:>8}{resultado['train']:>8}{resultado['val']:>8}{resultado['test']:>8}")

    linhas += [
        "-" * 60,
        f"{'TOTAL':<20}{total:>8}{total_treino:>8}{total_validacao:>8}{total_teste:>8}",
        "-" * 60, "",
        "DISTRIBUICAO PERCENTUAL POR CONJUNTO",
        "-" * 60,
    ]

    for conjunto, total_conjunto in (("train", total_treino), ("val", total_validacao), ("test", total_teste)):
        linhas.append(f"\n{conjunto.upper()}:")

        for classe, resultado in resultados.items():
            percentual = 100 * resultado[conjunto] / total_conjunto
            linhas.append(f"  {classe:<15}: {resultado[conjunto]:>4} arquivos ({percentual:>5.1f}%)")

    linhas += ["", "Estrutura criada:", "", "dataset_processado_split/"]

    for conjunto in ("train", "val", "test"):
        linhas.append(f"    {conjunto}/")
        for rotulo in ROTULOS: linhas.append(f"        {rotulo}/")

    linhas += [
        "",
        "Nenhum dado foi modificado durante esta etapa.",
        "Os arquivos NPY foram apenas copiados entre treino, validacao e teste.",
    ]

    (DATASET_SPLIT / "README.txt").write_text("\n".join(linhas), encoding="utf-8")


def mostrar_resultado_final(resultados, problemas, shape_imagem):
    total = sum(r["total"] for r in resultados.values())
    total_treino = sum(r["train"] for r in resultados.values())
    total_validacao = sum(r["val"] for r in resultados.values())
    total_teste = sum(r["test"] for r in resultados.values())

    print("\n" + "=" * 78)
    print("VERIFICACAO FINAL")
    print("=" * 78)

    print(f"\nShape          : {shape_imagem}")
    print(f"Total original : {total}")
    print(f"Train          : {total_treino}")
    print(f"Val            : {total_validacao}")
    print(f"Test           : {total_teste}")
    print(f"Total final    : {total_treino + total_validacao + total_teste}")

    if problemas:
        print(f"\nProblemas encontrados: {len(problemas)}")
        for problema in problemas: print(f"  - {problema}")
        return

    print("\nOK - A divisao foi realizada corretamente.")
    print("Nenhum arquivo esta presente em mais de um conjunto.")
    print("Todos os arquivos originais estao presentes no dataset dividido.")
    print(f"\nDataset salvo em:\n{DATASET_SPLIT}")
    print(f"\nREADME salvo em:\n{DATASET_SPLIT / 'README.txt'}")
    print(f"\nGrafico salvo em:\n{DATASET_SPLIT / 'distribuicao_classes.png'}")


def main():
    if not DATASET_ORIGINAL.exists(): raise FileNotFoundError(f"Dataset processado nao encontrado:\n{DATASET_ORIGINAL}")

    # Valida tudo antes de recriar o split
    corretos, erros, shape_imagem = validar_dataset()

    if erros:
        mostrar_plano_divisao({}, corretos, erros, shape_imagem)
        print("\nDivisao cancelada. Corrija os erros antes de continuar.")
        return

    if shape_imagem is None: raise ValueError("Nenhum arquivo NPY valido foi encontrado.")

    # Mostra a divisao antes de alterar os arquivos
    resultados = analisar_dataset()
    mostrar_plano_divisao(resultados, corretos, erros, shape_imagem)

    if not confirmar_divisao():
        print("\nDivisao cancelada pelo usuario.")
        return

    criar_estrutura()

    print("\n" + "=" * 78)
    print("DIVIDINDO DATASET")
    print("=" * 78)

    distribuicao = dividir_dataset()
    problemas = verificar_dataset(distribuicao)

    # Arquivos auxiliares somente se a verificacao estiver correta
    if not problemas:
        criar_readme(resultados, shape_imagem)
        criar_grafico_distribuicao(resultados)

    mostrar_resultado_final(resultados, problemas, shape_imagem)


if __name__ == "__main__":
    main()