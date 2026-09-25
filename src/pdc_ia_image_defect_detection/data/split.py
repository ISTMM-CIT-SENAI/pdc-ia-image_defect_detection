import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Caminhos do projeto
BASE_DIR = Path(__file__).resolve().parents[3]
DATASET_ORIGINAL = BASE_DIR / "dataset_processado"
DATASET_SPLIT = BASE_DIR / "dataset_processado_split"

# Proporcoes da divisao
PROPORCAO_TREINO = 0.70
PROPORCAO_VALIDACAO = 0.15
PROPORCAO_TESTE = 0.15

# Semente usada para garantir reprodutibilidade do split
SEED = 42


def obter_classes(root):
    """Retorna as classes encontradas no dataset"""
    return sorted([pasta for pasta in root.iterdir() if pasta.is_dir()])


def obter_arquivos(pasta):
    """Retorna os arquivos NPY encontrados em uma pasta"""
    return sorted([arquivo for arquivo in pasta.iterdir() if arquivo.is_file() and arquivo.suffix.lower() == ".npy"])


def calcular_divisao(quantidade):
    """Calcula as quantidades de treino, validacao e teste"""

    n_treino = round(quantidade * PROPORCAO_TREINO)
    n_validacao = round(quantidade * PROPORCAO_VALIDACAO)
    n_teste = quantidade - n_treino - n_validacao

    return n_treino, n_validacao, n_teste


def analisar_dataset():
    """Calcula previamente como cada classe sera dividida"""

    resultados = {}

    for pasta_classe in obter_classes(DATASET_ORIGINAL):
        quantidade = len(obter_arquivos(pasta_classe))
        n_treino, n_validacao, n_teste = calcular_divisao(quantidade)

        resultados[pasta_classe.name] = {
            "total": quantidade,
            "train": n_treino,
            "val": n_validacao,
            "test": n_teste,
        }

    return resultados


def mostrar_plano_divisao(resultados):
    """Mostra como sera feito o split"""

    print("\n" + "=" * 78)
    print("PLANO DE DIVISAO DO DATASET")
    print("=" * 78)

    print(f"\nOrigem : {DATASET_ORIGINAL}")
    print(f"Destino: {DATASET_SPLIT}")
    print(
        f"\nProporcao: {PROPORCAO_TREINO:.0%} treino | "
        f"{PROPORCAO_VALIDACAO:.0%} validacao | {PROPORCAO_TESTE:.0%} teste"
    )
    print(f"Semente aleatoria: {SEED}")

    print("\n" + "-" * 78)
    print(f"{'Classe':<20}{'Total':>10}{'Train':>10}{'Val':>10}{'Test':>10}")
    print("-" * 78)

    for classe, resultado in resultados.items():
        print(
            f"{classe:<20}{resultado['total']:>10}{resultado['train']:>10}"
            f"{resultado['val']:>10}{resultado['test']:>10}"
        )

    total = sum(resultado["total"] for resultado in resultados.values())
    total_treino = sum(resultado["train"] for resultado in resultados.values())
    total_validacao = sum(resultado["val"] for resultado in resultados.values())
    total_teste = sum(resultado["test"] for resultado in resultados.values())

    print("-" * 78)
    print(f"{'TOTAL':<20}{total:>10}{total_treino:>10}{total_validacao:>10}{total_teste:>10}")
    print("-" * 78)

    if DATASET_SPLIT.exists():
        print(
            "\nATENCAO: a pasta dataset_processado_split ja existe.\n"
            "Ela sera recriada caso o processamento seja confirmado."
        )


def confirmar_divisao():
    """Pergunta ao usuario se deseja realizar o split"""

    while True:
        resposta = input("\nDeseja continuar com esta divisao? [Y/N]: ").strip().upper()

        if resposta == "Y":
            return True

        if resposta == "N":
            return False

        print("Opcao invalida. Digite Y ou N.")


def criar_estrutura():
    """Cria a estrutura de pastas do novo dataset"""

    if DATASET_SPLIT.exists():
        shutil.rmtree(DATASET_SPLIT)

    for conjunto in ("train", "val", "test"):
        for pasta_classe in obter_classes(DATASET_ORIGINAL):
            (DATASET_SPLIT / conjunto / pasta_classe.name).mkdir(parents=True, exist_ok=True)


def dividir_dataset():
    """Embaralha e divide cada classe entre treino, validacao e teste"""

    random.seed(SEED)
    distribuicao = {}

    for pasta_classe in obter_classes(DATASET_ORIGINAL):
        arquivos = obter_arquivos(pasta_classe)
        random.shuffle(arquivos)

        n_treino, n_validacao, _ = calcular_divisao(len(arquivos))

        arquivos_treino = arquivos[:n_treino]
        arquivos_validacao = arquivos[n_treino:n_treino + n_validacao]
        arquivos_teste = arquivos[n_treino + n_validacao:]

        conjuntos = {
            "train": arquivos_treino,
            "val": arquivos_validacao,
            "test": arquivos_teste,
        }

        distribuicao[pasta_classe.name] = {
            conjunto: [arquivo.name for arquivo in lista_arquivos]
            for conjunto, lista_arquivos in conjuntos.items()
        }

        print(f"\n{pasta_classe.name}:")

        for conjunto, lista_arquivos in conjuntos.items():
            pasta_saida = DATASET_SPLIT / conjunto / pasta_classe.name

            for indice, caminho_arquivo in enumerate(lista_arquivos, start=1):
                shutil.copy2(caminho_arquivo, pasta_saida / caminho_arquivo.name)
                print(f"\r  {conjunto}: {indice}/{len(lista_arquivos)}", end="", flush=True)

            print()

    return distribuicao


def verificar_dataset(distribuicao):
    """Verifica quantidades, sobreposicoes e arquivos faltantes"""

    problemas = []
    total_original = 0
    total_final = 0

    for pasta_classe in obter_classes(DATASET_ORIGINAL):
        classe = pasta_classe.name

        originais = {arquivo.name for arquivo in obter_arquivos(pasta_classe)}
        treino = {arquivo.name for arquivo in obter_arquivos(DATASET_SPLIT / "train" / classe)}
        validacao = {arquivo.name for arquivo in obter_arquivos(DATASET_SPLIT / "val" / classe)}
        teste = {arquivo.name for arquivo in obter_arquivos(DATASET_SPLIT / "test" / classe)}

        total_original += len(originais)
        total_final += len(treino) + len(validacao) + len(teste)

        if treino & validacao:
            problemas.append(f"{classe}: existem arquivos presentes em treino e validacao.")

        if treino & teste:
            problemas.append(f"{classe}: existem arquivos presentes em treino e teste.")

        if validacao & teste:
            problemas.append(f"{classe}: existem arquivos presentes em validacao e teste.")

        if originais != treino | validacao | teste:
            problemas.append(f"{classe}: o conjunto final de arquivos nao corresponde ao conjunto original.")

        if len(treino) != len(distribuicao[classe]["train"]):
            problemas.append(f"{classe}: quantidade incorreta no treino.")

        if len(validacao) != len(distribuicao[classe]["val"]):
            problemas.append(f"{classe}: quantidade incorreta na validacao.")

        if len(teste) != len(distribuicao[classe]["test"]):
            problemas.append(f"{classe}: quantidade incorreta no teste.")

    if total_original != total_final:
        problemas.append(f"Total original ({total_original}) diferente do total final ({total_final}).")

    return problemas


def criar_grafico_distribuicao(resultados):
    """Salva a distribuicao das classes nos conjuntos de treino, validacao e teste"""

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


def criar_readme(resultados):
    """Cria um arquivo descrevendo como o dataset foi dividido"""

    total = sum(resultado["total"] for resultado in resultados.values())
    total_treino = sum(resultado["train"] for resultado in resultados.values())
    total_validacao = sum(resultado["val"] for resultado in resultados.values())
    total_teste = sum(resultado["test"] for resultado in resultados.values())

    linhas = [
        "DATASET PROCESSADO - DIVISAO TREINO / VALIDACAO / TESTE",
        "=" * 60,
        "",
        "Este dataset foi criado a partir da pasta 'dataset_processado'.",
        "",
        "Os dados ja foram pre-processados antes da divisao:",
        "  - grayscale",
        "  - shape (1, 1024, 1024)",
        "  - dtype float32",
        "  - valores entre 0.0 e 1.0",
        "  - formato NPY",
        "",
        "A divisao foi realizada separadamente para cada classe:",
        "",
        f"Treino    : {PROPORCAO_TREINO:.0%}",
        f"Validacao : {PROPORCAO_VALIDACAO:.0%}",
        f"Teste     : {PROPORCAO_TESTE:.0%}",
        "",
        f"Semente aleatoria utilizada: {SEED}",
        "",
        "Os arquivos de cada classe foram ordenados, embaralhados de forma pseudoaleatoria",
        "e depois distribuidos entre treino, validacao e teste.",
        "",
        "O numero de arquivos de treino e validacao foi arredondado para o inteiro mais",
        "proximo. Os arquivos restantes foram destinados ao conjunto de teste.",
        "",
        "-" * 60,
        f"{'Classe':<20}{'Total':>8}{'Train':>8}{'Val':>8}{'Test':>8}",
        "-" * 60,
    ]

    for classe, resultado in resultados.items():
        linhas.append(
            f"{classe:<20}{resultado['total']:>8}{resultado['train']:>8}"
            f"{resultado['val']:>8}{resultado['test']:>8}"
        )

    linhas.extend([
        "-" * 60,
        f"{'TOTAL':<20}{total:>8}{total_treino:>8}{total_validacao:>8}{total_teste:>8}",
        "-" * 60,
        "",
        "DISTRIBUICAO PERCENTUAL POR CONJUNTO",
        "-" * 60,
    ])

    for conjunto, total_conjunto in [("train", total_treino), ("val", total_validacao), ("test", total_teste)]:
        linhas.append(f"\n{conjunto.upper()}:")

        for classe, resultado in resultados.items():
            percentual = 100 * resultado[conjunto] / total_conjunto
            linhas.append(f"  {classe:<15}: {resultado[conjunto]:>4} arquivos ({percentual:>5.1f}%)")

    linhas.extend([
        "",
        "Estrutura criada:",
        "",
        "dataset_processado_split/",
    ])

    for conjunto in ("train", "val", "test"):
        linhas.append(f"    {conjunto}/")

        for classe in resultados:
            linhas.append(f"        {classe}/")

    linhas.extend([
        "",
        "Nenhum dado foi modificado durante esta etapa.",
        "Os arquivos NPY foram apenas copiados entre treino, validacao e teste.",
    ])

    (DATASET_SPLIT / "README.txt").write_text("\n".join(linhas), encoding="utf-8")


def mostrar_resultado_final(resultados, problemas):
    """Mostra o resumo final da divisao"""

    total = sum(resultado["total"] for resultado in resultados.values())
    total_treino = sum(resultado["train"] for resultado in resultados.values())
    total_validacao = sum(resultado["val"] for resultado in resultados.values())
    total_teste = sum(resultado["test"] for resultado in resultados.values())

    print("\n" + "=" * 78)
    print("VERIFICACAO FINAL")
    print("=" * 78)

    print(f"\nTotal original : {total}")
    print(f"Train          : {total_treino}")
    print(f"Val            : {total_validacao}")
    print(f"Test           : {total_teste}")
    print(f"Total final    : {total_treino + total_validacao + total_teste}")

    if problemas:
        print("\nATENCAO - Foram encontrados problemas:")

        for problema in problemas:
            print(f"  - {problema}")

        return

    print("\nOK - A divisao foi realizada corretamente.")
    print("Nenhum arquivo esta presente em mais de um conjunto.")
    print("Todos os arquivos originais estao presentes no dataset dividido.")
    print(f"\nDataset salvo em:\n{DATASET_SPLIT}")
    print(f"\nREADME salvo em:\n{DATASET_SPLIT / 'README.txt'}")
    print(f"\nGrafico salvo em:\n{DATASET_SPLIT / 'distribuicao_classes.png'}")


def main():

    if not DATASET_ORIGINAL.exists():
        raise FileNotFoundError(f"Dataset processado nao encontrado:\n{DATASET_ORIGINAL}")

    # Analise da divisao proposta
    resultados = analisar_dataset()
    mostrar_plano_divisao(resultados)

    if not confirmar_divisao():
        print("\nDivisao cancelada pelo usuario.")
        return

    # Criacao do novo dataset
    criar_estrutura()

    print("\n" + "=" * 78)
    print("DIVIDINDO DATASET")
    print("=" * 78)

    distribuicao = dividir_dataset()

    # Verificacao final
    problemas = verificar_dataset(distribuicao)

    if not problemas:
        criar_readme(resultados)
        criar_grafico_distribuicao(resultados)

    mostrar_resultado_final(resultados, problemas)


if __name__ == "__main__":
    main()