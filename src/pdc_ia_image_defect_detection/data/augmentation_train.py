"""
augmentation_train.py

Objetivo:
- Gera arquivos de data augmentation somente no conjunto de treino.
- Le a configuracao individual de cada classe em utils/augmention.py.
- Permite escolher quais classes recebem augmentation.
- Permite definir quantas copias gerar por imagem e quais transformacoes
  utilizar em cada classe.
- Remove augmentations antigos antes de gerar uma nova configuracao.
- Verifica os arquivos gerados ao final.

Usado depois de:
- pre_processing_data.py
- split.py

Entrada:
- dataset_processado_split/train/<classe>/*.npy

Saida:
- dataset_processado_split/train/<classe>/*_augNN.npy

Parametros ajustaveis:
- As configuracoes de classes, quantidade e transformacoes ficam em:
  pdc_ia_image_defect_detection/utils/augmention.py
- SEED_AUGMENTATION e CONFIRMAR_ANTES ficam neste arquivo.

Observacoes:
- Apenas train recebe augmentation.
- Val e test nunca sao modificados.
- Os arquivos originais do treino nunca sao alterados.
- Augmentations antigos sao removidos antes da nova geracao.
- Os arquivos gerados permanecem float32, shape (1,H,W) e intervalo [0,1].
"""

from pathlib import Path
from collections import Counter
import random
import re

import numpy as np
import torch

from pdc_ia_image_defect_detection.utils.augmention import (
    CONFIG_AUGMENTATION,
    AUGMENTATIONS_DISPONIVEIS,
    aplicar_augmentation_classe,
    deve_gerar_augmentation,
    quantidade_augmentation,
)


# =============================================================================
# PARAMETROS AJUSTAVEIS
# =============================================================================

# Mantem a geracao reproduzivel entre execucoes.
SEED_AUGMENTATION = 42

# Pede confirmacao antes de apagar augmentations antigos e gerar os novos.
CONFIRMAR_ANTES = True


# =============================================================================
# CAMINHOS
# =============================================================================

def encontrar_raiz_projeto():
    """
    Procura a raiz do projeto pela presenca de pyproject.toml.
    Funciona mesmo se este script mudar de subpasta dentro de src/.
    """

    caminho_atual = Path(__file__).resolve()

    for pasta in caminho_atual.parents:
        if (pasta / "pyproject.toml").exists():
            return pasta

    raise FileNotFoundError(
        "Nao foi possivel localizar a raiz do projeto "
        "(pyproject.toml nao encontrado)."
    )


BASE_DIR = encontrar_raiz_projeto()

DATASET_SPLIT = BASE_DIR / "dataset_processado_split"
TRAIN_DIR = DATASET_SPLIT / "train"
VAL_DIR = DATASET_SPLIT / "val"
TEST_DIR = DATASET_SPLIT / "test"


# Reconhece:
# arquivo_aug01.npy
# arquivo_aug001.npy
# arquivo_aug_01.npy
# arquivo_aug_1.npy
PADRAO_ARQUIVO_AUG = re.compile(
    r"_aug_?\d+$",
    flags=re.IGNORECASE
)


# =============================================================================
# FUNCOES AUXILIARES
# =============================================================================

def fixar_seed():
    random.seed(SEED_AUGMENTATION)
    np.random.seed(SEED_AUGMENTATION)
    torch.manual_seed(SEED_AUGMENTATION)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED_AUGMENTATION)


def eh_augmentation(caminho):
    return bool(
        PADRAO_ARQUIVO_AUG.search(
            caminho.stem
        )
    )


def obter_classes_dataset():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Pasta de treino nao encontrada:\n{TRAIN_DIR}"
        )

    classes = sorted([
        pasta.name
        for pasta in TRAIN_DIR.iterdir()
        if pasta.is_dir()
    ])

    if not classes:
        raise FileNotFoundError(
            f"Nenhuma classe encontrada em:\n{TRAIN_DIR}"
        )

    return classes


def verificar_classes(classes):
    classes_dataset = set(classes)
    classes_config = set(CONFIG_AUGMENTATION)

    faltando_config = classes_dataset - classes_config
    sobrando_config = classes_config - classes_dataset

    if faltando_config:
        raise ValueError(
            "Classes presentes no dataset sem configuracao de augmentation: "
            f"{sorted(faltando_config)}"
        )

    if sobrando_config:
        raise ValueError(
            "Classes configuradas que nao existem no conjunto de treino: "
            f"{sorted(sobrando_config)}"
        )


def listar_originais(classe):
    pasta = TRAIN_DIR / classe

    return sorted([
        caminho
        for caminho in pasta.glob("*.npy")
        if not eh_augmentation(caminho)
    ])


def listar_augmentations(classe):
    pasta = TRAIN_DIR / classe

    return sorted([
        caminho
        for caminho in pasta.glob("*.npy")
        if eh_augmentation(caminho)
    ])


def validar_array_original(array, caminho):
    if array.dtype != np.float32:
        raise ValueError(
            f"Dtype invalido em {caminho.name}: "
            f"{array.dtype}. Esperado: float32."
        )

    if array.ndim != 3 or array.shape[0] != 1:
        raise ValueError(
            f"Shape invalido em {caminho.name}: "
            f"{array.shape}. Esperado: (1,H,W)."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"NaN ou Inf encontrado em {caminho.name}."
        )

    minimo = float(array.min())
    maximo = float(array.max())

    if minimo < 0.0 or maximo > 1.0:
        raise ValueError(
            f"Valores fora de [0,1] em {caminho.name}: "
            f"min={minimo:.6f}, max={maximo:.6f}."
        )


def carregar_tensor(caminho):
    array = np.load(
        caminho,
        allow_pickle=False
    )

    validar_array_original(
        array,
        caminho
    )

    return torch.from_numpy(
        array
    ).float()


def nome_augmentation(caminho_original, indice):
    return (
        caminho_original.parent
        / f"{caminho_original.stem}_aug{indice:02d}.npy"
    )


# =============================================================================
# LIMPEZA
# =============================================================================

def remover_augmentations_antigos(classes):
    removidos_por_classe = Counter()

    for classe in classes:
        arquivos = listar_augmentations(
            classe
        )

        for caminho in arquivos:
            caminho.unlink()

        removidos_por_classe[classe] = len(
            arquivos
        )

    return removidos_por_classe


# =============================================================================
# PLANO
# =============================================================================

def transformacoes_ativas(classe):
    config = CONFIG_AUGMENTATION[
        classe
    ]

    return [
        nome
        for nome in AUGMENTATIONS_DISPONIVEIS
        if config[nome]
    ]


def mostrar_plano(classes):
    print("\n" + "=" * 80)
    print("PLANO DE DATA AUGMENTATION")
    print("=" * 80)

    print(f"\nDataset split : {DATASET_SPLIT}")
    print(f"Treino        : {TRAIN_DIR}")
    print(f"Seed          : {SEED_AUGMENTATION}")

    print(
        "\nSomente a pasta train sera modificada. "
        "Val e test permanecerao intactos."
    )

    total_originais = 0
    total_previsto = 0
    total_antigos = 0

    print("\n" + "-" * 80)

    for classe in classes:
        config = CONFIG_AUGMENTATION[
            classe
        ]

        originais = listar_originais(
            classe
        )

        antigos = listar_augmentations(
            classe
        )

        quantidade = quantidade_augmentation(
            classe
        )

        gerar = deve_gerar_augmentation(
            classe
        )

        previsto = (
            len(originais) * quantidade
            if gerar
            else 0
        )

        ativas = transformacoes_ativas(
            classe
        )

        total_originais += len(originais)
        total_previsto += previsto
        total_antigos += len(antigos)

        print(f"\nClasse: {classe}")
        print(f"  Originais              : {len(originais)}")
        print(f"  Augmentations antigos  : {len(antigos)}")
        print(f"  Gerar                  : {config['gerar']}")
        print(f"  Copias por original    : {quantidade}")
        print(f"  Novos previstos        : {previsto}")

        if ativas:
            print(
                "  Transformacoes         : "
                + ", ".join(ativas)
            )
        else:
            print(
                "  Transformacoes         : nenhuma"
            )

    print("\n" + "-" * 80)
    print(f"Originais totais        : {total_originais}")
    print(f"Augmentations antigos   : {total_antigos}")
    print(f"Novos previstos         : {total_previsto}")
    print(
        f"Total final no train    : "
        f"{total_originais + total_previsto}"
    )

    return {
        "originais": total_originais,
        "antigos": total_antigos,
        "previstos": total_previsto,
    }


# =============================================================================
# GERACAO
# =============================================================================

def gerar_augmentation_classe(classe):
    if not deve_gerar_augmentation(
        classe
    ):
        return 0

    quantidade = quantidade_augmentation(
        classe
    )

    originais = listar_originais(
        classe
    )

    gerados = 0

    for numero, caminho in enumerate(
        originais,
        start=1
    ):
        tensor_original = carregar_tensor(
            caminho
        )

        for indice_aug in range(
            1,
            quantidade + 1
        ):
            tensor_aug = aplicar_augmentation_classe(
                tensor_original,
                classe
            )

            array_aug = (
                tensor_aug
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False
                )
            )

            destino = nome_augmentation(
                caminho,
                indice_aug
            )

            np.save(
                destino,
                array_aug
            )

            gerados += 1

        print(
            f"\r  {classe:<15} "
            f"{numero:>3}/{len(originais):<3} "
            f"| gerados: {gerados}",
            end="",
            flush=True
        )

    if originais:
        print()

    return gerados


def gerar_augmentations(classes):
    gerados_por_classe = Counter()

    print("\n" + "=" * 80)
    print("GERANDO AUGMENTATIONS")
    print("=" * 80)

    for classe in classes:
        if not deve_gerar_augmentation(
            classe
        ):
            print(
                f"\n{classe}: augmentation desativado."
            )

            gerados_por_classe[
                classe
            ] = 0

            continue

        print(
            f"\nProcessando classe: {classe}"
        )

        gerados_por_classe[
            classe
        ] = gerar_augmentation_classe(
            classe
        )

    return gerados_por_classe


# =============================================================================
# VERIFICACAO
# =============================================================================

def verificar_arquivo_augmentation(caminho):
    erros = []

    try:
        array = np.load(
            caminho,
            allow_pickle=False
        )

    except Exception as erro:
        return [
            f"erro de leitura: {erro}"
        ]

    if array.dtype != np.float32:
        erros.append(
            f"dtype={array.dtype}"
        )

    if array.ndim != 3 or array.shape[0] != 1:
        erros.append(
            f"shape={array.shape}"
        )

    if not np.isfinite(array).all():
        erros.append(
            "possui NaN/Inf"
        )

    if array.size > 0:
        minimo = float(
            array.min()
        )

        maximo = float(
            array.max()
        )

        if minimo < 0.0:
            erros.append(
                f"min={minimo:.6f}"
            )

        if maximo > 1.0:
            erros.append(
                f"max={maximo:.6f}"
            )

    return erros


def verificar_resultado(classes):
    print("\n" + "=" * 80)
    print("VERIFICACAO FINAL")
    print("=" * 80)

    total_originais = 0
    total_aug = 0
    erros = []

    for classe in classes:
        originais = listar_originais(
            classe
        )

        augmentations = listar_augmentations(
            classe
        )

        gerar = deve_gerar_augmentation(
            classe
        )

        quantidade = quantidade_augmentation(
            classe
        )

        esperado = (
            len(originais) * quantidade
            if gerar
            else 0
        )

        total_originais += len(
            originais
        )

        total_aug += len(
            augmentations
        )

        if len(augmentations) != esperado:
            erros.append(
                f"{classe}: esperado={esperado}, "
                f"encontrado={len(augmentations)}"
            )

        for caminho in augmentations:
            problemas = verificar_arquivo_augmentation(
                caminho
            )

            for problema in problemas:
                erros.append(
                    f"{classe}/{caminho.name}: "
                    f"{problema}"
                )

        status = (
            "OK"
            if len(augmentations) == esperado
            else "ERRO"
        )

        print(
            f"{classe:<15} | "
            f"originais={len(originais):>3} | "
            f"aug={len(augmentations):>4} | "
            f"esperado={esperado:>4} | "
            f"{status}"
        )

    print("-" * 80)

    print(
        f"Originais no train      : "
        f"{total_originais}"
    )

    print(
        f"Augmentations no train  : "
        f"{total_aug}"
    )

    print(
        f"Total de arquivos train : "
        f"{total_originais + total_aug}"
    )

    if erros:
        print(
            "\nForam encontrados problemas:"
        )

        for erro in erros:
            print(
                f"- {erro}"
            )

        raise RuntimeError(
            f"Verificacao falhou com "
            f"{len(erros)} problema(s)."
        )

    print(
        "\nVerificacao concluida sem erros."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    classes = obter_classes_dataset()

    verificar_classes(
        classes
    )

    fixar_seed()

    plano = mostrar_plano(
        classes
    )

    if CONFIRMAR_ANTES:
        print(
            "\nATENCAO:"
        )

        print(
            "- todos os arquivos de augmentation antigos "
            "do conjunto train serao apagados;"
        )

        print(
            "- os arquivos originais nao serao modificados;"
        )

        print(
            "- val e test nao serao modificados."
        )

        resposta = input(
            "\nContinuar? [S/N]: "
        ).strip().lower()

        if resposta not in {
            "s",
            "sim",
            "y",
            "yes"
        }:
            print(
                "\nOperacao cancelada."
            )

            return

    print("\n" + "=" * 80)
    print("REMOVENDO AUGMENTATIONS ANTIGOS")
    print("=" * 80)

    removidos = remover_augmentations_antigos(
        classes
    )

    for classe in classes:
        print(
            f"{classe:<15}: "
            f"{removidos[classe]} removido(s)"
        )

    print(
        f"\nTotal removido: "
        f"{sum(removidos.values())}"
    )

    gerados = gerar_augmentations(
        classes
    )

    print("\n" + "=" * 80)
    print("GERACAO CONCLUIDA")
    print("=" * 80)

    for classe in classes:
        print(
            f"{classe:<15}: "
            f"{gerados[classe]} gerado(s)"
        )

    print(
        f"\nTotal gerado: "
        f"{sum(gerados.values())}"
    )

    if sum(gerados.values()) != plano[
        "previstos"
    ]:
        raise RuntimeError(
            "Quantidade total gerada diferente "
            "da quantidade prevista."
        )

    verificar_resultado(
        classes
    )

    print("\n" + "=" * 80)
    print("FINALIZADO")
    print("=" * 80)

    print(
        f"\nDataset de treino:\n"
        f"{TRAIN_DIR}"
    )

    print(
        "\nSomente o conjunto train recebeu augmentation."
    )

    print(
        "Os conjuntos val e test permaneceram inalterados."
    )


if __name__ == "__main__":
    main()