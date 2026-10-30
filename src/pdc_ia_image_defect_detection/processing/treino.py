"""
treino.py

Objetivo:
- Treinar a arquitetura selecionada usando somente train e val.
- Configurar separadamente LeNet5 (010-M), VGG16 (035-F) e VGG16 ImageNet.
- Selecionar bases interna/externa, balanceamento e augmentation existente.
- Manter a logica de treinamento em pre_processing/treino_class.py.

Regras:
- O conjunto test nao participa do treinamento.
- Augmentation e utilizada somente quando ja existe no diretorio train.
- Quando desativado, nao importa as dependencias de treinamento.
"""

import sys

sys.dont_write_bytecode = True


# =============================================================================
# PARAMETROS AJUSTAVEIS
# =============================================================================

EXECUTAR_TREINO = True
MODELO = "vgg16_imagenet"  # "lenet5", "vgg16" ou "vgg16_imagenet"

USAR_DATASET_INTERNO = True
USAR_DATASET_EXTERNO = False
BALANCEAR_ORIGENS = False
USAR_AUGMENTATION = False

# LeNet5: configuracao selecionada no Grid Search, experimento 010-M.
CONFIG_LENET5 = {
    "kernel_size": 5,
    "neuronios_fc1": 320,
    "neuronios_fc2": 224,
    "learning_rate": 1e-3,
    "num_epochs": 300,
    "batch_size": 64,
    "intervalo_validacao": 5,
    "paciencia": 10,
}

# VGG16: configuracao selecionada no Grid Search, experimento 035-F.
# Os kernels convolucionais (3x3) e de pooling (2x2) sao fixos na arquitetura.
CONFIG_VGG16 = {
    "neuronios_fc1": 4096,
    "neuronios_fc2": 4096,
    "dropout": 0.5,
    "learning_rate": 1e-4,
    "num_epochs": 300,
    "batch_size": 64,
    "intervalo_validacao": 5,
    "paciencia": 3,
}

# VGG16 ImageNet: arquitetura oficial, saida para 4 classes e pesos pre-treinados.
# Dados esperados: NPY grayscale float32, 224x224 e valores em [0,1].
# A duplicacao dos canais e a normalizacao ImageNet ocorrem dentro do modelo.
# Modo completo atualiza todos os pesos; parcial atualiza o ultimo bloco e FCs;
# classificador atualiza somente a camada final de saida.
CONFIG_VGG16_IMAGENET = {
    "modo_fine_tuning": "completo",
    "learning_rate": 1e-5,
    "num_epochs": 300,
    "batch_size": 64,
    "intervalo_validacao": 5,
    "paciencia": 3,
}

SEED_TREINO = 42
NUM_WORKERS = 0
USAR_CUDA = True


def executar():
    """Configura e executa o treinamento da arquitetura selecionada."""
    if not EXECUTAR_TREINO:
        print("\nTREINO desativado: EXECUTAR_TREINO=False.")
        return {}

    nome_modelo = MODELO.strip().lower()
    configuracoes = {
        "lenet5": (CONFIG_LENET5, ("kernel_size", "neuronios_fc1", "neuronios_fc2")),
        "vgg16": (CONFIG_VGG16, ("neuronios_fc1", "neuronios_fc2", "dropout")),
        "vgg16_imagenet": (CONFIG_VGG16_IMAGENET, ("modo_fine_tuning",)),
    }
    if nome_modelo not in configuracoes:
        raise ValueError(f"Modelo nao suportado: {MODELO!r}. Use 'lenet5', 'vgg16' ou 'vgg16_imagenet'.")

    config, nomes_parametros = configuracoes[nome_modelo]
    parametros_modelo = {nome: config[nome] for nome in nomes_parametros}

    from pdc_ia_image_defect_detection.pre_processing.treino_class import TreinoProcessor
    from pdc_ia_image_defect_detection.utils.paths import encontrar_raiz_projeto

    base_dir = encontrar_raiz_projeto(__file__)
    bases = {
        "INTERNA": {
            "ativa": USAR_DATASET_INTERNO,
            "train": base_dir / "dataset_processado_split" / "train",
            "val": base_dir / "dataset_processado_split" / "val",
        },
        "EXTERNA": {
            "ativa": USAR_DATASET_EXTERNO,
            "train": base_dir / "dataset_externo_processado_split" / "train",
            "val": base_dir / "dataset_externo_processado_split" / "val",
        },
    }

    print(f"\nModelo selecionado: {nome_modelo}")
    print(f"Parametros da arquitetura: {parametros_modelo}")

    return TreinoProcessor(
        nome_modelo=nome_modelo,
        bases=bases,
        pasta_resultados=base_dir / "resultados" / "resultados_treino",
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        intervalo_validacao=config["intervalo_validacao"],
        paciencia=config["paciencia"],
        seed=SEED_TREINO,
        num_workers=NUM_WORKERS,
        usar_cuda=USAR_CUDA,
        balancear_origens=BALANCEAR_ORIGENS,
        usar_augmentation=USAR_AUGMENTATION,
        parametros_modelo=parametros_modelo,
    ).executar()


if __name__ == "__main__":
    executar()
