"""
grid_search.py

Objetivo:
- Executar Grid Search da LeNet5 ou VGG16, com parametros separados por modelo.
- Retomar rodadas interrompidas ou continuar epocas na pasta original.
- Alterar a arquitetura do classificador VGG16 sem modificar convolucoes/poolings.
"""

import sys
sys.dont_write_bytecode = True

# =============================================================================
# PARAMETROS AJUSTAVEIS
# =============================================================================

EXECUTAR_GRID_SEARCH = False
MODELO_GRID = "vgg16"  # "lenet5" ou "vgg16"

CONTINUAR_TREINOS = False
PASTA_GRID_CONTINUAR = "resultados_grid_vgg16_inicio_2026-10-28_14-55-46_fim_2026-10-28_18-45-25"
EPOCAS_EXTRAS = 600

RETOMAR_EXECUCAO = False
PASTA_EXECUCAO_RETOMAR = ""
REEXECUTAR_FALHAS = False

# Cada modelo possui combinacoes e limites independentes.
CONFIG_LENET5 = {
    "learning_rates": (1e-3,),
    "kernels": (3, 4, 5),
    "pares_neuronios": ((200, 140), (240, 168), (280, 196),
                        (320, 224), (360, 252), (400, 280)),
    "num_epochs": 600,
    "batch_size": 64,
    "intervalo_validacao": 5,
    "paciencia": 5,
}

CONFIG_VGG16 = {
    "learning_rates": (1e-4, 1e-5),

    "pares_neuronios": ((320, 224),(512, 358),(1024, 717),(2048, 1434),(2560, 1792),(3072, 2150),(3584, 2509),(4096, 2867),(4096, 4096),),
    "dropouts": (0.3, 0.5),
    "num_epochs": 800,
    "batch_size": 64,
    "intervalo_validacao": 5,
    "paciencia": 5,
}

USAR_DATASET_INTERNO = True
USAR_DATASET_EXTERNO = False
BALANCEAR_ORIGENS = False
USAR_AUGMENTATION = False
SEED_TREINO = 42
NUM_WORKERS = 0
USAR_CUDA = True


def executar():
    """Executa a busca do modelo selecionado quando habilitada."""
    if not EXECUTAR_GRID_SEARCH:
        print("\nGRID SEARCH desativado: EXECUTAR_GRID_SEARCH=False.")
        return {}

    from pdc_ia_image_defect_detection.pre_processing.grid_search_class import GridSearchProcessor
    from pdc_ia_image_defect_detection.models.modelos import normalizar_nome_modelo
    from pdc_ia_image_defect_detection.utils.paths import encontrar_raiz_projeto

    modelo = normalizar_nome_modelo(MODELO_GRID)
    configs = {"lenet5": CONFIG_LENET5, "vgg16": CONFIG_VGG16}
    if modelo not in configs:
        raise ValueError(f"MODELO_GRID invalido: {modelo}. Escolha lenet5 ou vgg16.")
    config = configs[modelo]

    raiz = encontrar_raiz_projeto(__file__)
    bases = {
        "INTERNA": {
            "ativa": USAR_DATASET_INTERNO,
            "train": raiz / "dataset_processado_split" / "train",
            "val": raiz / "dataset_processado_split" / "val",
        },
        "EXTERNA": {
            "ativa": USAR_DATASET_EXTERNO,
            "train": raiz / "dataset_externo_processado_split" / "train",
            "val": raiz / "dataset_externo_processado_split" / "val",
        },
    }
    return GridSearchProcessor(
        raiz_projeto=raiz,
        bases=bases,
        modelo=modelo,
        learning_rates=config["learning_rates"],
        kernels=config.get("kernels", ()),
        pares_neuronios=config["pares_neuronios"],
        dropouts=config.get("dropouts", ()),
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        intervalo_validacao=config["intervalo_validacao"],
        paciencia=config["paciencia"],
        seed=SEED_TREINO,
        num_workers=NUM_WORKERS,
        usar_cuda=USAR_CUDA,
        balancear_origens=BALANCEAR_ORIGENS,
        usar_augmentation=USAR_AUGMENTATION,
        continuar_treinos=CONTINUAR_TREINOS,
        pasta_grid_continuar=PASTA_GRID_CONTINUAR,
        epocas_extras=EPOCAS_EXTRAS,
        retomar_execucao=RETOMAR_EXECUCAO,
        pasta_execucao_retomar=PASTA_EXECUCAO_RETOMAR,
        reexecutar_falhas=REEXECUTAR_FALHAS,
    ).executar()


if __name__ == "__main__":
    executar()
