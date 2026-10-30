"""
grid_test.py

Objetivo:
- Testar modelo_melhor.pt e modelo_final.pt de um Grid Search LeNet5 ou VGG16.
- Salvar metricas, resultados por classe, rankings e Top 20 em pasta separada.
- Selecionar modelo e parametros de teste no bloco correspondente.
"""

import sys
from pathlib import Path
sys.dont_write_bytecode = True

# =============================================================================
# PARAMETROS AJUSTAVEIS
# =============================================================================

EXECUTAR_GRID_TEST = False
MODELO_GRID_TEST = "vgg16"  # "lenet5" ou "vgg16"

CONFIG_TEST_LENET5 = {
    "pasta_grid_treino": "resultados_grid_vgg16_inicio_2026-10-28_14-55-46_fim_2026-10-28_18-45-25",  # Nome da pasta da busca em resultados/
    "batch_size": 64,
}

CONFIG_TEST_VGG16 = {
    "pasta_grid_treino": "resultados_grid_vgg16_inicio_2026-10-29_08-32-38_fim_2026-10-29_11-09-01",  # Nome da pasta da busca VGG16 em resultados/
    "batch_size": 64,
}

USAR_DATASET_INTERNO = True
USAR_DATASET_EXTERNO = False
NUM_WORKERS = 0
USAR_CUDA = True
SEED_TEST = 42
SALVAR_MATRIZES_CONFUSAO = True
TOP_GRAFICOS = 20
TOP_PARAMETROS = 20


def executar():
    """Avalia todos os checkpoints da busca selecionada."""
    if not EXECUTAR_GRID_TEST:
        print("\nGRID TEST desativado: EXECUTAR_GRID_TEST=False.")
        return {}
    from pdc_ia_image_defect_detection.models.modelos import normalizar_nome_modelo
    from pdc_ia_image_defect_detection.pre_processing.grid_test_class import GridTestProcessor
    from pdc_ia_image_defect_detection.utils.paths import encontrar_raiz_projeto

    modelo = normalizar_nome_modelo(MODELO_GRID_TEST)
    configs = {"lenet5": CONFIG_TEST_LENET5, "vgg16": CONFIG_TEST_VGG16}
    if modelo not in configs:
        raise ValueError(f"MODELO_GRID_TEST invalido: {modelo}.")
    config = configs[modelo]
    nome = str(config["pasta_grid_treino"]).strip()
    if not nome:
        raise ValueError(f"Informe pasta_grid_treino no bloco {modelo}.")
    raiz = encontrar_raiz_projeto(__file__)
    pasta = Path(nome)
    if not pasta.is_absolute():
        pasta = raiz / "resultados" / pasta
    bases = {
        "INTERNA": {"ativa": USAR_DATASET_INTERNO,
                    "test": raiz / "dataset_processado_split" / "test"},
        "EXTERNA": {"ativa": USAR_DATASET_EXTERNO,
                    "test": raiz / "dataset_externo_processado_split" / "test"},
    }
    return GridTestProcessor(
        pasta_grid=pasta,
        raiz_resultados=raiz / "resultados",
        bases=bases,
        modelo=modelo,
        batch_size=config["batch_size"],
        num_workers=NUM_WORKERS,
        usar_cuda=USAR_CUDA,
        seed=SEED_TEST,
        salvar_matrizes_confusao=SALVAR_MATRIZES_CONFUSAO,
        top_graficos=TOP_GRAFICOS,
        top_parametros=TOP_PARAMETROS,
    ).executar()


if __name__ == "__main__":
    executar()
