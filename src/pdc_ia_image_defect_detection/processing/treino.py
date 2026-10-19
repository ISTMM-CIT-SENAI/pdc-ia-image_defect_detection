"""
treino.py

Objetivo:
- Configurar e iniciar o treinamento do modelo.
- Utilizar somente train e val das bases selecionadas.
- Permitir combinar a base interna e a base externa.
- Permitir balancear as duas origens por classe quando ambas estiverem ativas.
- Permitir incluir ou ignorar os arquivos de augmentation ja existentes em train.
- Manter os parametros ajustaveis neste executavel e a logica em treino_class.py.

Bases:
- Interna  -> dataset_processado_split/train e val
- Externa  -> dataset_externo_processado_split/train e val

Regras:
- Test nao e carregado nem utilizado pelo treinamento.
- O conjunto test sera usado exclusivamente pelo futuro script de predicao/teste.
- Augmentation, quando existente, ja esta materializado dentro de train.
- USAR_AUGMENTATION apenas decide se esses arquivos entram no treino; nao gera augmentation.
- Se EXECUTAR_TREINO=False, a etapa termina antes de importar PyTorch, modelo ou
  classe de treinamento.
"""

import sys
sys.dont_write_bytecode = True


# =============================================================================
# PARAMETROS AJUSTAVEIS
# =============================================================================

# Ativa ou desativa completamente a etapa de treinamento.
EXECUTAR_TREINO = False

# Define quais bases participam do treinamento e da validacao.
USAR_DATASET_INTERNO = True
USAR_DATASET_EXTERNO = True

# Quando as duas bases estao ativas, iguala a quantidade INTERNA/EXTERNA por
# classe separadamente em train e val. Se uma origem nao possuir determinada
# classe, todas as amostras disponiveis da outra origem sao mantidas.
BALANCEAR_ORIGENS = True

# True inclui no train os arquivos gerados pelo augmentation.
# False usa somente as imagens originais, mesmo que arquivos _augXX.npy existam.
USAR_AUGMENTATION = True

# Hiperparametros principais.
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
INTERVALO_VALIDACAO = 5
SEED_TREINO = 42

# Configuracao do DataLoader. Zero e a opcao mais segura no Windows.
NUM_WORKERS = 0

# Usa CUDA quando disponivel. False forca CPU.
USAR_CUDA = True


def executar():
    """Executa o treinamento somente quando a etapa estiver ativada."""

    if not EXECUTAR_TREINO:
        print("\nTREINO desativado: EXECUTAR_TREINO=False.")
        return {}

    # Imports pesados e internos ocorrem somente quando o treino sera executado.
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

    return TreinoProcessor(
        bases=bases,
        pasta_resultados=base_dir / "resultados" / "resultados_treino",
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        intervalo_validacao=INTERVALO_VALIDACAO,
        seed=SEED_TREINO,
        num_workers=NUM_WORKERS,
        usar_cuda=USAR_CUDA,
        balancear_origens=BALANCEAR_ORIGENS,
        usar_augmentation=USAR_AUGMENTATION,
    ).executar()


if __name__ == "__main__":
    executar()
