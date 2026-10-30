"""
main.py

Objetivo:
- Orquestrar as etapas principais do projeto na ordem correta.
- Cada processing decide internamente, pela propria flag EXECUTAR_..., se deve
  rodar ou ser ignorado.
- Interromper o pipeline quando uma etapa ativa apresentar erro.

Ordem atual:
1. SAM
2. PREPARA_IMG
3. SPLIT
4. AUGMENTATION
5. TREINO
6. GRID_SEARCH
7. GRID_TEST
8. TEST

Observacoes:
- O treinamento utiliza somente train e val.
- O teste e executado separadamente e nunca atualiza os pesos do modelo.
- Parametros especificos permanecem nos respectivos arquivos de processing.
"""

import sys
from importlib import import_module

sys.dont_write_bytecode = True


# =============================================================================
# ETAPAS DO PIPELINE
# =============================================================================

ETAPAS = (
    ("SAM", "pdc_ia_image_defect_detection.processing.sam"),
    ("PREPARA_IMG", "pdc_ia_image_defect_detection.processing.prepara_img"),
    ("SPLIT", "pdc_ia_image_defect_detection.processing.split"),
    ("AUGMENTATION", "pdc_ia_image_defect_detection.processing.augmentation"),
    ("TREINO", "pdc_ia_image_defect_detection.processing.treino"),
    ("GRID_SEARCH", "pdc_ia_image_defect_detection.processing.grid_search"),
    ("GRID_TEST", "pdc_ia_image_defect_detection.processing.grid_test"),
    ("TEST", "pdc_ia_image_defect_detection.processing.test"),
)


# =============================================================================
# EXECUCAO
# =============================================================================

def executar():
    """Executa todas as etapas registradas em ETAPAS na ordem definida."""

    print("\n" + "=" * 80)
    print("PIPELINE PDC-IA")
    print("=" * 80)

    resultados = {}

    for nome, caminho_modulo in ETAPAS:
        print("\n" + "=" * 80)
        print(f"ETAPA: {nome}")
        print("=" * 80)

        try:
            modulo = import_module(caminho_modulo)
            funcao_executar = getattr(modulo, "executar", None)

            if not callable(funcao_executar):
                raise AttributeError(
                    f"O modulo {caminho_modulo} nao possui uma funcao executar()."
                )

            resultados[nome.lower()] = funcao_executar()

        except Exception as erro:
            print(f"\nERRO NA ETAPA {nome}: {erro}")
            print("Pipeline interrompido.")
            raise

    print("\n" + "=" * 80)
    print("PIPELINE FINALIZADO")
    print("=" * 80)

    return resultados


if __name__ == "__main__":
    executar()
