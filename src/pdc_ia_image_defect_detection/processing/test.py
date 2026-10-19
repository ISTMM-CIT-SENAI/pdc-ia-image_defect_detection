"""
test.py

Objetivo:
- Configurar e iniciar a avaliacao final de um modelo treinado.
- Permitir usar o conjunto test criado pelo split ou uma pasta de imagens
  escolhida manualmente.
- Quando o test do split estiver ativo, permitir selecionar independentemente
  a base interna e a base externa.
- Quando uma pasta manual for usada, preparar as imagens em memoria com o mesmo
  formato numerico esperado pelo modelo treinado.
- Obter o rotulo real exclusivamente pela pasta de classe.
- Selecionar uma pasta de treinamento e escolher entre modelo_melhor.pt e
  modelo_final.pt.
- Manter neste executavel apenas parametros, caminhos e a chamada da classe.

Modos de teste:
1. USAR_TEST_SPLIT=True
   - usa dataset_processado_split/test e/ou
     dataset_externo_processado_split/test;
   - os NPY do split ja estao preparados e nao recebem novo processamento;
   - USAR_DATASET_INTERNO e USAR_DATASET_EXTERNO controlam quais bases entram.

2. USAR_TEST_SPLIT=False
   - ignora as flags de base interna/externa;
   - usa apenas PASTA_TESTE_ESCOLHIDA;
   - a pasta escolhida deve possuir uma subpasta por classe;
   - aceita imagens nos formatos definidos por EXTENSOES_IMAGEM;
   - corrige orientacao EXIF;
   - verifica dimensoes e modo;
   - aplica crop quadrado central quando a imagem for retangular;
   - converte para grayscale;
   - redimensiona para a resolucao registrada no checkpoint;
   - converte para float32, normaliza para [0,1] e adiciona o canal.

Regras:
- Test nunca altera pesos do modelo.
- Nenhum augmentation e aplicado durante o teste.
- O rotulo real e determinado pela pasta de classe.
- Se EXECUTAR_TEST=False, a etapa termina antes de importar PyTorch, o modelo
  ou a classe de avaliacao.
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True


# =============================================================================
# PARAMETROS AJUSTAVEIS
# =============================================================================

# Ativa ou desativa completamente a etapa de teste.
EXECUTAR_TEST = True

# True usa os conjuntos test gerados pelo split.
# False usa somente PASTA_TESTE_ESCOLHIDA.
USAR_TEST_SPLIT = True

# Usados somente quando USAR_TEST_SPLIT=True.
USAR_DATASET_INTERNO = True
USAR_DATASET_EXTERNO = True

# Pasta usada somente quando USAR_TEST_SPLIT=False.
# Deve conter uma subpasta por classe com as imagens de teste.
# Pode ser caminho absoluto ou relativo a raiz do projeto.
PASTA_TESTE_ESCOLHIDA = Path("dataset_teste_escolhido")

# Formatos aceitos na pasta escolhida manualmente.
EXTENSOES_IMAGEM = (".png", ".jpg", ".jpeg", ".bmp")

# Pasta de um treinamento concluido.
# Deve conter modelo_melhor.pt e/ou modelo_final.pt.
# Pode ser caminho absoluto ou relativo a raiz do projeto.
PASTA_TREINO = (
    Path("resultados")
    / "resultados_treino"
    / "treino_lenet5_inicio_2026-10-19_11-17-37"
)

# True usa modelo_melhor.pt. False usa modelo_final.pt.
USAR_MODELO_MELHOR = True

# Parametros de avaliacao.
BATCH_SIZE = 32
NUM_WORKERS = 0
USAR_CUDA = True
SEED_TEST = 42

# Quantidade de exemplos salva em amostras_predicoes.png.
NUM_AMOSTRAS_GRAFICO = 12

# Quantidade de colunas no grafico de exemplos.
COLUNAS_AMOSTRAS = 4


def executar():
    """Executa o teste somente quando a etapa estiver ativada."""

    # A flag geral e verificada antes dos imports pesados do processo.
    if not EXECUTAR_TEST:
        print("\nTEST desativado: EXECUTAR_TEST=False.")
        return {}

    # Imports internos e dependentes de PyTorch ocorrem apenas quando necessario.
    from pdc_ia_image_defect_detection.pre_processing.test_class import TestProcessor
    from pdc_ia_image_defect_detection.utils.paths import encontrar_raiz_projeto

    base_dir = encontrar_raiz_projeto(__file__)

    # Resolve a pasta do treinamento em relacao a raiz do projeto.
    pasta_treino = PASTA_TREINO
    if not pasta_treino.is_absolute():
        pasta_treino = base_dir / pasta_treino

    # Resolve a pasta manual de teste em relacao a raiz do projeto.
    pasta_escolhida = PASTA_TESTE_ESCOLHIDA
    if not pasta_escolhida.is_absolute():
        pasta_escolhida = base_dir / pasta_escolhida

    # Define os dois conjuntos test gerados pelo split.
    bases_split = {
        "INTERNA": {
            "ativa": USAR_DATASET_INTERNO,
            "raiz_split": base_dir / "dataset_processado_split",
            "test": base_dir / "dataset_processado_split" / "test",
        },
        "EXTERNA": {
            "ativa": USAR_DATASET_EXTERNO,
            "raiz_split": base_dir / "dataset_externo_processado_split",
            "test": base_dir / "dataset_externo_processado_split" / "test",
        },
    }

    return TestProcessor(
        usar_test_split=USAR_TEST_SPLIT,
        bases_split=bases_split,
        pasta_teste_escolhida=pasta_escolhida,
        extensoes_imagem=EXTENSOES_IMAGEM,
        pasta_treino=pasta_treino,
        usar_modelo_melhor=USAR_MODELO_MELHOR,
        pasta_resultados=base_dir / "resultados" / "resultados_test",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        usar_cuda=USAR_CUDA,
        seed=SEED_TEST,
        num_amostras_grafico=NUM_AMOSTRAS_GRAFICO,
        colunas_amostras=COLUNAS_AMOSTRAS,
    ).executar()


if __name__ == "__main__":
    executar()
