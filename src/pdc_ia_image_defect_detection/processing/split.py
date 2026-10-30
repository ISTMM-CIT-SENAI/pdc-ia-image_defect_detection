"""
split.py

Objetivo:
- Configurar e executar a divisao dos datasets processados em train, val e test.
- Permitir dividir independentemente a base interna e a base externa.
- Aplicar as mesmas proporcoes separadamente em cada classe.
- Manter neste arquivo apenas parametros, caminhos e chamada do processamento.

Bases de entrada:
- Interna: dataset_processado
- Externa: dataset_externo_processado

Bases de saida:
- Interna: dataset_processado_split
- Externa: dataset_externo_processado_split

Funcionamento:
- Cada classe e dividida separadamente para preservar sua distribuicao.
- A quantidade de train e val e calculada pelas proporcoes configuradas.
- Test recebe todas as imagens restantes da classe.
- A selecao das imagens e aleatoria e reproduzivel pela seed configurada.
- As bases interna e externa sao divididas de forma independente.
- Classes presentes na base interna sao usadas como referencia do projeto.
- A base externa pode nao possuir imagens de alguma classe interna; nesse caso
  as pastas correspondentes sao criadas vazias no split externo.
- Uma classe externa que nao exista na referencia interna gera erro.
- Todos os arquivos das bases selecionadas sao validados antes da remocao de
  qualquer split antigo.
- Um split antigo da base selecionada e removido antes da nova divisao.
- Os datasets processados de entrada nunca sao modificados.
- EXECUTAR_SPLIT=False encerra a etapa antes dos imports internos.
- Se as duas bases estiverem desativadas, nenhum arquivo e alterado.

Configuracao:
- Todos os parametros sao definidos manualmente neste arquivo.
- O terminal e utilizado somente para exibir andamento e resultados.
- A logica do split fica em pre_processing/split_class.py.
- Funcoes genericas permanecem em utils.

Observacao:
- A escrita de bytecode e desativada para evitar arquivos .pyc e
  pastas __pycache__.
"""

import sys

# Desativa a escrita de bytecode Python durante a execucao.
sys.dont_write_bytecode = True

# =============================================================================
# PARAMETROS AJUSTAVEIS
# =============================================================================

# Ativa ou desativa completamente esta etapa do pipeline.
EXECUTAR_SPLIT = False

# Ativa a divisao da base interna.
DIVIDIR_DATASET_INTERNO = True

# Ativa a divisao da base externa.
DIVIDIR_DATASET_EXTERNO = True

# Proporcao destinada ao conjunto de treinamento.
PROPORCAO_TREINO = 0.60

# Proporcao destinada ao conjunto de validacao.
PROPORCAO_VALIDACAO = 0.20

# Proporcao destinada ao conjunto de teste.
PROPORCAO_TESTE = 0.20

# Define a seed usada no embaralhamento das imagens.
SEED_SPLIT = 42


# =============================================================================
# EXECUCAO
# =============================================================================

def executar():
    """Executa o split somente quando a etapa estiver ativada."""

    # Encerra antes de importar a classe especifica ou acessar os datasets.
    if not EXECUTAR_SPLIT:
        print("\nSPLIT desativado: EXECUTAR_SPLIT=False.")
        return {}

    # Imports internos ocorrem somente quando esta etapa realmente sera executada.
    from pdc_ia_image_defect_detection.pre_processing.split_class import SplitProcessor
    from pdc_ia_image_defect_detection.utils.paths import encontrar_raiz_projeto

    # Localiza a raiz do projeto somente apos a verificacao da flag geral.
    base_dir = encontrar_raiz_projeto(__file__)

    # Define entradas processadas e saidas split das duas bases.
    dataset_interno_dir = base_dir / "dataset_processado"
    dataset_interno_split_dir = base_dir / "dataset_processado_split"
    dataset_externo_dir = base_dir / "dataset_externo_processado"
    dataset_externo_split_dir = base_dir / "dataset_externo_processado_split"

    # Associa cada base aos caminhos utilizados pelo processamento.
    bases_split = {
        "INTERNA": {
            "usar": DIVIDIR_DATASET_INTERNO,
            "origem": dataset_interno_dir,
            "destino": dataset_interno_split_dir,
        },
        "EXTERNA": {
            "usar": DIVIDIR_DATASET_EXTERNO,
            "origem": dataset_externo_dir,
            "destino": dataset_externo_split_dir,
        },
    }

    # Reune as proporcoes aplicadas separadamente em cada classe.
    proporcoes_split = {
        "train": PROPORCAO_TREINO,
        "val": PROPORCAO_VALIDACAO,
        "test": PROPORCAO_TESTE,
    }

    return SplitProcessor.executar_bases(
        bases=bases_split,
        proporcoes=proporcoes_split,
        seed=SEED_SPLIT,
        referencia_classes=dataset_interno_dir,
    )


# Executa o processo quando este arquivo for iniciado diretamente.
if __name__ == "__main__":
    executar()
