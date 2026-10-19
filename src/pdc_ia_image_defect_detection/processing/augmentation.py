"""
augmentation.py

Objetivo:
- Configurar e iniciar o data augmentation dos conjuntos train selecionados.
- Permitir ativar independentemente a base interna e a base externa.
- Permitir configurar augmentation separadamente por classe.
- Manter neste executavel apenas parametros e a chamada do processamento.

Base interna:
- dataset_processado_split/train

Base externa:
- dataset_externo_processado_split/train

Configuracao:
- Todos os parametros sao alterados manualmente neste arquivo.
- O script nao pergunta configuracoes pelo terminal.
- O terminal e usado somente para mostrar o andamento e o resultado.

Observacoes:
- Somente train pode ser modificado.
- Val e test nunca recebem augmentation.
- A logica especifica de augmentation fica em augmentation_class.py.
- Funcoes genericas permanecem em utils.
- EXECUTAR_AUGMENTATION=False encerra a etapa antes dos imports internos.
- Se nenhuma base for selecionada, nada e modificado.
- Se nenhuma classe estiver com gerar=True, nada e modificado.
- A escrita de bytecode e desativada para evitar __pycache__ e arquivos .pyc.
"""

import sys

# Evita a criacao de arquivos .pyc e pastas __pycache__ nesta execucao.
# Esta linha precisa aparecer antes dos imports dos modulos do projeto.
sys.dont_write_bytecode = True

# =============================================================================
# PARAMETROS AJUSTAVEIS
# =============================================================================

# Ativa ou desativa completamente esta etapa do pipeline.
EXECUTAR_AUGMENTATION = False

# -----------------------------------------------------------------------------
# BASES QUE RECEBERAO AUGMENTATION
#
# True / False  -> somente base interna
# False / True  -> somente base externa
# True / True   -> ambas as bases
# False / False -> nenhuma base e modificada
# -----------------------------------------------------------------------------
USAR_DATASET_INTERNO = True
USAR_DATASET_EXTERNO = False

# Seed usada pelos sorteios das transformacoes aleatorias.
SEED_AUGMENTATION = 42

# Configuracao independente de cada classe.
# "gerar" decide se a classe recebera novos arquivos.
# "quantidade" define quantas copias serao criadas por imagem original.
CONFIG_AUGMENTATION = {
    "amassamento": {
        "gerar": False,
        "quantidade": 3,
        "flip_horizontal": True,
        "rotacao": True,
        "brilho": False,
        "contraste": False,
        "ruido": False,
        "blur": False,
    },
    "corrosao": {
        "gerar": True,
        "quantidade": 3,
        "flip_horizontal": False,
        "rotacao": False,
        "brilho": True,
        "contraste": True,
        "ruido": False,
        "blur": False,
    },
    "risco": {
        "gerar": False,
        "quantidade": 3,
        "flip_horizontal": True,
        "rotacao": True,
        "brilho": False,
        "contraste": False,
        "ruido": False,
        "blur": False,
    },
    "sem_falha": {
        "gerar": False,
        "quantidade": 3,
        "flip_horizontal": True,
        "rotacao": True,
        "brilho": False,
        "contraste": False,
        "ruido": False,
        "blur": False,
    },
}

# Intensidades utilizadas quando a transformacao correspondente estiver ativa.
PARAMETROS_AUGMENTATION = {
    # Probabilidade de aplicar flip horizontal em cada copia.
    "prob_flip_horizontal": 0.50,

    # Rotacao aleatoria em [-rotacao_graus, +rotacao_graus].
    "rotacao_graus": 15.0,

    # Variacao de brilho em torno de 1.0.
    "brilho": 0.15,

    # Variacao de contraste em torno de 1.0.
    "contraste": 0.15,

    # Desvio padrao do ruido gaussiano para imagens no intervalo [0,1].
    "ruido_std": 0.02,

    # Kernel do Gaussian Blur; deve ser positivo e impar.
    "blur_kernel_size": 5,

    # Intervalo de sigma utilizado no Gaussian Blur.
    "blur_sigma_min": 0.10,
    "blur_sigma_max": 1.50,
}


# =============================================================================
# EXECUCAO
# =============================================================================

def executar():
    """Executa o augmentation somente quando a etapa estiver ativada."""

    # Encerra antes de importar PyTorch/torchvision pela classe de augmentation.
    if not EXECUTAR_AUGMENTATION:
        print("\nAUGMENTATION desativado: EXECUTAR_AUGMENTATION=False.")
        return {}

    # Encerra sem acessar os datasets se nenhuma base estiver selecionada.
    if not USAR_DATASET_INTERNO and not USAR_DATASET_EXTERNO:
        print("\nAUGMENTATION ignorado: nenhuma base foi selecionada.")
        return {}

    # Encerra antes de qualquer limpeza se nenhuma classe solicitar augmentation.
    if not any(config.get("gerar", False) for config in CONFIG_AUGMENTATION.values()):
        print("\nAUGMENTATION ignorado: nenhuma classe esta com gerar=True.")
        return {}

    # Imports internos ocorrem somente quando esta etapa realmente sera executada.
    from pdc_ia_image_defect_detection.pre_processing.augmentation_class import (
        AugmentationProcessor,
    )
    from pdc_ia_image_defect_detection.utils.paths import encontrar_raiz_projeto

    # Localiza a raiz do projeto depois das verificacoes de encerramento rapido.
    base_dir = encontrar_raiz_projeto(__file__)

    # Informa quais conjuntos train participam desta execucao.
    bases_augmentation = {
        "INTERNA": base_dir / "dataset_processado_split" / "train"
        if USAR_DATASET_INTERNO else None,
        "EXTERNA": base_dir / "dataset_externo_processado_split" / "train"
        if USAR_DATASET_EXTERNO else None,
    }

    return AugmentationProcessor.executar_bases(
        bases=bases_augmentation,
        config_classes=CONFIG_AUGMENTATION,
        parametros=PARAMETROS_AUGMENTATION,
        seed=SEED_AUGMENTATION,
    )


# Permite testar este processing isoladamente durante o desenvolvimento.
if __name__ == "__main__":
    executar()
