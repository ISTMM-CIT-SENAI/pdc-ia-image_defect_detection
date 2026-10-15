"""
augmention.py

Objetivo:
- Centraliza a configuracao de data augmentation por classe.
- Define quais classes geram augmentation, quantas copias gerar e quais
  transformacoes usar em cada classe.
- Define separadamente quais augmentations poderao ser usados no treinamento.
- Mantem a normalizacao opcional por media/desvio padrao.

Usado por:
- augmentation_train.py: gera fisicamente os arquivos augmentados.
- images_dataset.py: posteriormente filtrara augmentations por classe no treino.
- main.py: posteriormente registrara a configuracao usada no treinamento.

Parametros ajustaveis:
- CONFIG_AUGMENTATION: geracao e transformacoes por classe.
- USAR_AUGMENTATION_POR_CLASSE: uso dos arquivos augmentados no treinamento.
- ROTATION_DEGREES, BRIGHTNESS, CONTRAST, NOISE_STD e parametros de blur.
- APLICAR_NORMALIZACAO, MEAN e STD.

Observacoes:
- Os arquivos .npy ja chegam em float32, formato (1,H,W) e intervalo [0,1].
- Augmentation e gerado fisicamente antes do treinamento.
- Nenhum augmentation aleatorio e aplicado em runtime pelo DataLoader.
- Validacao e teste nunca usam augmentation.
"""

import random

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF


# =============================================================================
# PARAMETROS AJUSTAVEIS
# =============================================================================

# -----------------------------------------------------------------------------
# GERACAO DE AUGMENTATION POR CLASSE
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# USO DOS AUGMENTATIONS NO TREINAMENTO
#
# Esta configuracao NAO controla a geracao.
# Ela sera utilizada pelo images_dataset.py para decidir quais arquivos
# *_aug*.npy entram no treinamento.
# -----------------------------------------------------------------------------

USAR_AUGMENTATION_POR_CLASSE = {
    "amassamento": False,
    "corrosao": True,
    "risco": False,
    "sem_falha": False,
}


# -----------------------------------------------------------------------------
# PARAMETROS DAS TRANSFORMACOES
# -----------------------------------------------------------------------------

# Flip horizontal
PROB_HORIZONTAL_FLIP = 0.50

# Rotacao aleatoria no intervalo [-ROTATION_DEGREES, +ROTATION_DEGREES]
ROTATION_DEGREES = 15.0

# Brilho: fator sorteado entre 1-BRIGHTNESS e 1+BRIGHTNESS
BRIGHTNESS = 0.15

# Contraste: fator sorteado entre 1-CONTRAST e 1+CONTRAST
CONTRAST = 0.15

# Ruido gaussiano aplicado ao tensor normalizado em [0,1]
NOISE_STD = 0.02

# Gaussian blur
BLUR_KERNEL_SIZE = 5
BLUR_SIGMA_MIN = 0.10
BLUR_SIGMA_MAX = 1.50


# -----------------------------------------------------------------------------
# NORMALIZACAO OPCIONAL DURANTE O CARREGAMENTO
#
# A normalizacao principal 0-255 -> 0-1 ja ocorre no pre-processamento.
# -----------------------------------------------------------------------------

APLICAR_NORMALIZACAO = False

MEAN = [0.5]
STD = [0.5]


# =============================================================================
# VALIDACAO DA CONFIGURACAO
# =============================================================================

CLASSES_ESPERADAS = {
    "amassamento",
    "corrosao",
    "risco",
    "sem_falha",
}

AUGMENTATIONS_DISPONIVEIS = (
    "flip_horizontal",
    "rotacao",
    "brilho",
    "contraste",
    "ruido",
    "blur",
)


def validar_configuracao():
    classes_config = set(CONFIG_AUGMENTATION)
    classes_treino = set(USAR_AUGMENTATION_POR_CLASSE)

    if classes_config != CLASSES_ESPERADAS:
        raise ValueError(
            "Classes invalidas em CONFIG_AUGMENTATION. "
            f"Esperado={sorted(CLASSES_ESPERADAS)}, encontrado={sorted(classes_config)}"
        )

    if classes_treino != CLASSES_ESPERADAS:
        raise ValueError(
            "Classes invalidas em USAR_AUGMENTATION_POR_CLASSE. "
            f"Esperado={sorted(CLASSES_ESPERADAS)}, encontrado={sorted(classes_treino)}"
        )

    for classe, config in CONFIG_AUGMENTATION.items():
        if not isinstance(config["gerar"], bool):
            raise TypeError(f"{classe}: 'gerar' deve ser True ou False.")

        if not isinstance(config["quantidade"], int) or config["quantidade"] < 0:
            raise ValueError(f"{classe}: 'quantidade' deve ser inteiro >= 0.")

        if config["gerar"] and config["quantidade"] == 0:
            raise ValueError(
                f"{classe}: gerar=True, mas quantidade=0."
            )

        for nome in AUGMENTATIONS_DISPONIVEIS:
            if not isinstance(config[nome], bool):
                raise TypeError(
                    f"{classe}: '{nome}' deve ser True ou False."
                )

        if config["gerar"] and not any(
            config[nome]
            for nome in AUGMENTATIONS_DISPONIVEIS
        ):
            raise ValueError(
                f"{classe}: gerar=True, mas nenhuma transformacao esta ativada."
            )

    if not 0.0 <= PROB_HORIZONTAL_FLIP <= 1.0:
        raise ValueError("PROB_HORIZONTAL_FLIP deve estar entre 0 e 1.")

    if ROTATION_DEGREES < 0:
        raise ValueError("ROTATION_DEGREES deve ser >= 0.")

    if BRIGHTNESS < 0 or CONTRAST < 0 or NOISE_STD < 0:
        raise ValueError(
            "BRIGHTNESS, CONTRAST e NOISE_STD devem ser >= 0."
        )

    if BLUR_KERNEL_SIZE <= 0 or BLUR_KERNEL_SIZE % 2 == 0:
        raise ValueError(
            "BLUR_KERNEL_SIZE deve ser inteiro positivo e impar."
        )

    if not 0 < BLUR_SIGMA_MIN <= BLUR_SIGMA_MAX:
        raise ValueError(
            "BLUR_SIGMA_MIN/MAX invalidos."
        )


# =============================================================================
# ACESSO A CONFIGURACAO
# =============================================================================

def obter_configuracao_classe(classe):
    if classe not in CONFIG_AUGMENTATION:
        raise KeyError(
            f"Classe sem configuracao de augmentation: {classe}"
        )

    return CONFIG_AUGMENTATION[classe]


def deve_gerar_augmentation(classe):
    config = obter_configuracao_classe(classe)

    return (
        config["gerar"]
        and config["quantidade"] > 0
    )


def quantidade_augmentation(classe):
    config = obter_configuracao_classe(classe)

    if not config["gerar"]:
        return 0

    return config["quantidade"]


def usar_augmentation_no_treino(classe):
    if classe not in USAR_AUGMENTATION_POR_CLASSE:
        raise KeyError(
            f"Classe sem configuracao de uso no treino: {classe}"
        )

    return USAR_AUGMENTATION_POR_CLASSE[classe]


# =============================================================================
# AUGMENTATION
# =============================================================================

def aplicar_augmentation_classe(tensor, classe):
    """
    Aplica somente as transformacoes habilitadas para a classe.

    Entrada:
        tensor: torch.Tensor float32 com shape (1,H,W), normalmente em [0,1].

    Saida:
        novo tensor float32 com mesmo shape e valores limitados a [0,1].
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            "A entrada do augmentation deve ser torch.Tensor."
        )

    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError(
            f"Formato esperado (1,H,W), recebido {tuple(tensor.shape)}."
        )

    config = obter_configuracao_classe(classe)

    if not config["gerar"]:
        return tensor.clone()

    saida = tensor.clone().float()

    # Flip horizontal
    if (
        config["flip_horizontal"]
        and random.random() < PROB_HORIZONTAL_FLIP
    ):
        saida = TF.hflip(saida)

    # Rotacao
    if config["rotacao"]:
        angulo = random.uniform(
            -ROTATION_DEGREES,
            ROTATION_DEGREES
        )

        saida = TF.rotate(
            saida,
            angle=angulo,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0.0
        )

    # Brilho
    if config["brilho"]:
        fator = random.uniform(
            max(0.0, 1.0 - BRIGHTNESS),
            1.0 + BRIGHTNESS
        )

        saida = TF.adjust_brightness(
            saida,
            fator
        )

    # Contraste
    if config["contraste"]:
        fator = random.uniform(
            max(0.0, 1.0 - CONTRAST),
            1.0 + CONTRAST
        )

        saida = TF.adjust_contrast(
            saida,
            fator
        )

    # Ruido gaussiano
    if config["ruido"]:
        ruido = torch.randn_like(saida) * NOISE_STD
        saida = saida + ruido

    # Gaussian blur
    if config["blur"]:
        sigma = random.uniform(
            BLUR_SIGMA_MIN,
            BLUR_SIGMA_MAX
        )

        saida = TF.gaussian_blur(
            saida,
            kernel_size=[BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE],
            sigma=[sigma, sigma]
        )

    return saida.clamp(0.0, 1.0).contiguous()


# =============================================================================
# TRANSFORMACOES BASICAS DO DATALOADER
# =============================================================================

def transformacoes_basicas():
    """
    Transformacoes aplicadas ao tensor carregado do .npy.

    O arquivo ja esta:
    - grayscale;
    - redimensionado;
    - float32;
    - em [0,1].

    Portanto, apenas a normalizacao adicional e opcional.
    """

    transformacoes = []

    if APLICAR_NORMALIZACAO:
        transformacoes.append(
            transforms.Normalize(
                mean=MEAN,
                std=STD
            )
        )

    return transforms.Compose(
        transformacoes
    )


def transformacoes_augmentation():
    """
    Mantido por compatibilidade com a estrutura antiga.

    Augmentation nao e mais aplicado aleatoriamente em runtime.
    A geracao ocorre fisicamente antes do treinamento por
    augmentation_train.py.
    """

    return transforms.Compose([])


def transformar_imagens(treino=False):
    """
    Retorna as transformacoes usadas pelo Dataset.

    treino=True nao aplica augmentation em runtime.
    Os arquivos augmentados ja estarao salvos no conjunto de treino.
    """

    return transformacoes_basicas()


# Valida imediatamente ao importar o modulo
validar_configuracao()