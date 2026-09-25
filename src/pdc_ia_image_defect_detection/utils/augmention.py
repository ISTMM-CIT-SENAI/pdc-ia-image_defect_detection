from torchvision import transforms


# Ativacao das transformacoes
APLICAR_HORIZONTAL_FLIP = False
APLICAR_ROTATION = False
APLICAR_COLOR_JITTER = False
APLICAR_NORMALIZACAO = False


# Parametros do data augmentation
HORIZONTAL_FLIP = 0.5
ROTATION_DEGREES = 10
BRIGHTNESS = 0.2
CONTRAST = 0.2


# Parametros da normalizacao
MEAN = [0.5]
STD = [0.5]


def transformacoes_basicas():
    """Define as transformacoes aplicadas em treino, validacao e teste"""

    transformacoes = []

    if APLICAR_NORMALIZACAO:
        transformacoes.append(transforms.Normalize(mean=MEAN, std=STD))

    return transformacoes


def transformacoes_augmentation():
    """Define apenas as transformacoes de data augmentation"""

    transformacoes = []

    if APLICAR_HORIZONTAL_FLIP:
        transformacoes.append(transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP))

    if APLICAR_ROTATION:
        transformacoes.append(transforms.RandomRotation(degrees=ROTATION_DEGREES))

    if APLICAR_COLOR_JITTER:
        transformacoes.append(transforms.ColorJitter(brightness=BRIGHTNESS, contrast=CONTRAST))

    return transformacoes


def transformar_imagens():
    """Monta as transformacoes utilizadas pelos datasets"""

    transform_treino = transforms.Compose(transformacoes_augmentation() + transformacoes_basicas())
    transform_validacao = transforms.Compose(transformacoes_basicas())

    return transform_treino, transform_validacao