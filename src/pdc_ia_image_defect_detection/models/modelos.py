"""
modelos.py

Objetivo:
- Centralizar os modelos disponiveis no projeto.
- Criar a arquitetura escolhida a partir de um identificador textual.
- Evitar imports especificos de arquiteturas dentro de treino e teste.

Regras:
- Cada modelo registrado deve herdar de torch.nn.Module.
- Cada modelo deve aceitar num_classes no construtor.
- A entrada esperada pelo pipeline atual e (N,1,H,W); a VGG16 ImageNet exige 224x224.
- A saida esperada e (N,num_classes), contendo logits.
- O modulo do modelo e importado somente quando aquele modelo e solicitado.
"""

from importlib import import_module
import torch.nn as nn


# =============================================================================
# MODELOS DISPONIVEIS
# =============================================================================

# Cada entrada associa o identificador usado nos parametros/checkpoints ao
# modulo e a classe que implementam a arquitetura.
MODELOS = {
    "lenet5": (
        "pdc_ia_image_defect_detection.models.lenet5",
        "LeNet5",
    ),
    "vgg16": (
        "pdc_ia_image_defect_detection.models.vgg16",
        "VGG16",
    ),
    "vgg16_imagenet": (
        "pdc_ia_image_defect_detection.models.vgg16_imagenet",
        "VGG16ImageNet",
    ),
}


# =============================================================================
# FUNCOES PUBLICAS
# =============================================================================

def normalizar_nome_modelo(nome):
    """
    Normaliza o identificador textual de um modelo.

    Parameters
    ----------
    nome : str
        Nome informado no processing, checkpoint ou outra etapa.

    Returns
    -------
    str
        Nome em minusculas e sem espacos laterais.
    """

    nome = str(nome).strip().lower()

    if not nome:
        raise ValueError("O nome do modelo nao pode ser vazio.")

    return nome


def modelos_disponiveis():
    """Retorna os identificadores dos modelos registrados."""

    return tuple(sorted(MODELOS))


def obter_classe_modelo(nome):
    """
    Localiza e importa a classe correspondente ao modelo solicitado.

    Parameters
    ----------
    nome : str
        Identificador registrado em MODELOS.

    Returns
    -------
    type[torch.nn.Module]
        Classe da arquitetura solicitada.

    Raises
    ------
    ValueError
        Quando o identificador nao esta registrado.

    ImportError / AttributeError
        Quando o modulo ou a classe registrada nao podem ser carregados.

    TypeError
        Quando a classe registrada nao herda de torch.nn.Module.
    """

    nome = normalizar_nome_modelo(nome)

    if nome not in MODELOS:
        disponiveis = ", ".join(modelos_disponiveis())
        raise ValueError(
            f"Modelo desconhecido: '{nome}'. "
            f"Modelos disponiveis: {disponiveis}."
        )

    modulo_nome, classe_nome = MODELOS[nome]
    modulo = import_module(modulo_nome)
    classe = getattr(modulo, classe_nome)

    if not isinstance(classe, type) or not issubclass(classe, nn.Module):
        raise TypeError(
            f"A classe registrada para '{nome}' nao herda de torch.nn.Module."
        )

    return classe


def criar_modelo(nome, num_classes, **parametros_modelo):
    """
    Instancia um modelo registrado com a quantidade informada de classes.

    Parameters
    ----------
    nome : str
        Identificador do modelo, por exemplo "lenet5" ou "vgg16".

    num_classes : int
        Quantidade de classes de saida.
    **parametros_modelo
        Configuracao opcional especifica da arquitetura selecionada.

    Returns
    -------
    torch.nn.Module
        Modelo instanciado, ainda sem ser movido para CPU/GPU.
    """

    num_classes = int(num_classes)

    if num_classes <= 0:
        raise ValueError("num_classes deve ser maior que zero.")

    classe = obter_classe_modelo(nome)
    return classe(num_classes=num_classes, **parametros_modelo)
