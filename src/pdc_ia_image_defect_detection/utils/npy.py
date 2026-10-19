"""
npy.py

Objetivo:
- Reunir funcoes genericas para leitura, validacao e escrita de imagens
  armazenadas em arquivos NPY.
- Padronizar o formato numerico usado nas diferentes etapas do projeto.
- Evitar duplicacao das mesmas verificacoes em augmentation, treinamento,
  predicao e demais processos.

Formato esperado das imagens:
- tipo: numpy.ndarray
- dtype: float32
- formato: (1, H, W)
- canal: 1 canal grayscale
- valores: finitos
- intervalo: [0, 1]

Funcoes disponiveis:
- validar_imagem_npy:
    valida um array ja carregado em memoria.

- carregar_imagem_npy:
    carrega um arquivo .npy e valida seu conteudo.

- salvar_imagem_npy:
    valida um array e salva o arquivo .npy.

Observacoes:
- Este modulo nao conhece classes, datasets, augmentation ou modelos.
- As funcoes podem ser reutilizadas por qualquer processo que trabalhe
  com o formato NPY padrao do projeto.
- Nenhuma normalizacao ou conversao automatica e feita neste modulo.
- Dados invalidos geram erro em vez de serem corrigidos silenciosamente.
"""

from pathlib import Path

import numpy as np


# =============================================================================
# VALIDACAO
# =============================================================================

def validar_imagem_npy(array, origem=None):
    """
    Valida um array conforme o formato padrao de imagem NPY do projeto.

    Parameters
    ----------
    array : numpy.ndarray
        Array da imagem que sera validado.

    origem : str ou Path ou None
        Identificacao opcional usada apenas nas mensagens de erro.
        Pode ser o caminho do arquivo que originou o array.

    Validacoes realizadas
    ---------------------
    - o objeto precisa ser numpy.ndarray;
    - dtype precisa ser float32;
    - shape precisa seguir (1, H, W);
    - as dimensoes espaciais precisam ser maiores que zero;
    - o array nao pode conter NaN ou Inf;
    - todos os valores precisam pertencer ao intervalo [0, 1].

    Returns
    -------
    None
        A funcao nao altera nem retorna o array.
        A ausencia de excecao indica que o dado e valido.

    Raises
    ------
    TypeError
        Quando o objeto ou dtype sao invalidos.

    ValueError
        Quando shape, dimensoes, finitude ou intervalo sao invalidos.
    """

    # Cria uma identificacao curta para mensagens de erro.
    referencia = (
        Path(origem).name
        if origem is not None
        else "array"
    )

    # Garante que o objeto recebido seja realmente um array NumPy.
    if not isinstance(array, np.ndarray):
        raise TypeError(
            f"{referencia}: esperado numpy.ndarray, "
            f"recebido {type(array).__name__}."
        )

    # Todo o pipeline numerico do projeto utiliza float32.
    if array.dtype != np.float32:
        raise TypeError(
            f"{referencia}: dtype={array.dtype}. "
            "Esperado: float32."
        )

    # As imagens seguem a convencao de canais primeiro: (C,H,W).
    # Como as imagens sao grayscale, C precisa ser igual a 1.
    if array.ndim != 3 or array.shape[0] != 1:
        raise ValueError(
            f"{referencia}: shape={array.shape}. "
            "Esperado: (1,H,W)."
        )

    # Altura e largura precisam representar uma imagem espacial valida.
    if array.shape[1] <= 0 or array.shape[2] <= 0:
        raise ValueError(
            f"{referencia}: dimensoes espaciais invalidas "
            f"{array.shape[1]}x{array.shape[2]}."
        )

    # Arrays sem elementos nao representam uma imagem valida.
    if array.size == 0:
        raise ValueError(
            f"{referencia}: array vazio."
        )

    # NaN e infinito tornam calculos numericos posteriores invalidos.
    if not np.isfinite(array).all():
        raise ValueError(
            f"{referencia}: possui valores NaN ou Inf."
        )

    # Calcula o intervalo numerico efetivamente presente na imagem.
    minimo = float(
        array.min()
    )
    maximo = float(
        array.max()
    )

    # O pre-processamento do projeto normaliza pixels para o intervalo [0,1].
    if minimo < 0.0 or maximo > 1.0:
        raise ValueError(
            f"{referencia}: valores fora de [0,1]. "
            f"min={minimo:.6f}, max={maximo:.6f}."
        )


# =============================================================================
# LEITURA
# =============================================================================

def carregar_imagem_npy(caminho):
    """
    Carrega e valida uma imagem armazenada em arquivo NPY.

    Parameters
    ----------
    caminho : str ou Path
        Caminho do arquivo .npy que sera carregado.

    Returns
    -------
    numpy.ndarray
        Array validado no formato padrao do projeto.

    Raises
    ------
    FileNotFoundError
        Quando o arquivo nao existe.

    IsADirectoryError
        Quando o caminho recebido aponta para uma pasta.

    ValueError ou TypeError
        Quando o conteudo do arquivo nao atende ao formato esperado.
    """

    # Padroniza o caminho recebido como objeto Path.
    caminho = Path(
        caminho
    )

    # Verifica se o caminho realmente existe.
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo NPY nao encontrado:\n{caminho}"
        )

    # Impede o uso de uma pasta como se fosse arquivo.
    if caminho.is_dir():
        raise IsADirectoryError(
            f"O caminho informado e um diretorio:\n{caminho}"
        )

    # Restringe esta funcao a arquivos com extensao .npy.
    if caminho.suffix.lower() != ".npy":
        raise ValueError(
            f"Extensao invalida para imagem NPY: {caminho.suffix}"
        )

    # Carrega somente arrays numericos.
    # allow_pickle=False impede desserializacao de objetos Python.
    array = np.load(
        caminho,
        allow_pickle=False,
    )

    # Valida o conteudo antes de disponibiliza-lo para outras etapas.
    validar_imagem_npy(
        array,
        origem=caminho,
    )

    return array


# =============================================================================
# ESCRITA
# =============================================================================

def salvar_imagem_npy(caminho, array):
    """
    Valida e salva uma imagem no formato NPY padrao do projeto.

    Parameters
    ----------
    caminho : str ou Path
        Caminho completo do arquivo de destino.

    array : numpy.ndarray
        Imagem que sera validada e salva.

    Returns
    -------
    Path
        Caminho final do arquivo salvo.

    Raises
    ------
    TypeError ou ValueError
        Quando o array nao atende ao formato esperado.
    """

    # Padroniza o caminho recebido como objeto Path.
    caminho = Path(
        caminho
    )

    # Exige extensao .npy para manter a nomenclatura previsivel.
    if caminho.suffix.lower() != ".npy":
        raise ValueError(
            f"O arquivo de destino deve possuir extensao .npy:\n{caminho}"
        )

    # Valida o array antes de criar ou sobrescrever qualquer arquivo.
    validar_imagem_npy(
        array,
        origem=caminho,
    )

    # Cria a pasta de destino quando ela ainda nao existir.
    caminho.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Salva exatamente o array validado.
    np.save(
        caminho,
        array,
    )

    return caminho