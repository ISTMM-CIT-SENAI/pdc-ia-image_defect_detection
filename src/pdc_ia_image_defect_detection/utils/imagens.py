"""
imagens.py

Objetivo:
- Reunir funcoes genericas para leitura, listagem e preparacao basica de imagens.
- Evitar repeticao das mesmas operacoes em preprocessamento, predicao e outros
  processos que trabalhem diretamente com arquivos de imagem.

Funcoes disponiveis:
- listar_imagens:
    lista recursivamente arquivos com extensoes permitidas.

- carregar_imagem_pil:
    abre uma imagem, seleciona o primeiro frame quando necessario e pode corrigir
    a orientacao EXIF.

- crop_quadrado_central:
    recorta uma imagem retangular para o maior quadrado central possivel.

- converter_imagem_para_npy:
    converte uma imagem PIL para grayscale, redimensiona, normaliza para [0,1],
    adiciona o canal e retorna um numpy.ndarray float32 no formato (1,H,W).

Observacoes:
- Este modulo nao conhece classes, datasets, augmentation ou modelos.
- As funcoes recebem os parametros necessarios e nao dependem de variaveis globais.
- A validacao final do array utiliza utils.npy para manter um unico padrao numerico.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from pdc_ia_image_defect_detection.utils.npy import validar_imagem_npy


# =============================================================================
# LISTAGEM DE IMAGENS
# =============================================================================

def listar_imagens(root, extensoes):
    """
    Lista recursivamente os arquivos de imagem existentes em uma pasta.

    Parameters
    ----------
    root : str ou Path
        Pasta a partir da qual a busca sera realizada.

    extensoes : iterable[str]
        Extensoes aceitas, com ou sem ponto e em qualquer combinacao de maiusculas
        e minusculas.

    Returns
    -------
    list[Path]
        Caminhos encontrados em ordem alfabetica.

    Raises
    ------
    FileNotFoundError
        Quando a pasta informada nao existe.

    NotADirectoryError
        Quando o caminho informado nao e um diretorio.
    """

    # Padroniza a raiz como Path para manter o mesmo comportamento entre scripts.
    root = Path(root)

    # A listagem exige uma pasta existente.
    if not root.exists():
        raise FileNotFoundError(
            f"Pasta de imagens nao encontrada:\n{root}"
        )

    # Impede a busca recursiva em um arquivo isolado.
    if not root.is_dir():
        raise NotADirectoryError(
            f"O caminho informado nao e um diretorio:\n{root}"
        )

    # Normaliza as extensoes para o formato '.ext' em minusculo.
    extensoes_normalizadas = {
        extensao.lower() if str(extensao).startswith(".") else f".{str(extensao).lower()}"
        for extensao in extensoes
    }

    # Mantem apenas arquivos cuja extensao pertence ao conjunto permitido.
    return sorted(
        caminho
        for caminho in root.rglob("*")
        if caminho.is_file() and caminho.suffix.lower() in extensoes_normalizadas
    )


# =============================================================================
# LEITURA PIL
# =============================================================================

def carregar_imagem_pil(caminho, corrigir_exif=False):
    """
    Carrega uma imagem como objeto PIL independente do arquivo original.

    Parameters
    ----------
    caminho : str ou Path
        Arquivo de imagem que sera aberto.

    corrigir_exif : bool
        Quando True, aplica ImageOps.exif_transpose antes de devolver a imagem.

    Returns
    -------
    tuple[PIL.Image.Image, dict]
        Imagem carregada e metadados basicos da leitura.

        O dicionario de metadados contem:
        - formato;
        - modo_original;
        - tamanho_original;
        - tamanho_final_leitura;
        - frames;
        - exif_corrigido.

    Raises
    ------
    FileNotFoundError
        Quando o arquivo nao existe.

    IsADirectoryError
        Quando o caminho aponta para uma pasta.
    """

    # Padroniza o caminho recebido.
    caminho = Path(caminho)

    # Verifica a existencia antes de delegar a abertura ao Pillow.
    if not caminho.exists():
        raise FileNotFoundError(
            f"Imagem nao encontrada:\n{caminho}"
        )

    # Evita tentar abrir um diretorio como imagem.
    if caminho.is_dir():
        raise IsADirectoryError(
            f"O caminho informado e um diretorio:\n{caminho}"
        )

    # Mantem o arquivo aberto somente durante a leitura e copia o resultado para
    # memoria antes de sair do bloco with.
    with Image.open(caminho) as imagem:
        # Registra o formato informado pelo Pillow ou usa a extensao como fallback.
        formato = imagem.format or caminho.suffix.replace(".", "").upper()

        # Guarda o modo original antes de qualquer conversao posterior.
        modo_original = imagem.mode

        # Imagens animadas ou multipagina podem possuir mais de um frame.
        frames = int(getattr(imagem, "n_frames", 1))

        # O pipeline utiliza somente o primeiro frame.
        if frames > 1:
            imagem.seek(0)

        # Registra o tamanho antes de qualquer correcao de orientacao.
        tamanho_original = tuple(imagem.size)

        # Corrige a orientacao somente quando solicitado pelo processo chamador.
        if corrigir_exif:
            imagem = ImageOps.exif_transpose(imagem)

        # Carrega os pixels antes de fechar o arquivo de origem.
        imagem.load()

        # A copia torna o objeto retornado independente do arquivo aberto no with.
        imagem_saida = imagem.copy()

    # O tamanho apos EXIF pode mudar quando largura e altura sao trocadas.
    tamanho_final_leitura = tuple(imagem_saida.size)

    metadata = {
        "formato": formato,
        "modo_original": modo_original,
        "tamanho_original": tamanho_original,
        "tamanho_final_leitura": tamanho_final_leitura,
        "frames": frames,
        "exif_corrigido": bool(corrigir_exif),
    }

    return imagem_saida, metadata


# =============================================================================
# CROP QUADRADO CENTRAL
# =============================================================================

def crop_quadrado_central(imagem):
    """
    Recorta uma imagem para o maior quadrado central possivel.

    Parameters
    ----------
    imagem : PIL.Image.Image
        Imagem que sera analisada.

    Returns
    -------
    tuple[PIL.Image.Image, tuple[int, int, int, int], bool]
        Imagem resultante, caixa utilizada no crop e indicador de aplicacao.

        Quando a imagem ja e quadrada:
        - nenhuma regiao e removida;
        - a caixa corresponde a imagem completa;
        - o indicador retorna False.
    """

    # Le largura e altura diretamente do objeto PIL.
    largura, altura = imagem.size

    # Imagens quadradas nao precisam ser alteradas.
    if largura == altura:
        caixa = (0, 0, largura, altura)
        return imagem.copy(), caixa, False

    # A menor dimensao define o lado maximo do quadrado sem criar pixels novos.
    lado = min(largura, altura)

    # Calcula o deslocamento necessario para centralizar o quadrado na largura.
    esquerda = (largura - lado) // 2

    # Calcula o deslocamento necessario para centralizar o quadrado na altura.
    superior = (altura - lado) // 2

    # Define os limites direito e inferior da caixa.
    direita = esquerda + lado
    inferior = superior + lado

    # A caixa segue a convencao Pillow: esquerda, superior, direita, inferior.
    caixa = (esquerda, superior, direita, inferior)

    # Executa o crop e retorna uma nova imagem.
    imagem_crop = imagem.crop(caixa)

    return imagem_crop, caixa, True


# =============================================================================
# CONVERSAO PARA O FORMATO NUMERICO DO PROJETO
# =============================================================================

def converter_imagem_para_npy(imagem, tamanho_final):
    """
    Converte uma imagem PIL para o formato numerico padrao do projeto.

    Fluxo
    -----
    PIL
        -> grayscale
        -> resize LANCZOS
        -> numpy float32
        -> divisao por 255
        -> adicao do canal
        -> validacao

    Parameters
    ----------
    imagem : PIL.Image.Image
        Imagem que sera convertida.

    tamanho_final : tuple[int, int]
        Resolucao final no formato (largura, altura).

    Returns
    -------
    numpy.ndarray
        Array float32 com shape (1,H,W) e valores em [0,1].

    Raises
    ------
    ValueError
        Quando a resolucao final e invalida ou o resultado nao possui o formato
        numerico esperado.
    """

    # Converte a resolucao para inteiros para evitar valores numericos ambiguos.
    largura_final = int(tamanho_final[0])
    altura_final = int(tamanho_final[1])

    # Dimensoes nulas ou negativas nao representam uma imagem valida.
    if largura_final <= 0 or altura_final <= 0:
        raise ValueError(
            f"Tamanho final invalido: {tamanho_final}"
        )

    # O modelo trabalha com um unico canal, portanto a conversao para L e sempre
    # aplicada antes do resize.
    imagem = imagem.convert("L")

    # LANCZOS preserva melhor detalhes durante reducoes de resolucao.
    imagem = imagem.resize(
        (largura_final, altura_final),
        Image.Resampling.LANCZOS,
    )

    # Converte os pixels 0-255 diretamente para float32.
    array = np.asarray(
        imagem,
        dtype=np.float32,
    )

    # Normaliza os pixels para o intervalo usado pelo restante do pipeline.
    array = array / 255.0

    # Adiciona a dimensao de canal, convertendo (H,W) para (1,H,W).
    array = np.expand_dims(
        array,
        axis=0,
    )

    # Garante explicitamente float32 mesmo apos as operacoes numericas.
    array = array.astype(
        np.float32,
        copy=False,
    )

    # Reutiliza a validacao central do formato NPY do projeto.
    validar_imagem_npy(array)

    return array
