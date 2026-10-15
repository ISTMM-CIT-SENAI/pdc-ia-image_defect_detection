"""
crop_dataset_sam.py

Objetivo:
- Aplica SAM 2.1 em todas as imagens de dataset_atual.
- Seleciona a peca usando criterios geometricos robustos.
- Gera crop quadrado com margem de 10%.
- Salva em dataset_atual_crop mantendo nomes e estrutura de pastas.
- Marca automaticamente casos suspeitos como REVISAR.
- Gera diagnosticos apenas para os casos REVISAR.
- Salva no CSV os dados das tres mascaras candidatas.

Criterios principais:
- Mascara deve conter o ponto central.
- Area da mascara dentro de limites plausiveis.
- Preenchimento minimo da bounding box.
- Bounding box nao pode ocupar area excessiva da imagem.
- Crop resultante nao pode ficar excessivamente grande.
- Score SAM e apenas auxiliar; nao invalida sozinho uma mascara.

Observacoes:
- dataset_atual nunca e modificada.
- dataset_atual_crop e recriada quando LIMPAR_PASTA_SAIDA=True.
- Casos REVISAR tambem sao salvos para permitir inspecao visual.
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
from collections import Counter
import csv
import shutil

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from PIL import Image, ImageOps

from sam2.sam2_image_predictor import SAM2ImagePredictor


# =============================================================================
# PARAMETROS AJUSTAVEIS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PASTA_ENTRADA = BASE_DIR / "dataset_rgb"
PASTA_SAIDA = BASE_DIR / "dataset_rgb_crop"

MODELO_SAM = "facebook/sam2.1-hiera-small"

BUSCAR_SUBPASTAS = True
CONFIRMAR_ANTES = True
LIMPAR_PASTA_SAIDA = True

# Margem adicionada ao redor da bounding box
MARGEM_CROP = 0.10

# -----------------------------------------------------------------------------
# LIMITES DUROS PARA UMA MASCARA SER CANDIDATA
# -----------------------------------------------------------------------------

AREA_MINIMA_CANDIDATA = 0.015          # 1,5%
AREA_MAXIMA_CANDIDATA = 0.12           # 12%

PREENCHIMENTO_BBOX_MINIMO = 0.50       # 50%
AREA_BBOX_MAXIMA_CANDIDATA = 0.16      # 16% da imagem

# Crop quadrado nao pode ultrapassar 65% do menor lado da imagem
CROP_RELATIVO_MAXIMO_CANDIDATA = 0.65

# -----------------------------------------------------------------------------
# FAIXA CONSERVADORA PARA MARCAR REVISAR
# A mascara pode ser valida, mas valores proximos dos limites merecem inspeção.
# -----------------------------------------------------------------------------

AREA_REVISAR_MINIMA = 0.020             # abaixo de 2%
AREA_REVISAR_MAXIMA = 0.11              # acima de 11%

PREENCHIMENTO_REVISAR_MINIMO = 0.60     # abaixo de 60%
AREA_BBOX_REVISAR_MAXIMA = 0.14         # acima de 14%

CROP_RELATIVO_REVISAR_MAXIMO = 0.60     # acima de 60%
CENTRALIDADE_REVISAR_MINIMA = 0.80

# Score baixo e apenas informativo
SCORE_SAM_BAIXO = 0.70

# -----------------------------------------------------------------------------
# SELECAO ENTRE CANDIDATAS VALIDAS
# -----------------------------------------------------------------------------

PESO_PREENCHIMENTO = 0.55
PESO_CENTRALIDADE = 0.35
PESO_SCORE = 0.10

# Se duas candidatas tiverem criterios praticamente iguais,
# prefere a de maior area para evitar selecionar apenas um defeito interno.
TOLERANCIA_EMPATE = 0.015

SALVAR_DIAGNOSTICOS_REVISAR = True
DPI_GRAFICOS = 160


# =============================================================================
# CAMINHOS
# =============================================================================

INICIO = datetime.now()

PASTA_RESULTADOS = (
    BASE_DIR
    / "resultados"
    / "crop_dataset_sam"
    / f"execucao_{INICIO:%Y-%m-%d_%H-%M-%S}"
)

PASTA_DIAGNOSTICOS = PASTA_RESULTADOS / "revisar"

PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

if SALVAR_DIAGNOSTICOS_REVISAR:
    PASTA_DIAGNOSTICOS.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# IMAGENS
# =============================================================================

def listar_imagens():
    if not PASTA_ENTRADA.exists():
        raise FileNotFoundError(f"Pasta de entrada nao encontrada:\n{PASTA_ENTRADA}")

    Image.init()
    extensoes = {ext.lower() for ext in Image.registered_extensions()}
    arquivos = PASTA_ENTRADA.rglob("*") if BUSCAR_SUBPASTAS else PASTA_ENTRADA.iterdir()

    imagens = sorted([
        arquivo for arquivo in arquivos
        if arquivo.is_file() and arquivo.suffix.lower() in extensoes
    ])

    if not imagens:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em:\n{PASTA_ENTRADA}")

    return imagens


def carregar_imagem(caminho):
    with Image.open(caminho) as imagem:
        imagem = ImageOps.exif_transpose(imagem)
        return imagem.copy()


def preparar_pasta_saida():
    if PASTA_ENTRADA.resolve() == PASTA_SAIDA.resolve():
        raise ValueError("PASTA_ENTRADA e PASTA_SAIDA nao podem ser iguais.")

    if LIMPAR_PASTA_SAIDA and PASTA_SAIDA.exists():
        shutil.rmtree(PASTA_SAIDA)

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)


def salvar_crop(crop, caminho_origem):
    relativo = caminho_origem.relative_to(PASTA_ENTRADA)
    destino = PASTA_SAIDA / relativo
    destino.parent.mkdir(parents=True, exist_ok=True)

    imagem_salvar = crop

    if destino.suffix.lower() in {".jpg", ".jpeg"} and imagem_salvar.mode not in {"RGB", "L"}:
        imagem_salvar = imagem_salvar.convert("RGB")

    imagem_salvar.save(destino)
    return destino


# =============================================================================
# SAM
# =============================================================================

def contexto_autocast():
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    return nullcontext()


def carregar_sam():
    print("\nCarregando SAM 2.1...")
    print(f"Modelo      : {MODELO_SAM}")
    print(f"Dispositivo : {device}")

    predictor = SAM2ImagePredictor.from_pretrained(
        MODELO_SAM,
        device=device
    )

    print("SAM 2.1 carregado.")
    return predictor


# =============================================================================
# GEOMETRIA DA MASCARA
# =============================================================================

def obter_bbox(mask):
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1
    )


def calcular_centralidade(bbox, largura, altura):
    if bbox is None:
        return 0.0

    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    dx = (cx - largura / 2) / (largura / 2)
    dy = (cy - altura / 2) / (altura / 2)

    distancia = np.sqrt(dx ** 2 + dy ** 2)

    return float(max(0.0, 1.0 - distancia / np.sqrt(2)))


def calcular_preenchimento_bbox(mask, bbox):
    if bbox is None:
        return 0.0

    x1, y1, x2, y2 = bbox
    area_bbox = (x2 - x1) * (y2 - y1)

    if area_bbox <= 0:
        return 0.0

    return float(mask.sum() / area_bbox)


def calcular_crop_relativo_bbox(bbox, largura, altura):
    if bbox is None:
        return 1.0

    x1, y1, x2, y2 = bbox

    largura_bbox = x2 - x1
    altura_bbox = y2 - y1

    lado_objeto = max(largura_bbox, altura_bbox)
    lado_desejado = lado_objeto * (1 + 2 * MARGEM_CROP)

    menor_dimensao = min(largura, altura)

    return float(min(lado_desejado, menor_dimensao) / menor_dimensao)


def motivos_invalidade_candidata(candidata):
    motivos = []

    if not candidata["contem_centro"]:
        motivos.append("nao contem centro")

    if candidata["area"] < AREA_MINIMA_CANDIDATA:
        motivos.append("area muito pequena")

    if candidata["area"] > AREA_MAXIMA_CANDIDATA:
        motivos.append("area muito grande")

    if candidata["preenchimento_bbox"] < PREENCHIMENTO_BBOX_MINIMO:
        motivos.append("fill baixo")

    if candidata["bbox_area_relativa"] > AREA_BBOX_MAXIMA_CANDIDATA:
        motivos.append("bbox muito grande")

    if candidata["crop_relativo_estimado"] > CROP_RELATIVO_MAXIMO_CANDIDATA:
        motivos.append("crop estimado muito grande")

    return motivos


def analisar_candidata(mask, score, largura, altura):
    area = float(mask.mean())
    bbox = obter_bbox(mask)

    if bbox is None:
        return {
            "score": float(score),
            "area": 0.0,
            "centralidade": 0.0,
            "preenchimento_bbox": 0.0,
            "bbox_area_relativa": 0.0,
            "crop_relativo_estimado": 1.0,
            "contem_centro": False,
            "criterio": 0.0,
            "valida": False,
            "motivos_invalidade": "mascara vazia",
            "bbox": None
        }

    x1, y1, x2, y2 = bbox

    bbox_area = (x2 - x1) * (y2 - y1)
    bbox_area_relativa = bbox_area / (largura * altura)

    xc = min(int(largura / 2), largura - 1)
    yc = min(int(altura / 2), altura - 1)

    contem_centro = bool(mask[yc, xc])

    preenchimento = calcular_preenchimento_bbox(mask, bbox)
    centralidade = calcular_centralidade(bbox, largura, altura)
    crop_relativo = calcular_crop_relativo_bbox(bbox, largura, altura)

    # O score SAM e limitado a [0, 1] somente para o criterio combinado.
    score_normalizado = float(np.clip(score, 0.0, 1.0))

    criterio = (
        PESO_PREENCHIMENTO * preenchimento
        + PESO_CENTRALIDADE * centralidade
        + PESO_SCORE * score_normalizado
    )

    candidata = {
        "score": float(score),
        "area": area,
        "centralidade": centralidade,
        "preenchimento_bbox": preenchimento,
        "bbox_area_relativa": bbox_area_relativa,
        "crop_relativo_estimado": crop_relativo,
        "contem_centro": contem_centro,
        "criterio": float(criterio),
        "bbox": bbox
    }

    motivos = motivos_invalidade_candidata(candidata)

    candidata["valida"] = len(motivos) == 0
    candidata["motivos_invalidade"] = ", ".join(motivos)

    return candidata


# =============================================================================
# SELECAO DA MASCARA
# =============================================================================

def selecionar_mascara(masks, scores, largura, altura):
    candidatas = [
        analisar_candidata(mask, score, largura, altura)
        for mask, score in zip(masks, scores)
    ]

    validas = [
        i for i, candidata in enumerate(candidatas)
        if candidata["valida"]
    ]

    if not validas:
        indice = int(np.argmax([
            candidata["criterio"]
            for candidata in candidatas
        ]))

        return indice, candidatas, True

    melhor_criterio = max(
        candidatas[i]["criterio"]
        for i in validas
    )

    # Candidatas praticamente empatadas.
    proximas = [
        i for i in validas
        if melhor_criterio - candidatas[i]["criterio"] <= TOLERANCIA_EMPATE
    ]

    # Se geometricamente forem equivalentes, prefere a maior mascara.
    indice = max(
        proximas,
        key=lambda i: candidatas[i]["area"]
    )

    return indice, candidatas, False


# =============================================================================
# SEGMENTACAO
# =============================================================================

def segmentar(predictor, imagem):
    imagem_rgb = imagem.convert("RGB")
    imagem_np = np.asarray(imagem_rgb).copy()

    altura, largura = imagem_np.shape[:2]

    ponto = np.array([
        [largura / 2.0, altura / 2.0]
    ], dtype=np.float32)

    rotulo = np.array([1], dtype=np.int32)

    with torch.inference_mode():
        with contexto_autocast():

            predictor.set_image(imagem_np)

            masks, scores, _ = predictor.predict(
                point_coords=ponto,
                point_labels=rotulo,
                multimask_output=True
            )

    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores, dtype=np.float32)

    indice, candidatas, fallback = selecionar_mascara(
        masks,
        scores,
        largura,
        altura
    )

    return masks, scores, indice, candidatas, fallback, ponto[0]


# =============================================================================
# CROP
# =============================================================================

def criar_crop_quadrado(imagem, bbox):
    x1, y1, x2, y2 = bbox

    largura_bbox = x2 - x1
    altura_bbox = y2 - y1

    lado_objeto = max(
        largura_bbox,
        altura_bbox
    )

    margem = round(
        lado_objeto * MARGEM_CROP
    )

    lado_desejado = lado_objeto + 2 * margem

    largura_img, altura_img = imagem.size
    menor_dimensao = min(largura_img, altura_img)

    lado_crop = min(
        lado_desejado,
        menor_dimensao
    )

    limitado = lado_crop < lado_desejado

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    esquerda = int(round(
        cx - lado_crop / 2
    ))

    superior = int(round(
        cy - lado_crop / 2
    ))

    esquerda = max(
        0,
        min(
            esquerda,
            largura_img - lado_crop
        )
    )

    superior = max(
        0,
        min(
            superior,
            altura_img - lado_crop
        )
    )

    direita = esquerda + lado_crop
    inferior = superior + lado_crop

    contem_bbox = (
        esquerda <= x1
        and superior <= y1
        and direita >= x2
        and inferior >= y2
    )

    crop = imagem.crop(
        (
            esquerda,
            superior,
            direita,
            inferior
        )
    )

    if crop.width != crop.height:
        raise ValueError(
            f"Crop nao ficou quadrado: {crop.width}x{crop.height}"
        )

    return crop, {
        "lado_desejado": lado_desejado,
        "lado_crop": lado_crop,
        "margem_px": margem,
        "crop_relativo": lado_crop / menor_dimensao,
        "caixa_crop": (
            esquerda,
            superior,
            direita,
            inferior
        ),
        "limitado": limitado,
        "contem_bbox": contem_bbox
    }


# =============================================================================
# VERIFICACAO FINAL / REVISAR
# =============================================================================

def avaliar_resultado(candidata, fallback, info_crop):
    motivos = []

    # Falha na selecao normal sempre exige revisao.
    if fallback:
        motivos.append(
            "nenhuma candidata passou pelos filtros geometricos"
        )

    # Verificacoes absolutas.
    if not candidata["contem_centro"]:
        motivos.append(
            "mascara nao contem ponto central"
        )

    if candidata["area"] < AREA_MINIMA_CANDIDATA:
        motivos.append(
            "area abaixo do limite da candidata"
        )

    if candidata["area"] > AREA_MAXIMA_CANDIDATA:
        motivos.append(
            "area acima do limite da candidata"
        )

    if candidata["preenchimento_bbox"] < PREENCHIMENTO_BBOX_MINIMO:
        motivos.append(
            "fill abaixo do limite da candidata"
        )

    if candidata["bbox_area_relativa"] > AREA_BBOX_MAXIMA_CANDIDATA:
        motivos.append(
            "bounding box excessivamente grande"
        )

    if candidata["crop_relativo_estimado"] > CROP_RELATIVO_MAXIMO_CANDIDATA:
        motivos.append(
            "crop estimado excessivamente grande"
        )

    # Faixa conservadora.
    if candidata["area"] < AREA_REVISAR_MINIMA:
        motivos.append(
            "area proxima do limite inferior"
        )

    if candidata["area"] > AREA_REVISAR_MAXIMA:
        motivos.append(
            "area proxima do limite superior"
        )

    if candidata["preenchimento_bbox"] < PREENCHIMENTO_REVISAR_MINIMO:
        motivos.append(
            "fill baixo para o padrao do dataset"
        )

    if candidata["bbox_area_relativa"] > AREA_BBOX_REVISAR_MAXIMA:
        motivos.append(
            "bbox grande para o padrao do dataset"
        )

    if candidata["centralidade"] < CENTRALIDADE_REVISAR_MINIMA:
        motivos.append(
            "centralidade abaixo do padrao esperado"
        )

    if info_crop["crop_relativo"] > CROP_RELATIVO_REVISAR_MAXIMO:
        motivos.append(
            "crop final grande para o padrao do dataset"
        )

    # Integridade geometrica.
    if info_crop["limitado"]:
        motivos.append(
            "margem do crop limitada pela imagem"
        )

    if not info_crop["contem_bbox"]:
        motivos.append(
            "crop nao contem toda a bounding box"
        )

    # Remove motivos duplicados preservando ordem.
    motivos = list(dict.fromkeys(motivos))

    status = "OK" if not motivos else "REVISAR"

    return status, motivos


# =============================================================================
# DIAGNOSTICOS
# =============================================================================

def nome_diagnostico(caminho):
    relativo = caminho.relative_to(
        PASTA_ENTRADA
    ).with_suffix("")

    return "__".join(
        relativo.parts
    )


def salvar_diagnostico(
    caminho,
    imagem,
    masks,
    candidatas,
    indice,
    ponto,
    crop,
    info_crop,
    motivos
):
    imagem_np = np.asarray(
        imagem.convert("RGB")
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11)
    )

    axes = axes.ravel()

    # Original
    axes[0].imshow(imagem_np)

    axes[0].scatter(
        ponto[0],
        ponto[1],
        s=100,
        marker="*"
    )

    axes[0].set_title(
        f"Original + ponto central\n"
        f"{imagem.width}x{imagem.height}"
    )

    axes[0].axis("off")

    # Tres candidatas
    for i in range(
        min(3, len(masks))
    ):
        candidata = candidatas[i]

        axes[i + 1].imshow(
            imagem_np
        )

        axes[i + 1].imshow(
            masks[i],
            alpha=0.45
        )

        selecionada = (
            " - ESCOLHIDA"
            if i == indice
            else ""
        )

        validade = (
            "VALIDA"
            if candidata["valida"]
            else "INVALIDA"
        )

        axes[i + 1].set_title(
            f"Mask {i + 1}{selecionada}\n"
            f"SAM={candidata['score']:.4f} | "
            f"Area={candidata['area']:.2%}\n"
            f"Fill={candidata['preenchimento_bbox']:.2%} | "
            f"BBox={candidata['bbox_area_relativa']:.2%}\n"
            f"Crop={candidata['crop_relativo_estimado']:.2%} | "
            f"Centro={candidata['centralidade']:.3f}\n"
            f"Criterio={candidata['criterio']:.3f} | {validade}"
        )

        axes[i + 1].axis("off")

    candidata = candidatas[indice]

    # Escolhida + bbox + crop
    axes[4].imshow(
        imagem_np
    )

    axes[4].imshow(
        masks[indice],
        alpha=0.35
    )

    if candidata["bbox"] is not None:
        x1, y1, x2, y2 = candidata["bbox"]

        axes[4].add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=2
            )
        )

    esquerda, superior, direita, inferior = (
        info_crop["caixa_crop"]
    )

    axes[4].add_patch(
        Rectangle(
            (esquerda, superior),
            direita - esquerda,
            inferior - superior,
            fill=False,
            linewidth=2,
            linestyle="--"
        )
    )

    axes[4].set_title(
        f"Selecionada\n"
        f"Area={candidata['area']:.2%} | "
        f"Fill={candidata['preenchimento_bbox']:.2%}\n"
        f"BBox={candidata['bbox_area_relativa']:.2%} | "
        f"Crop={info_crop['crop_relativo']:.2%}"
    )

    axes[4].axis("off")

    # Crop
    axes[5].imshow(
        crop.convert("RGB")
    )

    axes[5].set_title(
        f"REVISAR\n"
        f"{crop.width}x{crop.height}"
    )

    axes[5].axis("off")

    fig.suptitle(
        f"{caminho.relative_to(PASTA_ENTRADA)}\n"
        + " | ".join(motivos),
        fontsize=11
    )

    fig.tight_layout()

    fig.savefig(
        PASTA_DIAGNOSTICOS
        / f"{nome_diagnostico(caminho)}_diagnostico.png",
        dpi=DPI_GRAFICOS
    )

    plt.close(fig)


# =============================================================================
# PROCESSAMENTO
# =============================================================================

def processar_imagem(predictor, caminho):
    imagem = carregar_imagem(
        caminho
    )

    (
        masks,
        scores,
        indice,
        candidatas,
        fallback,
        ponto
    ) = segmentar(
        predictor,
        imagem
    )

    candidata = candidatas[
        indice
    ]

    if candidata["bbox"] is None:
        raise ValueError(
            "Mascara selecionada esta vazia."
        )

    crop, info_crop = criar_crop_quadrado(
        imagem,
        candidata["bbox"]
    )

    status, motivos = avaliar_resultado(
        candidata,
        fallback,
        info_crop
    )

    destino = salvar_crop(
        crop,
        caminho
    )

    if (
        status == "REVISAR"
        and SALVAR_DIAGNOSTICOS_REVISAR
    ):
        salvar_diagnostico(
            caminho,
            imagem,
            masks,
            candidatas,
            indice,
            ponto,
            crop,
            info_crop,
            motivos
        )

    x1, y1, x2, y2 = candidata["bbox"]

    resultado = {
        "imagem": str(
            caminho.relative_to(
                PASTA_ENTRADA
            )
        ),

        "saida": str(
            destino.relative_to(
                BASE_DIR
            )
        ),

        "largura_original": imagem.width,
        "altura_original": imagem.height,

        "mascara_escolhida": indice + 1,

        "score_sam": candidata["score"],

        "score_sam_baixo": (
            candidata["score"]
            < SCORE_SAM_BAIXO
        ),

        "area_mascara": candidata["area"],

        "fill_bbox": candidata[
            "preenchimento_bbox"
        ],

        "area_bbox": candidata[
            "bbox_area_relativa"
        ],

        "crop_relativo_estimado": candidata[
            "crop_relativo_estimado"
        ],

        "crop_relativo_final": info_crop[
            "crop_relativo"
        ],

        "centralidade": candidata[
            "centralidade"
        ],

        "criterio": candidata[
            "criterio"
        ],

        "fallback": fallback,

        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,

        "bbox_largura": x2 - x1,
        "bbox_altura": y2 - y1,

        "margem_px": info_crop[
            "margem_px"
        ],

        "lado_crop": info_crop[
            "lado_crop"
        ],

        "status": status,

        "motivos": ", ".join(
            motivos
        )
    }

    # Salva detalhes das tres candidatas no CSV.
    for i in range(3):
        if i < len(candidatas):
            c = candidatas[i]

            resultado[f"score_{i + 1}"] = c["score"]
            resultado[f"area_{i + 1}"] = c["area"]
            resultado[f"fill_{i + 1}"] = c["preenchimento_bbox"]
            resultado[f"bbox_area_{i + 1}"] = c["bbox_area_relativa"]
            resultado[f"crop_rel_{i + 1}"] = c["crop_relativo_estimado"]
            resultado[f"centro_{i + 1}"] = c["centralidade"]
            resultado[f"criterio_{i + 1}"] = c["criterio"]
            resultado[f"valida_{i + 1}"] = c["valida"]
            resultado[f"invalidade_{i + 1}"] = c["motivos_invalidade"]

        else:
            resultado[f"score_{i + 1}"] = ""
            resultado[f"area_{i + 1}"] = ""
            resultado[f"fill_{i + 1}"] = ""
            resultado[f"bbox_area_{i + 1}"] = ""
            resultado[f"crop_rel_{i + 1}"] = ""
            resultado[f"centro_{i + 1}"] = ""
            resultado[f"criterio_{i + 1}"] = ""
            resultado[f"valida_{i + 1}"] = ""
            resultado[f"invalidade_{i + 1}"] = ""

    return resultado


# =============================================================================
# CSV E RESUMO
# =============================================================================

def salvar_csv(resultados):
    caminho = (
        PASTA_RESULTADOS
        / "resultados.csv"
    )

    campos = [
        "imagem",
        "saida",
        "largura_original",
        "altura_original",
        "mascara_escolhida",

        "score_1",
        "area_1",
        "fill_1",
        "bbox_area_1",
        "crop_rel_1",
        "centro_1",
        "criterio_1",
        "valida_1",
        "invalidade_1",

        "score_2",
        "area_2",
        "fill_2",
        "bbox_area_2",
        "crop_rel_2",
        "centro_2",
        "criterio_2",
        "valida_2",
        "invalidade_2",

        "score_3",
        "area_3",
        "fill_3",
        "bbox_area_3",
        "crop_rel_3",
        "centro_3",
        "criterio_3",
        "valida_3",
        "invalidade_3",

        "score_sam",
        "score_sam_baixo",
        "area_mascara",
        "fill_bbox",
        "area_bbox",
        "crop_relativo_estimado",
        "crop_relativo_final",
        "centralidade",
        "criterio",
        "fallback",

        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_largura",
        "bbox_altura",

        "margem_px",
        "lado_crop",

        "status",
        "motivos"
    ]

    with open(
        caminho,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as arquivo:

        writer = csv.DictWriter(
            arquivo,
            fieldnames=campos,
            delimiter=";"
        )

        writer.writeheader()
        writer.writerows(
            resultados
        )

    return caminho


def salvar_resumo(resultados, erros):
    caminho = (
        PASTA_RESULTADOS
        / "resumo.txt"
    )

    ok = sum(
        r["status"] == "OK"
        for r in resultados
    )

    revisar = sum(
        r["status"] == "REVISAR"
        for r in resultados
    )

    contador_motivos = Counter()

    for resultado in resultados:
        if resultado["status"] == "REVISAR":
            for motivo in resultado["motivos"].split(", "):
                if motivo:
                    contador_motivos[motivo] += 1

    with open(
        caminho,
        "w",
        encoding="utf-8"
    ) as arquivo:

        arquivo.write(
            "CROP DATASET COM SAM 2.1\n"
        )

        arquivo.write(
            "=" * 70
            + "\n\n"
        )

        arquivo.write(
            f"Entrada    : {PASTA_ENTRADA}\n"
        )

        arquivo.write(
            f"Saida      : {PASTA_SAIDA}\n"
        )

        arquivo.write(
            f"Processadas: {len(resultados)}\n"
        )

        arquivo.write(
            f"OK         : {ok}\n"
        )

        arquivo.write(
            f"REVISAR    : {revisar}\n"
        )

        arquivo.write(
            f"ERROS      : {len(erros)}\n\n"
        )

        arquivo.write(
            "LIMITES DAS CANDIDATAS\n"
        )

        arquivo.write(
            f"Area       : "
            f"{AREA_MINIMA_CANDIDATA:.2%} "
            f"a {AREA_MAXIMA_CANDIDATA:.2%}\n"
        )

        arquivo.write(
            f"Fill minimo: "
            f"{PREENCHIMENTO_BBOX_MINIMO:.2%}\n"
        )

        arquivo.write(
            f"BBox maxima: "
            f"{AREA_BBOX_MAXIMA_CANDIDATA:.2%}\n"
        )

        arquivo.write(
            f"Crop maximo: "
            f"{CROP_RELATIVO_MAXIMO_CANDIDATA:.2%}\n\n"
        )

        if contador_motivos:
            arquivo.write(
                "MOTIVOS DE REVISAO\n"
            )

            for motivo, quantidade in contador_motivos.most_common():
                arquivo.write(
                    f"{quantidade:3d} - {motivo}\n"
                )

        if erros:
            arquivo.write(
                "\nERROS\n"
            )

            for erro in erros:
                arquivo.write(
                    f"- {erro}\n"
                )

    return caminho


# =============================================================================
# MAIN
# =============================================================================

def main():
    imagens = listar_imagens()

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CROP AUTOMATICO DO DATASET COM SAM 2.1 - REVISAO ROBUSTA"
    )

    print(
        "=" * 80
    )

    print(
        f"\nEntrada                  : {PASTA_ENTRADA}"
    )

    print(
        f"Saida                    : {PASTA_SAIDA}"
    )

    print(
        f"Imagens encontradas      : {len(imagens)}"
    )

    print(
        f"Margem                   : {MARGEM_CROP:.0%}"
    )

    print(
        f"Area candidata           : "
        f"{AREA_MINIMA_CANDIDATA:.1%} a "
        f"{AREA_MAXIMA_CANDIDATA:.0%}"
    )

    print(
        f"Fill minimo candidata    : "
        f"{PREENCHIMENTO_BBOX_MINIMO:.0%}"
    )

    print(
        f"BBox maxima candidata    : "
        f"{AREA_BBOX_MAXIMA_CANDIDATA:.0%}"
    )

    print(
        f"Crop maximo candidata    : "
        f"{CROP_RELATIVO_MAXIMO_CANDIDATA:.0%}"
    )

    print(
        f"Dispositivo              : {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU                      : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA                     : "
            f"{torch.version.cuda}"
        )

        print(
            f"BF16                     : "
            f"{torch.cuda.is_bf16_supported()}"
        )

    if PASTA_SAIDA.exists() and LIMPAR_PASTA_SAIDA:
        print(
            "\nATENCAO: dataset_atual_crop existente sera apagada "
            "e recriada."
        )

    if CONFIRMAR_ANTES:
        resposta = input(
            "\nProcessar todas as imagens? [S/N]: "
        ).strip().lower()

        if resposta not in {
            "s",
            "sim",
            "y",
            "yes"
        }:
            print(
                "\nProcessamento cancelado."
            )

            return

    preparar_pasta_saida()

    predictor = carregar_sam()

    resultados = []
    erros = []

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PROCESSANDO"
    )

    print(
        "=" * 80
    )

    for numero, caminho in enumerate(
        imagens,
        start=1
    ):
        relativo = caminho.relative_to(
            PASTA_ENTRADA
        )

        try:
            resultado = processar_imagem(
                predictor,
                caminho
            )

            resultados.append(
                resultado
            )

            print(
                f"[{numero:03d}/{len(imagens):03d}] "
                f"{resultado['status']:<7} | "
                f"{str(relativo):<48} | "
                f"Area {resultado['area_mascara']:.1%} | "
                f"Fill {resultado['fill_bbox']:.1%} | "
                f"BBox {resultado['area_bbox']:.1%} | "
                f"Crop {resultado['crop_relativo_final']:.1%}"
            )

        except Exception as erro:
            mensagem = (
                f"{relativo}: {erro}"
            )

            erros.append(
                mensagem
            )

            print(
                f"[{numero:03d}/{len(imagens):03d}] "
                f"ERRO    | {mensagem}"
            )

    caminho_csv = salvar_csv(
        resultados
    )

    caminho_resumo = salvar_resumo(
        resultados,
        erros
    )

    ok = sum(
        resultado["status"] == "OK"
        for resultado in resultados
    )

    revisar = sum(
        resultado["status"] == "REVISAR"
        for resultado in resultados
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "RESULTADO FINAL"
    )

    print(
        "=" * 80
    )

    print(
        f"\nImagens encontradas : {len(imagens)}"
    )

    print(
        f"Processadas         : {len(resultados)}"
    )

    print(
        f"OK                  : {ok}"
    )

    print(
        f"REVISAR             : {revisar}"
    )

    print(
        f"ERROS               : {len(erros)}"
    )

    print(
        f"\nDataset recortado:\n"
        f"{PASTA_SAIDA}"
    )

    print(
        f"\nCSV:\n"
        f"{caminho_csv}"
    )

    print(
        f"\nResumo:\n"
        f"{caminho_resumo}"
    )

    if SALVAR_DIAGNOSTICOS_REVISAR:
        print(
            f"\nDiagnosticos REVISAR:\n"
            f"{PASTA_DIAGNOSTICOS}"
        )

    print(
        "\ndataset_atual nao foi modificada."
    )


if __name__ == "__main__":
    main()