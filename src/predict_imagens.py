"""
predict_imagens.py

Objetivo:
- Executa predicoes em uma pasta de imagens externas.
- Todas as imagens da pasta devem possuir o mesmo rotulo real, informado pelo usuario no terminal.
- Converte automaticamente as imagens para grayscale.
- Transforma imagens retangulares em quadradas por crop central usando a menor dimensao.
- Redimensiona obrigatoriamente para a resolucao utilizada pelo modelo treinado.
- Normaliza os pixels de 0-255 para 0-1 e converte para float32.
- Calcula confianca, segunda classe, margem, entropia e metricas de classificacao.
- Gera CSV, HTML, resumo, graficos e visualizador interativo.

Usa:
- models/lenet5.py: recria a arquitetura da rede treinada.

Parametros ajustaveis:
- PASTA_MODELO: checkpoint utilizado na predicao.
- DIRETORIO_IMAGENS: pasta contendo imagens de um mesmo rotulo.
- BUSCAR_SUBPASTAS: inclui imagens existentes em subpastas.
- LIMIAR_CONFIANCA: limite para destacar predicoes de baixa confianca.
- DPI_GRAFICOS: resolucao dos graficos salvos.

Fluxo:
imagem -> EXIF -> grayscale -> crop quadrado central -> resize do modelo
-> float32 -> /255 -> tensor (1,H,W) -> predicao

Observacoes:
- O rotulo real e informado apenas uma vez e aplicado a todas as imagens da pasta.
- O crop quadrado usa sempre a menor dimensao da imagem.
- Imagens que ja sao quadradas nao sao recortadas.
- A resolucao final e obtida diretamente do checkpoint.
- Nenhuma imagem original e modificada.
"""

from pathlib import Path
from datetime import datetime
from html import escape
import csv
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from pdc_ia_image_defect_detection.models.lenet5 import LeNet5


# =============================================================================
# PARAMETROS AJUSTAVEIS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Modelo utilizado
PASTA_MODELO = BASE_DIR / "resultados" / "resultados_treino" / "treino_400_aug_inicio_2026-10-06_11-46_fim_2026-10-06_11-52"

# Pasta contendo imagens de UM MESMO rotulo
DIRETORIO_IMAGENS = Path(r"C:\PDC-dataset\dataset_externo\CR7\Risco")

BUSCAR_SUBPASTAS = True
LIMIAR_CONFIANCA = 0.70
DPI_GRAFICOS = 300


# Caminhos derivados
CAMINHO_MODELO = PASTA_MODELO / "modelo_treinado.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INICIO_EXECUCAO = datetime.now()
PASTA_RESULTADOS = BASE_DIR / "resultados" / "predict_diretorio_resultados" / f"inicio_{INICIO_EXECUCAO:%Y-%m-%d_%H-%M}_em_andamento"
PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)


def obter_info_pasta():
    # Aceita treino_DIMENSAO_... e treino_DIMENSAO_aug_...
    correspondencia = re.match(r"^treino_(\d+)(_aug)?_", PASTA_MODELO.name)
    if correspondencia is None: raise ValueError(f"Nao foi possivel identificar dimensao/augmentation:\n{PASTA_MODELO.name}")
    return int(correspondencia.group(1)), correspondencia.group(2) == "_aug"


def verificar_caminhos():
    if not PASTA_MODELO.exists(): raise FileNotFoundError(f"Pasta do modelo nao encontrada:\n{PASTA_MODELO}")
    if not CAMINHO_MODELO.exists(): raise FileNotFoundError(f"Modelo nao encontrado:\n{CAMINHO_MODELO}")
    if not DIRETORIO_IMAGENS.exists(): raise FileNotFoundError(f"Diretorio nao encontrado:\n{DIRETORIO_IMAGENS}")
    if not DIRETORIO_IMAGENS.is_dir(): raise NotADirectoryError(f"O caminho informado nao e um diretorio:\n{DIRETORIO_IMAGENS}")


def extensoes_suportadas():
    Image.init()
    return {ext.lower() for ext in Image.registered_extensions()}


def listar_imagens():
    extensoes = extensoes_suportadas()
    arquivos = DIRETORIO_IMAGENS.rglob("*") if BUSCAR_SUBPASTAS else DIRETORIO_IMAGENS.iterdir()
    arquivos = sorted([arquivo for arquivo in arquivos if arquivo.is_file() and arquivo.suffix.lower() in extensoes])

    if not arquivos: raise FileNotFoundError(f"Nenhuma imagem suportada encontrada em:\n{DIRETORIO_IMAGENS}")
    return arquivos, extensoes


def carregar_checkpoint():
    checkpoint = torch.load(CAMINHO_MODELO, map_location=device)
    ausentes = {"model_state_dict", "classes", "dimensao_imagem"} - checkpoint.keys()

    if ausentes: raise KeyError(f"Campos ausentes no checkpoint: {sorted(ausentes)}")

    classes = checkpoint["classes"]
    if not classes: raise ValueError("O checkpoint nao possui classes validas.")

    dimensao_checkpoint = int(checkpoint["dimensao_imagem"])
    dimensao_pasta, aug_pasta = obter_info_pasta()

    if dimensao_checkpoint != dimensao_pasta:
        raise ValueError(
            f"Dimensao inconsistente: pasta={dimensao_pasta}x{dimensao_pasta}, "
            f"checkpoint={dimensao_checkpoint}x{dimensao_checkpoint}"
        )

    if "usar_augmentation" in checkpoint and bool(checkpoint["usar_augmentation"]) != aug_pasta:
        raise ValueError(f"Augmentation inconsistente: pasta={aug_pasta}, checkpoint={bool(checkpoint['usar_augmentation'])}")

    return checkpoint, classes, dimensao_checkpoint, aug_pasta


def selecionar_rotulo_real(classes):
    # Um unico rotulo real para toda a pasta
    print("\n" + "=" * 70)
    print("ROTULO REAL DAS IMAGENS")
    print("=" * 70)

    for indice, classe in enumerate(classes, start=1): print(f"{indice} - {classe}")

    while True:
        resposta = input(f"\nEscolha o rotulo real [1-{len(classes)}]: ").strip()

        if resposta.isdigit():
            indice = int(resposta)

            if 1 <= indice <= len(classes):
                rotulo = classes[indice - 1]
                print(f"Rotulo selecionado: {rotulo}")
                return rotulo

        print("Opcao invalida.")


def crop_quadrado_central(imagem):
    # Usa a menor dimensao como lado e remove apenas o excesso da maior
    largura, altura = imagem.size
    lado = min(largura, altura)

    esquerda = (largura - lado) // 2
    superior = (altura - lado) // 2
    direita = esquerda + lado
    inferior = superior + lado

    caixa = (esquerda, superior, direita, inferior)
    crop_aplicado = largura != altura

    if crop_aplicado: imagem = imagem.crop(caixa)

    return imagem, caixa, crop_aplicado


def processar_imagem(caminho, dimensao_modelo):
    # EXIF -> grayscale -> quadrado central -> resize -> float32 -> 0-1
    with Image.open(caminho) as imagem:
        formato = imagem.format or caminho.suffix.replace(".", "").upper()
        modo_original = imagem.mode
        largura_original, altura_original = imagem.size
        frames = getattr(imagem, "n_frames", 1)

        if frames > 1: imagem.seek(0)

        # Corrige orientacao antes de verificar largura e altura
        imagem = ImageOps.exif_transpose(imagem)

        largura_exif, altura_exif = imagem.size

        # Converte qualquer modo para grayscale
        convertido_grayscale = imagem.mode != "L"
        imagem = imagem.convert("L")

        # Garante proporcao 1:1 antes do resize
        imagem, caixa_crop, crop_aplicado = crop_quadrado_central(imagem)
        largura_crop, altura_crop = imagem.size

        if largura_crop != altura_crop:
            raise ValueError(f"Falha no crop quadrado: {largura_crop}x{altura_crop}")

        # Resize para a dimensao exigida pelo checkpoint
        imagem = imagem.resize((dimensao_modelo, dimensao_modelo), Image.Resampling.LANCZOS)

        if imagem.mode != "L": raise ValueError(f"Modo final invalido: {imagem.mode}. Esperado: L.")
        if imagem.size != (dimensao_modelo, dimensao_modelo):
            raise ValueError(f"Resize incorreto: {imagem.size}. Esperado: {(dimensao_modelo, dimensao_modelo)}")

        # Normalizacao 0-255 -> 0-1
        array = np.asarray(imagem, dtype=np.float32) / 255.0
        array = np.expand_dims(array, axis=0)

    tensor = torch.from_numpy(array)
    esperado = (1, dimensao_modelo, dimensao_modelo)

    if tuple(tensor.shape) != esperado: raise ValueError(f"Shape inesperado: {tuple(tensor.shape)}. Esperado: {esperado}")
    if tensor.dtype != torch.float32: raise TypeError(f"Dtype inesperado: {tensor.dtype}. Esperado: torch.float32.")
    if not torch.isfinite(tensor).all(): raise ValueError("A imagem possui NaN ou infinito.")

    minimo, maximo = tensor.min().item(), tensor.max().item()
    if minimo < 0.0 or maximo > 1.0: raise ValueError(f"Pixels fora de 0-1: [{minimo:.6f}, {maximo:.6f}]")

    return tensor, {
        "formato": formato,
        "modo_original": modo_original,
        "convertido_grayscale": convertido_grayscale,
        "largura_original": largura_original,
        "altura_original": altura_original,
        "largura_exif": largura_exif,
        "altura_exif": altura_exif,
        "largura_crop": largura_crop,
        "altura_crop": altura_crop,
        "caixa_crop": caixa_crop,
        "crop_aplicado": crop_aplicado,
        "frames": frames,
        "dimensao_final": dimensao_modelo,
    }


def carregar_modelo(checkpoint, classes, dimensao_imagem):
    model = LeNet5(num_classes=len(classes)).to(device)
    amostra = torch.zeros(1, 1, dimensao_imagem, dimensao_imagem, dtype=torch.float32, device=device)

    # Inicializa LazyLinear na resolucao correta
    with torch.no_grad(): model(amostra)

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as erro:
        raise RuntimeError("Checkpoint incompativel com a arquitetura ou dimensao atual da LeNet5.") from erro

    model.eval()
    return model


def predizer(model, classes, imagem):
    with torch.no_grad():
        probabilidades = torch.softmax(model(imagem.unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()

    ordem = np.argsort(probabilidades)[::-1]
    indice1, indice2 = int(ordem[0]), int(ordem[1])
    confianca, segunda_prob = float(probabilidades[indice1]), float(probabilidades[indice2])
    margem = confianca - segunda_prob
    entropia = float(-(probabilidades * np.log(probabilidades + 1e-12)).sum() / np.log(len(classes))) if len(classes) > 1 else 0.0

    return {
        "previsto": classes[indice1],
        "confianca": confianca,
        "segunda_classe": classes[indice2],
        "segunda_probabilidade": segunda_prob,
        "margem": margem,
        "entropia": entropia,
        "probabilidades": {classe: float(probabilidades[i]) for i, classe in enumerate(classes)},
    }


def calcular_metricas(resultados, classes):
    ok = [r for r in resultados if r["status"] == "OK"]
    if not ok: return None, None, None

    y_true = [r["real"] for r in ok]
    y_pred = [r["previsto"] for r in ok]

    metricas = {
        "imagens": len(ok),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0),
        "top2_accuracy": np.mean([r["real"] in (r["previsto"], r["segunda_classe"]) for r in ok]),
    }

    relatorio = classification_report(y_true, y_pred, labels=classes, target_names=classes, zero_division=0)
    matriz = confusion_matrix(y_true, y_pred, labels=classes)

    return metricas, relatorio, matriz


def salvar_csv(resultados, classes):
    cabecalho = [
        "Imagem", "Caminho relativo", "Rotulo real", "Rotulo predito", "Correto",
        "Formato", "Modo original", "Convertida para grayscale",
        "Resolucao original", "Resolucao apos EXIF", "Resolucao crop",
        "Crop aplicado", "Caixa crop", "Resolucao modelo",
        "Confianca", "Segunda classe", "Segunda probabilidade",
        "Margem", "Entropia", "Status", "Erro",
    ] + [f"Prob_{classe}" for classe in classes]

    with open(PASTA_RESULTADOS / "predicoes_diretorio.csv", "w", newline="", encoding="utf-8-sig") as arquivo:
        writer = csv.writer(arquivo, delimiter=";")
        writer.writerow(cabecalho)

        for r in resultados:
            linha = [
                r["nome"], r["relativo"], r["real"], r["previsto"], r["correto"],
                r["formato"], r["modo_original"], r["convertido_grayscale"],
                r["resolucao_original"], r["resolucao_exif"], r["resolucao_crop"],
                r["crop_aplicado"], r["caixa_crop"], r["resolucao_modelo"],
                f"{r['confianca']:.6f}" if r["confianca"] is not None else "",
                r["segunda_classe"],
                f"{r['segunda_probabilidade']:.6f}" if r["segunda_probabilidade"] is not None else "",
                f"{r['margem']:.6f}" if r["margem"] is not None else "",
                f"{r['entropia']:.6f}" if r["entropia"] is not None else "",
                r["status"], r["erro"],
            ]

            linha += [f"{r['probabilidades'].get(classe, 0):.6f}" if r["status"] == "OK" else "" for classe in classes]
            writer.writerow(linha)


def salvar_tabela_html(resultados):
    linhas = [
        "<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Predicoes</title>",
        "<style>body{font-family:Arial;margin:30px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:7px;text-align:left}th{background:#eee}.erro{background:#f8d7da}.incerto{background:#fff3cd}</style></head><body>",
        "<h2>Predicoes do diretorio</h2>",
        f"<p><b>Diretorio:</b> {escape(str(DIRETORIO_IMAGENS))}</p>",
        f"<p><b>Modelo:</b> {escape(str(CAMINHO_MODELO))}</p>",
        "<table><tr><th>Imagem</th><th>Real</th><th>Previsto</th><th>Confianca</th><th>2ª classe</th><th>Margem</th><th>Entropia</th><th>Status</th></tr>",
    ]

    for r in resultados:
        classe = "erro" if r["status"] != "OK" else ("incerto" if r["confianca"] < LIMIAR_CONFIANCA else "")
        confianca = f"{r['confianca']:.2%}" if r["confianca"] is not None else ""
        margem = f"{r['margem']:.2%}" if r["margem"] is not None else ""
        entropia = f"{r['entropia']:.4f}" if r["entropia"] is not None else ""

        linhas.append(
            f"<tr class='{classe}'><td>{escape(r['relativo'])}</td><td>{escape(r['real'])}</td>"
            f"<td>{escape(r['previsto'])}</td><td>{confianca}</td><td>{escape(r['segunda_classe'])}</td>"
            f"<td>{margem}</td><td>{entropia}</td><td>{escape(r['status'])}</td></tr>"
        )

    linhas.append("</table></body></html>")
    (PASTA_RESULTADOS / "predicoes_diretorio.html").write_text("\n".join(linhas), encoding="utf-8")


def salvar_resumo(resultados, classes, dimensao_imagem, aug_modelo, rotulo_real, metricas, relatorio):
    ok = [r for r in resultados if r["status"] == "OK"]
    erros = [r for r in resultados if r["status"] != "OK"]
    confiancas = np.array([r["confianca"] for r in ok]) if ok else np.array([])
    margens = np.array([r["margem"] for r in ok]) if ok else np.array([])
    entropias = np.array([r["entropia"] for r in ok]) if ok else np.array([])
    contagem = {classe: sum(r["previsto"] == classe for r in ok) for classe in classes}
    convertidas = sum(r["convertido_grayscale"] for r in ok)
    crops = sum(r["crop_aplicado"] for r in ok)

    linhas = [
        "RESUMO DAS PREDICOES", "=" * 70, "",
        f"Modelo: {CAMINHO_MODELO}",
        f"Diretorio: {DIRETORIO_IMAGENS}",
        f"Rotulo real informado: {rotulo_real}",
        f"Dimensao utilizada pelo modelo: {dimensao_imagem}x{dimensao_imagem}",
        f"Modelo treinado com augmentation: {aug_modelo}",
        "Processamento: grayscale -> crop quadrado central -> resize -> float32 -> normalizacao 0-1",
        f"Total de arquivos: {len(resultados)}",
        f"Predicoes realizadas: {len(ok)}",
        f"Convertidas para grayscale: {convertidas}",
        f"Ja estavam grayscale: {len(ok) - convertidas}",
        f"Imagens que precisaram de crop: {crops}",
        f"Imagens que ja eram quadradas: {len(ok) - crops}",
        f"Arquivos com erro: {len(erros)}",
    ]

    if ok:
        linhas += [
            "", "CONFIANCA E INCERTEZA", "-" * 70,
            f"Confianca media: {confiancas.mean():.4f}",
            f"Confianca mediana: {np.median(confiancas):.4f}",
            f"Confianca minima: {confiancas.min():.4f}",
            f"Confianca maxima: {confiancas.max():.4f}",
            f"Margem media entre 1a e 2a classe: {margens.mean():.4f}",
            f"Entropia normalizada media: {entropias.mean():.4f}",
            f"Predicoes abaixo de {LIMIAR_CONFIANCA:.0%}: {sum(confiancas < LIMIAR_CONFIANCA)}",
            "", "DISTRIBUICAO DAS PREDICOES", "-" * 70,
        ]

        linhas += [f"{classe}: {contagem[classe]} ({contagem[classe] / len(ok):.2%})" for classe in classes]

    if metricas:
        linhas += ["", "METRICAS", "-" * 70]

        for nome, valor in metricas.items():
            linhas.append(f"{nome}: {valor:.6f}" if isinstance(valor, (float, np.floating)) else f"{nome}: {valor}")

        linhas += ["", "RELATORIO DE CLASSIFICACAO", "-" * 70, relatorio]

    if erros:
        linhas += ["", "ARQUIVOS COM ERRO", "-" * 70]
        linhas += [f"{r['relativo']} -> {r['erro']}" for r in erros]

    (PASTA_RESULTADOS / "resumo_predicoes.txt").write_text("\n".join(linhas), encoding="utf-8")


def salvar_graficos(resultados, classes, matriz=None):
    ok = [r for r in resultados if r["status"] == "OK"]
    if not ok: return

    plt.figure(figsize=(9, 5))
    plt.hist([r["confianca"] for r in ok], bins=20)
    plt.axvline(LIMIAR_CONFIANCA, linestyle="--")
    plt.xlabel("Confianca")
    plt.ylabel("Quantidade")
    plt.title("Distribuicao da confianca")
    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / "distribuicao_confianca.png", dpi=DPI_GRAFICOS)
    plt.close()

    contagens = [sum(r["previsto"] == classe for r in ok) for classe in classes]

    plt.figure(figsize=(9, 5))
    plt.bar(classes, contagens)
    plt.ylabel("Quantidade")
    plt.title("Distribuicao das classes preditas")
    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / "distribuicao_classes_preditas.png", dpi=DPI_GRAFICOS)
    plt.close()

    if matriz is not None:
        plt.figure(figsize=(8, 7))
        plt.imshow(matriz)
        plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
        plt.yticks(range(len(classes)), classes)
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.title("Matriz de confusao")

        for i in range(len(classes)):
            for j in range(len(classes)): plt.text(j, i, matriz[i, j], ha="center", va="center")

        plt.tight_layout()
        plt.savefig(PASTA_RESULTADOS / "matriz_confusao.png", dpi=DPI_GRAFICOS)
        plt.close()


def carregar_imagem_exibicao(caminho):
    with Image.open(caminho) as imagem:
        if getattr(imagem, "n_frames", 1) > 1: imagem.seek(0)
        return ImageOps.exif_transpose(imagem).copy()


def visualizar_resultados(resultados):
    resultados_ok = [r for r in resultados if r["status"] == "OK"]

    if not resultados_ok:
        print("Nenhuma imagem disponivel para exibicao.")
        return

    indice = 0
    fig, ax = plt.subplots(figsize=(11, 8))

    def mostrar_imagem():
        ax.clear()
        r = resultados_ok[indice]
        imagem = carregar_imagem_exibicao(r["caminho"])

        if imagem.mode in ("L", "I", "I;16", "F"): ax.imshow(imagem, cmap="gray")
        else: ax.imshow(imagem.convert("RGB"))

        ax.set_title(
            f"{r['relativo']}\n"
            f"Real: {r['real']} | Previsto: {r['previsto']} ({r['confianca']:.2%}) | Correto: {r['correto']}\n"
            f"2ª: {r['segunda_classe']} ({r['segunda_probabilidade']:.2%}) | "
            f"Margem: {r['margem']:.2%} | Entropia: {r['entropia']:.3f}\n"
            f"Original: {r['resolucao_original']} -> Crop: {r['resolucao_crop']} -> "
            f"Modelo: {r['resolucao_modelo']} | Imagem {indice + 1}/{len(resultados_ok)}",
            fontsize=10
        )

        ax.axis("off")
        fig.suptitle("→ / N: proxima   |   ← / P: anterior   |   Q / ESC: sair", fontsize=10)
        fig.canvas.draw_idle()

    def ao_pressionar(event):
        nonlocal indice

        if event.key in ("right", "n", "d", " "):
            indice = (indice + 1) % len(resultados_ok)
            mostrar_imagem()
        elif event.key in ("left", "p", "a"):
            indice = (indice - 1) % len(resultados_ok)
            mostrar_imagem()
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", ao_pressionar)
    mostrar_imagem()
    plt.show()


def main():
    verificar_caminhos()

    arquivos, extensoes = listar_imagens()
    checkpoint, classes, dimensao_imagem, aug_modelo = carregar_checkpoint()

    rotulo_real = selecionar_rotulo_real(classes)

    print("\n" + "=" * 70)
    print("CONFIGURACAO DA PREDICAO")
    print("=" * 70)
    print(f"\nDispositivo: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    print(f"Modelo: {CAMINHO_MODELO}")
    print(f"Diretorio: {DIRETORIO_IMAGENS}")
    print(f"Rotulo real: {rotulo_real}")
    print(f"Imagens encontradas: {len(arquivos)}")
    print(f"Classes do modelo: {classes}")
    print(f"Dimensao obrigatoria: {dimensao_imagem}x{dimensao_imagem}")
    print("Canal utilizado: grayscale - 1 canal")
    print("Crop: quadrado central pela menor dimensao")
    print("Normalizacao: pixels / 255 -> 0-1")
    print(f"Treinado com augmentation: {aug_modelo}")

    model = carregar_modelo(checkpoint, classes, dimensao_imagem)

    print("\n" + "=" * 70)
    print("PREDICOES")
    print("=" * 70)

    resultados = []

    for caminho in arquivos:
        try:
            imagem, metadata = processar_imagem(caminho, dimensao_imagem)
            predicao = predizer(model, classes, imagem)

            resultado = {
                "caminho": caminho,
                "relativo": str(caminho.relative_to(DIRETORIO_IMAGENS)),
                "nome": caminho.name,
                "real": rotulo_real,
                "correto": predicao["previsto"] == rotulo_real,
                "formato": metadata["formato"],
                "modo_original": metadata["modo_original"],
                "convertido_grayscale": metadata["convertido_grayscale"],
                "resolucao_original": f"{metadata['largura_original']}x{metadata['altura_original']}",
                "resolucao_exif": f"{metadata['largura_exif']}x{metadata['altura_exif']}",
                "resolucao_crop": f"{metadata['largura_crop']}x{metadata['altura_crop']}",
                "crop_aplicado": metadata["crop_aplicado"],
                "caixa_crop": str(metadata["caixa_crop"]),
                "resolucao_modelo": f"{dimensao_imagem}x{dimensao_imagem}",
                "status": "OK",
                "erro": "",
                **predicao,
            }

            resultados.append(resultado)

            conversao = f"{metadata['modo_original']} -> L" if metadata["convertido_grayscale"] else "L"
            crop = (
                f"{resultado['resolucao_exif']} -> {resultado['resolucao_crop']}"
                if metadata["crop_aplicado"]
                else f"{resultado['resolucao_crop']} (ja quadrada)"
            )

            print(
                f"{resultado['relativo']} -> real={rotulo_real} | previsto={resultado['previsto']} "
                f"({resultado['confianca']:.2%}) | {conversao} | crop={crop} | "
                f"modelo={resultado['resolucao_modelo']} | correto={resultado['correto']}"
            )

        except Exception as erro:
            resultados.append({
                "caminho": caminho,
                "relativo": str(caminho.relative_to(DIRETORIO_IMAGENS)),
                "nome": caminho.name,
                "real": rotulo_real,
                "correto": False,
                "formato": "",
                "modo_original": "",
                "convertido_grayscale": False,
                "resolucao_original": "",
                "resolucao_exif": "",
                "resolucao_crop": "",
                "crop_aplicado": False,
                "caixa_crop": "",
                "resolucao_modelo": "",
                "previsto": "",
                "confianca": None,
                "segunda_classe": "",
                "segunda_probabilidade": None,
                "margem": None,
                "entropia": None,
                "probabilidades": {},
                "status": "ERRO",
                "erro": str(erro),
            })

            print(f"{caminho.name} -> ERRO: {erro}")

    metricas, relatorio, matriz = calcular_metricas(resultados, classes)

    salvar_csv(resultados, classes)
    salvar_tabela_html(resultados)
    salvar_resumo(resultados, classes, dimensao_imagem, aug_modelo, rotulo_real, metricas, relatorio)
    salvar_graficos(resultados, classes, matriz)

    fim_execucao = datetime.now()
    pasta_final = BASE_DIR / "resultados" / "predict_diretorio_resultados" / f"inicio_{INICIO_EXECUCAO:%Y-%m-%d_%H-%M}_fim_{fim_execucao:%Y-%m-%d_%H-%M}"

    global PASTA_RESULTADOS
    pasta_antiga = PASTA_RESULTADOS
    pasta_antiga.rename(pasta_final)
    PASTA_RESULTADOS = pasta_final

    resultados_ok = [r for r in resultados if r["status"] == "OK"]
    acertos = sum(r["correto"] for r in resultados_ok)
    crops = sum(r["crop_aplicado"] for r in resultados_ok)

    print("\n" + "=" * 70)
    print("RESULTADO FINAL")
    print("=" * 70)
    print(f"\nRotulo real: {rotulo_real}")
    print(f"Dimensao utilizada: {dimensao_imagem}x{dimensao_imagem}")
    print("Canal: grayscale - 1 canal")
    print("Crop: central pela menor dimensao")
    print("Normalizacao: 0-1")
    print(f"Imagens encontradas: {len(resultados)}")
    print(f"Predicoes realizadas: {len(resultados_ok)}")
    print(f"Imagens recortadas: {crops}")
    print(f"Imagens que ja eram quadradas: {len(resultados_ok) - crops}")
    print(f"Acertos: {acertos}")
    print(f"Erros de classificacao: {len(resultados_ok) - acertos}")
    print(f"Arquivos com erro de processamento: {len(resultados) - len(resultados_ok)}")

    if resultados_ok:
        confiancas = np.array([r["confianca"] for r in resultados_ok])
        print(f"Acuracia: {acertos / len(resultados_ok):.2%}")
        print(f"Confianca media: {confiancas.mean():.2%}")
        print(f"Predicoes abaixo de {LIMIAR_CONFIANCA:.0%}: {sum(confiancas < LIMIAR_CONFIANCA)}")

    if metricas:
        print(f"F1 weighted: {metricas['f1_weighted']:.4f}")
        print(f"Top-2 accuracy: {metricas['top2_accuracy']:.2%}")

    print(f"\nResultados salvos em:\n{PASTA_RESULTADOS}")

    print("\nControles da visualizacao:")
    print("  →, N, D ou ESPACO = proxima imagem")
    print("  ←, P ou A          = imagem anterior")
    print("  Q ou ESC           = fechar")

    visualizar_resultados(resultados)


if __name__ == "__main__":
    main()