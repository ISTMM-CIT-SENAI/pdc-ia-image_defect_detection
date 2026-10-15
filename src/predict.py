"""
predict.py

Objetivo:
- Avalia um modelo treinado no conjunto test reservado pelo split.
- Carrega o checkpoint, verifica sua compatibilidade com o dataset e calcula as metricas finais.
- Gera CSV, tabela HTML, graficos, matrizes de confusao, resumo e exemplos de predicoes.
- Exibe nas amostras as imagens originais correspondentes aos arquivos NPY do conjunto test.

Usa:
- data/images_dataset.py: carrega o conjunto test.
- models/lenet5.py: recria a arquitetura da rede.
- training.py: calcula loss e acuracia.
- metrics.py: metricas, relatorio e matrizes de confusao.
- utils/nomenclatura.py: localiza a imagem original pelo codigo MATERIAL_CP_RAC.

Executar depois de:
- main.py, utilizando um checkpoint de treinamento concluido.

Parametros ajustaveis:
- PASTA_MODELO: treinamento que sera avaliado.
- NUM_IMAGENS: quantidade de exemplos mostrados/salvos.
- BATCH_SIZE: tamanho dos lotes durante a avaliacao.
- COLUNAS_AMOSTRAS: quantidade de imagens por linha no grafico de exemplos.
- DPI_GRAFICOS: resolucao dos graficos salvos.
- EXTENSOES_ORIGINAIS: formatos pesquisados no dataset original.

Observacoes:
- Este script utiliza apenas dataset_processado_split/test.
- O conjunto test nao participa do treinamento nem da validacao.
- A resolucao e obtida automaticamente do checkpoint e conferida com o nome da pasta.
- Pastas de modelos treinados com ou sem augmentation sao reconhecidas automaticamente.
- O augmentation utilizado no treinamento nao altera o conjunto test.
"""

from pathlib import Path
from datetime import datetime
import csv
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from pdc_ia_image_defect_detection.data.images_dataset import DatasetDeImagens
from pdc_ia_image_defect_detection.models.lenet5 import LeNet5
from pdc_ia_image_defect_detection.utils.nomenclatura import MATERIAIS, CONDICOES, interpretar_nome
from training import evaluate
from metrics import get_predictions, calculate_metrics, show_metrics, save_metrics, plot_confusion_matrices


# =============================================================================
# PARAMETROS AJUSTAVEIS
# =============================================================================

# Modelo que sera avaliado
PASTA_MODELO = Path(__file__).resolve().parent.parent / "resultados" / "resultados_treino" / "treino_400_aug_inicio_2026-10-06_11-46_fim_2026-10-06_11-52"

NUM_IMAGENS = 10               # Quantidade de exemplos de predicao
BATCH_SIZE = 32                # Lote usado somente durante a avaliacao
COLUNAS_AMOSTRAS = 5           # Imagens por linha no grafico de exemplos
DPI_GRAFICOS = 300             # Resolucao dos PNG salvos

EXTENSOES_ORIGINAIS = (".png", ".jpg", ".jpeg", ".bmp")  # Formatos procurados no dataset original


# Caminhos fixos do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "dataset_processado_split" / "test"
DATASET_ORIGINAL = BASE_DIR / "dataset_atual"
CAMINHO_MODELO = PASTA_MODELO / "modelo_treinado.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Pasta desta execucao
INICIO_EXECUCAO = datetime.now()
PASTA_RESULTADOS = BASE_DIR / "resultados" / "predict_resultados" / f"inicio_{INICIO_EXECUCAO:%Y-%m-%d_%H-%M}_em_andamento"
PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)


def obter_info_pasta():
    # Reconhece treino_DIMENSAO_... e treino_DIMENSAO_aug_...
    correspondencia = re.match(r"^treino_(\d+)(_aug)?_", PASTA_MODELO.name)

    if correspondencia is None:
        raise ValueError(f"Nao foi possivel identificar dimensao/augmentation pelo nome da pasta:\n{PASTA_MODELO.name}")

    return int(correspondencia.group(1)), correspondencia.group(2) == "_aug"


def buscar_imagem_original(caminho_npy):
    # Localiza a imagem original usando MATERIAL_CP_RAC
    dados = interpretar_nome(caminho_npy)
    pasta_material = next((pasta for pasta, codigo in MATERIAIS.items() if codigo == dados["material"]), None)

    if pasta_material is None: raise ValueError(f"Material desconhecido em: {caminho_npy.name}")

    pastas_condicao = [pasta for pasta, (_, rac) in CONDICOES.items() if rac == dados["rac"]]
    if not pastas_condicao: raise ValueError(f"RAC desconhecido em: {caminho_npy.name}")

    for pasta_condicao in pastas_condicao:
        pasta_original = DATASET_ORIGINAL / pasta_material / pasta_condicao

        for extensao in EXTENSOES_ORIGINAIS:
            caminho = pasta_original / f"{caminho_npy.stem}{extensao}"
            if caminho.exists(): return caminho

    raise FileNotFoundError(f"Imagem original nao encontrada para: {caminho_npy.name}")


def verificar_caminhos():
    if not TEST_DIR.exists(): raise FileNotFoundError(f"Pasta de teste nao encontrada:\n{TEST_DIR}")
    if not DATASET_ORIGINAL.exists(): raise FileNotFoundError(f"Dataset original nao encontrado:\n{DATASET_ORIGINAL}")
    if not PASTA_MODELO.exists(): raise FileNotFoundError(f"Pasta do modelo nao encontrada:\n{PASTA_MODELO}")
    if not CAMINHO_MODELO.exists(): raise FileNotFoundError(f"Modelo nao encontrado:\n{CAMINHO_MODELO}")


def carregar_checkpoint():
    checkpoint = torch.load(CAMINHO_MODELO, map_location=device)
    campos_necessarios = {"model_state_dict", "classes", "dimensao_imagem"}
    campos_ausentes = campos_necessarios - checkpoint.keys()

    if campos_ausentes: raise KeyError(f"Campos ausentes no checkpoint: {sorted(campos_ausentes)}")

    classes = checkpoint["classes"]
    dimensao_checkpoint = int(checkpoint["dimensao_imagem"])
    dimensao_pasta, augmentation_pasta = obter_info_pasta()

    if not classes: raise ValueError("O checkpoint nao possui classes validas.")

    if dimensao_checkpoint != dimensao_pasta:
        raise ValueError(
            "Dimensao inconsistente entre pasta e checkpoint:\n"
            f"  Pasta      : {dimensao_pasta}x{dimensao_pasta}\n"
            f"  Checkpoint : {dimensao_checkpoint}x{dimensao_checkpoint}"
        )

    # Confere augmentation quando essa informacao existir no checkpoint
    if "usar_augmentation" in checkpoint and bool(checkpoint["usar_augmentation"]) != augmentation_pasta:
        raise ValueError(
            "Augmentation inconsistente entre pasta e checkpoint:\n"
            f"  Pasta      : {augmentation_pasta}\n"
            f"  Checkpoint : {bool(checkpoint['usar_augmentation'])}"
        )

    return checkpoint, classes, dimensao_checkpoint, augmentation_pasta


def verificar_dataset(dataset_teste, classes, dimensao_imagem):
    if len(dataset_teste) == 0: raise ValueError("O conjunto de teste esta vazio.")

    if dataset_teste.classes != classes:
        raise ValueError(
            "As classes do conjunto de teste nao correspondem as classes do modelo.\n"
            f"Modelo: {classes}\n"
            f"Teste : {dataset_teste.classes}"
        )

    contagem = {classe: 0 for classe in classes}
    for _, rotulo in dataset_teste.amostras: contagem[classes[rotulo]] += 1

    classes_vazias = [classe for classe, quantidade in contagem.items() if quantidade == 0]
    if classes_vazias: raise ValueError(f"Classes sem imagens no conjunto de teste: {classes_vazias}")

    # Valida uma amostra contra a configuracao do checkpoint
    imagem, _ = dataset_teste[0]
    shape_esperado = (1, dimensao_imagem, dimensao_imagem)

    if tuple(imagem.shape) != shape_esperado:
        raise ValueError(f"Dimensao do dataset incompativel com o modelo:\n  Dataset : {tuple(imagem.shape)}\n  Modelo  : {shape_esperado}")

    if imagem.dtype != torch.float32: raise TypeError(f"Tipo inesperado: {imagem.dtype}. Esperado: torch.float32.")
    if not torch.isfinite(imagem).all(): raise ValueError("A imagem de teste possui valores NaN ou infinitos.")
    if imagem.min().item() < 0.0 or imagem.max().item() > 1.0:
        raise ValueError(f"Pixels fora do intervalo 0-1: {imagem.min().item():.6f} a {imagem.max().item():.6f}")

    print("\nDistribuicao do conjunto de teste:")
    for classe in classes: print(f"  {classe}: {contagem[classe]} imagens")


def carregar_modelo(checkpoint, classes, dataset_teste):
    model = LeNet5(num_classes=len(classes)).to(device)
    imagem, _ = dataset_teste[0]

    # Inicializa LazyLinear antes de carregar o checkpoint
    with torch.no_grad(): model(imagem.unsqueeze(0).to(device))

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as erro:
        raise RuntimeError("O checkpoint nao e compativel com a arquitetura atual da LeNet5.") from erro

    model.eval()
    return model


def criar_resultados(dataset_teste, all_labels, all_preds, classes):
    if len(dataset_teste) != len(all_labels) or len(all_labels) != len(all_preds):
        raise RuntimeError("Quantidade inconsistente entre arquivos, rotulos e predicoes.")

    resultados = []

    for i, ((caminho, rotulo_dataset), rotulo_real, predicao) in enumerate(zip(dataset_teste.amostras, all_labels, all_preds)):
        if rotulo_dataset != rotulo_real: raise RuntimeError(f"Inconsistencia de rotulo na amostra {i}: {caminho.name}")

        classe_real, classe_prevista = classes[rotulo_real], classes[predicao]

        resultados.append({
            "arquivo": caminho,
            "nome": caminho.stem,
            "real": classe_real,
            "previsto": classe_prevista,
            "acerto": classe_real == classe_prevista,
        })

    return resultados


def salvar_csv(resultados):
    # Tabela completa das predicoes
    with open(PASTA_RESULTADOS / "predicoes.csv", "w", newline="", encoding="utf-8-sig") as arquivo:
        writer = csv.writer(arquivo, delimiter=";")
        writer.writerow(["Imagem", "Rotulo real", "Rotulo predito", "Resultado"])

        for resultado in resultados:
            writer.writerow([
                resultado["nome"], resultado["real"], resultado["previsto"],
                "Acerto" if resultado["acerto"] else "Erro",
            ])


def salvar_tabela_html(resultados):
    linhas = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='UTF-8'>",
        "<title>Predicoes do modelo</title>",
        "<style>",
        "body{font-family:Arial;margin:30px}",
        "table{border-collapse:collapse;width:100%}",
        "th,td{border:1px solid #ccc;padding:8px;text-align:left}",
        "th{background:#eee}",
        ".acerto{background:#d4edda}",
        ".erro{background:#f8d7da}",
        "</style></head><body>",
        "<h2>Predicoes do conjunto de teste</h2>",
        "<table>",
        "<tr><th>Imagem</th><th>Rotulo real</th><th>Rotulo predito</th><th>Resultado</th></tr>",
    ]

    for resultado in resultados:
        classe = "acerto" if resultado["acerto"] else "erro"
        situacao = "Acerto" if resultado["acerto"] else "Erro"
        linhas.append(
            f"<tr class='{classe}'><td>{resultado['nome']}</td><td>{resultado['real']}</td>"
            f"<td>{resultado['previsto']}</td><td>{situacao}</td></tr>"
        )

    linhas.extend(["</table>", "</body></html>"])
    (PASTA_RESULTADOS / "tabela_predicoes.html").write_text("\n".join(linhas), encoding="utf-8")


def salvar_grafico_total(resultados):
    acertos = sum(resultado["acerto"] for resultado in resultados)
    erros = len(resultados) - acertos

    fig, ax = plt.subplots(figsize=(7, 5))
    barras = ax.bar(["Acertos", "Erros"], [acertos, erros])

    ax.set_ylabel("Quantidade de imagens")
    ax.set_title("Acertos e erros - conjunto de teste")
    ax.bar_label(barras)

    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / "acertos_erros_totais.png", dpi=DPI_GRAFICOS)
    plt.close(fig)


def salvar_grafico_classes(resultados, classes):
    acertos, erros = [], []

    for classe in classes:
        resultados_classe = [resultado for resultado in resultados if resultado["real"] == classe]
        acertos_classe = sum(resultado["acerto"] for resultado in resultados_classe)
        acertos.append(acertos_classe)
        erros.append(len(resultados_classe) - acertos_classe)

    x, largura = np.arange(len(classes)), 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    barras_acertos = ax.bar(x - largura / 2, acertos, largura, label="Acertos")
    barras_erros = ax.bar(x + largura / 2, erros, largura, label="Erros")

    ax.set_xlabel("Classes")
    ax.set_ylabel("Quantidade de imagens")
    ax.set_title("Acertos e erros por classe - conjunto de teste")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.bar_label(barras_acertos)
    ax.bar_label(barras_erros)

    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / "acertos_erros_por_classe.png", dpi=DPI_GRAFICOS)
    plt.close(fig)


def criar_amostras(resultados):
    # Sorteia exemplos e recupera as imagens originais
    quantidade = min(NUM_IMAGENS, len(resultados))
    indices = np.random.choice(len(resultados), quantidade, replace=False)
    selecionados = [resultados[i] for i in indices]

    colunas = min(COLUNAS_AMOSTRAS, quantidade)
    linhas = math.ceil(quantidade / colunas)

    fig, axes = plt.subplots(linhas, colunas, figsize=(4 * colunas, 4.5 * linhas))
    axes = np.atleast_1d(axes).ravel()

    for ax, resultado in zip(axes, selecionados):
        caminho_original = buscar_imagem_original(resultado["arquivo"])

        with Image.open(caminho_original) as imagem:
            imagem_exibicao = imagem.copy()

        if imagem_exibicao.mode in ("L", "I", "I;16", "F"): ax.imshow(imagem_exibicao, cmap="gray", vmin=0, vmax=255)
        else: ax.imshow(imagem_exibicao.convert("RGB"))

        ax.set_title(f"Real: {resultado['real']}\nPrevisto: {resultado['previsto']}", fontsize=11)
        ax.axis("off")

    for ax in axes[quantidade:]: ax.axis("off")

    fig.suptitle("Predicoes do modelo - conjunto de teste")
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / "amostras_predicoes.png", dpi=DPI_GRAFICOS)

    return fig


def salvar_resumo(resultados, classes, test_loss, metricas, dimensao_imagem, augmentation_modelo):
    acertos = sum(resultado["acerto"] for resultado in resultados)
    erros = len(resultados) - acertos

    linhas = [
        "RESUMO DAS PREDICOES",
        "=" * 50, "",
        f"Modelo: {CAMINHO_MODELO}",
        f"Conjunto: {TEST_DIR}",
        f"Dimensao do modelo: {dimensao_imagem}x{dimensao_imagem}",
        f"Modelo treinado com augmentation: {augmentation_modelo}",
        f"Total de imagens: {len(resultados)}",
        f"Acertos: {acertos}",
        f"Erros: {erros}",
        f"Loss: {test_loss:.6f}",
        f"Acuracia: {metricas['accuracy']:.2%}",
        f"Precisao macro: {metricas['precision_macro']:.4f}",
        f"Recall macro: {metricas['recall_macro']:.4f}",
        f"F1-score macro: {metricas['f1_macro']:.4f}",
        "", "RESULTADOS POR CLASSE", "-" * 50,
    ]

    for classe in classes:
        resultados_classe = [resultado for resultado in resultados if resultado["real"] == classe]
        acertos_classe = sum(resultado["acerto"] for resultado in resultados_classe)
        erros_classe = len(resultados_classe) - acertos_classe
        acuracia_classe = acertos_classe / len(resultados_classe) if resultados_classe else 0

        linhas.append(
            f"{classe}: total={len(resultados_classe)} | acertos={acertos_classe} | "
            f"erros={erros_classe} | acuracia={acuracia_classe:.2%}"
        )

    (PASTA_RESULTADOS / "resumo_predicoes.txt").write_text("\n".join(linhas), encoding="utf-8")


# =============================================================================
# AVALIACAO
# =============================================================================

# Verificacoes
verificar_caminhos()
checkpoint, classes, DIMENSAO_IMAGEM, AUGMENTATION_MODELO = carregar_checkpoint()

dataset_teste = DatasetDeImagens(root=TEST_DIR)
verificar_dataset(dataset_teste, classes, DIMENSAO_IMAGEM)

test_loader = DataLoader(dataset_teste, batch_size=BATCH_SIZE, shuffle=False)


# Informacoes
print(f"\nUsando dispositivo: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

print(f"Modelo: {CAMINHO_MODELO}")
print(f"Dimensao do modelo: {DIMENSAO_IMAGEM}x{DIMENSAO_IMAGEM}")
print(f"Treinado com augmentation: {AUGMENTATION_MODELO}")
print(f"Teste: {len(dataset_teste)} imagens")
print(f"Batch size: {BATCH_SIZE}")


# Modelo
model = carregar_modelo(checkpoint, classes, dataset_teste)


# Avaliacao do conjunto test
criterion = nn.CrossEntropyLoss()
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
all_labels, all_preds = get_predictions(model, test_loader, device)
metricas, relatorio = calculate_metrics(all_labels, all_preds, classes)

if abs(test_acc - metricas["accuracy"]) > 1e-10:
    raise RuntimeError(f"Inconsistencia entre as acuracias: evaluate={test_acc:.6f} | metrics={metricas['accuracy']:.6f}")

resultados = criar_resultados(dataset_teste, all_labels, all_preds, classes)


# Metricas
print("\nAvaliacao final - teste")
print(f"Perda teste: {test_loss:.4f}")
print(f"Acuracia teste: {test_acc:.2%}")

show_metrics(metricas, relatorio, "teste")
save_metrics(metricas, relatorio, test_loss, PASTA_RESULTADOS, "teste")
plot_confusion_matrices(all_labels, all_preds, classes, PASTA_RESULTADOS, "teste")


# Arquivos de resultados
salvar_csv(resultados)
salvar_tabela_html(resultados)
salvar_grafico_total(resultados)
salvar_grafico_classes(resultados, classes)
salvar_resumo(resultados, classes, test_loss, metricas, DIMENSAO_IMAGEM, AUGMENTATION_MODELO)
fig_amostras = criar_amostras(resultados)


# Finalizacao
FIM_EXECUCAO = datetime.now()
PASTA_FINAL = BASE_DIR / "resultados" / "predict_resultados" / f"inicio_{INICIO_EXECUCAO:%Y-%m-%d_%H-%M}_fim_{FIM_EXECUCAO:%Y-%m-%d_%H-%M}"
PASTA_RESULTADOS.rename(PASTA_FINAL)
PASTA_RESULTADOS = PASTA_FINAL


# Resumo final
acertos = sum(resultado["acerto"] for resultado in resultados)
erros = len(resultados) - acertos

print("\nPredicao concluida")
print(f"Dimensao: {DIMENSAO_IMAGEM}x{DIMENSAO_IMAGEM}")
print(f"Treinado com augmentation: {AUGMENTATION_MODELO}")
print(f"Total: {len(resultados)}")
print(f"Acertos: {acertos}")
print(f"Erros: {erros}")
print(f"Acuracia: {metricas['accuracy']:.2%}")
print(f"F1-score macro: {metricas['f1_macro']:.4f}")
print(f"\nResultados salvos em:\n{PASTA_RESULTADOS}")

plt.show()
plt.close(fig_amostras)