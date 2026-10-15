"""
metrics.py

Objetivo:
- Centraliza metricas, relatorios e graficos usados na avaliacao do modelo.
- Calcula accuracy, precision, recall e F1-score nas versoes macro e weighted.
- Gera curvas de treinamento, matrizes de confusao absoluta/normalizada e arquivos TXT.
- Registra as principais configuracoes utilizadas em cada treinamento.

Usado por:
- main.py: metricas e graficos da validacao, historico e informacoes do treinamento.
- predict.py: metricas e matrizes de confusao do conjunto de teste.

Parametros ajustaveis:
- DPI_GRAFICOS: resolucao dos graficos salvos.
- FIGSIZE_HISTORICO: tamanho da figura das curvas de treinamento.
- FIGSIZE_MATRIZ: tamanho das matrizes de confusao.
- CMAP_MATRIZ: mapa de cores das matrizes.

Observacoes:
- As metricas usam zero_division=0 para evitar erros quando uma classe nao possui predicoes.
- Macro atribui o mesmo peso a todas as classes; weighted considera a quantidade de amostras.
- Este arquivo nao define parametros de treinamento como batch size, epochs ou learning rate.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# =============================================================================
# PARAMETROS AJUSTAVEIS - SAIDA DOS GRAFICOS
# =============================================================================

DPI_GRAFICOS = 300               # Resolucao dos arquivos PNG
FIGSIZE_HISTORICO = (12, 4)      # Tamanho das curvas de treino/validacao
FIGSIZE_MATRIZ = (7, 7)          # Tamanho das matrizes de confusao
CMAP_MATRIZ = "Blues"            # Mapa de cores da matriz


def visualize(history, num_epochs, output_dir=None):
    # Curvas de perda e acuracia
    epochs = range(1, num_epochs + 1)
    indices_validacao = [i for i, valor in enumerate(history["val_loss"]) if not math.isnan(valor)]
    epochs_validacao = [i + 1 for i in indices_validacao]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_HISTORICO)

    axes[0].plot(epochs, history["train_loss"], label="Treino")
    axes[0].plot(epochs_validacao, [history["val_loss"][i] for i in indices_validacao], marker="o", label="Validacao")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Perda")
    axes[0].set_title("Perda por epoca")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Treino")
    axes[1].plot(epochs_validacao, [history["val_acc"][i] for i in indices_validacao], marker="o", label="Validacao")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Acuracia")
    axes[1].set_title("Acuracia por epoca")
    axes[1].legend()

    fig.tight_layout()
    if output_dir is not None: fig.savefig(Path(output_dir) / "historico_treinamento.png", dpi=DPI_GRAFICOS)
    plt.close(fig)


def get_predictions(model, loader, device):
    # Coleta rotulos reais e previstos
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_labels, all_preds


def calculate_metrics(all_labels, all_preds, classes):
    # Metricas globais
    metricas = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "precision_weighted": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }

    # Metricas individuais por classe
    relatorio = classification_report(
        all_labels, all_preds, labels=range(len(classes)), target_names=classes, digits=4, zero_division=0
    )

    return metricas, relatorio


def show_metrics(metricas, relatorio, conjunto):
    # Exibe resultados no terminal
    print(f"\nMetricas finais - conjunto de {conjunto}")
    print(f"Acuracia:          {metricas['accuracy']:.2%}")
    print(f"Precisao macro:    {metricas['precision_macro']:.4f}")
    print(f"Recall macro:      {metricas['recall_macro']:.4f}")
    print(f"F1-score macro:    {metricas['f1_macro']:.4f}")
    print(f"Precisao weighted: {metricas['precision_weighted']:.4f}")
    print(f"Recall weighted:   {metricas['recall_weighted']:.4f}")
    print(f"F1-score weighted: {metricas['f1_weighted']:.4f}")
    print("\nMetricas por classe:")
    print(relatorio)


def save_metrics(metricas, relatorio, loss, output_dir, conjunto):
    # Salva metricas e relatorio em TXT
    caminho = Path(output_dir) / f"metricas_{conjunto}.txt"

    with open(caminho, "w", encoding="utf-8") as arquivo:
        arquivo.write(f"METRICAS FINAIS - CONJUNTO DE {conjunto.upper()}\n")
        arquivo.write("=" * 50 + "\n\n")
        arquivo.write(f"Loss:              {loss:.6f}\n")
        arquivo.write(f"Acuracia:          {metricas['accuracy']:.6f}\n")
        arquivo.write(f"Precisao macro:    {metricas['precision_macro']:.6f}\n")
        arquivo.write(f"Recall macro:      {metricas['recall_macro']:.6f}\n")
        arquivo.write(f"F1-score macro:    {metricas['f1_macro']:.6f}\n")
        arquivo.write(f"Precisao weighted: {metricas['precision_weighted']:.6f}\n")
        arquivo.write(f"Recall weighted:   {metricas['recall_weighted']:.6f}\n")
        arquivo.write(f"F1-score weighted: {metricas['f1_weighted']:.6f}\n\n")
        arquivo.write("METRICAS POR CLASSE\n")
        arquivo.write("=" * 50 + "\n\n")
        arquivo.write(relatorio)


def plot_confusion_matrices(all_labels, all_preds, classes, output_dir=None, conjunto="validacao"):
    # Matrizes absoluta e normalizada
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes)))
    cm_normalizada = confusion_matrix(all_labels, all_preds, labels=range(len(classes)), normalize="true")

    fig, ax = plt.subplots(figsize=FIGSIZE_MATRIZ)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(
        ax=ax, cmap=CMAP_MATRIZ, colorbar=False, values_format="d"
    )
    ax.set_title(f"Matriz de confusao - {conjunto}")
    fig.tight_layout()
    if output_dir is not None: fig.savefig(Path(output_dir) / f"matriz_confusao_{conjunto}.png", dpi=DPI_GRAFICOS)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE_MATRIZ)
    ConfusionMatrixDisplay(confusion_matrix=cm_normalizada, display_labels=classes).plot(
        ax=ax, cmap=CMAP_MATRIZ, colorbar=False, values_format=".2f"
    )
    ax.set_title(f"Matriz de confusao normalizada - {conjunto}")
    fig.tight_layout()
    if output_dir is not None: fig.savefig(Path(output_dir) / f"matriz_confusao_normalizada_{conjunto}.png", dpi=DPI_GRAFICOS)
    plt.close(fig)


def save_training_info(
    output_dir, inicio, fim, dataset_dir, dataset_treino, dataset_validacao,
    classes, model, criterion, optimizer, batch_size, num_epochs, eval_interval, device
):
    # Registra configuracoes utilizadas no experimento
    caminho = Path(output_dir) / "informacoes_treinamento.txt"
    duracao = fim - inicio

    with open(caminho, "w", encoding="utf-8") as arquivo:
        arquivo.write("INFORMACOES DO TREINAMENTO\n")
        arquivo.write("=" * 60 + "\n\n")
        arquivo.write(f"Inicio: {inicio:%Y-%m-%d %H:%M:%S}\n")
        arquivo.write(f"Fim:    {fim:%Y-%m-%d %H:%M:%S}\n")
        arquivo.write(f"Duracao: {duracao}\n\n")

        arquivo.write("DADOS\n")
        arquivo.write("-" * 60 + "\n")
        arquivo.write(f"Dataset: {dataset_dir}\n")
        arquivo.write(f"Classes: {classes}\n")
        arquivo.write(f"Treino: {len(dataset_treino)} imagens\n")
        arquivo.write(f"Validacao: {len(dataset_validacao)} imagens\n")
        arquivo.write("Transformacoes em runtime: nenhuma\n\n")

        arquivo.write("CONFIGURACOES DO TREINAMENTO\n")
        arquivo.write("-" * 60 + "\n")
        arquivo.write(f"Batch size: {batch_size}\n")
        arquivo.write(f"Numero de epocas: {num_epochs}\n")
        arquivo.write(f"Intervalo de avaliacao: {eval_interval}\n")
        arquivo.write(f"Funcao de erro: {criterion}\n")
        arquivo.write(f"Otimizador: {optimizer.__class__.__name__}\n")
        arquivo.write(f"Learning rate: {optimizer.param_groups[0]['lr']}\n")
        arquivo.write(f"Dispositivo: {device}\n")

        if device.type == "cuda":
            arquivo.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            arquivo.write(f"CUDA: {torch.version.cuda}\n")

        arquivo.write("\nMODELO\n")
        arquivo.write("-" * 60 + "\n")
        arquivo.write(str(model))