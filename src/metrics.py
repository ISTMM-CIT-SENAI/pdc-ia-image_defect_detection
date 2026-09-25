from pathlib import Path
import math

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def visualize(history, num_epochs, output_dir=None):
    """Exibe e salva as curvas de perda e acuracia"""

    epochs = range(1, num_epochs + 1)
    indices_validacao = [i for i, valor in enumerate(history["test_loss"]) if not math.isnan(valor)]
    epochs_validacao = [i + 1 for i in indices_validacao]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Treino")
    axes[0].plot(epochs_validacao, [history["test_loss"][i] for i in indices_validacao], marker="o", label="Validacao")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Perda")
    axes[0].set_title("Perda por epoca")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Treino")
    axes[1].plot(epochs_validacao, [history["test_acc"][i] for i in indices_validacao], marker="o", label="Validacao")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Acuracia")
    axes[1].set_title("Acuracia por epoca")
    axes[1].legend()

    fig.tight_layout()

    if output_dir is not None:
        fig.savefig(Path(output_dir) / "historico_treinamento.png", dpi=300)

    plt.close(fig)


def get_predictions(model, loader, device):
    """Retorna os rotulos reais e previstos"""

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_labels, all_preds


def calculate_metrics(all_labels, all_preds, classes):
    """Calcula as metricas finais de classificacao"""

    metricas = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "precision_weighted": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }

    relatorio = classification_report(
        all_labels, all_preds,
        labels=range(len(classes)),
        target_names=classes,
        digits=4,
        zero_division=0
    )

    return metricas, relatorio


def show_metrics(metricas, relatorio):
    """Exibe as metricas finais no terminal"""

    print("\nMetricas finais - conjunto de teste")
    print(f"Acuracia:          {metricas['accuracy']:.2%}")
    print(f"Precisao macro:    {metricas['precision_macro']:.4f}")
    print(f"Recall macro:      {metricas['recall_macro']:.4f}")
    print(f"F1-score macro:    {metricas['f1_macro']:.4f}")
    print(f"Precisao weighted: {metricas['precision_weighted']:.4f}")
    print(f"Recall weighted:   {metricas['recall_weighted']:.4f}")
    print(f"F1-score weighted: {metricas['f1_weighted']:.4f}")
    print("\nMetricas por classe:")
    print(relatorio)


def save_metrics(metricas, relatorio, test_loss, output_dir):
    """Salva as metricas finais em arquivo TXT"""

    caminho = Path(output_dir) / "metricas_teste.txt"

    with open(caminho, "w", encoding="utf-8") as arquivo:
        arquivo.write("METRICAS FINAIS - CONJUNTO DE TESTE\n")
        arquivo.write("=" * 50 + "\n\n")

        arquivo.write(f"Loss:              {test_loss:.6f}\n")
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


def plot_confusion_matrices(all_labels, all_preds, classes, output_dir=None):
    """Exibe e salva as matrizes de confusao absoluta e normalizada"""

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes)))
    cm_normalizada = confusion_matrix(all_labels, all_preds, labels=range(len(classes)), normalize="true")

    fig, ax = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Matriz de confusao - conjunto de teste")
    fig.tight_layout()

    if output_dir is not None:
        fig.savefig(Path(output_dir) / "matriz_confusao.png", dpi=300)

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay(confusion_matrix=cm_normalizada, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f")
    ax.set_title("Matriz de confusao normalizada - conjunto de teste")
    fig.tight_layout()

    if output_dir is not None:
        fig.savefig(Path(output_dir) / "matriz_confusao_normalizada.png", dpi=300)

    plt.close(fig)


def save_training_info(output_dir, inicio, fim, dataset_dir, dataset_treino, dataset_validacao, dataset_teste,
                       classes, transform_treino, transform_validacao, model, criterion, optimizer,
                       batch_size, num_epochs, eval_interval, device):
    """Salva as configuracoes utilizadas no experimento"""

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
        arquivo.write(f"Teste: {len(dataset_teste)} imagens\n\n")

        arquivo.write("TRANSFORMACOES DE TREINO\n")
        arquivo.write("-" * 60 + "\n")
        arquivo.write(str(transform_treino) + "\n\n")

        arquivo.write("TRANSFORMACOES DE VALIDACAO/TESTE\n")
        arquivo.write("-" * 60 + "\n")
        arquivo.write(str(transform_validacao) + "\n\n")

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