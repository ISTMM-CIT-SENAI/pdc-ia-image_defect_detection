"""
metrics.py

Objetivo:
- Reunir metricas e visualizacoes genericas reutilizaveis entre treinamento
  e teste.
- Salvar historico de treinamento.
- Calcular accuracy, precision, recall e F1-score.
- Gerar relatorio por classe e matrizes de confusao absoluta e normalizada.

Observacoes:
- Este modulo nao conhece datasets, modelos ou caminhos fixos do projeto.
- As funcoes recebem dados e destinos explicitamente.
- As metricas de classificacao usam zero_division=0 para manter comportamento
  definido quando uma classe nao possui predicoes ou suporte no conjunto.
"""

from pathlib import Path
import csv
import math

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


# =============================================================================
# HISTORICO DE TREINAMENTO
# =============================================================================

CAMPOS_HISTORICO = (
    "epoca",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
)


def salvar_historico_csv(historico, caminho):
    """
    Salva o historico de treinamento em CSV.

    Parameters
    ----------
    historico : iterable[dict]
        Registros contendo os campos definidos em CAMPOS_HISTORICO.

    caminho : str ou Path
        Arquivo CSV de destino.
    """

    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=CAMPOS_HISTORICO)
        escritor.writeheader()

        for registro in historico:
            escritor.writerow(
                {
                    campo: registro.get(campo)
                    for campo in CAMPOS_HISTORICO
                }
            )


def plotar_historico(historico, caminho):
    """
    Gera curvas de loss e accuracy para treino e validacao.

    Valores NaN de validacao sao ignorados visualmente. Isso permite executar
    validacao a cada N epocas sem alterar o comprimento do historico.
    """

    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    epocas = [registro["epoca"] for registro in historico]
    train_loss = [registro["train_loss"] for registro in historico]
    train_acc = [registro["train_acc"] for registro in historico]

    epocas_val = []
    val_loss = []
    val_acc = []

    # Mantem somente epocas em que a validacao foi realmente executada.
    for registro in historico:
        perda = registro["val_loss"]
        acuracia = registro["val_acc"]

        if not math.isnan(perda) and not math.isnan(acuracia):
            epocas_val.append(registro["epoca"])
            val_loss.append(perda)
            val_acc.append(acuracia)

    figura, eixos = plt.subplots(1, 2, figsize=(12, 5))

    eixos[0].plot(epocas, train_loss, label="train")
    if epocas_val:
        eixos[0].plot(epocas_val, val_loss, label="val")
    eixos[0].set_title("Loss")
    eixos[0].set_xlabel("Epoca")
    eixos[0].set_ylabel("Loss")
    eixos[0].grid(True, alpha=0.25)
    eixos[0].legend()

    eixos[1].plot(epocas, train_acc, label="train")
    if epocas_val:
        eixos[1].plot(epocas_val, val_acc, label="val")
    eixos[1].set_title("Accuracy")
    eixos[1].set_xlabel("Epoca")
    eixos[1].set_ylabel("Accuracy")
    eixos[1].set_ylim(0.0, 1.0)
    eixos[1].grid(True, alpha=0.25)
    eixos[1].legend()

    figura.tight_layout()
    figura.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close(figura)


# =============================================================================
# METRICAS DE CLASSIFICACAO
# =============================================================================

def calcular_metricas_classificacao(rotulos_reais, predicoes, classes):
    """
    Calcula metricas globais e relatorio detalhado por classe.

    Parameters
    ----------
    rotulos_reais : sequence[int]
        Indices reais das classes.

    predicoes : sequence[int]
        Indices previstos pelo modelo.

    classes : sequence[str]
        Nomes das classes na mesma ordem utilizada pelo modelo.

    Returns
    -------
    tuple[dict, str]
        Dicionario de metricas globais e texto do classification_report.
    """

    # A lista explicita de indices mantem todas as classes no relatorio mesmo
    # quando alguma delas nao aparece no conjunto avaliado.
    indices = list(range(len(classes)))

    metricas = {
        "accuracy": accuracy_score(rotulos_reais, predicoes),
        "balanced_accuracy": balanced_accuracy_score(rotulos_reais, predicoes),
        "mcc": matthews_corrcoef(rotulos_reais, predicoes),
        "cohen_kappa": cohen_kappa_score(rotulos_reais, predicoes),
        "precision_macro": precision_score(
            rotulos_reais,
            predicoes,
            labels=indices,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            rotulos_reais,
            predicoes,
            labels=indices,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            rotulos_reais,
            predicoes,
            labels=indices,
            average="macro",
            zero_division=0,
        ),
        "precision_weighted": precision_score(
            rotulos_reais,
            predicoes,
            labels=indices,
            average="weighted",
            zero_division=0,
        ),
        "recall_weighted": recall_score(
            rotulos_reais,
            predicoes,
            labels=indices,
            average="weighted",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            rotulos_reais,
            predicoes,
            labels=indices,
            average="weighted",
            zero_division=0,
        ),
    }

    relatorio = classification_report(
        rotulos_reais,
        predicoes,
        labels=indices,
        target_names=list(classes),
        digits=4,
        zero_division=0,
    )

    return metricas, relatorio


def salvar_metricas_classificacao(metricas, relatorio, loss, caminho, titulo="TESTE"):
    """
    Salva loss, metricas globais e relatorio por classe em TXT.

    Parameters
    ----------
    metricas : dict
        Resultado de calcular_metricas_classificacao().

    relatorio : str
        Relatorio textual por classe.

    loss : float
        Loss media obtida no conjunto avaliado.

    caminho : str ou Path
        Arquivo TXT de destino.

    titulo : str
        Identificacao textual do conjunto avaliado.
    """

    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    linhas = [
        f"METRICAS FINAIS - {titulo.upper()}",
        "=" * 60,
        "",
        f"Loss:               {float(loss):.6f}",
        f"Acuracia:           {metricas['accuracy']:.6f}",
        f"Acuracia balanceada:{metricas['balanced_accuracy']:>10.6f}",
        f"MCC:                {metricas['mcc']:.6f}",
        f"Cohen kappa:        {metricas['cohen_kappa']:.6f}",
        f"Precisao macro:     {metricas['precision_macro']:.6f}",
        f"Recall macro:       {metricas['recall_macro']:.6f}",
        f"F1-score macro:     {metricas['f1_macro']:.6f}",
        f"Precisao weighted:  {metricas['precision_weighted']:.6f}",
        f"Recall weighted:    {metricas['recall_weighted']:.6f}",
        f"F1-score weighted:  {metricas['f1_weighted']:.6f}",
        "",
        "METRICAS POR CLASSE",
        "=" * 60,
        "",
        relatorio,
    ]

    caminho.write_text("\n".join(linhas), encoding="utf-8")


def _salvar_matriz_csv(matriz, classes, caminho, formato):
    """Salva uma matriz quadrada com nomes de classes em CSV."""

    caminho = Path(caminho)
    with caminho.open("w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.writer(arquivo, delimiter=";")
        escritor.writerow(["real/predito", *classes])

        for classe, linha in zip(classes, matriz):
            escritor.writerow(
                [
                    classe,
                    *(formato(valor) for valor in linha),
                ]
            )


def plotar_matrizes_confusao(rotulos_reais, predicoes, classes, pasta_saida, prefixo="teste"):
    """
    Gera e salva matrizes de confusao absoluta e normalizada em PNG e CSV.

    O prefixo identifica o conjunto avaliado, por exemplo:
    ``teste``, ``treino_final`` ou ``validacao_final``.
    """

    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    indices = list(range(len(classes)))
    titulo = str(prefixo).replace("_", " ").strip().title()

    matriz = confusion_matrix(
        rotulos_reais,
        predicoes,
        labels=indices,
    )

    matriz_normalizada = confusion_matrix(
        rotulos_reais,
        predicoes,
        labels=indices,
        normalize="true",
    )

    _salvar_matriz_csv(
        matriz,
        classes,
        pasta_saida / f"matriz_confusao_{prefixo}.csv",
        lambda valor: str(int(valor)),
    )

    _salvar_matriz_csv(
        matriz_normalizada,
        classes,
        pasta_saida / f"matriz_confusao_normalizada_{prefixo}.csv",
        lambda valor: f"{float(valor):.6f}",
    )

    figura, eixo = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=list(classes),
    ).plot(
        ax=eixo,
        colorbar=False,
        values_format="d",
    )
    eixo.set_title(f"Matriz de confusao - {titulo}")
    figura.tight_layout()
    figura.savefig(
        pasta_saida / f"matriz_confusao_{prefixo}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figura)

    figura, eixo = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(
        confusion_matrix=matriz_normalizada,
        display_labels=list(classes),
    ).plot(
        ax=eixo,
        colorbar=False,
        values_format=".2f",
    )
    eixo.set_title(f"Matriz de confusao normalizada - {titulo}")
    figura.tight_layout()
    figura.savefig(
        pasta_saida / f"matriz_confusao_normalizada_{prefixo}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figura)

    return matriz, matriz_normalizada
