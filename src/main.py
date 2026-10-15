"""
main.py

Objetivo:
- Treina a LeNet5 utilizando os conjuntos fixos train e val.
- Permite utilizar augmentation seletivamente por classe.
- Mostra a composicao real do conjunto de treino antes do treinamento.
- Registra no checkpoint quais classes utilizaram augmentation.
- Mantem o conjunto test completamente fora do treinamento.

Fluxo:
- dataset_processado_split/train -> treinamento
- dataset_processado_split/val   -> validacao
- dataset_processado_split/test  -> NAO utilizado aqui

Configuracao de augmentation:
- Geracao dos arquivos: utils/augmention.py -> CONFIG_AUGMENTATION
- Uso no treinamento: utils/augmention.py -> USAR_AUGMENTATION_POR_CLASSE

Observacoes:
- Arquivos originais sempre entram no treino.
- Augmentations entram somente nas classes habilitadas.
- Val nunca utiliza augmentation.
- Test permanece reservado aos scripts de predicao/avaliacao final.
"""

from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pdc_ia_image_defect_detection.data.images_dataset import DatasetDeImagens
from pdc_ia_image_defect_detection.models.lenet5 import LeNet5
from pdc_ia_image_defect_detection.utils.augmention import (
    CONFIG_AUGMENTATION,
    USAR_AUGMENTATION_POR_CLASSE,
    transformar_imagens,
)
from pdc_ia_image_defect_detection.utils.utils import desnormalizar

from training import train_one_epoch, evaluate

from metrics import (
    visualize,
    get_predictions,
    calculate_metrics,
    show_metrics,
    save_metrics,
    plot_confusion_matrices,
    save_training_info,
)


# =============================================================================
# PARAMETROS AJUSTAVEIS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset_processado_split"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

BATCH_SIZE = 512
NUM_EPOCHS = 100
EVAL_INTERVAL = 5

LEARNING_RATE = 1e-3

SALVAR_PREVIA_LOTE = True
NUM_IMAGENS_PREVIA = 8


# =============================================================================
# DISPOSITIVO
# =============================================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# HORARIO
# =============================================================================

INICIO_EXECUCAO = datetime.now()


# =============================================================================
# VERIFICACOES
# =============================================================================

def verificar_caminhos():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset split nao encontrado:\n"
            f"{DATASET_DIR}"
        )

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Pasta de treino nao encontrada:\n"
            f"{TRAIN_DIR}"
        )

    if not VAL_DIR.exists():
        raise FileNotFoundError(
            f"Pasta de validacao nao encontrada:\n"
            f"{VAL_DIR}"
        )


def verificar_classes(
    dataset_treino,
    dataset_validacao
):
    if (
        dataset_treino.class_to_idx
        != dataset_validacao.class_to_idx
    ):
        raise ValueError(
            "Mapeamento de classes diferente entre "
            "treino e validacao.\n"
            f"Treino: {dataset_treino.class_to_idx}\n"
            f"Validacao: {dataset_validacao.class_to_idx}"
        )


def verificar_lote(imagens):
    if imagens.ndim != 4:
        raise ValueError(
            f"Lote deveria possuir 4 dimensoes, "
            f"mas recebeu shape {tuple(imagens.shape)}."
        )

    if imagens.shape[1] != 1:
        raise ValueError(
            f"Esperado 1 canal, recebido "
            f"{imagens.shape[1]}."
        )

    if imagens.shape[2] != imagens.shape[3]:
        raise ValueError(
            f"Imagens nao sao quadradas: "
            f"{imagens.shape[2]}x{imagens.shape[3]}."
        )

    if imagens.dtype != torch.float32:
        raise TypeError(
            f"Dtype inesperado: {imagens.dtype}. "
            f"Esperado: torch.float32."
        )

    if not torch.isfinite(imagens).all():
        raise ValueError(
            "O lote possui valores NaN ou Inf."
        )

    return int(
        imagens.shape[-1]
    )


# =============================================================================
# AUGMENTATION
# =============================================================================

def obter_resumo_augmentation(
    dataset_treino
):
    return dataset_treino.resumo_augmentation()


def augmentation_efetivamente_usado(
    resumo
):
    return any(
        dados["augmentation_usados"] > 0
        for dados in resumo.values()
    )


def mostrar_resumo_augmentation(
    resumo
):
    print("\n" + "=" * 80)
    print("COMPOSICAO DO CONJUNTO DE TREINO")
    print("=" * 80)

    total_originais = 0
    total_aug = 0
    total_ignorados = 0

    for classe, dados in resumo.items():
        originais = dados[
            "originais"
        ]

        aug_usados = dados[
            "augmentation_usados"
        ]

        aug_ignorados = dados[
            "augmentation_ignorados"
        ]

        total = dados[
            "total_usado"
        ]

        total_originais += originais
        total_aug += aug_usados
        total_ignorados += aug_ignorados

        configurado = (
            USAR_AUGMENTATION_POR_CLASSE[
                classe
            ]
        )

        print(
            f"\n{classe}"
        )

        print(
            f"  Usar augmentation     : "
            f"{configurado}"
        )

        print(
            f"  Originais             : "
            f"{originais}"
        )

        print(
            f"  Augmentation usados   : "
            f"{aug_usados}"
        )

        print(
            f"  Augmentation ignorados: "
            f"{aug_ignorados}"
        )

        print(
            f"  Total usado no treino : "
            f"{total}"
        )

    print("\n" + "-" * 80)

    print(
        f"Originais usados         : "
        f"{total_originais}"
    )

    print(
        f"Augmentations usados     : "
        f"{total_aug}"
    )

    print(
        f"Augmentations ignorados  : "
        f"{total_ignorados}"
    )

    print(
        f"Total efetivo de treino  : "
        f"{total_originais + total_aug}"
    )


# =============================================================================
# RESULTADOS
# =============================================================================

def criar_pasta_resultados(
    dimensao_imagem,
    usar_augmentation
):
    sufixo_aug = (
        "_aug"
        if usar_augmentation
        else ""
    )

    nome = (
        f"treino_{dimensao_imagem}"
        f"{sufixo_aug}"
        f"_inicio_{INICIO_EXECUCAO:%Y-%m-%d_%H-%M}"
        f"_em_andamento"
    )

    pasta = (
        BASE_DIR
        / "resultados"
        / "resultados_treino"
        / nome
    )

    pasta.mkdir(
        parents=True,
        exist_ok=False
    )

    return pasta


def renomear_pasta_resultados(
    pasta_atual,
    dimensao_imagem,
    usar_augmentation,
    fim_execucao
):
    sufixo_aug = (
        "_aug"
        if usar_augmentation
        else ""
    )

    nome_final = (
        f"treino_{dimensao_imagem}"
        f"{sufixo_aug}"
        f"_inicio_{INICIO_EXECUCAO:%Y-%m-%d_%H-%M}"
        f"_fim_{fim_execucao:%Y-%m-%d_%H-%M}"
    )

    destino = (
        pasta_atual.parent
        / nome_final
    )

    if destino.exists():
        raise FileExistsError(
            f"Pasta final ja existe:\n"
            f"{destino}"
        )

    pasta_atual.rename(
        destino
    )

    return destino


# =============================================================================
# PREVIA DO LOTE
# =============================================================================

def converter_para_plot(
    tensor
):
    imagem = desnormalizar(
        tensor
    )

    if torch.is_tensor(
        imagem
    ):
        imagem = (
            imagem
            .detach()
            .cpu()
            .numpy()
        )

    imagem = np.asarray(
        imagem
    )

    return np.squeeze(
        imagem
    )


def salvar_previa_lote(
    imagens,
    rotulos,
    classes,
    pasta_resultados
):
    quantidade = min(
        NUM_IMAGENS_PREVIA,
        len(imagens)
    )

    if quantidade <= 0:
        return

    colunas = 4

    linhas = int(
        np.ceil(
            quantidade / colunas
        )
    )

    fig, axes = plt.subplots(
        linhas,
        colunas,
        figsize=(
            12,
            3 * linhas
        )
    )

    axes = np.array(
        axes
    ).reshape(-1)

    for indice in range(
        len(axes)
    ):
        axes[indice].axis(
            "off"
        )

    for indice in range(
        quantidade
    ):
        imagem = converter_para_plot(
            imagens[indice]
        )

        rotulo = int(
            rotulos[indice].item()
        )

        axes[indice].imshow(
            imagem,
            cmap="gray"
        )

        axes[indice].set_title(
            classes[
                rotulo
            ]
        )

        axes[indice].axis(
            "off"
        )

    fig.tight_layout()

    fig.savefig(
        pasta_resultados
        / "output_lote_treino.png",
        dpi=200
    )

    plt.close(
        fig
    )


# =============================================================================
# REGISTRO DO AUGMENTATION
# =============================================================================

def anexar_info_augmentation(
    pasta_resultados,
    resumo_augmentation,
    usar_augmentation
):
    caminho = (
        pasta_resultados
        / "informacoes_treinamento.txt"
    )

    with open(
        caminho,
        "a",
        encoding="utf-8"
    ) as arquivo:

        arquivo.write(
            "\n\n"
            + "=" * 70
            + "\n"
        )

        arquivo.write(
            "AUGMENTATION POR CLASSE\n"
        )

        arquivo.write(
            "=" * 70
            + "\n"
        )

        arquivo.write(
            f"\nAugmentation usado no treinamento: "
            f"{usar_augmentation}\n"
        )

        arquivo.write(
            "\nCONFIGURACAO DE USO\n"
        )

        for classe in sorted(
            USAR_AUGMENTATION_POR_CLASSE
        ):
            arquivo.write(
                f"{classe}: "
                f"{USAR_AUGMENTATION_POR_CLASSE[classe]}\n"
            )

        arquivo.write(
            "\nCOMPOSICAO EFETIVA DO TREINO\n"
        )

        for classe in sorted(
            resumo_augmentation
        ):
            dados = resumo_augmentation[
                classe
            ]

            arquivo.write(
                f"\n{classe}\n"
            )

            arquivo.write(
                f"  originais: "
                f"{dados['originais']}\n"
            )

            arquivo.write(
                f"  augmentation usados: "
                f"{dados['augmentation_usados']}\n"
            )

            arquivo.write(
                f"  augmentation ignorados: "
                f"{dados['augmentation_ignorados']}\n"
            )

            arquivo.write(
                f"  total usado: "
                f"{dados['total_usado']}\n"
            )

        arquivo.write(
            "\nCONFIGURACAO DE GERACAO DISPONIVEL\n"
        )

        for classe in sorted(
            CONFIG_AUGMENTATION
        ):
            config = CONFIG_AUGMENTATION[
                classe
            ]

            arquivo.write(
                f"\n{classe}\n"
            )

            for nome, valor in config.items():
                arquivo.write(
                    f"  {nome}: {valor}\n"
                )


# =============================================================================
# MAIN
# =============================================================================

def main():
    verificar_caminhos()

    print("\n" + "=" * 80)
    print("TREINAMENTO LENET5")
    print("=" * 80)

    print(
        f"\nDataset:\n"
        f"{DATASET_DIR}"
    )

    print(
        f"\nDispositivo: "
        f"{device}"
    )

    if device.type == "cuda":
        print(
            f"GPU        : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA       : "
            f"{torch.version.cuda}"
        )

    # -------------------------------------------------------------------------
    # Transformacoes
    # -------------------------------------------------------------------------

    transform_treino = transformar_imagens(
        treino=True
    )

    transform_validacao = transformar_imagens(
        treino=False
    )

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------

    dataset_treino = DatasetDeImagens(
        TRAIN_DIR,
        transform=transform_treino
    )

    dataset_validacao = DatasetDeImagens(
        VAL_DIR,
        transform=transform_validacao
    )

    verificar_classes(
        dataset_treino,
        dataset_validacao
    )

    classes = dataset_treino.classes

    # -------------------------------------------------------------------------
    # Augmentation efetivamente carregado
    # -------------------------------------------------------------------------

    resumo_augmentation = obter_resumo_augmentation(
        dataset_treino
    )

    usar_augmentation = augmentation_efetivamente_usado(
        resumo_augmentation
    )

    mostrar_resumo_augmentation(
        resumo_augmentation
    )

    # -------------------------------------------------------------------------
    # Quantidades
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("DATASETS")
    print("=" * 80)

    print(
        f"\nClasses   : "
        f"{classes}"
    )

    print(
        f"Treino    : "
        f"{len(dataset_treino)}"
    )

    print(
        f"Validacao : "
        f"{len(dataset_validacao)}"
    )

    print(
        "\nO conjunto test nao e utilizado pelo main.py."
    )

    # -------------------------------------------------------------------------
    # DataLoaders
    # -------------------------------------------------------------------------

    train_loader = DataLoader(
        dataset_treino,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset_validacao,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # -------------------------------------------------------------------------
    # Primeiro lote
    # -------------------------------------------------------------------------

    imagens_lote, rotulos_lote = next(
        iter(
            train_loader
        )
    )

    dimensao_imagem = verificar_lote(
        imagens_lote
    )

    print("\n" + "=" * 80)
    print("VERIFICACAO DO LOTE")
    print("=" * 80)

    print(
        f"\nShape      : "
        f"{tuple(imagens_lote.shape)}"
    )

    print(
        f"Dtype      : "
        f"{imagens_lote.dtype}"
    )

    print(
        f"Minimo     : "
        f"{imagens_lote.min().item():.6f}"
    )

    print(
        f"Maximo     : "
        f"{imagens_lote.max().item():.6f}"
    )

    print(
        f"Dimensao   : "
        f"{dimensao_imagem}x{dimensao_imagem}"
    )

    # -------------------------------------------------------------------------
    # Pasta de resultados
    # -------------------------------------------------------------------------

    pasta_resultados = criar_pasta_resultados(
        dimensao_imagem,
        usar_augmentation
    )

    print(
        f"\nResultados:\n"
        f"{pasta_resultados}"
    )

    if SALVAR_PREVIA_LOTE:
        salvar_previa_lote(
            imagens_lote,
            rotulos_lote,
            classes,
            pasta_resultados
        )

    # -------------------------------------------------------------------------
    # Modelo
    # -------------------------------------------------------------------------

    model = LeNet5(
        num_classes=len(
            classes
        )
    ).to(
        device
    )

    # Inicializa LazyLinear
    with torch.no_grad():
        saida_teste = model(
            imagens_lote[:1].to(
                device
            )
        )

    print("\n" + "=" * 80)
    print("MODELO")
    print("=" * 80)

    print(
        f"\nEntrada exemplo : "
        f"{tuple(imagens_lote[:1].shape)}"
    )

    print(
        f"Saida exemplo   : "
        f"{tuple(saida_teste.shape)}"
    )

    print(
        f"LazyLinear      : "
        f"{model.classifier[0].in_features} entradas"
    )

    print(
        f"\n{model}"
    )

    # -------------------------------------------------------------------------
    # Loss e otimizador
    # -------------------------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -------------------------------------------------------------------------
    # Historico
    # -------------------------------------------------------------------------

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    # -------------------------------------------------------------------------
    # Treinamento
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TREINAMENTO")
    print("=" * 80)

    print(
        f"\nBatch size         : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epocas             : "
        f"{NUM_EPOCHS}"
    )

    print(
        f"Intervalo validacao: "
        f"{EVAL_INTERVAL}"
    )

    print(
        f"Learning rate      : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Augmentation       : "
        f"{usar_augmentation}"
    )

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        history[
            "train_loss"
        ].append(
            train_loss
        )

        history[
            "train_acc"
        ].append(
            train_acc
        )

        validar = (
            epoch == 1
            or epoch % EVAL_INTERVAL == 0
            or epoch == NUM_EPOCHS
        )

        if validar:
            val_loss, val_acc = evaluate(
                model,
                val_loader,
                criterion,
                device
            )

            history[
                "val_loss"
            ].append(
                val_loss
            )

            history[
                "val_acc"
            ].append(
                val_acc
            )

            print(
                f"Epoca "
                f"{epoch:03d}/{NUM_EPOCHS:03d} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        else:
            history[
                "val_loss"
            ].append(
                np.nan
            )

            history[
                "val_acc"
            ].append(
                np.nan
            )

            print(
                f"Epoca "
                f"{epoch:03d}/{NUM_EPOCHS:03d} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Train Acc: {train_acc:.4f}"
            )

    # -------------------------------------------------------------------------
    # Validacao final
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("VALIDACAO FINAL")
    print("=" * 80)

    val_loss, val_acc = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    all_labels, all_preds = get_predictions(
        model,
        val_loader,
        device
    )

    metricas, relatorio = calculate_metrics(
        all_labels,
        all_preds,
        classes
    )

    show_metrics(
        metricas,
        relatorio,
        "validacao"
    )

    # -------------------------------------------------------------------------
    # Graficos e metricas
    # -------------------------------------------------------------------------

    visualize(
        history,
        NUM_EPOCHS,
        pasta_resultados
    )

    plot_confusion_matrices(
        all_labels,
        all_preds,
        classes,
        pasta_resultados,
        "validacao"
    )

    save_metrics(
        metricas,
        relatorio,
        val_loss,
        pasta_resultados,
        "validacao"
    )

    # -------------------------------------------------------------------------
    # Checkpoint
    # -------------------------------------------------------------------------

    checkpoint = {
        "model_state_dict": model.state_dict(),

        "classes": classes,

        "dimensao_imagem": dimensao_imagem,

        # Mantido para compatibilidade com predict_imagens.py
        # e demais scripts existentes.
        "usar_augmentation": usar_augmentation,

        # Nova informacao detalhada por classe.
        "usar_augmentation_por_classe": dict(
            USAR_AUGMENTATION_POR_CLASSE
        ),

        "config_augmentation_geracao": {
            classe: dict(
                config
            )
            for classe, config
            in CONFIG_AUGMENTATION.items()
        },

        "resumo_augmentation_treino": resumo_augmentation,

        "quantidade_treino": len(
            dataset_treino
        ),

        "quantidade_validacao": len(
            dataset_validacao
        ),

        "batch_size": BATCH_SIZE,

        "num_epochs": NUM_EPOCHS,

        "eval_interval": EVAL_INTERVAL,

        "learning_rate": LEARNING_RATE,
    }

    caminho_modelo = (
        pasta_resultados
        / "modelo_treinado.pt"
    )

    torch.save(
        checkpoint,
        caminho_modelo
    )

    # -------------------------------------------------------------------------
    # Informacoes do treinamento
    # -------------------------------------------------------------------------

    FIM_EXECUCAO = datetime.now()

    save_training_info(
        pasta_resultados,
        INICIO_EXECUCAO,
        FIM_EXECUCAO,
        DATASET_DIR,
        dataset_treino,
        dataset_validacao,
        classes,
        model,
        criterion,
        optimizer,
        BATCH_SIZE,
        NUM_EPOCHS,
        EVAL_INTERVAL,
        device
    )

    # Acrescenta as novas informacoes de augmentation sem exigir
    # alteracao imediata no metrics.py.
    anexar_info_augmentation(
        pasta_resultados,
        resumo_augmentation,
        usar_augmentation
    )

    # -------------------------------------------------------------------------
    # Finalizacao
    # -------------------------------------------------------------------------

    pasta_final = renomear_pasta_resultados(
        pasta_resultados,
        dimensao_imagem,
        usar_augmentation,
        FIM_EXECUCAO
    )

    print("\n" + "=" * 80)
    print("TREINAMENTO FINALIZADO")
    print("=" * 80)

    print(
        f"\nValidacao Loss : "
        f"{val_loss:.6f}"
    )

    print(
        f"Validacao Acc  : "
        f"{val_acc:.4f}"
    )

    print(
        f"\nModelo salvo em:\n"
        f"{pasta_final / 'modelo_treinado.pt'}"
    )

    print(
        f"\nResultados:\n"
        f"{pasta_final}"
    )

    print(
        "\nO conjunto test permaneceu sem uso durante "
        "todo o treinamento."
    )


if __name__ == "__main__":
    main()