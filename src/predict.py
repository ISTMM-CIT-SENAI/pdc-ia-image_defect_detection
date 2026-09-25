from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from pdc_ia_image_defect_detection.models.lenet5 import LeNet5


# Caminhos
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "dataset_processado_split" / "test"
DATASET_ORIGINAL = BASE_DIR / "dataset_falhas"

PASTA_MODELO = BASE_DIR / "resultados" / "inicio_2026-09-25_15-25-32_fim_2026-09-25_15-43-43"
CAMINHO_MODELO = PASTA_MODELO / "modelo_treinado.pt"

# Configuracoes
NUM_IMAGENS = 10
EXTENSOES_ORIGINAIS = (".png", ".jpg", ".jpeg", ".bmp")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def buscar_imagem_original(caminho_npy):
    """Localiza a imagem original correspondente ao arquivo NPY"""

    classe = caminho_npy.parent.name
    pasta_original = DATASET_ORIGINAL / classe

    for extensao in EXTENSOES_ORIGINAIS:
        caminho = pasta_original / f"{caminho_npy.stem}{extensao}"

        if caminho.exists():
            return caminho

    raise FileNotFoundError(f"Imagem original nao encontrada para: {caminho_npy.name}")


# Carrega checkpoint
checkpoint = torch.load(CAMINHO_MODELO, map_location=device)
classes = checkpoint["classes"]

# Localiza arquivos do conjunto de teste
arquivos_teste = sorted(TEST_DIR.glob("*/*.npy"))

if len(arquivos_teste) < NUM_IMAGENS:
    raise ValueError(f"O conjunto de teste possui apenas {len(arquivos_teste)} arquivos.")

# Inicializa o modelo usando uma amostra real
amostra = torch.from_numpy(np.load(arquivos_teste[0], allow_pickle=False)).unsqueeze(0).to(device)

model = LeNet5(num_classes=len(classes)).to(device)

with torch.no_grad():
    model(amostra)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Sorteia 10 arquivos do conjunto de teste
indices = np.random.choice(len(arquivos_teste), NUM_IMAGENS, replace=False)
arquivos_sorteados = [arquivos_teste[i] for i in indices]

# Predicoes
fig, axes = plt.subplots(2, 5, figsize=(20, 9))

with torch.no_grad():
    for ax, caminho_npy in zip(axes.flat, arquivos_sorteados):
        imagem = torch.from_numpy(np.load(caminho_npy, allow_pickle=False)).unsqueeze(0).to(device)

        output = model(imagem)
        predicao = output.argmax(dim=1).item()
        classe_prevista = classes[predicao]

        caminho_original = buscar_imagem_original(caminho_npy)

        with Image.open(caminho_original) as imagem_original:
            ax.imshow(imagem_original, cmap="gray")

        classe_real = caminho_npy.parent.name

        ax.set_title(
            f"Real: {classe_real}\n"
            f"Previsto: {classe_prevista}",
            fontsize=11
        )
        ax.axis("off")

plt.suptitle("Predicoes do modelo - conjunto de teste")
plt.tight_layout()
plt.show()