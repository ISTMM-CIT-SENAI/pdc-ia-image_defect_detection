import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from pdc_ia_image_defect_detection.data.images_dataset import DatasetDeImagens
from pdc_ia_image_defect_detection.utils.utils import desnormalizar, transformar_imagens
from pdc_ia_image_defect_detection.models.lenet5 import LeNet5
from training import train_one_epoch, evaluate, visualize

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

transform_basico = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

BASE_DIR = Path(__file__).resolve().parent.parent  # ajuste conforme a profundidade
DATASET_DIR = BASE_DIR / "dataset_falhas"

for item in os.listdir(DATASET_DIR):
    print(item)

dataset_completo = datasets.ImageFolder(root=str(DATASET_DIR), transform=transform_basico)

print(f"Total de imagens: {len(dataset_completo)}")
print(f"Classes encontradas: {dataset_completo.classes}")
print(f"Mapeamento classe -> índice: {dataset_completo.class_to_idx}")

imagem, rotulo = dataset_completo[0]
print(f"Formato do tensor da imagem: {imagem.shape}")  # [canais, altura, largura]
print(f"Rótulo (índice): {rotulo} -> classe: {dataset_completo.classes[rotulo]}")

dataset_custom = DatasetDeImagens(root=str(DATASET_DIR), transform=transform_basico)

print(f"Total de imagens: {len(dataset_custom)}")
print(f"Classes: {dataset_custom.classes}")

imagem, rotulo = dataset_custom[0]
print(f"Formato da imagem: {imagem.shape}, rótulo: {rotulo} ({dataset_custom.classes[rotulo]})")

assert len(dataset_custom) == len(dataset_completo)
print("\nDataset customizado tem o mesmo tamanho que o ImageFolder ✔")

transform_treino, transform_validacao = transformar_imagens()
#
# dataset_aug = DatasetDeImagens(root=str(DATASET_DIR), transform=transform_treino)
# caminho_exemplo, rotulo_exemplo = dataset_aug.amostras[0]
#
# fig, axes = plt.subplots(1, 5, figsize=(12, 3))
# for i in range(5):
#     img_transformada, _ = dataset_aug[0]  # mesma imagem, transformação re-sorteada a cada chamada
#     axes[i].imshow(desnormalizar(img_transformada))
#     axes[i].axis("off")
# axes[2].set_title(f"5 versões de uma imagem da classe '{dataset_aug.classes[rotulo_exemplo]}' após data augmentation")
# plt.tight_layout()
# plt.show()

PROPORCAO_TREINO = 0.8

dataset_base = DatasetDeImagens(root=str(DATASET_DIR))  # sem transform ainda

n_treino = int(len(dataset_base) * PROPORCAO_TREINO)
n_validacao = len(dataset_base) - n_treino

gerador = torch.Generator().manual_seed(42)  # reprodutibilidade da divisão
indices_treino, indices_validacao = random_split(
    range(len(dataset_base)), [n_treino, n_validacao], generator=gerador
)

print(f"Exemplos de treino:    {len(indices_treino)}")
print(f"Exemplos de validação: {len(indices_validacao)}")

from torch.utils.data import Subset

dataset_treino = Subset(DatasetDeImagens(root=str(DATASET_DIR), transform=transform_treino), indices_treino.indices)
dataset_validacao = Subset(DatasetDeImagens(root=str(DATASET_DIR), transform=transform_validacao), indices_validacao.indices)

print(f"dataset_treino:    {len(dataset_treino)} imagens (com data augmentation)")
print(f"dataset_validacao: {len(dataset_validacao)} imagens (sem data augmentation)")

BATCH_SIZE = 128

train_loader = DataLoader(dataset_treino, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset_validacao, batch_size=BATCH_SIZE, shuffle=False)

imagens, rotulos = next(iter(train_loader))
print(f"Formato do lote de imagens: {imagens.shape}")  # [batch, canais, altura, largura]
print(f"Formato do lote de rótulos: {rotulos.shape}")

classes = dataset_base.classes

fig, axes = plt.subplots(2, 8, figsize=(32, 9))
for i, ax in enumerate(axes.flat):
    if i < len(imagens):
        ax.imshow(desnormalizar(imagens[i]))
        ax.set_title(classes[rotulos[i]], fontsize=9)
    ax.axis("off")
plt.suptitle("Exemplo de um lote de treino (com data augmentation)")
plt.tight_layout()
plt.savefig(str(BASE_DIR / "output_lote_treino.png"), dpi=300)
#plt.show()

#instanciacao do modelo
model = LeNet5(num_classes=len(dataset_completo.classes)).to(device)
print(model)

with torch.no_grad():
    sample_output = model(imagens.to(device))

print(f"Formato da entrada:  {imagens.shape}")
print(f"Formato da saída:    {sample_output.shape}")  # [batch, num_classes]

#funcao de erro e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

NUM_EPOCHS = 30

history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, val_loader, criterion, device)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)

    print(
        f"Época {epoch:02d}/{NUM_EPOCHS} | "
        f"Perda treino: {train_loss:.4f} | Acurácia treino: {train_acc:.2%} | "
        f"Perda teste: {test_loss:.4f} | Acurácia teste: {test_acc:.2%}"
    )

visualize(history, NUM_EPOCHS)
