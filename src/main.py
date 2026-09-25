import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from pdc_ia_image_defect_detection.data.images_dataset import DatasetDeImagens
from pdc_ia_image_defect_detection.utils.utils import desnormalizar
from pdc_ia_image_defect_detection.utils.augmention import transformar_imagens
from pdc_ia_image_defect_detection.models.lenet5 import LeNet5
from training import train_one_epoch, evaluate
from metrics import visualize, get_predictions, calculate_metrics, show_metrics, save_metrics, plot_confusion_matrices, save_training_info
from pathlib import Path
from datetime import datetime


# Configuracoes gerais
#torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")


# Caminhos do dataset
BASE_DIR = Path(__file__).resolve().parent.parent

INICIO_EXECUCAO = datetime.now()
PASTA_RESULTADOS = BASE_DIR / "resultados" / f"inicio_{INICIO_EXECUCAO:%Y-%m-%d_%H-%M-%S}_em_andamento"
PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

DATASET_DIR = BASE_DIR / "dataset_processado_split"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"


# Transformacoes das imagens
transform_treino, transform_validacao = transformar_imagens()


# Datasets
dataset_treino = DatasetDeImagens(root=str(TRAIN_DIR), transform=transform_treino)
dataset_validacao = DatasetDeImagens(root=str(VAL_DIR), transform=transform_validacao)
dataset_teste = DatasetDeImagens(root=str(TEST_DIR), transform=transform_validacao)

assert dataset_treino.classes == dataset_validacao.classes == dataset_teste.classes

classes = dataset_treino.classes

print(f"Classes: {classes}")
print(f"Treino: {len(dataset_treino)} imagens")
print(f"Validacao: {len(dataset_validacao)} imagens")
print(f"Teste: {len(dataset_teste)} imagens")


# DataLoaders
BATCH_SIZE = 512

train_loader = DataLoader(dataset_treino, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset_validacao, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset_teste, batch_size=BATCH_SIZE, shuffle=False)


# Verificacao de um lote
imagens, rotulos = next(iter(train_loader))

print(f"Formato do lote de imagens: {imagens.shape}")
print(f"Formato do lote de rótulos: {rotulos.shape}")
print(f"Tipo dos dados: {imagens.dtype}")
print(f"Intervalo dos pixels: {imagens.min().item():.6f} a {imagens.max().item():.6f}")


# Visualizacao de um lote de treino
fig, axes = plt.subplots(2, 8, figsize=(32, 9))

for i, ax in enumerate(axes.flat):
    if i < len(imagens):
        ax.imshow(desnormalizar(imagens[i]), cmap="gray")
        ax.set_title(classes[rotulos[i]], fontsize=9)
    ax.axis("off")

plt.suptitle("Exemplo de um lote de treino")
plt.tight_layout()
plt.savefig(PASTA_RESULTADOS / "output_lote_treino.png", dpi=300)
plt.close()


# Instanciacao do modelo
model = LeNet5(num_classes=len(classes)).to(device)
print(model)
print(f"Dispositivo do modelo: {next(model.parameters()).device}")


# Verificacao da entrada e saida do modelo
with torch.no_grad():
    sample_output = model(imagens[:1].to(device))

print(f"Entrada da primeira camada Linear: {model.classifier[0].in_features}")
print(f"Formato da entrada de teste: {imagens[:1].shape}")
print(f"Formato da saída de teste: {sample_output.shape}")


# Funcao de erro e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Configuracoes do treinamento
NUM_EPOCHS = 30
EVAL_INTERVAL = 5

history = {
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": []
}


# Treinamento
for epoch in range(1, NUM_EPOCHS + 1):

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)

    # Avalia na primeira epoca, no intervalo escolhido e na ultima epoca
    if epoch == 1 or epoch % EVAL_INTERVAL == 0 or epoch == NUM_EPOCHS:

        test_loss, test_acc = evaluate(model, val_loader, criterion, device)

        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Época {epoch:03d}/{NUM_EPOCHS} | "
            f"Perda treino: {train_loss:.4f} | Acurácia treino: {train_acc:.2%} | "
            f"Perda validação: {test_loss:.4f} | Acurácia validação: {test_acc:.2%}"
        )

    else:

        history["test_loss"].append(float("nan"))
        history["test_acc"].append(float("nan"))

        print(
            f"Época {epoch:03d}/{NUM_EPOCHS} | "
            f"Perda treino: {train_loss:.4f} | Acurácia treino: {train_acc:.2%}"
        )

# Avaliacao final no conjunto de teste
test_loss, test_acc = evaluate(model, test_loader, criterion, device)

print("\nAvaliacao final")
print(f"Perda teste: {test_loss:.4f}")
print(f"Acuracia teste: {test_acc:.2%}")


# Predicoes e metricas finais
all_labels, all_preds = get_predictions(model, test_loader, device)
metricas, relatorio = calculate_metrics(all_labels, all_preds, classes)

show_metrics(metricas, relatorio)


# Salva os resultados
visualize(history, NUM_EPOCHS, PASTA_RESULTADOS)
plot_confusion_matrices(all_labels, all_preds, classes, PASTA_RESULTADOS)
save_metrics(metricas, relatorio, test_loss, PASTA_RESULTADOS)

# Salva o modelo treinado
torch.save({
    "model_state_dict": model.state_dict(),
    "classes": classes,
}, PASTA_RESULTADOS / "modelo_treinado.pt")

# Finalizacao da execucao
FIM_EXECUCAO = datetime.now()



save_training_info(
    PASTA_RESULTADOS, INICIO_EXECUCAO, FIM_EXECUCAO, DATASET_DIR,
    dataset_treino, dataset_validacao, dataset_teste, classes,
    transform_treino, transform_validacao, model, criterion, optimizer,
    BATCH_SIZE, NUM_EPOCHS, EVAL_INTERVAL, device
)

PASTA_FINAL = BASE_DIR / "resultados" / f"inicio_{INICIO_EXECUCAO:%Y-%m-%d_%H-%M-%S}_fim_{FIM_EXECUCAO:%Y-%m-%d_%H-%M-%S}"
PASTA_RESULTADOS.rename(PASTA_FINAL)
PASTA_RESULTADOS = PASTA_FINAL

print(f"\nResultados salvos em:\n{PASTA_RESULTADOS}")