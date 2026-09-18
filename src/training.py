import torch
import matplotlib.pyplot as plt

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()  # coloca o modelo em modo de treino (ativa dropout, batchnorm, etc., se houver)

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # 1. Zera os gradientes acumulados
        optimizer.zero_grad()

        # 2. Forward pass
        outputs = model(images)

        # 3. Calcula a perda
        loss = criterion(outputs, labels)

        # 4. Backward pass (calcula os gradientes)
        loss.backward()

        # 5. Atualiza os pesos
        optimizer.step()

        # Estatísticas para acompanhar o treinamento
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def visualize(history, num_epochs):

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs_range = range(1, num_epochs + 1)

    axes[0].plot(epochs_range, history["train_loss"], label="Treino")
    axes[0].plot(epochs_range, history["test_loss"], label="Teste")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Perda")
    axes[0].set_title("Perda por época")
    axes[0].legend()

    axes[1].plot(epochs_range, history["train_acc"], label="Treino")
    axes[1].plot(epochs_range, history["test_acc"], label="Teste")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia")
    axes[1].set_title("Acurácia por época")
    axes[1].legend()

    plt.tight_layout()
    plt.show()