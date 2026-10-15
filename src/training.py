"""
training.py

Objetivo:
- Define as rotinas basicas de treinamento e avaliacao da rede.
- train_one_epoch(): executa uma epoca de treino com forward, backward e atualizacao dos pesos.
- evaluate(): calcula loss e acuracia sem alterar os pesos do modelo.

Usado por:
- main.py: treinamento por epoca e validacao periodica.
- predict.py: avaliacao final no conjunto test.

Parametros:
- model: modelo PyTorch utilizado.
- loader: DataLoader do conjunto avaliado.
- criterion: funcao de loss definida externamente.
- optimizer: otimizador usado somente no treinamento.
- device: CPU ou GPU.

Observacoes:
- Este arquivo nao possui parametros ajustaveis diretamente.
- Batch size, learning rate, numero de epocas e intervalo de avaliacao sao definidos em main.py.
- evaluate() utiliza torch.no_grad(), portanto nao calcula gradientes nem modifica os pesos.
"""

import torch


def train_one_epoch(model, loader, criterion, optimizer, device):
    # Treinamento: forward -> loss -> backward -> atualizacao dos pesos
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    # Avaliacao: forward sem gradientes e sem alterar os pesos
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(dim=1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total