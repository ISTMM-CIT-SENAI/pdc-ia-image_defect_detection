"""
lenet5.py

Objetivo:
- Define a arquitetura LeNet5 utilizada para classificacao das imagens.
- Recebe imagens grayscale com 1 canal e retorna os logits das classes.
- Usa LazyLinear para adaptar automaticamente a primeira camada totalmente conectada ao tamanho espacial gerado pelas convolucoes.

Usado por:
- main.py: cria o modelo para treinamento.
- predict.py e predict_imagens.py: recriam a mesma arquitetura para carregar o checkpoint.

Parametro:
- num_classes: quantidade de classes de saida; normalmente definida automaticamente a partir do dataset/checkpoint.

Observacoes:
- A entrada deve possuir formato (batch, 1, H, W).
- A arquitetura usa Tanh e Average Pooling, seguindo a estrutura escolhida para a LeNet5 do projeto.
- Alterar esta arquitetura torna checkpoints anteriores potencialmente incompativeis.
"""

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Extracao de caracteristicas: 1 canal -> 6 -> 16
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=5, stride=5),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=5, stride=5),
        )

        # Classificacao final
        self.classifier = nn.Sequential(
            nn.LazyLinear(120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)