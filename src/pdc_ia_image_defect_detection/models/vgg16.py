"""
vgg16.py

Objetivo:
- Definir exclusivamente a arquitetura VGG16 adaptada ao projeto.
- Receber imagens grayscale no formato (N,1,H,W).
- Retornar logits no formato (N,num_classes).

Observacoes:
- Possui 13 convolucoes, 5 max-poolings e 3 camadas lineares.
- O pooling adaptativo fixa a entrada do classificador em 512x7x7.
- Inicializa os pesos conforme a VGG16 oficial do Torchvision.
- Permite variar os neuronios densos e dropout sem alterar o extrator convolucional.
- Este arquivo nao depende de datasets, treinamento ou checkpoints.
"""

import torch.nn as nn


class VGG16(nn.Module):
    """Implementa a arquitetura VGG16 para classificacao de imagens grayscale."""

    def __init__(self, num_classes=4, neuronios_fc1=4096, neuronios_fc2=4096, dropout=0.5):
        super().__init__()
        neuronios_fc1 = int(neuronios_fc1)
        neuronios_fc2 = int(neuronios_fc2)
        dropout = float(dropout)
        if neuronios_fc1 <= 0 or neuronios_fc2 <= 0:
            raise ValueError("Neuronios das camadas densas devem ser positivos.")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout deve estar no intervalo [0, 1).")

        # Extrator convolucional da VGG16 (configuracao D).
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classificador VGG16 adaptado a quantidade de classes do projeto.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, neuronios_fc1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(neuronios_fc1, neuronios_fc2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(neuronios_fc2, int(num_classes)),
        )

        # Inicializacao de pesos utilizada pela VGG16 do Torchvision.
        for camada in self.modules():
            if isinstance(camada, nn.Conv2d):
                nn.init.kaiming_normal_(camada.weight, mode="fan_out", nonlinearity="relu")
                if camada.bias is not None:
                    nn.init.constant_(camada.bias, 0)
            elif isinstance(camada, nn.Linear):
                nn.init.normal_(camada.weight, mean=0, std=0.01)
                nn.init.constant_(camada.bias, 0)

    def forward(self, x):
        """Executa extracao de caracteristicas e classificacao."""
        x = self.features(x)
        return self.classifier(x)
