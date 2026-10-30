"""
lenet5.py

Objetivo:
- Definir exclusivamente a arquitetura LeNet5 utilizada pelo projeto.
- Receber imagens grayscale no formato (N,1,H,W).
- Manter a dimensao da primeira camada totalmente conectada independente da
  resolucao espacial usando nn.LazyLinear.

Observacoes:
- Este arquivo nao conhece datasets, caminhos, treinamento ou checkpoints.
- Classes, kernel e neuronios possuem valores padrao compativeis com o modelo original.
- A entrada possui um unico canal porque o pre-processamento final e grayscale.
"""

import torch.nn as nn


class LeNet5(nn.Module):
    """
    Implementa a rede convolucional LeNet5 adaptada ao projeto.

    Parameters
    ----------
    num_classes : int
        Quantidade de classes de saida do classificador.

    kernel_size : int
        Tamanho comum dos kernels das convolucoes e poolings.
    neuronios_fc1, neuronios_fc2 : int
        Neuronios das duas camadas densas ocultas.

    Notes
    -----
    A primeira camada Linear utiliza LazyLinear para inferir automaticamente a
    quantidade de atributos produzida pelas camadas convolucionais. Assim, a
    arquitetura nao precisa manter um valor fixo dependente da resolucao da
    imagem.
    """

    def __init__(self, num_classes=4, kernel_size=5, neuronios_fc1=120, neuronios_fc2=84):
        super().__init__()

        kernel_size = int(kernel_size)
        neuronios_fc1 = int(neuronios_fc1)
        neuronios_fc2 = int(neuronios_fc2)
        if min(kernel_size, neuronios_fc1, neuronios_fc2) <= 0:
            raise ValueError("Kernel e neuronios devem ser maiores que zero.")

        # Extrator convolucional para imagens grayscale com um unico canal.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=kernel_size),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=kernel_size),
        )

        # Classificador final. LazyLinear calcula in_features no primeiro forward.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(neuronios_fc1),
            nn.ReLU(inplace=True),
            nn.Linear(neuronios_fc1, neuronios_fc2),
            nn.ReLU(inplace=True),
            nn.Linear(neuronios_fc2, int(num_classes)),
        )

    def forward(self, x):
        """Executa extracao de atributos e classificacao."""
        x = self.features(x)
        return self.classifier(x)
