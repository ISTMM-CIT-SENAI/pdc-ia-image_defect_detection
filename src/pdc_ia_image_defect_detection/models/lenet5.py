"""
lenet5.py

Objetivo:
- Definir exclusivamente a arquitetura LeNet5 utilizada pelo projeto.
- Receber imagens grayscale no formato (N,1,H,W).
- Manter a dimensao da primeira camada totalmente conectada independente da
  resolucao espacial usando nn.LazyLinear.

Observacoes:
- Este arquivo nao conhece datasets, caminhos, treinamento ou checkpoints.
- A quantidade de classes e recebida no construtor.
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

    Notes
    -----
    A primeira camada Linear utiliza LazyLinear para inferir automaticamente a
    quantidade de atributos produzida pelas camadas convolucionais. Assim, a
    arquitetura nao precisa manter um valor fixo dependente da resolucao da
    imagem.
    """

    def __init__(self, num_classes=4):
        super().__init__()

        # Extrator convolucional para imagens grayscale com um unico canal.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=5, stride=5),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=5, stride=5),
        )

        # Classificador final. LazyLinear calcula in_features no primeiro forward.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, int(num_classes)),
        )

    def forward(self, x):
        """Executa extracao de atributos e classificacao."""
        x = self.features(x)
        return self.classifier(x)
