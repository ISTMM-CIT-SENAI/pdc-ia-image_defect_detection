"""
vgg16_imagenet.py

Objetivo:
- Utilizar a VGG16 oficial do Torchvision com pesos do ImageNet-1K.
- Receber NPY grayscale float32 em [0,1] e resolucao 224x224.
- Replicar o canal e aplicar a normalizacao RGB do ImageNet no forward.
- Adaptar apenas a ultima camada para as classes do projeto.
- Permitir fine-tuning completo, parcial ou apenas da camada de saida.

Regras:
- Nao realiza crop, resize ou divisao por 255.
- Pesos ImageNet sao carregados somente em novos treinamentos.
- Checkpoints sao reconstruidos sem download e recebem seus pesos salvos.
"""

import torch
import torch.nn as nn


class VGG16ImageNet(nn.Module):
    """VGG16 oficial pre-treinada, adaptada para imagens grayscale."""

    def __init__(
        self,
        num_classes=4,
        modo_fine_tuning="completo",
        carregar_pesos_imagenet=False,
    ):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        num_classes = int(num_classes)
        if num_classes <= 0:
            raise ValueError("num_classes deve ser maior que zero.")

        self.modo_fine_tuning = str(modo_fine_tuning).strip().lower()
        if self.modo_fine_tuning not in ("completo", "parcial", "classificador"):
            raise ValueError(
                "modo_fine_tuning deve ser 'completo', 'parcial' ou 'classificador'."
            )

        pesos = VGG16_Weights.IMAGENET1K_V1 if carregar_pesos_imagenet else None
        self.vgg = vgg16(weights=pesos)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)

        # Media e desvio-padrao oficiais de VGG16_Weights.IMAGENET1K_V1.
        self.register_buffer(
            "_media_imagenet",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_desvio_imagenet",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        if self.modo_fine_tuning == "classificador":
            for parametro in self.vgg.parameters():
                parametro.requires_grad_(False)
            for parametro in self.vgg.classifier[6].parameters():
                parametro.requires_grad_(True)

        elif self.modo_fine_tuning == "parcial":
            for parametro in self.vgg.features.parameters():
                parametro.requires_grad_(False)
            for camada in list(self.vgg.features.children())[24:]:
                for parametro in camada.parameters():
                    parametro.requires_grad_(True)
            # Todas as camadas densas permanecem treinaveis.

        # Em modo 'completo', todas as camadas permanecem treinaveis.

    def forward(self, x):
        """Adapta grayscale em [0,1] para a VGG16 RGB e retorna logits."""
        if x.ndim != 4 or x.shape[1] != 1 or tuple(x.shape[-2:]) != (224, 224):
            raise ValueError(
                "VGG16 ImageNet espera tensor (N,1,224,224) float32 normalizado em [0,1]. "
                f"Recebido: {tuple(x.shape)}."
            )

        # O dataset ja esta em [0,1]; nao repetir a divisao por 255.
        x = x.repeat(1, 3, 1, 1)
        x = (x - self._media_imagenet) / self._desvio_imagenet
        return self.vgg(x)
