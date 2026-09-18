from pathlib import Path
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

def mostrar_arvore(root, max_arquivos_por_pasta=3):
    root = Path(root)
    for pasta_classe in sorted(root.iterdir()):
        if pasta_classe.is_dir():
            arquivos = sorted(pasta_classe.glob("*"))
            print(f"{pasta_classe.name}/  ({len(arquivos)} arquivos)")
            for arq in arquivos[:max_arquivos_por_pasta]:
                print(f"    {arq.name}")
            if len(arquivos) > max_arquivos_por_pasta:
                print(f"    ... (+{len(arquivos) - max_arquivos_por_pasta} arquivos)")

def transformar_imagens():
    transform_treino = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    transform_validacao = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    return transform_treino, transform_validacao

def desnormalizar(tensor_img):
    """Desfaz a normalização só para exibição com matplotlib"""
    img = tensor_img.clone().permute(1, 2, 0).numpy()
    img = img * 0.5 + 0.5  # inverte a normalização (mean=0.5, std=0.5)
    return np.clip(img, 0, 1)



