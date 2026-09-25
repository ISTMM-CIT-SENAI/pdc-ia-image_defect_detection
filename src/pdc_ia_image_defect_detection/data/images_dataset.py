from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetDeImagens(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root: caminho da pasta raiz, contendo uma subpasta por classe
            transform: transformacoes a serem aplicadas em cada imagem (opcional)
        """

        self.root = Path(root)
        self.transform = transform

        # Descobre as classes a partir dos nomes das subpastas
        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_to_idx = {nome: i for i, nome in enumerate(self.classes)}

        # Monta a lista de exemplos: (caminho_do_arquivo, indice_da_classe)
        self.amostras = []

        for nome_classe in self.classes:
            pasta_classe = self.root / nome_classe

            for caminho_img in sorted(pasta_classe.glob("*.npy")):
                self.amostras.append((caminho_img, self.class_to_idx[nome_classe]))

    def __len__(self):
        return len(self.amostras)

    def __getitem__(self, idx):
        caminho_img, rotulo = self.amostras[idx]

        # Carrega o array ja pre-processado e converte para tensor
        imagem = torch.from_numpy(np.load(caminho_img, allow_pickle=False))

        if self.transform is not None:
            imagem = self.transform(imagem)

        return imagem, rotulo