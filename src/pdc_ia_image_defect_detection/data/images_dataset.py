from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

class DatasetDeImagens(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root: caminho da pasta raiz, contendo uma subpasta por classe
            transform: transformações a serem aplicadas em cada imagem (opcional)
        """
        self.root = Path(root)
        self.transform = transform

        # Descobre as classes a partir dos nomes das subpastas (ordenadas para reprodutibilidade)
        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_to_idx = {nome: i for i, nome in enumerate(self.classes)}

        # Monta a lista de exemplos: (caminho_da_imagem, indice_da_classe)
        # Note que aqui só guardamos os CAMINHOS, não as imagens carregadas —
        # isso evita estourar a memória RAM com datasets grandes
        self.amostras = []
        extensoes_validas = (".png", ".jpg", ".jpeg", ".bmp")
        for nome_classe in self.classes:
            pasta_classe = self.root / nome_classe
            for caminho_img in pasta_classe.iterdir():
                if caminho_img.suffix.lower() in extensoes_validas:
                    self.amostras.append((caminho_img, self.class_to_idx[nome_classe]))

    def __len__(self):
        return len(self.amostras)

    def __getitem__(self, idx):
        caminho_img, rotulo = self.amostras[idx]

        # A imagem só é carregada aqui, no momento em que é solicitada
        # (carregamento "preguiçoso" / lazy loading)
        imagem = Image.open(caminho_img).convert("RGB")

        if self.transform is not None:
            imagem = self.transform(imagem)

        return imagem, rotulo