"""
utils.py

Objetivo:
- Reune funcoes auxiliares simples utilizadas pelo projeto.
- mostrar_arvore(): exibe resumidamente a estrutura de uma pasta de dataset.
- desnormalizar(): prepara um tensor de imagem para exibicao com matplotlib.

Usado por:
- main.py: utiliza desnormalizar() para exibir imagens durante o treinamento.
- mostrar_arvore() pode ser utilizado em scripts de diagnostico ou verificacao do dataset.

Parametros:
- root: pasta cuja estrutura sera exibida.
- max_arquivos_por_pasta: quantidade maxima de nomes mostrados por pasta; padrao = 3.
- tensor_img: tensor da imagem que sera convertido para NumPy para exibicao.

Observacoes:
- As imagens atuais ja estao na escala 0-1; por isso desnormalizar() apenas remove dimensoes,
  transfere o tensor para CPU e limita os valores ao intervalo 0-1.
- O nome desnormalizar e mantido para preservar a nomenclatura usada pelo projeto.
"""

from pathlib import Path

import numpy as np


def mostrar_arvore(root, max_arquivos_por_pasta=3):
    # max_arquivos_por_pasta pode ser ajustado na chamada da funcao
    root = Path(root)

    for pasta_classe in sorted(root.iterdir()):
        if not pasta_classe.is_dir(): continue

        arquivos = sorted(pasta_classe.glob("*"))
        print(f"{pasta_classe.name}/  ({len(arquivos)} arquivos)")

        for arquivo in arquivos[:max_arquivos_por_pasta]: print(f"    {arquivo.name}")

        if len(arquivos) > max_arquivos_por_pasta:
            print(f"    ... (+{len(arquivos) - max_arquivos_por_pasta} arquivos)")


def desnormalizar(tensor_img):
    # Converte o tensor para formato adequado a exibicao
    img = tensor_img.detach().cpu().squeeze(0).numpy()
    return np.clip(img, 0, 1)