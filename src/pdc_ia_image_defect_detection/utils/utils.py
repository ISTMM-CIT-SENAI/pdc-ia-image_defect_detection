from pathlib import Path

import numpy as np

from pdc_ia_image_defect_detection.utils.augmention import APLICAR_NORMALIZACAO, MEAN, STD


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


def desnormalizar(tensor_img):
    """Prepara a imagem para exibicao"""

    img = tensor_img.detach().cpu().squeeze(0).numpy()

    if APLICAR_NORMALIZACAO:
        img = img * STD[0] + MEAN[0]

    return np.clip(img, 0, 1)