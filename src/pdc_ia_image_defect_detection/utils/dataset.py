"""
dataset.py

Objetivo:
- Disponibilizar um Dataset PyTorch generico para arquivos NPY do projeto.
- Receber uma lista de amostras ja selecionadas por outras etapas.
- Carregar cada NPY utilizando a validacao central de utils.npy.
- Converter a imagem para tensor float32 e o rotulo para torch.long.
- Permitir opcionalmente retornar o caminho do arquivo para uso futuro em
  avaliacao e predicao.

Formato esperado das amostras:
{
    "caminho": Path(...),
    "indice": 0,
}

Observacoes:
- Este modulo nao procura datasets nem decide quais arquivos usar.
- Balanceamento, selecao de origem e divisao train/val pertencem ao processo
  que monta a lista de amostras.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from pdc_ia_image_defect_detection.utils.npy import carregar_imagem_npy


class DatasetNPY(Dataset):
    """
    Dataset generico para imagens NPY previamente processadas.

    Parameters
    ----------
    amostras : iterable[dict]
        Registros contendo pelo menos as chaves ``caminho`` e ``indice``.

    retornar_caminho : bool, default=False
        Quando True, __getitem__ retorna tambem o caminho textual da imagem.
        Essa opcao sera util posteriormente no script de predicao.
    """

    def __init__(self, amostras, retornar_caminho=False):
        # Materializa a sequencia para que __len__ e __getitem__ sejam estaveis.
        self.amostras = list(amostras)
        self.retornar_caminho = bool(retornar_caminho)

        # Valida a estrutura minima de todos os registros antes do primeiro lote.
        for indice, amostra in enumerate(self.amostras):
            if not isinstance(amostra, dict):
                raise TypeError(
                    f"Amostra {indice} deve ser dict, recebido {type(amostra).__name__}."
                )

            if "caminho" not in amostra or "indice" not in amostra:
                raise ValueError(
                    f"Amostra {indice} deve conter as chaves 'caminho' e 'indice'."
                )

            caminho = Path(amostra["caminho"])
            if caminho.suffix.lower() != ".npy":
                raise ValueError(
                    f"Amostra {indice} nao aponta para NPY: {caminho}"
                )

            if int(amostra["indice"]) < 0:
                raise ValueError(
                    f"Indice de classe invalido na amostra {indice}: {amostra['indice']}"
                )

    def __len__(self):
        """Retorna a quantidade de amostras disponiveis."""
        return len(self.amostras)

    def __getitem__(self, indice):
        """
        Carrega uma amostra, valida o NPY e converte para tensores PyTorch.

        Returns
        -------
        tuple
            ``(imagem, rotulo)`` ou ``(imagem, rotulo, caminho)``.
        """

        registro = self.amostras[indice]
        caminho = Path(registro["caminho"])

        # utils.npy garante float32, shape (1,H,W), valores finitos e [0,1].
        array = carregar_imagem_npy(caminho)

        # torch.from_numpy evita copia desnecessaria quando o array e contiguo.
        imagem = torch.from_numpy(array)
        rotulo = torch.tensor(int(registro["indice"]), dtype=torch.long)

        if self.retornar_caminho:
            return imagem, rotulo, str(caminho)

        return imagem, rotulo
