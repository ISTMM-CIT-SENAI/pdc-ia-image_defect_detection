"""
images_dataset.py

Objetivo:
- Carrega arquivos .npy ja pre-processados para PyTorch.
- Define automaticamente as classes pelas subpastas.
- Permite incluir ou ignorar arquivos de augmentation por classe no treino.
- Garante que validacao e teste nunca utilizem augmentation.

Usado por:
- main.py para criar train_dataset, val_dataset e test_dataset.

Configuracao de augmentation:
- USAR_AUGMENTATION_POR_CLASSE fica em:
  utils/augmention.py

Observacoes:
- Arquivos originais entram sempre.
- Arquivos *_augNN.npy entram apenas no conjunto train e somente quando
  a classe correspondente estiver habilitada.
- Val e test ignoram qualquer arquivo de augmentation que exista por engano.
- Os .npy ja devem estar em float32, shape (1,H,W) e normalmente em [0,1].
"""

from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from pdc_ia_image_defect_detection.utils.augmention import (
    usar_augmentation_no_treino,
)


# =============================================================================
# PADRAO DOS ARQUIVOS AUGMENTADOS
# =============================================================================

# Reconhece, por exemplo:
# A36_23_003_aug01.npy
# A36_23_003_aug02.npy
# A36_23_003_aug_01.npy
PADRAO_ARQUIVO_AUG = re.compile(
    r"_aug_?\d+$",
    flags=re.IGNORECASE
)


def eh_augmentation(caminho):
    """Retorna True quando o arquivo possui sufixo de augmentation."""
    return bool(
        PADRAO_ARQUIVO_AUG.search(
            caminho.stem
        )
    )


class DatasetDeImagens(Dataset):
    """
    Dataset para arquivos .npy pre-processados.

    Estrutura esperada:

    dataset_processado_split/
        train/
            amassamento/
            corrosao/
            risco/
            sem_falha/

        val/
            ...

        test/
            ...

    No conjunto train:
    - originais entram sempre;
    - augmentations entram conforme USAR_AUGMENTATION_POR_CLASSE.

    Em val/test:
    - augmentations nunca entram.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Diretorio do dataset nao encontrado:\n"
                f"{self.root_dir}"
            )

        if not self.root_dir.is_dir():
            raise NotADirectoryError(
                f"O caminho nao e um diretorio:\n"
                f"{self.root_dir}"
            )

        # Nome esperado: train, val ou test
        self.conjunto = self.root_dir.name.lower()

        # Classes definidas pelas subpastas, mantendo ordem alfabetica
        self.classes = sorted([
            pasta.name
            for pasta in self.root_dir.iterdir()
            if pasta.is_dir()
        ])

        if not self.classes:
            raise FileNotFoundError(
                f"Nenhuma classe encontrada em:\n"
                f"{self.root_dir}"
            )

        self.class_to_idx = {
            classe: indice
            for indice, classe in enumerate(self.classes)
        }

        self.amostras = []

        # Contadores uteis para diagnostico no main
        self.originais_por_classe = {
            classe: 0
            for classe in self.classes
        }

        self.augmentation_por_classe = {
            classe: 0
            for classe in self.classes
        }

        self.augmentation_ignorados_por_classe = {
            classe: 0
            for classe in self.classes
        }

        self._carregar_lista_amostras()

        if not self.amostras:
            raise FileNotFoundError(
                f"Nenhum arquivo .npy valido encontrado em:\n"
                f"{self.root_dir}"
            )


    def _carregar_lista_amostras(self):
        """
        Monta a lista (caminho, indice_classe).

        Originais:
        - entram sempre.

        Augmentations:
        - train: entram apenas se a classe estiver habilitada;
        - val/test: sao sempre ignorados.
        """

        for classe in self.classes:
            pasta_classe = (
                self.root_dir
                / classe
            )

            indice_classe = self.class_to_idx[
                classe
            ]

            arquivos = sorted(
                pasta_classe.glob("*.npy")
            )

            for caminho in arquivos:
                arquivo_aug = eh_augmentation(
                    caminho
                )

                # -------------------------------------------------------------
                # ARQUIVO ORIGINAL
                # -------------------------------------------------------------

                if not arquivo_aug:
                    self.amostras.append(
                        (
                            caminho,
                            indice_classe
                        )
                    )

                    self.originais_por_classe[
                        classe
                    ] += 1

                    continue

                # -------------------------------------------------------------
                # AUGMENTATION EM VAL/TEST
                # Nunca deve entrar.
                # -------------------------------------------------------------

                if self.conjunto != "train":
                    self.augmentation_ignorados_por_classe[
                        classe
                    ] += 1

                    continue

                # -------------------------------------------------------------
                # AUGMENTATION NO TRAIN
                # Consulta configuracao individual da classe.
                # -------------------------------------------------------------

                if usar_augmentation_no_treino(
                    classe
                ):
                    self.amostras.append(
                        (
                            caminho,
                            indice_classe
                        )
                    )

                    self.augmentation_por_classe[
                        classe
                    ] += 1

                else:
                    self.augmentation_ignorados_por_classe[
                        classe
                    ] += 1


    def __len__(self):
        return len(
            self.amostras
        )


    def __getitem__(self, indice):
        caminho, rotulo = self.amostras[
            indice
        ]

        array = np.load(
            caminho,
            allow_pickle=False
        )

        # Garante float32 mesmo se algum arquivo estiver inconsistente
        if array.dtype != np.float32:
            array = array.astype(
                np.float32
            )

        tensor = torch.from_numpy(
            array
        )

        if self.transform is not None:
            tensor = self.transform(
                tensor
            )

        return tensor, rotulo


    def resumo_augmentation(self):
        """
        Retorna resumo das amostras efetivamente carregadas.

        Util para main.py registrar exatamente quais classes
        utilizaram augmentation no treinamento.
        """

        resumo = {}

        for classe in self.classes:
            originais = self.originais_por_classe[
                classe
            ]

            augmentation = self.augmentation_por_classe[
                classe
            ]

            ignorados = self.augmentation_ignorados_por_classe[
                classe
            ]

            resumo[classe] = {
                "originais": originais,
                "augmentation_usados": augmentation,
                "augmentation_ignorados": ignorados,
                "total_usado": originais + augmentation,
            }

        return resumo