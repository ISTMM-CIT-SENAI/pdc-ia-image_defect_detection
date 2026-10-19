"""
test_class.py

Objetivo:
- Implementar toda a logica especifica da avaliacao final do modelo.
- Carregar um checkpoint criado pelo treino atual.
- Avaliar o modelo sem atualizar pesos.
- Aceitar o test do split ou uma pasta escolhida manualmente.
- Obter os rotulos reais exclusivamente pelas pastas de classe.
- Combinar test interno e externo quando ambos estiverem selecionados.
- Salvar metricas, matrizes de confusao, predicoes e resumos detalhados.
- Registrar a proporcao observada do split quando o modo split for utilizado.

Regras:
- No modo split, os NPY ja preparados sao usados sem novo pre-processamento.
- No modo pasta escolhida, imagens comuns sao preparadas em memoria antes da
  inferencia usando crop quadrado central, grayscale, resize LANCZOS, float32
  e normalizacao para [0,1].
- Test nunca recebe augmentation em runtime.
- O modelo e importado exclusivamente de models/lenet5.py.
- As classes sao conferidas contra o checkpoint.
- Pastas de classe ausentes sao permitidas, mas pastas desconhecidas geram erro.
- O conjunto test pode conter zero imagens em alguma classe.
- O relatorio de proporcao do split ignora arquivos _augNN.npy existentes em
  train, pois augmentation ocorre depois do split e nao faz parte da divisao.
"""

from collections import Counter, defaultdict
from datetime import datetime
from html import escape
from pathlib import Path
import csv
import math
import random
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pdc_ia_image_defect_detection.models.lenet5 import LeNet5
from pdc_ia_image_defect_detection.utils.metrics import (
    calcular_metricas_classificacao,
    plotar_matrizes_confusao,
    salvar_metricas_classificacao,
)
from pdc_ia_image_defect_detection.utils.imagens import (
    carregar_imagem_pil,
    converter_imagem_para_npy,
    crop_quadrado_central,
    listar_imagens,
)
from pdc_ia_image_defect_detection.utils.npy import carregar_imagem_npy
from pdc_ia_image_defect_detection.utils.nomenclatura import ROTULOS


class _DatasetTeste(Dataset):
    """
    Dataset interno da etapa de teste.

    A classe suporta dois tipos de registro:
    - npy: arquivo ja preparado pelo split, carregado sem novo processamento;
    - imagem: arquivo comum da pasta manual, preparado em memoria antes de ser
      convertido para tensor.
    """

    def __init__(self, registros, input_shape):
        """Armazena os registros e o shape esperado pelo checkpoint."""
        self.registros = list(registros)
        self.input_shape = tuple(int(valor) for valor in input_shape)

    def __len__(self):
        """Retorna a quantidade total de amostras."""
        return len(self.registros)

    @staticmethod
    def carregar_imagem_manual(caminho, input_shape):
        """
        Converte uma imagem comum para o formato numerico usado pelo modelo.

        Returns
        -------
        tuple[np.ndarray, dict]
            Array final e metadados do pre-processamento realizado.
        """

        caminho = Path(caminho)
        input_shape = tuple(int(valor) for valor in input_shape)

        imagem, _ = carregar_imagem_pil(
            caminho,
            corrigir_exif=True,
        )

        largura, altura = imagem.size

        if largura <= 0 or altura <= 0:
            raise ValueError(
                f"Resolucao invalida em {caminho.name}: {imagem.size}"
            )

        modo_entrada = imagem.mode
        resolucao_entrada = imagem.size
        aplicar_crop = largura != altura

        # O maior quadrado central possui lado igual a menor dimensao.
        if aplicar_crop:
            imagem = crop_quadrado_central(imagem)

        # input_shape usa (canal, altura, largura), enquanto o util recebe
        # tamanho_final no formato (largura, altura).
        tamanho_final = (
            int(input_shape[2]),
            int(input_shape[1]),
        )

        # O util executa grayscale, resize LANCZOS, float32, divisao por 255
        # e expansao para o formato (1,H,W).
        array = converter_imagem_para_npy(
            imagem,
            tamanho_final,
        )

        if tuple(array.shape) != input_shape:
            raise ValueError(
                f"Shape final inesperado para {caminho.name}: "
                f"{tuple(array.shape)}, esperado={input_shape}"
            )

        info = {
            "formato": caminho.suffix.lower(),
            "modo": modo_entrada,
            "resolucao": resolucao_entrada,
            "convertida_grayscale": modo_entrada != "L",
            "crop_quadrado": aplicar_crop,
        }

        return array, info

    @classmethod
    def carregar_array(cls, registro, input_shape):
        """Carrega um registro NPY ou uma imagem manual conforme seu tipo."""

        if registro["tipo"] == "npy":
            array = carregar_imagem_npy(registro["caminho"])

            if tuple(array.shape) != tuple(input_shape):
                raise ValueError(
                    f"shape={tuple(array.shape)}, esperado={tuple(input_shape)}"
                )

            return array

        if registro["tipo"] == "imagem":
            array, _ = cls.carregar_imagem_manual(
                registro["caminho"],
                input_shape,
            )
            return array

        raise ValueError(
            f"Tipo de registro desconhecido: {registro['tipo']}"
        )

    def __getitem__(self, indice):
        """Carrega uma amostra e retorna tensor, rotulo e metadados basicos."""

        registro = self.registros[indice]
        array = self.carregar_array(
            registro,
            self.input_shape,
        )

        # Garante memoria contigua antes de compartilhar o array com PyTorch.
        tensor = torch.from_numpy(np.ascontiguousarray(array))

        return (
            tensor,
            int(registro["indice"]),
            str(registro["caminho"]),
            str(registro["origem"]),
            str(registro["tipo"]),
        )


class TestProcessor:
    """Coordena selecao dos dados, pre-processamento, inferencia e relatorios."""

    # Reconhece augmentations materializados pelo augmentation_class.py.
    PADRAO_AUGMENTATION = re.compile(r"_aug_?\d+$", flags=re.IGNORECASE)

    def __init__(
        self,
        usar_test_split,
        bases_split,
        pasta_teste_escolhida,
        extensoes_imagem,
        pasta_treino,
        usar_modelo_melhor,
        pasta_resultados,
        batch_size,
        num_workers,
        usar_cuda,
        seed,
        num_amostras_grafico,
        colunas_amostras,
    ):
        """Armazena a configuracao recebida de processing/test.py."""

        self.usar_test_split = bool(usar_test_split)
        self.bases_split = dict(bases_split)
        self.pasta_teste_escolhida = Path(pasta_teste_escolhida)
        self.extensoes_imagem = tuple(
            extensao.lower() if str(extensao).startswith(".")
            else f".{str(extensao).lower()}"
            for extensao in extensoes_imagem
        )
        self.pasta_treino = Path(pasta_treino)
        self.usar_modelo_melhor = bool(usar_modelo_melhor)
        self.caminho_modelo = None
        self.pasta_resultados = Path(pasta_resultados)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.usar_cuda = bool(usar_cuda)
        self.seed = int(seed)
        self.num_amostras_grafico = int(num_amostras_grafico)
        self.colunas_amostras = int(colunas_amostras)

        # Preenchidos durante executar().
        self.device = None
        self.checkpoint = None
        self.classes = []
        self.class_to_idx = {}
        self.input_shape = None
        self.modelo = None
        self.amostras = []
        self.bases_split_ativas = {}
        self.pasta_execucao = None
        self.inicio = None
        self.estatisticas_pasta_manual = {}

    # =========================================================================
    # VALIDACAO E SELECAO DOS DADOS
    # =========================================================================

    @classmethod
    def _eh_augmentation(cls, caminho):
        """Retorna True quando o stem termina em _augNN ou _aug_NN."""
        return bool(cls.PADRAO_AUGMENTATION.search(Path(caminho).stem))

    @staticmethod
    def _formatar_counter(counter):
        """Formata um Counter para exibicao compacta e deterministica."""
        itens = sorted(counter.items(), key=lambda item: str(item[0]))
        return ", ".join(
            f"{valor}={quantidade}"
            for valor, quantidade in itens
        ) or "nenhum"

    @staticmethod
    def _formatar_resolucoes(counter):
        """Formata um Counter de resolucoes no formato LARGURAxALTURA."""
        itens = sorted(counter.items())
        return ", ".join(
            f"{largura}x{altura}={quantidade}"
            for (largura, altura), quantidade in itens
        ) or "nenhuma"

    def _validar_parametros(self):
        """Valida configuracao basica antes de criar arquivos de resultado."""

        if self.batch_size <= 0:
            raise ValueError("BATCH_SIZE deve ser maior que zero.")

        if self.num_workers < 0:
            raise ValueError("NUM_WORKERS nao pode ser negativo.")

        if self.num_amostras_grafico < 0:
            raise ValueError("NUM_AMOSTRAS_GRAFICO nao pode ser negativo.")

        if self.colunas_amostras <= 0:
            raise ValueError("COLUNAS_AMOSTRAS deve ser maior que zero.")

        # A pasta de treinamento deve existir antes de selecionar o checkpoint.
        if not self.pasta_treino.exists():
            raise FileNotFoundError(
                f"Pasta de treinamento nao encontrada:\n{self.pasta_treino}"
            )

        if not self.pasta_treino.is_dir():
            raise NotADirectoryError(
                f"PASTA_TREINO nao e um diretorio:\n{self.pasta_treino}"
            )

        # A flag escolhe explicitamente entre o melhor checkpoint e o final.
        nome_modelo = (
            "modelo_melhor.pt"
            if self.usar_modelo_melhor
            else "modelo_final.pt"
        )
        self.caminho_modelo = self.pasta_treino / nome_modelo

        if not self.caminho_modelo.exists() or not self.caminho_modelo.is_file():
            raise FileNotFoundError(
                f"Modelo selecionado nao encontrado:\n{self.caminho_modelo}"
            )

        # A pasta manual depende de uma lista explicita de formatos aceitos.
        if not self.extensoes_imagem:
            raise ValueError("EXTENSOES_IMAGEM nao pode ser vazio.")

        # No modo split, pelo menos uma das duas bases precisa estar ativa.
        if self.usar_test_split:
            self.bases_split_ativas = {
                nome: dados
                for nome, dados in self.bases_split.items()
                if bool(dados.get("ativa", False))
            }

            if not self.bases_split_ativas:
                raise ValueError(
                    "USAR_TEST_SPLIT=True, mas nenhuma base de test foi selecionada."
                )

            erros = []
            for nome, dados in self.bases_split_ativas.items():
                pasta_test = Path(dados["test"])
                raiz_split = Path(dados["raiz_split"])

                if not raiz_split.exists() or not raiz_split.is_dir():
                    erros.append(f"[{nome}] raiz do split invalida: {raiz_split}")

                if not pasta_test.exists() or not pasta_test.is_dir():
                    erros.append(f"[{nome}] pasta test invalida: {pasta_test}")

            if erros:
                raise FileNotFoundError(
                    "Falha ao validar os conjuntos test do split:\n- "
                    + "\n- ".join(erros)
                )

        # No modo manual, apenas a pasta escolhida participa.
        else:
            if not self.pasta_teste_escolhida.exists():
                raise FileNotFoundError(
                    "PASTA_TESTE_ESCOLHIDA nao encontrada:\n"
                    f"{self.pasta_teste_escolhida}"
                )

            if not self.pasta_teste_escolhida.is_dir():
                raise NotADirectoryError(
                    "PASTA_TESTE_ESCOLHIDA nao e um diretorio:\n"
                    f"{self.pasta_teste_escolhida}"
                )

    def _configurar_dispositivo(self):
        """Seleciona CUDA quando permitido e disponivel; caso contrario usa CPU."""

        usar_cuda_real = self.usar_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if usar_cuda_real else "cpu")

        print(f"Dispositivo            : {self.device}")

        if self.device.type == "cuda":
            print(f"GPU                    : {torch.cuda.get_device_name(0)}")
            print(f"CUDA                   : {torch.version.cuda}")

    def _carregar_checkpoint(self):
        """Carrega e valida os metadados essenciais do checkpoint atual."""

        self.checkpoint = torch.load(
            self.caminho_modelo,
            map_location=self.device,
        )

        campos = {
            "model_state_dict",
            "classes",
            "input_shape",
            "num_classes",
        }
        ausentes = campos - set(self.checkpoint)

        if ausentes:
            raise KeyError(
                "Checkpoint incompatível com o treino atual. Campos ausentes: "
                f"{sorted(ausentes)}"
            )

        self.classes = list(self.checkpoint["classes"])
        self.input_shape = tuple(int(valor) for valor in self.checkpoint["input_shape"])

        if not self.classes:
            raise ValueError("Checkpoint nao possui classes.")

        if int(self.checkpoint["num_classes"]) != len(self.classes):
            raise ValueError(
                "num_classes do checkpoint nao corresponde a lista de classes."
            )

        # O projeto atual utiliza uma ordem fixa de classes para treino e teste.
        if tuple(self.classes) != tuple(ROTULOS):
            raise ValueError(
                "Classes do checkpoint nao correspondem ao padrao atual do projeto.\n"
                f"Checkpoint: {self.classes}\n"
                f"Projeto   : {list(ROTULOS)}"
            )

        if len(self.input_shape) != 3 or self.input_shape[0] != 1:
            raise ValueError(
                f"input_shape invalido no checkpoint: {self.input_shape}"
            )

        self.class_to_idx = {
            classe: indice
            for indice, classe in enumerate(self.classes)
        }

    def _validar_pastas_classes(self, raiz):
        """
        Valida os diretorios de primeiro nivel usados como rotulos reais.

        Pastas de classe ausentes sao permitidas. Qualquer diretorio de primeiro
        nivel cujo nome nao corresponda a uma classe do checkpoint gera erro.
        """

        raiz = Path(raiz)

        pastas = sorted(
            caminho
            for caminho in raiz.iterdir()
            if caminho.is_dir()
        )

        desconhecidas = [
            pasta.name
            for pasta in pastas
            if pasta.name not in self.class_to_idx
        ]

        if desconhecidas:
            raise ValueError(
                f"Pastas de classe desconhecidas em {raiz}: {desconhecidas}. "
                f"Classes validas: {self.classes}"
            )

    def _coletar_pasta_classes_npy(self, raiz, origem):
        """
        Coleta NPY do test criado pelo split sem aplicar novo processamento.

        Parameters
        ----------
        raiz : str ou Path
            Pasta test que contem subpastas de classe.

        origem : str
            Identificacao da base usada nos relatorios.

        Returns
        -------
        list[dict]
            Registros prontos para o Dataset de teste.
        """

        raiz = Path(raiz)
        self._validar_pastas_classes(raiz)
        registros = []

        for classe in self.classes:
            pasta_classe = raiz / classe

            if not pasta_classe.exists():
                continue

            arquivos = sorted(
                caminho
                for caminho in pasta_classe.rglob("*.npy")
                if caminho.is_file()
            )

            for caminho in arquivos:
                # Test nao deve conter copias artificiais produzidas no train.
                if self._eh_augmentation(caminho):
                    raise ValueError(
                        "Arquivo de augmentation encontrado no conjunto de teste: "
                        f"{caminho}"
                    )

                registros.append(
                    {
                        "caminho": caminho,
                        "rotulo": classe,
                        "indice": self.class_to_idx[classe],
                        "origem": origem,
                        "tipo": "npy",
                    }
                )

        return registros

    def _coletar_pasta_classes_imagens(self, raiz):
        """
        Coleta imagens comuns da pasta escolhida manualmente.

        A pasta de primeiro nivel define o rotulo real. Dentro de cada classe,
        subpastas adicionais sao permitidas e pesquisadas recursivamente.
        """

        raiz = Path(raiz)
        self._validar_pastas_classes(raiz)
        registros = []

        for classe in self.classes:
            pasta_classe = raiz / classe

            if not pasta_classe.exists():
                continue

            arquivos = listar_imagens(
                pasta_classe,
                self.extensoes_imagem,
            )

            for caminho in arquivos:
                registros.append(
                    {
                        "caminho": caminho,
                        "rotulo": classe,
                        "indice": self.class_to_idx[classe],
                        "origem": "ESCOLHIDA",
                        "tipo": "imagem",
                    }
                )

        return registros

    def _coletar_amostras(self):
        """Seleciona as amostras conforme o modo configurado no processing."""

        registros = []

        if self.usar_test_split:
            for origem, dados in self.bases_split_ativas.items():
                registros.extend(
                    self._coletar_pasta_classes_npy(
                        dados["test"],
                        origem,
                    )
                )
        else:
            registros.extend(
                self._coletar_pasta_classes_imagens(
                    self.pasta_teste_escolhida,
                )
            )

        if not registros:
            tipo = "NPY" if self.usar_test_split else "imagem"
            raise RuntimeError(
                f"Nenhum arquivo de {tipo} foi encontrado para o teste."
            )

        self.amostras = registros

    def _preprocessar_imagem_manual(self, caminho, registrar_estatistica=False):
        """
        Prepara uma imagem manual e opcionalmente registra suas caracteristicas.

        A conversao efetiva e centralizada em _DatasetTeste para garantir que a
        validacao inicial e a inferencia usem exatamente o mesmo procedimento.
        """

        array, info = _DatasetTeste.carregar_imagem_manual(
            caminho,
            self.input_shape,
        )

        # As estatisticas sao registradas somente durante a validacao inicial
        # para nao duplicar contagens quando a imagem for carregada na inferencia.
        if registrar_estatistica:
            self.estatisticas_pasta_manual["formatos"][info["formato"]] += 1
            self.estatisticas_pasta_manual["modos"][info["modo"]] += 1
            self.estatisticas_pasta_manual["resolucoes"][info["resolucao"]] += 1

            if info["convertida_grayscale"]:
                self.estatisticas_pasta_manual["convertidas_grayscale"] += 1

            if info["crop_quadrado"]:
                self.estatisticas_pasta_manual["crop_quadrado"] += 1
            else:
                self.estatisticas_pasta_manual["ja_quadradas"] += 1

        return array

    def _carregar_array_registro(self, registro):
        """Carrega um registro usando a mesma implementacao usada pelo Dataset."""

        return _DatasetTeste.carregar_array(
            registro,
            self.input_shape,
        )

    def _validar_amostras(self):
        """
        Valida todas as entradas antes de criar a pasta de resultados.

        No modo split, apenas reabre os NPY e confere o shape do checkpoint.
        No modo pasta escolhida, executa todo o pre-processamento em memoria para
        garantir que cada imagem pode ser convertida corretamente antes do teste.
        """

        erros = []

        if not self.usar_test_split:
            self.estatisticas_pasta_manual = {
                "formatos": Counter(),
                "modos": Counter(),
                "resolucoes": Counter(),
                "convertidas_grayscale": 0,
                "crop_quadrado": 0,
                "ja_quadradas": 0,
            }

        for registro in self.amostras:
            caminho = registro["caminho"]

            try:
                if registro["tipo"] == "npy":
                    self._carregar_array_registro(registro)
                else:
                    self._preprocessar_imagem_manual(
                        caminho,
                        registrar_estatistica=True,
                    )

            except Exception as erro:
                erros.append(f"{caminho}: {erro}")

        if erros:
            raise RuntimeError(
                "Falha na validacao das imagens de teste:\n- "
                + "\n- ".join(erros)
            )

    # =========================================================================
    # MODELO E INFERENCIA
    # =========================================================================

    def _inicializar_modelo(self):
        """Cria a LeNet5, materializa LazyLinear e carrega os pesos."""

        self.modelo = LeNet5(
            num_classes=len(self.classes)
        ).to(self.device)

        # O forward com tensor vazio materializa os parametros da LazyLinear
        # usando exatamente a dimensao registrada pelo treinamento.
        exemplo = torch.zeros(
            (1, *self.input_shape),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            saida = self.modelo(exemplo)

        if tuple(saida.shape) != (1, len(self.classes)):
            raise RuntimeError(
                f"Saida inesperada do modelo: {tuple(saida.shape)}"
            )

        try:
            self.modelo.load_state_dict(
                self.checkpoint["model_state_dict"]
            )
        except RuntimeError as erro:
            raise RuntimeError(
                "O checkpoint nao e compativel com models/lenet5.py atual."
            ) from erro

        self.modelo.eval()

    def _executar_inferencia(self):
        """
        Executa o teste e retorna loss media, acuracia e detalhes por imagem.
        """

        dataset = _DatasetTeste(
            self.amostras,
            self.input_shape,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )

        criterio = nn.CrossEntropyLoss()
        perda_acumulada = 0.0
        total = 0
        acertos = 0
        detalhes = []

        self.modelo.eval()

        with torch.no_grad():
            for imagens, rotulos, caminhos, origens, tipos in loader:
                imagens = imagens.to(
                    self.device,
                    non_blocking=self.device.type == "cuda",
                )
                rotulos = rotulos.to(
                    self.device,
                    non_blocking=self.device.type == "cuda",
                )

                logits = self.modelo(imagens)
                perda = criterio(logits, rotulos)
                probabilidades = torch.softmax(logits, dim=1)

                valores_top2, indices_top2 = probabilidades.topk(
                    k=min(2, len(self.classes)),
                    dim=1,
                )

                predicoes = logits.argmax(dim=1)
                tamanho_lote = rotulos.size(0)

                perda_acumulada += perda.item() * tamanho_lote
                acertos += (predicoes == rotulos).sum().item()
                total += tamanho_lote

                # Move somente os dados necessarios para CPU ao montar o relatorio.
                rotulos_cpu = rotulos.cpu().tolist()
                predicoes_cpu = predicoes.cpu().tolist()
                probs_cpu = probabilidades.cpu().numpy()
                valores_top2_cpu = valores_top2.cpu().numpy()
                indices_top2_cpu = indices_top2.cpu().numpy()

                for indice_local, caminho_texto in enumerate(caminhos):
                    real = int(rotulos_cpu[indice_local])
                    previsto = int(predicoes_cpu[indice_local])
                    prob = probs_cpu[indice_local]

                    # Entropia em base natural; epsilon evita log(0).
                    entropia = float(
                        -np.sum(prob * np.log(np.clip(prob, 1e-12, 1.0)))
                    )

                    if len(self.classes) >= 2:
                        segunda_idx = int(indices_top2_cpu[indice_local][1])
                        segunda_prob = float(valores_top2_cpu[indice_local][1])
                    else:
                        segunda_idx = previsto
                        segunda_prob = 0.0

                    confianca = float(prob[previsto])
                    margem = confianca - segunda_prob

                    detalhes.append(
                        {
                            "caminho": Path(caminho_texto),
                            "origem": str(origens[indice_local]),
                            "tipo": str(tipos[indice_local]),
                            "real_indice": real,
                            "previsto_indice": previsto,
                            "real": self.classes[real],
                            "previsto": self.classes[previsto],
                            "acerto": real == previsto,
                            "confianca": confianca,
                            "segunda_classe": self.classes[segunda_idx],
                            "segunda_confianca": segunda_prob,
                            "margem": margem,
                            "entropia": entropia,
                        }
                    )

        if total == 0:
            raise RuntimeError("DataLoader de teste nao possui amostras.")

        return perda_acumulada / total, acertos / total, detalhes

    # =========================================================================
    # RESULTADOS E RELATORIOS
    # =========================================================================

    def _criar_pasta_resultados(self):
        """Cria uma pasta temporaria desta execucao antes de gerar resultados."""

        self.inicio = datetime.now()
        modo = "split" if self.usar_test_split else "pasta"

        self.pasta_execucao = (
            self.pasta_resultados
            / f"test_{modo}_inicio_{self.inicio:%Y-%m-%d_%H-%M-%S}_em_andamento"
        )

        self.pasta_execucao.mkdir(
            parents=True,
            exist_ok=False,
        )

    def _contagens_teste(self):
        """Retorna contagens por classe e por origem para o conjunto avaliado."""

        por_classe = Counter(
            registro["rotulo"]
            for registro in self.amostras
        )

        por_origem = Counter(
            registro["origem"]
            for registro in self.amostras
        )

        por_origem_classe = defaultdict(Counter)
        for registro in self.amostras:
            por_origem_classe[registro["origem"]][registro["rotulo"]] += 1

        return por_classe, por_origem, por_origem_classe

    def _salvar_predicoes_csv(self, detalhes):
        """Salva uma linha por imagem com previsao e indicadores de confianca."""

        caminho = self.pasta_execucao / "predicoes.csv"

        with caminho.open("w", newline="", encoding="utf-8-sig") as arquivo:
            escritor = csv.writer(arquivo, delimiter=";")
            escritor.writerow(
                [
                    "Arquivo",
                    "Origem",
                    "Tipo de entrada",
                    "Rotulo real",
                    "Rotulo previsto",
                    "Resultado",
                    "Confianca",
                    "Segunda classe",
                    "Segunda confianca",
                    "Margem",
                    "Entropia",
                ]
            )

            for item in detalhes:
                escritor.writerow(
                    [
                        str(item["caminho"]),
                        item["origem"],
                        item["tipo"],
                        item["real"],
                        item["previsto"],
                        "Acerto" if item["acerto"] else "Erro",
                        f"{item['confianca']:.8f}",
                        item["segunda_classe"],
                        f"{item['segunda_confianca']:.8f}",
                        f"{item['margem']:.8f}",
                        f"{item['entropia']:.8f}",
                    ]
                )

    def _salvar_tabela_html(self, detalhes):
        """Salva tabela navegavel das predicoes em HTML."""

        linhas = [
            "<!DOCTYPE html>",
            "<html><head><meta charset='UTF-8'>",
            "<title>Predicoes do teste</title>",
            "<style>",
            "body{font-family:Arial;margin:30px}",
            "table{border-collapse:collapse;width:100%}",
            "th,td{border:1px solid #ccc;padding:7px;text-align:left}",
            "th{background:#eee}",
            ".acerto{background:#dff0d8}",
            ".erro{background:#f2dede}",
            "</style></head><body>",
            "<h2>Predicoes do conjunto de teste</h2>",
            "<table>",
            "<tr><th>Arquivo</th><th>Origem</th><th>Tipo</th><th>Real</th><th>Previsto</th>"
            "<th>Resultado</th><th>Confianca</th><th>Segunda classe</th>"
            "<th>Margem</th><th>Entropia</th></tr>",
        ]

        for item in detalhes:
            css = "acerto" if item["acerto"] else "erro"
            resultado = "Acerto" if item["acerto"] else "Erro"

            linhas.append(
                f"<tr class='{css}'>"
                f"<td>{escape(str(item['caminho']))}</td>"
                f"<td>{escape(item['origem'])}</td>"
                f"<td>{escape(item['tipo'])}</td>"
                f"<td>{escape(item['real'])}</td>"
                f"<td>{escape(item['previsto'])}</td>"
                f"<td>{resultado}</td>"
                f"<td>{item['confianca']:.4f}</td>"
                f"<td>{escape(item['segunda_classe'])}</td>"
                f"<td>{item['margem']:.4f}</td>"
                f"<td>{item['entropia']:.4f}</td>"
                "</tr>"
            )

        linhas.extend(["</table>", "</body></html>"])

        (self.pasta_execucao / "tabela_predicoes.html").write_text(
            "\n".join(linhas),
            encoding="utf-8",
        )

    def _salvar_grafico_total(self, detalhes):
        """Gera grafico simples com total de acertos e erros."""

        acertos = sum(item["acerto"] for item in detalhes)
        erros = len(detalhes) - acertos

        figura, eixo = plt.subplots(figsize=(7, 5))
        barras = eixo.bar(["Acertos", "Erros"], [acertos, erros])
        eixo.set_ylabel("Quantidade de imagens")
        eixo.set_title("Acertos e erros - teste")
        eixo.bar_label(barras)
        figura.tight_layout()
        figura.savefig(
            self.pasta_execucao / "acertos_erros_totais.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figura)

    def _salvar_grafico_classes(self, detalhes):
        """Gera grafico de acertos e erros separados por classe real."""

        acertos = []
        erros = []

        for classe in self.classes:
            itens = [
                item
                for item in detalhes
                if item["real"] == classe
            ]
            acertos_classe = sum(item["acerto"] for item in itens)
            acertos.append(acertos_classe)
            erros.append(len(itens) - acertos_classe)

        x = np.arange(len(self.classes))
        largura = 0.35

        figura, eixo = plt.subplots(figsize=(10, 6))
        barras_acertos = eixo.bar(
            x - largura / 2,
            acertos,
            largura,
            label="Acertos",
        )
        barras_erros = eixo.bar(
            x + largura / 2,
            erros,
            largura,
            label="Erros",
        )

        eixo.set_xlabel("Classes")
        eixo.set_ylabel("Quantidade de imagens")
        eixo.set_title("Acertos e erros por classe - teste")
        eixo.set_xticks(x)
        eixo.set_xticklabels(self.classes)
        eixo.legend()
        eixo.bar_label(barras_acertos)
        eixo.bar_label(barras_erros)

        figura.tight_layout()
        figura.savefig(
            self.pasta_execucao / "acertos_erros_por_classe.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figura)

    def _salvar_amostras(self, detalhes):
        """Salva exemplos de predicoes exibindo diretamente os NPY avaliados."""

        if self.num_amostras_grafico == 0:
            return

        quantidade = min(
            self.num_amostras_grafico,
            len(detalhes),
        )

        if quantidade == 0:
            return

        rng = random.Random(self.seed)
        indices = rng.sample(
            range(len(detalhes)),
            quantidade,
        )

        selecionados = [
            detalhes[indice]
            for indice in indices
        ]

        colunas = min(self.colunas_amostras, quantidade)
        linhas = math.ceil(quantidade / colunas)

        figura, eixos = plt.subplots(
            linhas,
            colunas,
            figsize=(4 * colunas, 4 * linhas),
        )
        eixos = np.atleast_1d(eixos).ravel()

        for eixo, item in zip(eixos, selecionados):
            # Exibe exatamente a representacao numerica apresentada ao modelo.
            if item["tipo"] == "npy":
                array = carregar_imagem_npy(item["caminho"])
            else:
                array = self._preprocessar_imagem_manual(
                    item["caminho"],
                    registrar_estatistica=False,
                )

            imagem = array[0]

            eixo.imshow(
                imagem,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            eixo.set_title(
                f"Real: {item['real']}\n"
                f"Previsto: {item['previsto']}\n"
                f"Conf.: {item['confianca']:.2%}",
                fontsize=10,
            )
            eixo.axis("off")

        for eixo in eixos[quantidade:]:
            eixo.axis("off")

        figura.suptitle("Amostras de predicao - teste")
        figura.tight_layout()
        figura.savefig(
            self.pasta_execucao / "amostras_predicoes.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figura)

    def _salvar_resumo_predicoes(self, detalhes, loss, metricas):
        """Salva resumo global e por classe das predicoes."""

        acertos_total = sum(item["acerto"] for item in detalhes)
        erros_total = len(detalhes) - acertos_total

        linhas = [
            "RESUMO DAS PREDICOES",
            "=" * 60,
            "",
            f"Pasta do treino: {self.pasta_treino}",
            f"Modelo: {self.caminho_modelo}",
            f"Checkpoint escolhido: {'melhor' if self.usar_modelo_melhor else 'final'}",
            f"Modo: {'test do split' if self.usar_test_split else 'pasta escolhida'}",
            f"Input shape: {self.input_shape}",
            f"Total de imagens: {len(detalhes)}",
            f"Acertos: {acertos_total}",
            f"Erros: {erros_total}",
            f"Loss: {loss:.6f}",
            f"Acuracia: {metricas['accuracy']:.2%}",
            f"Precisao macro: {metricas['precision_macro']:.4f}",
            f"Recall macro: {metricas['recall_macro']:.4f}",
            f"F1-score macro: {metricas['f1_macro']:.4f}",
            f"Precisao weighted: {metricas['precision_weighted']:.4f}",
            f"Recall weighted: {metricas['recall_weighted']:.4f}",
            f"F1-score weighted: {metricas['f1_weighted']:.4f}",
            "",
            "RESULTADOS POR CLASSE",
            "-" * 60,
        ]

        for classe in self.classes:
            itens = [
                item
                for item in detalhes
                if item["real"] == classe
            ]
            acertos = sum(item["acerto"] for item in itens)
            erros = len(itens) - acertos
            acuracia = acertos / len(itens) if itens else 0.0

            linhas.append(
                f"{classe}: total={len(itens)} | acertos={acertos} | "
                f"erros={erros} | acuracia={acuracia:.2%}"
            )

        (self.pasta_execucao / "resumo_predicoes.txt").write_text(
            "\n".join(linhas),
            encoding="utf-8",
        )

    def _salvar_informacoes_teste(self, fim):
        """Registra configuracao, checkpoint, dispositivo e composicao do teste."""

        por_classe, por_origem, por_origem_classe = self._contagens_teste()

        config_treino = self.checkpoint.get("config_treino", {})

        linhas = [
            "INFORMACOES DO TESTE",
            "=" * 70,
            "",
            f"Inicio: {self.inicio.isoformat(sep=' ', timespec='seconds')}",
            f"Fim: {fim.isoformat(sep=' ', timespec='seconds')}",
            f"Duracao_s: {(fim - self.inicio).total_seconds():.3f}",
            "",
            "MODELO",
            "-" * 70,
            f"Pasta do treino: {self.pasta_treino}",
            f"Checkpoint: {self.caminho_modelo}",
            f"Checkpoint escolhido: {'melhor' if self.usar_modelo_melhor else 'final'}",
            f"Model name: {self.checkpoint.get('model_name', 'nao informado')}",
            f"Epoca do checkpoint: {self.checkpoint.get('epoca', 'nao informado')}",
            f"Input shape: {self.input_shape}",
            f"Classes: {', '.join(self.classes)}",
            f"Treinado com augmentation: {config_treino.get('usar_augmentation', 'nao informado')}",
            f"Treinado com balanceamento de origens: {config_treino.get('balancear_origens', 'nao informado')}",
            f"Bases usadas no treino: {config_treino.get('bases', 'nao informado')}",
            "",
            "CONFIGURACAO DO TESTE",
            "-" * 70,
            f"Modo: {'split' if self.usar_test_split else 'pasta escolhida'}",
            f"Batch size: {self.batch_size}",
            f"Num workers: {self.num_workers}",
            f"Device: {self.device}",
            f"Total de imagens: {len(self.amostras)}",
        ]

        if self.device.type == "cuda":
            linhas.append(f"GPU: {torch.cuda.get_device_name(0)}")
            linhas.append(f"CUDA: {torch.version.cuda}")

        if self.usar_test_split:
            linhas.append(
                f"Bases test selecionadas: {', '.join(self.bases_split_ativas)}"
            )
        else:
            linhas.append(f"Pasta escolhida: {self.pasta_teste_escolhida}")
            linhas.extend(
                [
                    "",
                    "PRE-PROCESSAMENTO DA PASTA ESCOLHIDA",
                    "-" * 70,
                    "Rotulo real: pasta de classe",
                    "Orientacao EXIF: corrigida",
                    "Crop: maior quadrado central quando necessario",
                    "Canal final: grayscale",
                    f"Resolucao final: {self.input_shape[2]}x{self.input_shape[1]}",
                    "Resize: LANCZOS",
                    "Dtype final: float32",
                    "Normalizacao: 0-255 para 0-1",
                    f"Imagens convertidas para grayscale: {self.estatisticas_pasta_manual['convertidas_grayscale']}",
                    f"Imagens com crop quadrado: {self.estatisticas_pasta_manual['crop_quadrado']}",
                    f"Imagens ja quadradas: {self.estatisticas_pasta_manual['ja_quadradas']}",
                    "Formatos de entrada: "
                    + self._formatar_counter(self.estatisticas_pasta_manual["formatos"]),
                    "Modos de entrada: "
                    + self._formatar_counter(self.estatisticas_pasta_manual["modos"]),
                    "Resolucoes de entrada: "
                    + self._formatar_resolucoes(self.estatisticas_pasta_manual["resolucoes"]),
                ]
            )

        linhas.extend(
            [
                "",
                "DISTRIBUICAO POR CLASSE",
                "-" * 70,
            ]
        )

        for classe in self.classes:
            linhas.append(f"{classe}: {por_classe[classe]}")

        linhas.extend(
            [
                "",
                "DISTRIBUICAO POR ORIGEM",
                "-" * 70,
            ]
        )

        for origem in sorted(por_origem):
            linhas.append(f"{origem}: {por_origem[origem]}")
            for classe in self.classes:
                linhas.append(
                    f"  {classe}: {por_origem_classe[origem][classe]}"
                )

        (self.pasta_execucao / "informacoes_teste.txt").write_text(
            "\n".join(linhas),
            encoding="utf-8",
        )

    def _contar_split_original(self, raiz_split):
        """
        Conta train/val/test por classe ignorando augmentations existentes em train.

        Returns
        -------
        dict
            Contagens por particao e classe.
        """

        raiz_split = Path(raiz_split)
        contagens = {
            particao: Counter()
            for particao in ("train", "val", "test")
        }

        for particao in ("train", "val", "test"):
            pasta_particao = raiz_split / particao

            if not pasta_particao.exists() or not pasta_particao.is_dir():
                raise FileNotFoundError(
                    f"Particao ausente ao calcular proporcao do split: {pasta_particao}"
                )

            for classe in self.classes:
                pasta_classe = pasta_particao / classe

                if not pasta_classe.exists():
                    continue

                arquivos = [
                    caminho
                    for caminho in pasta_classe.rglob("*.npy")
                    if caminho.is_file()
                ]

                # Augmentations existem somente depois do split e precisam ser
                # removidos desta contagem para recuperar a divisao original.
                if particao == "train":
                    arquivos = [
                        caminho
                        for caminho in arquivos
                        if not self._eh_augmentation(caminho)
                    ]

                contagens[particao][classe] = len(arquivos)

        return contagens

    def _salvar_proporcao_split(self):
        """
        Salva a proporcao observada do split em TXT.

        No modo de pasta escolhida nao existe uma divisao train/val/test associada
        ao conjunto avaliado; nesse caso o arquivo registra explicitamente que a
        informacao nao se aplica.
        """

        caminho = self.pasta_execucao / "proporcao_split.txt"

        if not self.usar_test_split:
            caminho.write_text(
                "PROPORCAO DO SPLIT\n"
                "=" * 70
                + "\n\nModo de teste: pasta escolhida.\n"
                "Nao existe um split train/val/test associado a este teste.\n",
                encoding="utf-8",
            )
            return

        linhas = [
            "PROPORCAO OBSERVADA DO SPLIT",
            "=" * 70,
            "",
            "As porcentagens sao calculadas a partir dos arquivos presentes em",
            "train, val e test. Arquivos de augmentation existentes em train sao",
            "ignorados porque foram criados depois da divisao original.",
            "",
        ]

        combinado = {
            particao: Counter()
            for particao in ("train", "val", "test")
        }

        for origem, dados in self.bases_split_ativas.items():
            contagens = self._contar_split_original(
                dados["raiz_split"]
            )

            linhas.extend(
                [
                    f"BASE {origem}",
                    "-" * 70,
                ]
            )

            total_base = sum(
                sum(contagens[particao].values())
                for particao in ("train", "val", "test")
            )

            for particao in ("train", "val", "test"):
                total_particao = sum(contagens[particao].values())
                percentual = (
                    100.0 * total_particao / total_base
                    if total_base
                    else 0.0
                )

                linhas.append(
                    f"{particao:<5}: {total_particao:>6} | {percentual:>7.3f}%"
                )

                combinado[particao].update(contagens[particao])

            linhas.append(f"Total: {total_base}")
            linhas.append("")

            linhas.append("Por classe:")
            for classe in self.classes:
                total_classe = sum(
                    contagens[particao][classe]
                    for particao in ("train", "val", "test")
                )

                valores = []
                for particao in ("train", "val", "test"):
                    quantidade = contagens[particao][classe]
                    percentual = (
                        100.0 * quantidade / total_classe
                        if total_classe
                        else 0.0
                    )
                    valores.append(
                        f"{particao}={quantidade} ({percentual:.2f}%)"
                    )

                linhas.append(
                    f"  {classe:<15}: " + " | ".join(valores)
                )

            linhas.append("")

        # Quando duas bases participam, registra tambem a proporcao combinada.
        if len(self.bases_split_ativas) > 1:
            linhas.extend(
                [
                    "BASES COMBINADAS",
                    "-" * 70,
                ]
            )

            total_combinado = sum(
                sum(combinado[particao].values())
                for particao in ("train", "val", "test")
            )

            for particao in ("train", "val", "test"):
                total_particao = sum(combinado[particao].values())
                percentual = (
                    100.0 * total_particao / total_combinado
                    if total_combinado
                    else 0.0
                )
                linhas.append(
                    f"{particao:<5}: {total_particao:>6} | {percentual:>7.3f}%"
                )

            linhas.append(f"Total: {total_combinado}")

        caminho.write_text(
            "\n".join(linhas),
            encoding="utf-8",
        )

    def _mostrar_plano(self):
        """Exibe a configuracao validada antes de iniciar inferencia."""

        por_classe, por_origem, _ = self._contagens_teste()

        print("\n" + "=" * 90)
        print("TESTE DO MODELO")
        print("=" * 90)
        print(f"Pasta do treino        : {self.pasta_treino}")
        print(f"Modelo                 : {self.caminho_modelo}")
        print(f"Checkpoint             : {'melhor' if self.usar_modelo_melhor else 'final'}")
        print(f"Modo                   : {'split' if self.usar_test_split else 'pasta escolhida'}")
        print(f"Input shape            : {self.input_shape}")
        print(f"Total de imagens       : {len(self.amostras)}")
        print(f"Batch size             : {self.batch_size}")

        if self.usar_test_split:
            print(f"Bases                  : {', '.join(self.bases_split_ativas)}")
        else:
            print(f"Pasta escolhida        : {self.pasta_teste_escolhida}")
            print("Pre-processamento      : crop central -> grayscale -> resize -> float32 -> [0,1]")
            print(f"Resolucao final        : {self.input_shape[2]}x{self.input_shape[1]}")
            print(f"Com crop quadrado      : {self.estatisticas_pasta_manual['crop_quadrado']}")
            print(f"Ja quadradas           : {self.estatisticas_pasta_manual['ja_quadradas']}")
            print(f"Convertidas grayscale  : {self.estatisticas_pasta_manual['convertidas_grayscale']}")

        print("\nIMAGENS POR CLASSE")
        for classe in self.classes:
            print(f"  {classe:<15}: {por_classe[classe]}")

        print("\nIMAGENS POR ORIGEM")
        for origem in sorted(por_origem):
            print(f"  {origem:<15}: {por_origem[origem]}")

    def _finalizar_pasta(self, fim):
        """Renomeia a pasta temporaria para registrar inicio e fim da execucao."""

        modo = "split" if self.usar_test_split else "pasta"
        pasta_final = (
            self.pasta_resultados
            / f"test_{modo}_inicio_{self.inicio:%Y-%m-%d_%H-%M-%S}"
              f"_fim_{fim:%Y-%m-%d_%H-%M-%S}"
        )

        if pasta_final.exists():
            raise FileExistsError(
                f"Pasta final de resultados ja existe: {pasta_final}"
            )

        self.pasta_execucao.rename(pasta_final)
        self.pasta_execucao = pasta_final

    # =========================================================================
    # FLUXO PRINCIPAL
    # =========================================================================

    def executar(self):
        """Executa o fluxo completo de avaliacao final."""

        # Todas as validacoes de entrada ocorrem antes da criacao de resultados.
        self._validar_parametros()
        self._configurar_dispositivo()
        self._carregar_checkpoint()
        self._coletar_amostras()
        self._validar_amostras()
        self._mostrar_plano()

        # Resultados so passam a ser criados depois de entradas e modelo validos.
        self._criar_pasta_resultados()
        self._inicializar_modelo()

        loss, acuracia_loader, detalhes = self._executar_inferencia()

        rotulos_reais = [
            item["real_indice"]
            for item in detalhes
        ]
        predicoes = [
            item["previsto_indice"]
            for item in detalhes
        ]

        metricas, relatorio = calcular_metricas_classificacao(
            rotulos_reais,
            predicoes,
            self.classes,
        )

        # Confere se a acuracia calculada durante os lotes coincide com sklearn.
        if abs(acuracia_loader - metricas["accuracy"]) > 1e-12:
            raise RuntimeError(
                "Inconsistencia interna entre as duas acuracias calculadas."
            )

        salvar_metricas_classificacao(
            metricas,
            relatorio,
            loss,
            self.pasta_execucao / "metricas_teste.txt",
            titulo="TESTE",
        )

        plotar_matrizes_confusao(
            rotulos_reais,
            predicoes,
            self.classes,
            self.pasta_execucao,
            prefixo="teste",
        )

        self._salvar_predicoes_csv(detalhes)
        self._salvar_tabela_html(detalhes)
        self._salvar_grafico_total(detalhes)
        self._salvar_grafico_classes(detalhes)
        self._salvar_amostras(detalhes)
        self._salvar_resumo_predicoes(
            detalhes,
            loss,
            metricas,
        )
        self._salvar_proporcao_split()

        fim = datetime.now()
        self._salvar_informacoes_teste(fim)
        self._finalizar_pasta(fim)

        acertos = sum(item["acerto"] for item in detalhes)
        erros = len(detalhes) - acertos

        print("\n" + "=" * 90)
        print("TESTE FINALIZADO")
        print("=" * 90)
        print(f"Total                  : {len(detalhes)}")
        print(f"Acertos                : {acertos}")
        print(f"Erros                  : {erros}")
        print(f"Loss                   : {loss:.6f}")
        print(f"Acuracia               : {metricas['accuracy']:.2%}")
        print(f"F1-score macro         : {metricas['f1_macro']:.4f}")
        print(f"Resultados             : {self.pasta_execucao}")

        return {
            "pasta_resultados": self.pasta_execucao,
            "total": len(detalhes),
            "acertos": acertos,
            "erros": erros,
            "loss": loss,
            "metricas": metricas,
            "classes": tuple(self.classes),
            "modo": "split" if self.usar_test_split else "pasta",
            "pasta_treino": self.pasta_treino,
            "modelo": self.caminho_modelo,
            "checkpoint_escolhido": "melhor" if self.usar_modelo_melhor else "final",
        }
