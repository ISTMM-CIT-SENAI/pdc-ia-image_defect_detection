"""
treino_class.py

Objetivo:
- Implementar toda a logica especifica do treinamento.
- Selecionar train e val das bases interna e/ou externa.
- Combinar as origens sem utilizar o conjunto test.
- Aplicar balanceamento opcional INTERNA/EXTERNA por classe.
- Criar DataLoaders, modelo, loss e otimizador.
- Treinar o modelo selecionado e validar em intervalos configuraveis.
- Aplicar early stopping por paciencia medida em eventos de validacao.
- Salvar melhor checkpoint, checkpoint final, historico e informacoes da execucao.

Regras importantes:
- Test nunca e carregado neste modulo.
- As imagens ja chegam em NPY float32, shape (1,H,W) e intervalo [0,1].
- Augmentation ja foi materializado anteriormente no diretorio train.
- A flag usar_augmentation decide se arquivos _augXX.npy entram no train.
- Val nunca depende dessa flag e deve permanecer sem augmentation.
- O modelo e criado exclusivamente por models/modelos.py.
- A ordem das classes vem de utils.nomenclatura.ROTULOS.
- O carregamento das imagens e reutilizavel e fica em utils.dataset.
- Historico e grafico ficam em utils.metrics.

Balanceamento de origens:
- So e aplicado quando duas ou mais origens estao ativas e
  balancear_origens=True.
- Para cada classe e particao, se INTERNA e EXTERNA possuem imagens, sao
  selecionadas quantidades iguais usando o menor total entre as origens.
- Se apenas uma origem possui a classe, suas amostras sao mantidas integralmente.
- O procedimento usa seed fixa para ser reproduzivel.
"""

from collections import Counter, defaultdict
import csv
import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path
import math
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pdc_ia_image_defect_detection.models.modelos import (
    criar_modelo,
    modelos_disponiveis,
    normalizar_nome_modelo,
)
from pdc_ia_image_defect_detection.utils.dataset import DatasetNPY
from pdc_ia_image_defect_detection.utils.metrics import (
    calcular_metricas_classificacao,
    plotar_historico,
    plotar_matrizes_confusao,
    salvar_historico_csv,
    salvar_metricas_classificacao,
)
from pdc_ia_image_defect_detection.utils.resultados import (
    criar_pasta_resultados,
    finalizar_pasta_resultados,
)
from pdc_ia_image_defect_detection.utils.nomenclatura import ROTULOS


class TreinoProcessor:
    """Coordena selecao de dados, treinamento, validacao e salvamento."""

    # Reconhece os nomes produzidos por augmentation_class.py.
    # Exemplos aceitos: imagem_aug01.npy e imagem_aug_01.npy.
    PADRAO_AUGMENTATION = re.compile(r"_aug_?\d+$", flags=re.IGNORECASE)

    def __init__(
        self,
        nome_modelo,
        bases,
        pasta_resultados,
        num_epochs,
        batch_size,
        learning_rate,
        intervalo_validacao,
        paciencia,
        seed,
        num_workers,
        usar_cuda,
        balancear_origens,
        usar_augmentation,
        parametros_modelo=None,
        detalhes_resultado=(),
        salvar_historico_incremental=False,
        continuar_de=None,
        nome_curto_grid=False,
    ):
        """Armazena toda a configuracao recebida do processing/treino.py."""

        self.nome_modelo = normalizar_nome_modelo(nome_modelo)
        self.bases = dict(bases)
        self.pasta_resultados = Path(pasta_resultados)
        self.num_epochs = int(num_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.intervalo_validacao = int(intervalo_validacao)
        self.paciencia = None if paciencia is None else int(paciencia)
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.usar_cuda = bool(usar_cuda)
        self.balancear_origens = bool(balancear_origens)
        self.usar_augmentation = bool(usar_augmentation)
        self.parametros_modelo = dict(parametros_modelo or {})
        self.detalhes_resultado = tuple(detalhes_resultado)
        self.nome_curto_grid = bool(nome_curto_grid)
        self.salvar_historico_incremental = bool(salvar_historico_incremental)
        self.continuar_de = None if continuar_de is None else Path(continuar_de)
        self.estado_otimizador_retomado = "nao_se_aplica"
        self.epocas_origem = 0
        self.gerador_train = None
        self.otimizador = None

        # A ordem fixa evita divergencia de indices entre treino e predicao.
        self.classes = tuple(ROTULOS)
        self.class_to_idx = {
            classe: indice
            for indice, classe in enumerate(self.classes)
        }

        # Estes atributos sao preenchidos durante executar().
        self.bases_ativas = {}
        self.amostras_train = []
        self.amostras_val = []
        self.device = None
        self.modelo = None
        self.pasta_execucao = None
        self.historico = []
        self.melhor_val_loss = math.inf
        self.melhor_epoca = None
        self.validacoes_sem_melhoria = 0
        self.interrompido_por_paciencia = False
        self.epocas_executadas = 0
        self.input_shape = None
        self.assinatura_dados = None

    def _validar_parametros(self):
        """Valida hiperparametros e caminhos sem modificar arquivos."""

        if self.num_epochs <= 0:
            raise ValueError("NUM_EPOCHS deve ser maior que zero.")

        if self.batch_size <= 0:
            raise ValueError("BATCH_SIZE deve ser maior que zero.")

        if self.learning_rate <= 0:
            raise ValueError("LEARNING_RATE deve ser maior que zero.")

        if self.intervalo_validacao <= 0:
            raise ValueError("INTERVALO_VALIDACAO deve ser maior que zero.")

        if self.paciencia is not None and self.paciencia <= 0:
            raise ValueError("PACIENCIA deve ser maior que zero ou None.")

        if self.num_workers < 0:
            raise ValueError("NUM_WORKERS nao pode ser negativo.")

        if self.nome_modelo not in modelos_disponiveis():
            disponiveis = ", ".join(modelos_disponiveis())
            raise ValueError(
                f"MODELO desconhecido: '{self.nome_modelo}'. "
                f"Disponiveis: {disponiveis}."
            )

        # Mantem somente as bases explicitamente ativadas no processing.
        self.bases_ativas = {
            nome: dados
            for nome, dados in self.bases.items()
            if bool(dados.get("ativa", False))
        }

        if not self.bases_ativas:
            raise ValueError("Nenhuma base foi selecionada para treinamento.")

        erros = []

        # Train e val precisam existir para toda base ativa.
        for nome, dados in self.bases_ativas.items():
            for particao in ("train", "val"):
                caminho = Path(dados[particao])

                if not caminho.exists():
                    erros.append(f"[{nome}] {particao} nao encontrado: {caminho}")
                elif not caminho.is_dir():
                    erros.append(f"[{nome}] {particao} nao e pasta: {caminho}")

        if erros:
            raise FileNotFoundError(
                "Falha ao validar as bases de treinamento:\n- "
                + "\n- ".join(erros)
            )

    def _fixar_seed(self):
        """Fixa as principais fontes de aleatoriedade utilizadas no processo."""

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    @classmethod
    def _eh_augmentation(cls, caminho):
        """
        Informa se um NPY segue o padrao de nome gerado pelo augmentation.

        Parameters
        ----------
        caminho : str ou Path
            Caminho do arquivo que sera analisado.

        Returns
        -------
        bool
            True quando o stem termina em _augNN ou _aug_NN.
        """

        return bool(cls.PADRAO_AUGMENTATION.search(Path(caminho).stem))

    def _coletar_particao(self, particao):
        """
        Lista NPY de todas as origens ativas para uma particao.

        Em train, arquivos de augmentation sao incluidos somente quando
        usar_augmentation=True. Em val, todos os NPY sao coletados normalmente,
        pois a etapa de augmentation do projeto nao modifica essa particao.

        Cada registro mantem caminho, rotulo, indice numerico e origem. A classe
        pode existir vazia em uma origem; isso nao e tratado como erro.
        """

        registros = []

        for origem, dados in self.bases_ativas.items():
            raiz = Path(dados[particao])

            for rotulo in self.classes:
                pasta_classe = raiz / rotulo

                # O split novo deve criar todas as classes. Se a pasta estiver
                # ausente, consideramos erro estrutural em vez de ignorar.
                if not pasta_classe.exists() or not pasta_classe.is_dir():
                    raise FileNotFoundError(
                        f"[{origem}] Pasta da classe ausente em {particao}: "
                        f"{pasta_classe}"
                    )

                arquivos = sorted(
                    caminho
                    for caminho in pasta_classe.glob("*.npy")
                    if caminho.is_file()
                )

                # A flag controla apenas train. Quando desativada, os NPY
                # gerados pelo augmentation permanecem no disco, mas sao
                # ignorados pelo treinamento.
                if particao == "train" and not self.usar_augmentation:
                    arquivos = [
                        caminho
                        for caminho in arquivos
                        if not self._eh_augmentation(caminho)
                    ]

                for caminho in arquivos:
                    registros.append(
                        {
                            "caminho": caminho,
                            "rotulo": rotulo,
                            "indice": self.class_to_idx[rotulo],
                            "origem": origem,
                        }
                    )

        if not registros:
            raise RuntimeError(
                f"Nenhuma imagem NPY foi encontrada na particao {particao}."
            )

        return registros

    def _balancear_por_origem(self, registros, particao):
        """
        Iguala as origens por classe sem eliminar classes exclusivas de uma origem.

        O balanceamento e feito sobre os registros ja separados em train ou val,
        portanto nao existe transferencia de imagens entre particoes.
        """

        if not self.balancear_origens or len(self.bases_ativas) < 2:
            return list(registros)

        por_classe_origem = defaultdict(lambda: defaultdict(list))

        for registro in registros:
            por_classe_origem[registro["rotulo"]][registro["origem"]].append(registro)

        selecionados = []
        origens = tuple(sorted(self.bases_ativas))

        # Usa seed diferente entre train e val sem perder reprodutibilidade.
        deslocamento = 0 if particao == "train" else 100000

        for indice_classe, rotulo in enumerate(self.classes):
            grupos = {
                origem: list(por_classe_origem[rotulo].get(origem, []))
                for origem in origens
            }

            grupos_nao_vazios = {
                origem: grupo
                for origem, grupo in grupos.items()
                if grupo
            }

            # Nenhuma origem possui essa classe.
            if not grupos_nao_vazios:
                continue

            # Se somente uma origem possui a classe, mantemos todas as amostras.
            if len(grupos_nao_vazios) == 1:
                selecionados.extend(next(iter(grupos_nao_vazios.values())))
                continue

            # Quando todas as origens envolvidas possuem a classe, cada uma usa
            # a menor quantidade observada entre elas.
            quantidade = min(len(grupo) for grupo in grupos_nao_vazios.values())

            for indice_origem, origem in enumerate(sorted(grupos_nao_vazios)):
                grupo = grupos_nao_vazios[origem]
                rng = random.Random(
                    self.seed + deslocamento + indice_classe * 100 + indice_origem
                )

                selecionados.extend(rng.sample(grupo, quantidade))

        # Ordenacao aleatoria final reproduzivel evita agrupar classes/origens.
        rng_final = random.Random(self.seed + deslocamento + 9999)
        rng_final.shuffle(selecionados)

        return selecionados

    def _contar(self, registros):
        """Conta amostras por classe e por origem para exibicao do plano."""

        por_classe = Counter(registro["rotulo"] for registro in registros)
        por_origem = Counter(registro["origem"] for registro in registros)
        por_classe_origem = Counter(
            (registro["rotulo"], registro["origem"])
            for registro in registros
        )

        return por_classe, por_origem, por_classe_origem

    def _mostrar_plano(self):
        """Mostra exatamente quais dados serao utilizados antes de treinar."""

        print("\n" + "=" * 90)
        print("TREINAMENTO")
        print("=" * 90)
        print(f"\nBases ativas          : {', '.join(self.bases_ativas)}")
        print(f"Balancear origens     : {self.balancear_origens}")
        print(f"Usar augmentation     : {self.usar_augmentation}")
        print(f"Epocas                : {self.num_epochs}")
        print(f"Batch size            : {self.batch_size}")
        print(f"Learning rate         : {self.learning_rate}")
        print(f"Intervalo validacao   : {self.intervalo_validacao}")
        print(
            f"Paciencia validacoes  : "
            f"{self.paciencia if self.paciencia is not None else 'desativada'}"
        )
        print(f"Seed                   : {self.seed}")
        print("Conjunto test         : NAO UTILIZADO")

        for nome, registros in (
            ("TRAIN", self.amostras_train),
            ("VAL", self.amostras_val),
        ):
            por_classe, por_origem, por_classe_origem = self._contar(registros)

            print("\n" + "-" * 90)
            print(f"{nome}: {len(registros)} imagens")
            print("-" * 90)

            print("Por origem:")
            for origem in sorted(self.bases_ativas):
                print(f"  {origem:<12}: {por_origem[origem]}")

            print("Por classe:")
            for rotulo in self.classes:
                detalhes = ", ".join(
                    f"{origem}={por_classe_origem[(rotulo, origem)]}"
                    for origem in sorted(self.bases_ativas)
                )
                print(f"  {rotulo:<15}: {por_classe[rotulo]:>5} | {detalhes}")

    def _detalhes_nome_resultado(self):
        """Retorna detalhes permanentes usados no nome da pasta de treino."""

        if self.nome_curto_grid:
            # O Grid registra modelo, resolucao e flags no JSON/CSV da rodada.
            # Nome curto evita caminhos superiores ao limite comum do Windows.
            return self.detalhes_resultado

        altura, largura = self.input_shape[1], self.input_shape[2]
        resolucao = f"r{altura}" if altura == largura else f"r{largura}x{altura}"
        return (self.nome_modelo, resolucao, *self.detalhes_resultado)

    def _indicadores_nome_resultado(self):
        """Retorna somente flags de experimento que podem aparecer no nome."""

        if self.nome_curto_grid:
            return ()

        return (
            ("int", "INTERNA" in self.bases_ativas),
            ("ext", "EXTERNA" in self.bases_ativas),
            ("aug", self.usar_augmentation),
            ("bal", self.balancear_origens),
        )

    def _criar_pasta_resultados(self, inicio):
        """Cria a pasta padronizada desta execucao em estado em_andamento."""

        self.pasta_execucao = criar_pasta_resultados(
            raiz=self.pasta_resultados,
            tipo="treino",
            inicio=inicio,
            detalhes=self._detalhes_nome_resultado(),
            indicadores=self._indicadores_nome_resultado(),
        )

    def _finalizar_pasta_resultados(self, inicio, fim):
        """Renomeia a pasta de treino incluindo o horario final."""

        self.pasta_execucao = finalizar_pasta_resultados(
            pasta_atual=self.pasta_execucao,
            tipo="treino",
            inicio=inicio,
            fim=fim,
            detalhes=self._detalhes_nome_resultado(),
            indicadores=self._indicadores_nome_resultado(),
        )

    def _criar_loaders(self):
        """Cria DatasetNPY e DataLoader para train e val."""

        dataset_train = DatasetNPY(self.amostras_train)
        dataset_val = DatasetNPY(self.amostras_val)

        # Generator dedicado torna o shuffle de train reproduzivel.
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        self.gerador_train = generator

        usar_pin_memory = self.device.type == "cuda"

        train_loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=usar_pin_memory,
            generator=generator,
        )

        val_loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=usar_pin_memory,
        )

        return train_loader, val_loader

    def _configurar_dispositivo(self):
        """Seleciona CUDA quando solicitada e disponivel; caso contrario usa CPU."""

        cuda_disponivel = torch.cuda.is_available()

        if self.usar_cuda and cuda_disponivel:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"\nDispositivo           : {self.device}")

        if self.device.type == "cuda":
            print(f"GPU                    : {torch.cuda.get_device_name(0)}")
            print(f"CUDA PyTorch           : {torch.version.cuda}")
        elif self.usar_cuda and not cuda_disponivel:
            print("CUDA solicitada, mas indisponivel. Treino sera executado em CPU.")

    def _inicializar_modelo(self, train_loader):
        """Cria o modelo selecionado e valida sua interface com um lote real."""

        # ImageNet e carregado apenas ao iniciar um novo treino da VGG16.
        # Continuacoes e testes devem restaurar somente os pesos do checkpoint.
        parametros_iniciais = {}
        if self.nome_modelo == "vgg16_imagenet":
            parametros_iniciais["carregar_pesos_imagenet"] = self.continuar_de is None

        self.modelo = criar_modelo(
            self.nome_modelo,
            num_classes=len(self.classes),
            **self.parametros_modelo,
            **parametros_iniciais,
        ).to(self.device)

        imagens, _ = next(iter(train_loader))
        imagens = imagens.to(self.device, non_blocking=self.device.type == "cuda")

        # Registra a geometria efetiva usada pelo treinamento.
        self.input_shape = tuple(int(valor) for valor in imagens.shape[1:])

        # O primeiro forward tambem materializa eventuais camadas Lazy*.
        with torch.no_grad():
            saida = self.modelo(imagens[:1])

        if saida.shape != (1, len(self.classes)):
            raise RuntimeError(
                f"Saida inesperada do modelo '{self.nome_modelo}': "
                f"{tuple(saida.shape)}. Esperado (1,{len(self.classes)})."
            )

        # Procura a primeira camada Linear apenas para informar sua entrada.
        primeira_linear = next(
            (modulo for modulo in self.modelo.modules() if isinstance(modulo, nn.Linear)),
            None,
        )

        print(f"Modelo                 : {self.nome_modelo}")
        print(f"Classe do modelo       : {self.modelo.__class__.__name__}")
        print(f"Parametros modelo      : {self.parametros_modelo}")
        print(f"Entrada do modelo      : {self.input_shape}")
        print(f"Classes de saida       : {len(self.classes)}")

        if primeira_linear is not None:
            print(f"Primeira Linear        : in_features={primeira_linear.in_features}")

    def _treinar_epoca(self, loader, criterio, otimizador):
        """Executa uma epoca completa com atualizacao dos pesos."""

        self.modelo.train()

        perda_acumulada = 0.0
        acertos = 0
        total = 0

        for imagens, rotulos in loader:
            imagens = imagens.to(
                self.device,
                non_blocking=self.device.type == "cuda",
            )
            rotulos = rotulos.to(
                self.device,
                non_blocking=self.device.type == "cuda",
            )

            otimizador.zero_grad(set_to_none=True)
            saidas = self.modelo(imagens)
            perda = criterio(saidas, rotulos)
            perda.backward()
            otimizador.step()

            tamanho_lote = rotulos.size(0)
            perda_acumulada += perda.item() * tamanho_lote
            acertos += (saidas.argmax(dim=1) == rotulos).sum().item()
            total += tamanho_lote

        if total == 0:
            raise RuntimeError("train_loader nao possui amostras.")

        return perda_acumulada / total, acertos / total

    def _avaliar(self, loader, criterio):
        """Avalia o modelo sem calculo de gradientes ou atualizacao dos pesos."""

        self.modelo.eval()

        perda_acumulada = 0.0
        acertos = 0
        total = 0

        with torch.no_grad():
            for imagens, rotulos in loader:
                imagens = imagens.to(
                    self.device,
                    non_blocking=self.device.type == "cuda",
                )
                rotulos = rotulos.to(
                    self.device,
                    non_blocking=self.device.type == "cuda",
                )

                saidas = self.modelo(imagens)
                perda = criterio(saidas, rotulos)

                tamanho_lote = rotulos.size(0)
                perda_acumulada += perda.item() * tamanho_lote
                acertos += (saidas.argmax(dim=1) == rotulos).sum().item()
                total += tamanho_lote

        if total == 0:
            raise RuntimeError("val_loader nao possui amostras.")

        return perda_acumulada / total, acertos / total

    def _obter_predicoes(self, loader):
        """Retorna rotulos reais e previsoes de um DataLoader sem alterar pesos."""

        self.modelo.eval()
        rotulos_reais = []
        predicoes = []

        with torch.no_grad():
            for imagens, rotulos in loader:
                imagens = imagens.to(
                    self.device,
                    non_blocking=self.device.type == "cuda",
                )
                saidas = self.modelo(imagens)

                rotulos_reais.extend(rotulos.cpu().tolist())
                predicoes.extend(saidas.argmax(dim=1).cpu().tolist())

        return rotulos_reais, predicoes

    def _salvar_metricas_finais(self, train_loader, val_loader, criterio):
        """Salva metricas e matrizes do modelo final em train e val."""

        resumo = {}

        for nome, loader in (("treino_final", train_loader), ("validacao_final", val_loader)):
            loss, _ = self._avaliar(loader, criterio)
            reais, previstos = self._obter_predicoes(loader)
            metricas, relatorio = calcular_metricas_classificacao(
                reais,
                previstos,
                self.classes,
            )

            salvar_metricas_classificacao(
                metricas,
                relatorio,
                loss,
                self.pasta_execucao / f"metricas_{nome}.txt",
                titulo=nome.replace("_", " ").upper(),
            )

            plotar_matrizes_confusao(
                reais,
                previstos,
                self.classes,
                self.pasta_execucao,
                prefixo=nome,
            )

            resumo[nome] = {
                "loss": float(loss),
                **{chave: float(valor) for chave, valor in metricas.items()},
            }

        return resumo

    def _assinar_amostras(self):
        """Identifica arquivos efetivamente usados em train e val."""
        resumo = hashlib.sha256()
        for particao, amostras in (("train", self.amostras_train),
                                   ("val", self.amostras_val)):
            for item in amostras:
                caminho = Path(item["caminho"]).resolve()
                dados = caminho.stat()
                registro = (particao, str(caminho), item["rotulo"],
                            str(dados.st_size), str(dados.st_mtime_ns))
                resumo.update("|".join(registro).encode("utf-8"))
                resumo.update(b"\n")
        return resumo.hexdigest()

    def _capturar_rng(self):
        """Registra RNGs para continuacoes mais reprodutiveis."""
        estado_numpy = np.random.get_state()
        return {
            "python": random.getstate(),
            "numpy": (estado_numpy[0], estado_numpy[1].tolist(),
                      int(estado_numpy[2]), int(estado_numpy[3]),
                      float(estado_numpy[4])),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "data_loader": self.gerador_train.get_state() if self.gerador_train else None,
        }

    def _restaurar_rng(self, estado):
        """Restaura RNGs disponiveis; checkpoints antigos podem nao possui-los."""
        if not estado:
            return
        random.setstate(estado["python"])
        n = estado["numpy"]
        np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
        torch.set_rng_state(estado["torch"].cpu())
        if self.device.type == "cuda" and estado.get("cuda"):
            torch.cuda.set_rng_state_all([item.cpu() for item in estado["cuda"]])
        if estado.get("data_loader") is not None:
            self.gerador_train.set_state(estado["data_loader"].cpu())

    def _carregar_continuacao(self):
        """Recupera pesos finais, historico, paciencia e estado do Adam, se disponivel."""
        origem = self.continuar_de
        if not origem or not origem.is_dir():
            raise FileNotFoundError(f"Pasta de origem invalida: {origem}")
        caminho_final = origem / "modelo_final.pt"
        caminho_melhor = origem / "modelo_melhor.pt"
        caminho_historico = origem / "historico_treinamento.csv"
        for caminho in (caminho_final, caminho_melhor, caminho_historico):
            if not caminho.is_file():
                raise FileNotFoundError(f"Arquivo de continuacao ausente: {caminho}")

        # Somente checkpoints locais e confiaveis do proprio projeto.
        checkpoint = torch.load(caminho_final, map_location="cpu", weights_only=False)
        if str(checkpoint.get("model_name", "lenet5")).lower() != self.nome_modelo:
            raise ValueError("Modelo da continuacao diferente do configurado.")
        if dict(checkpoint.get("config_modelo") or {}) != self.parametros_modelo:
            raise ValueError("Parametros da arquitetura divergentes do checkpoint.")
        if tuple(checkpoint["input_shape"]) != self.input_shape:
            raise ValueError("Dimensoes de entrada divergentes da origem.")
        if list(checkpoint["classes"]) != list(self.classes):
            raise ValueError("Classes divergentes da origem.")
        if checkpoint.get("assinatura_dados") != self.assinatura_dados:
            raise ValueError("Datasets ou arquivos alterados desde o treino de origem.")
        original = checkpoint.get("config_treino", {})
        esperado = {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "intervalo_validacao": self.intervalo_validacao,
            "paciencia": self.paciencia,
            "seed": self.seed,
            "balancear_origens": self.balancear_origens,
            "usar_augmentation": self.usar_augmentation,
            "bases": list(self.bases_ativas),
        }
        for chave, valor in esperado.items():
            if original.get(chave) != valor:
                raise ValueError(f"A configuracao '{chave}' difere da origem.")
        self.epocas_origem = int(checkpoint["epoca"])
        if self.num_epochs <= self.epocas_origem:
            raise ValueError("Limite de epocas nao supera o da origem.")
        if self.epocas_origem != int(original.get("num_epochs", -1)):
            raise ValueError("Origem nao chegou ao limite configurado de epocas.")
        if checkpoint.get("interrompido_por_paciencia"):
            raise ValueError("Nao e permitido continuar um treino encerrado por paciencia.")

        with caminho_historico.open("r", newline="", encoding="utf-8-sig") as arq:
            historico = []
            for linha in csv.DictReader(arq):
                historico.append({"epoca": int(linha["epoca"]),
                                  **{campo: float(linha[campo])
                                     for campo in ("train_loss", "train_acc", "val_loss", "val_acc")}})
        if not historico or historico[-1]["epoca"] != self.epocas_origem:
            raise ValueError("Historico nao corresponde a ultima epoca do checkpoint.")
        self.historico = historico
        self.epocas_executadas = self.epocas_origem
        melhor_salvo = checkpoint.get("melhor_val_loss")
        self.melhor_val_loss = float(
            melhor_salvo if melhor_salvo is not None else min(
                linha["val_loss"] for linha in historico
                if math.isfinite(linha["val_loss"])
            )
        )
        self.melhor_epoca = checkpoint.get("melhor_epoca")
        if self.melhor_epoca is None:
            validas = [linha for linha in historico if math.isfinite(linha["val_loss"])]
            self.melhor_epoca = min(validas, key=lambda linha: linha["val_loss"])["epoca"]
        self.validacoes_sem_melhoria = int(checkpoint.get("validacoes_sem_melhoria", 0))
        self.modelo.load_state_dict(checkpoint["model_state_dict"])
        estado_adam = checkpoint.get("optimizer_state_dict")
        if estado_adam:
            self.otimizador.load_state_dict(estado_adam)
            # Carregamento em CPU pode requerer transferencia dos buffers ao device.
            for estados in self.otimizador.state.values():
                for chave, valor in estados.items():
                    if isinstance(valor, torch.Tensor) and chave != "step":
                        estados[chave] = valor.to(self.device)
            self.estado_otimizador_retomado = "restaurado"
        else:
            self.estado_otimizador_retomado = "reinicializado_checkpoint_antigo"
        estado_rng = checkpoint.get("rng_state")
        if not estado_adam or not estado_rng:
            raise ValueError(
                "Checkpoint antigo sem estado completo de Adam/RNG. "
                "Refaca o Grid Search inicial com esta versao."
            )
        self._restaurar_rng(estado_rng)
        shutil.copy2(caminho_melhor, self.pasta_execucao / "modelo_melhor.pt")
        salvar_historico_csv(self.historico, self.pasta_execucao / "historico_treinamento.csv")
        self._salvar_dados_graficos()
        print(f"CONTINUACAO: epoca {self.epocas_origem} ate {self.num_epochs}; "
              f"Adam: {self.estado_otimizador_retomado}; paciencia atual: "
              f"{self.validacoes_sem_melhoria}/{self.paciencia}")

    def _salvar_checkpoint(self, caminho, epoca, val_loss=None, val_acc=None):
        """Salva pesos e metadados necessarios para reproduzir a inferencia."""

        checkpoint = {
            "model_name": self.nome_modelo,
            "model_class": self.modelo.__class__.__name__,
            "config_modelo": dict(self.parametros_modelo),
            "model_state_dict": self.modelo.state_dict(),
            "classes": list(self.classes),
            "class_to_idx": dict(self.class_to_idx),
            "input_shape": list(self.input_shape),
            "num_classes": len(self.classes),
            "assinatura_dados": self.assinatura_dados,
            "epoca": int(epoca),
            "epocas_executadas": int(self.epocas_executadas),
            "interrompido_por_paciencia": bool(self.interrompido_por_paciencia),
            "paciencia_atingida": bool(self.interrompido_por_paciencia),
            "limite_epocas_atingido": bool(self.epocas_executadas >= self.num_epochs),
            "motivo_parada": (
                "paciencia" if self.interrompido_por_paciencia
                else "limite_epocas" if self.epocas_executadas >= self.num_epochs
                else "em_andamento"
            ),
            "validacoes_sem_melhoria": int(self.validacoes_sem_melhoria),
            "val_loss": None if val_loss is None else float(val_loss),
            "val_acc": None if val_acc is None else float(val_acc),
            "melhor_val_loss": float(self.melhor_val_loss),
            "melhor_epoca": self.melhor_epoca,
            "optimizer_state_dict": (
                self.otimizador.state_dict() if self.salvar_historico_incremental
                and self.otimizador is not None else None
            ),
            "rng_state": (self._capturar_rng() if self.salvar_historico_incremental else None),
            "estado_otimizador_retomado": self.estado_otimizador_retomado,
            "config_treino": {
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "intervalo_validacao": self.intervalo_validacao,
                "paciencia": self.paciencia,
                "seed": self.seed,
                "balancear_origens": self.balancear_origens,
                "usar_augmentation": self.usar_augmentation,
                "bases": list(self.bases_ativas),
                "usar_cuda": self.usar_cuda,
                "num_workers": self.num_workers,
            },
        }

        temporario = Path(str(caminho) + ".tmp")
        torch.save(checkpoint, temporario)
        os.replace(temporario, caminho)

    def _salvar_informacoes(self, inicio, fim, metricas_finais):
        """Salva um resumo textual da configuracao e dos resultados principais."""

        linhas = [
            "TREINAMENTO PDC-IA",
            "",
            f"Inicio: {inicio.isoformat(sep=' ', timespec='seconds')}",
            f"Fim: {fim.isoformat(sep=' ', timespec='seconds')}",
            f"Duracao_s: {(fim - inicio).total_seconds():.3f}",
            "",
            "MODELO",
            f"Identificador: {self.nome_modelo}",
            f"Classe: {self.modelo.__class__.__name__}",
            f"Parametros da arquitetura: {self.parametros_modelo}",
            f"Input shape: {self.input_shape}",
            f"Classes: {', '.join(self.classes)}",
            "",
            "DADOS",
            f"Bases ativas: {', '.join(self.bases_ativas)}",
            f"Balancear origens: {self.balancear_origens}",
            f"Usar augmentation: {self.usar_augmentation}",
            f"Train: {len(self.amostras_train)}",
            f"Val: {len(self.amostras_val)}",
            "Test: nao utilizado",
            "",
            "HIPERPARAMETROS",
            f"Epocas: {self.num_epochs}",
            f"Batch size: {self.batch_size}",
            f"Learning rate: {self.learning_rate}",
            f"Intervalo validacao: {self.intervalo_validacao}",
            f"Paciencia: {self.paciencia if self.paciencia is not None else 'desativada'}",
            f"Epocas executadas: {self.epocas_executadas}",
            f"Interrompido por paciencia: {self.interrompido_por_paciencia}",
            f"Limite de epocas atingido: {self.epocas_executadas >= self.num_epochs}",
            f"Motivo da parada: {'paciencia' if self.interrompido_por_paciencia else 'limite_epocas' if self.epocas_executadas >= self.num_epochs else 'erro'}",
            f"Validacoes sem melhoria ao final: {self.validacoes_sem_melhoria}",
            f"Seed: {self.seed}",
            f"Continuacao de: {self.continuar_de if self.continuar_de else 'nao'}",
            f"Epocas de origem: {self.epocas_origem}",
            f"Estado do Adam na retomada: {self.estado_otimizador_retomado}",
            f"Device: {self.device}",
            "",
            "MELHOR VALIDACAO",
            f"Epoca: {self.melhor_epoca}",
            f"Val loss: {self.melhor_val_loss}",
            "",
            "MODELO FINAL - TREINO",
            f"Loss: {metricas_finais['treino_final']['loss']:.6f}",
            f"Accuracy: {metricas_finais['treino_final']['accuracy']:.6f}",
            f"Balanced accuracy: {metricas_finais['treino_final']['balanced_accuracy']:.6f}",
            f"F1 macro: {metricas_finais['treino_final']['f1_macro']:.6f}",
            f"MCC: {metricas_finais['treino_final']['mcc']:.6f}",
            "",
            "MODELO FINAL - VALIDACAO",
            f"Loss: {metricas_finais['validacao_final']['loss']:.6f}",
            f"Accuracy: {metricas_finais['validacao_final']['accuracy']:.6f}",
            f"Balanced accuracy: {metricas_finais['validacao_final']['balanced_accuracy']:.6f}",
            f"F1 macro: {metricas_finais['validacao_final']['f1_macro']:.6f}",
            f"MCC: {metricas_finais['validacao_final']['mcc']:.6f}",
        ]

        (self.pasta_execucao / "informacoes_treinamento.txt").write_text(
            "\n".join(linhas),
            encoding="utf-8",
        )

    def _salvar_dados_graficos(self):
        """Salva arrays de historico para regeneracao futura dos graficos."""

        np.savez_compressed(
            self.pasta_execucao / "dados_graficos.npz",
            **{
                campo: np.asarray([item[campo] for item in self.historico])
                for campo in ("epoca", "train_loss", "train_acc", "val_loss", "val_acc")
            },
        )

    def executar(self):
        """
        Executa o fluxo completo do treinamento e retorna um resumo estruturado.
        """

        # Toda validacao estrutural acontece antes da criacao da pasta de resultados.
        self._validar_parametros()
        self._fixar_seed()

        train_completo = self._coletar_particao("train")
        val_completo = self._coletar_particao("val")

        self.amostras_train = self._balancear_por_origem(train_completo, "train")
        self.amostras_val = self._balancear_por_origem(val_completo, "val")

        if not self.amostras_train:
            raise RuntimeError("Nenhuma amostra permaneceu em train.")

        if not self.amostras_val:
            raise RuntimeError("Nenhuma amostra permaneceu em val.")

        self.assinatura_dados = self._assinar_amostras()
        self._mostrar_plano()
        self._configurar_dispositivo()

        inicio = datetime.now()
        train_loader, val_loader = self._criar_loaders()
        self._inicializar_modelo(train_loader)
        self._criar_pasta_resultados(inicio)

        criterio = nn.CrossEntropyLoss()
        otimizador = torch.optim.Adam(
            self.modelo.parameters(),
            lr=self.learning_rate,
        )
        self.otimizador = otimizador
        if self.continuar_de is not None:
            self._carregar_continuacao()

        print("\n" + "=" * 90)
        print("INICIANDO TREINAMENTO")
        print("=" * 90)

        inicio_loop = time.perf_counter()

        for epoca in range(self.epocas_origem + 1, self.num_epochs + 1):
            train_loss, train_acc = self._treinar_epoca(
                train_loader,
                criterio,
                otimizador,
            )

            # Registra a ultima epoca realmente executada. Isso e importante
            # quando o early stopping encerra o treinamento antes de NUM_EPOCHS.
            self.epocas_executadas = epoca

            # Valida na primeira epoca, no intervalo configurado e na ultima.
            executar_validacao = (
                epoca == 1
                or epoca % self.intervalo_validacao == 0
                or epoca == self.num_epochs
            )

            houve_melhoria = False

            if executar_validacao:
                val_loss, val_acc = self._avaliar(val_loader, criterio)

                # O melhor modelo e definido pela menor loss de validacao.
                # Qualquer reducao estrita da val_loss conta como melhoria e
                # reinicia o contador de paciencia.
                if val_loss < self.melhor_val_loss:
                    houve_melhoria = True
                    self.melhor_val_loss = val_loss
                    self.melhor_epoca = epoca
                    self.validacoes_sem_melhoria = 0

                    self._salvar_checkpoint(
                        self.pasta_execucao / "modelo_melhor.pt",
                        epoca=epoca,
                        val_loss=val_loss,
                        val_acc=val_acc,
                    )
                else:
                    self.validacoes_sem_melhoria += 1
            else:
                val_loss = math.nan
                val_acc = math.nan

            self.historico.append(
                {
                    "epoca": epoca,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )

            if self.salvar_historico_incremental:
                salvar_historico_csv(
                    self.historico,
                    self.pasta_execucao / "historico_treinamento.csv",
                )
                self._salvar_dados_graficos()

            if executar_validacao:
                estado_melhoria = "melhorou" if houve_melhoria else (
                    f"sem melhora {self.validacoes_sem_melhoria}/"
                    f"{self.paciencia}"
                    if self.paciencia is not None
                    else "sem melhora"
                )

                print(
                    f"Epoca {epoca:>4}/{self.num_epochs} | "
                    f"train loss={train_loss:.6f} acc={train_acc:.4f} | "
                    f"val loss={val_loss:.6f} acc={val_acc:.4f} | "
                    f"{estado_melhoria}"
                )
            else:
                print(
                    f"Epoca {epoca:>4}/{self.num_epochs} | "
                    f"train loss={train_loss:.6f} acc={train_acc:.4f}"
                )

            # PACIENCIA conta validacoes consecutivas sem reducao da val_loss.
            # Portanto, com PACIENCIA=10 e INTERVALO_VALIDACAO=5, por exemplo,
            # o criterio pode abranger varias dezenas de epocas.
            if (
                executar_validacao
                and self.paciencia is not None
                and self.validacoes_sem_melhoria >= self.paciencia
            ):
                self.interrompido_por_paciencia = True
                print(
                    "\nEARLY STOPPING: treinamento interrompido apos "
                    f"{self.validacoes_sem_melhoria} validacoes consecutivas "
                    "sem melhoria da val_loss."
                )
                break

        duracao_loop = time.perf_counter() - inicio_loop

        # Salva sempre o estado final, independentemente do melhor checkpoint.
        ultima_validacao = next(
            (
                registro
                for registro in reversed(self.historico)
                if not math.isnan(registro["val_loss"])
            ),
            None,
        )

        self._salvar_checkpoint(
            self.pasta_execucao / "modelo_final.pt",
            epoca=self.epocas_executadas,
            val_loss=None if ultima_validacao is None else ultima_validacao["val_loss"],
            val_acc=None if ultima_validacao is None else ultima_validacao["val_acc"],
        )

        salvar_historico_csv(
            self.historico,
            self.pasta_execucao / "historico_treinamento.csv",
        )
        if self.salvar_historico_incremental:
            self._salvar_dados_graficos()
        plotar_historico(
            self.historico,
            self.pasta_execucao / "historico_treinamento.png",
        )

        metricas_finais = self._salvar_metricas_finais(
            train_loader,
            val_loader,
            criterio,
        )

        fim = datetime.now()
        self._salvar_informacoes(inicio, fim, metricas_finais)
        self._finalizar_pasta_resultados(inicio, fim)

        print("\n" + "=" * 90)
        print("TREINAMENTO FINALIZADO")
        print("=" * 90)
        print(f"Resultados             : {self.pasta_execucao}")
        print(f"Melhor epoca           : {self.melhor_epoca}")
        print(f"Melhor val loss        : {self.melhor_val_loss:.6f}")
        print(f"Epocas executadas      : {self.epocas_executadas}")
        print(f"Early stopping         : {self.interrompido_por_paciencia}")
        print(f"Tempo do loop [s]      : {duracao_loop:.3f}")
        print("Test                    : nao utilizado")

        return {
            "model_name": self.nome_modelo,
            "pasta_resultados": self.pasta_execucao,
            "modelo_melhor": self.pasta_execucao / "modelo_melhor.pt",
            "modelo_final": self.pasta_execucao / "modelo_final.pt",
            "historico": self.historico,
            "melhor_epoca": self.melhor_epoca,
            "melhor_val_loss": self.melhor_val_loss,
            "paciencia": self.paciencia,
            "epocas_executadas": self.epocas_executadas,
            "interrompido_por_paciencia": self.interrompido_por_paciencia,
            "paciencia_atingida": self.interrompido_por_paciencia,
            "limite_epocas_atingido": self.epocas_executadas >= self.num_epochs,
            "motivo_parada": (
                "paciencia" if self.interrompido_por_paciencia
                else "limite_epocas" if self.epocas_executadas >= self.num_epochs
                else "erro"
            ),
            "limite_epocas": self.num_epochs,
            "validacoes_sem_melhoria": self.validacoes_sem_melhoria,
            "classes": self.classes,
            "input_shape": self.input_shape,
            "train_total": len(self.amostras_train),
            "val_total": len(self.amostras_val),
            "metricas_finais": metricas_finais,
        }
