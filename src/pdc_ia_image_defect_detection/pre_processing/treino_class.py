"""
treino_class.py

Objetivo:
- Implementar toda a logica especifica do treinamento.
- Selecionar train e val das bases interna e/ou externa.
- Combinar as origens sem utilizar o conjunto test.
- Aplicar balanceamento opcional INTERNA/EXTERNA por classe.
- Criar DataLoaders, modelo, loss e otimizador.
- Treinar a LeNet5 e validar em intervalos configuraveis.
- Salvar melhor checkpoint, checkpoint final, historico e informacoes da execucao.

Regras importantes:
- Test nunca e carregado neste modulo.
- As imagens ja chegam em NPY float32, shape (1,H,W) e intervalo [0,1].
- Augmentation ja foi materializado anteriormente no diretorio train.
- A flag usar_augmentation decide se arquivos _augXX.npy entram no train.
- Val nunca depende dessa flag e deve permanecer sem augmentation.
- O modelo e importado exclusivamente de models/lenet5.py.
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

from pdc_ia_image_defect_detection.models.lenet5 import LeNet5
from pdc_ia_image_defect_detection.utils.dataset import DatasetNPY
from pdc_ia_image_defect_detection.utils.metrics import (
    plotar_historico,
    salvar_historico_csv,
)
from pdc_ia_image_defect_detection.utils.nomenclatura import ROTULOS


class TreinoProcessor:
    """Coordena selecao de dados, treinamento, validacao e salvamento."""

    # Reconhece os nomes produzidos por augmentation_class.py.
    # Exemplos aceitos: imagem_aug01.npy e imagem_aug_01.npy.
    PADRAO_AUGMENTATION = re.compile(r"_aug_?\d+$", flags=re.IGNORECASE)

    def __init__(
        self,
        bases,
        pasta_resultados,
        num_epochs,
        batch_size,
        learning_rate,
        intervalo_validacao,
        seed,
        num_workers,
        usar_cuda,
        balancear_origens,
        usar_augmentation,
    ):
        """Armazena toda a configuracao recebida do processing/treino.py."""

        self.bases = dict(bases)
        self.pasta_resultados = Path(pasta_resultados)
        self.num_epochs = int(num_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.intervalo_validacao = int(intervalo_validacao)
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.usar_cuda = bool(usar_cuda)
        self.balancear_origens = bool(balancear_origens)
        self.usar_augmentation = bool(usar_augmentation)

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
        self.input_shape = None

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

        if self.num_workers < 0:
            raise ValueError("NUM_WORKERS nao pode ser negativo.")

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

    def _criar_pasta_resultados(self):
        """Cria uma pasta unica para os arquivos desta execucao."""

        agora = datetime.now()
        marca = agora.strftime("%Y-%m-%d_%H-%M-%S")

        # A resolucao sera acrescentada ao resumo depois do primeiro lote.
        nome = f"treino_lenet5_inicio_{marca}"

        self.pasta_resultados.mkdir(parents=True, exist_ok=True)
        self.pasta_execucao = self.pasta_resultados / nome
        self.pasta_execucao.mkdir(parents=False, exist_ok=False)

        return agora

    def _criar_loaders(self):
        """Cria DatasetNPY e DataLoader para train e val."""

        dataset_train = DatasetNPY(self.amostras_train)
        dataset_val = DatasetNPY(self.amostras_val)

        # Generator dedicado torna o shuffle de train reproduzivel.
        generator = torch.Generator()
        generator.manual_seed(self.seed)

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
        """
        Cria LeNet5 e inicializa LazyLinear com um lote real antes do otimizador.
        """

        self.modelo = LeNet5(num_classes=len(self.classes)).to(self.device)

        imagens, _ = next(iter(train_loader))
        imagens = imagens.to(self.device, non_blocking=self.device.type == "cuda")

        # Registra a geometria efetiva usada pelo treinamento.
        self.input_shape = tuple(int(valor) for valor in imagens.shape[1:])

        # O primeiro forward materializa os parametros de nn.LazyLinear.
        with torch.no_grad():
            saida = self.modelo(imagens[:1])

        if saida.shape != (1, len(self.classes)):
            raise RuntimeError(
                f"Saida inesperada do modelo: {tuple(saida.shape)}. "
                f"Esperado (1,{len(self.classes)})."
            )

        print(f"Entrada do modelo      : {self.input_shape}")
        print(f"Classes de saida       : {len(self.classes)}")
        print(f"LazyLinear in_features : {self.modelo.classifier[1].in_features}")

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

    def _salvar_checkpoint(self, caminho, epoca, val_loss=None, val_acc=None):
        """Salva pesos e metadados necessarios para reproduzir a inferencia."""

        checkpoint = {
            "model_name": "LeNet5",
            "model_state_dict": self.modelo.state_dict(),
            "classes": list(self.classes),
            "class_to_idx": dict(self.class_to_idx),
            "input_shape": list(self.input_shape),
            "num_classes": len(self.classes),
            "epoca": int(epoca),
            "val_loss": None if val_loss is None else float(val_loss),
            "val_acc": None if val_acc is None else float(val_acc),
            "config_treino": {
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "intervalo_validacao": self.intervalo_validacao,
                "seed": self.seed,
                "balancear_origens": self.balancear_origens,
                "usar_augmentation": self.usar_augmentation,
                "bases": list(self.bases_ativas),
            },
        }

        torch.save(checkpoint, caminho)

    def _salvar_informacoes(self, inicio, fim):
        """Salva um resumo textual da configuracao e dos resultados principais."""

        linhas = [
            "TREINAMENTO PDC-IA",
            "",
            f"Inicio: {inicio.isoformat(sep=' ', timespec='seconds')}",
            f"Fim: {fim.isoformat(sep=' ', timespec='seconds')}",
            f"Duracao_s: {(fim - inicio).total_seconds():.3f}",
            "",
            "MODELO",
            "LeNet5",
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
            f"Seed: {self.seed}",
            f"Device: {self.device}",
            "",
            "MELHOR VALIDACAO",
            f"Epoca: {self.melhor_epoca}",
            f"Val loss: {self.melhor_val_loss}",
        ]

        (self.pasta_execucao / "informacoes_treinamento.txt").write_text(
            "\n".join(linhas),
            encoding="utf-8",
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

        self._mostrar_plano()
        self._configurar_dispositivo()

        inicio = self._criar_pasta_resultados()
        train_loader, val_loader = self._criar_loaders()
        self._inicializar_modelo(train_loader)

        criterio = nn.CrossEntropyLoss()
        otimizador = torch.optim.Adam(
            self.modelo.parameters(),
            lr=self.learning_rate,
        )

        print("\n" + "=" * 90)
        print("INICIANDO TREINAMENTO")
        print("=" * 90)

        inicio_loop = time.perf_counter()

        for epoca in range(1, self.num_epochs + 1):
            train_loss, train_acc = self._treinar_epoca(
                train_loader,
                criterio,
                otimizador,
            )

            # Valida na primeira epoca, no intervalo configurado e na ultima.
            executar_validacao = (
                epoca == 1
                or epoca % self.intervalo_validacao == 0
                or epoca == self.num_epochs
            )

            if executar_validacao:
                val_loss, val_acc = self._avaliar(val_loader, criterio)

                # O melhor modelo e definido pela menor loss de validacao.
                if val_loss < self.melhor_val_loss:
                    self.melhor_val_loss = val_loss
                    self.melhor_epoca = epoca
                    self._salvar_checkpoint(
                        self.pasta_execucao / "modelo_melhor.pt",
                        epoca=epoca,
                        val_loss=val_loss,
                        val_acc=val_acc,
                    )
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

            if executar_validacao:
                print(
                    f"Epoca {epoca:>4}/{self.num_epochs} | "
                    f"train loss={train_loss:.6f} acc={train_acc:.4f} | "
                    f"val loss={val_loss:.6f} acc={val_acc:.4f}"
                )
            else:
                print(
                    f"Epoca {epoca:>4}/{self.num_epochs} | "
                    f"train loss={train_loss:.6f} acc={train_acc:.4f}"
                )

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
            epoca=self.num_epochs,
            val_loss=None if ultima_validacao is None else ultima_validacao["val_loss"],
            val_acc=None if ultima_validacao is None else ultima_validacao["val_acc"],
        )

        salvar_historico_csv(
            self.historico,
            self.pasta_execucao / "historico_treinamento.csv",
        )
        plotar_historico(
            self.historico,
            self.pasta_execucao / "historico_treinamento.png",
        )

        fim = datetime.now()
        self._salvar_informacoes(inicio, fim)

        print("\n" + "=" * 90)
        print("TREINAMENTO FINALIZADO")
        print("=" * 90)
        print(f"Resultados             : {self.pasta_execucao}")
        print(f"Melhor epoca           : {self.melhor_epoca}")
        print(f"Melhor val loss        : {self.melhor_val_loss:.6f}")
        print(f"Tempo do loop [s]      : {duracao_loop:.3f}")
        print("Test                    : nao utilizado")

        return {
            "pasta_resultados": self.pasta_execucao,
            "modelo_melhor": self.pasta_execucao / "modelo_melhor.pt",
            "modelo_final": self.pasta_execucao / "modelo_final.pt",
            "historico": self.historico,
            "melhor_epoca": self.melhor_epoca,
            "melhor_val_loss": self.melhor_val_loss,
            "classes": self.classes,
            "input_shape": self.input_shape,
            "train_total": len(self.amostras_train),
            "val_total": len(self.amostras_val),
        }
