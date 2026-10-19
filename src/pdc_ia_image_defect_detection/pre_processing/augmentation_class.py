"""
augmentation_class.py

Objetivo:
- Implementar toda a logica especifica do processo de data augmentation.
- Trabalhar sobre conjuntos train ja processados e armazenados em arquivos NPY.
- Permitir processar uma ou mais bases na mesma execucao.
- Validar completamente a configuracao antes de modificar qualquer arquivo.
- Remover augmentations antigos antes de gerar a nova configuracao.
- Gerar novas imagens separadamente para cada classe.
- Verificar os arquivos gerados antes de finalizar o processo.

Entrada esperada:
- Uma pasta train contendo uma subpasta para cada classe.
- Cada imagem deve estar armazenada em arquivo .npy.
- Cada NPY original deve possuir:
    dtype   = float32
    formato = (1, H, W)
    valores = intervalo [0, 1]

Identificacao dos arquivos:
- Arquivos sem sufixo de augmentation sao considerados originais.
- Arquivos terminados em _aug01, _aug02, _aug_01, etc. sao considerados
  arquivos de augmentation.
- Arquivos de augmentation nunca sao utilizados como fonte para novas imagens.

Fluxo de uma base:
1. Validar pasta, classes, configuracao e parametros.
2. Fixar as seeds utilizadas pelas operacoes aleatorias.
3. Analisar os arquivos existentes e calcular o plano de geracao.
4. Remover todos os augmentations antigos do train.
5. Gerar novos augmentations somente para as classes ativadas.
6. Reabrir e validar todos os arquivos gerados.
7. Conferir se as quantidades geradas correspondem ao esperado.
8. Retornar um resumo completo da execucao.

Transformacoes disponiveis:
- flip_horizontal:
    aplica inversao horizontal de acordo com uma probabilidade configurada.

- rotacao:
    aplica rotacao com angulo sorteado entre os limites positivo e negativo
    definidos em rotacao_graus.

- brilho:
    aplica fator aleatorio de brilho ao redor de 1.0.

- contraste:
    aplica fator aleatorio de contraste ao redor de 1.0.

- ruido:
    adiciona ruido gaussiano ao tensor normalizado.

- blur:
    aplica Gaussian Blur utilizando sigma sorteado dentro do intervalo definido.

Regras:
- Cada classe possui configuracao independente.
- Uma classe com gerar=False nao recebe novos augmentations.
- Augmentations antigos sao removidos inclusive das classes com gerar=False.
- Uma classe com gerar=True deve possuir quantidade maior que zero.
- Uma classe com gerar=True deve possuir pelo menos uma transformacao ativa.
- Nenhum arquivo e removido antes da conclusao de todas as validacoes.
- Val e test nao sao utilizados por este modulo.
- Arquivos originais nunca sao removidos ou sobrescritos.

Arquitetura:
- Os parametros ajustaveis ficam em processing/augmentation.py.
- A logica especifica de augmentation fica neste arquivo.
- Operacoes genericas de leitura, validacao e escrita de NPY ficam em utils.
"""

from collections import Counter
from pathlib import Path
import random
import re

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF

from pdc_ia_image_defect_detection.utils.npy import (
    carregar_imagem_npy,
    salvar_imagem_npy,
)


# =============================================================================
# CONSTANTES DO PROCESSO
# =============================================================================

# Reconhece os formatos de nomenclatura de augmentation utilizados no projeto.
#
# Exemplos reconhecidos:
# imagem_aug01.npy
# imagem_aug001.npy
# imagem_aug_01.npy
#
# A expressao e aplicada somente ao stem, portanto a extensao .npy nao participa
# da verificacao.
PADRAO_AUGMENTATION = re.compile(
    r"_aug_?\d+$",
    flags=re.IGNORECASE,
)

# Define os nomes validos das transformacoes configuraveis por classe.
#
# Esta lista tambem e usada para:
# - verificar campos obrigatorios;
# - validar os tipos booleanos;
# - descobrir quais transformacoes estao ativas.
TRANSFORMACOES_DISPONIVEIS = (
    "flip_horizontal",
    "rotacao",
    "brilho",
    "contraste",
    "ruido",
    "blur",
)


# =============================================================================
# PROCESSADOR DE AUGMENTATION
# =============================================================================

class AugmentationProcessor:
    """
    Executa o fluxo completo de augmentation sobre uma pasta train.

    Cada instancia representa uma unica base:
    - train interno; ou
    - train externo.

    O metodo de classe executar_bases() permite criar e executar automaticamente
    uma instancia independente para cada base habilitada.
    """

    # =========================================================================
    # EXECUCAO DE MULTIPLAS BASES
    # =========================================================================

    @staticmethod
    def _alguma_classe_ativa(config_classes):
        """
        Verifica se existe pelo menos uma classe com gerar=True.

        Parameters
        ----------
        config_classes : dict
            Configuracao de augmentation de todas as classes.

        Returns
        -------
        bool
            True quando pelo menos uma classe esta ativa.
            False quando nenhuma classe deve gerar augmentation.
        """

        # Procura o campo "gerar" em todas as configuracoes.
        # get(..., False) evita erro nesta verificacao preliminar caso o campo
        # esteja ausente; a validacao detalhada ocorre posteriormente.
        return any(
            bool(config.get("gerar", False))
            for config in config_classes.values()
        )

    @classmethod
    def executar_bases(cls, bases, config_classes, parametros, seed=42):
        """
        Executa augmentation em todas as bases habilitadas.

        Parameters
        ----------
        bases : dict
            Dicionario no formato:
            {
                "INTERNA": Path ou None,
                "EXTERNA": Path ou None,
            }

            Uma base com valor None e considerada desativada.

        config_classes : dict
            Configuracao completa das classes e transformacoes.

        parametros : dict
            Parametros numericos utilizados pelas transformacoes.

        seed : int
            Seed usada pelas operacoes aleatorias.

        Returns
        -------
        dict
            Resultado separado por base processada.

            Exemplo:
            {
                "interna": {...},
                "externa": {...},
            }
        """

        # Remove da execucao todas as bases marcadas com None.
        # Os caminhos ativos sao convertidos para Path para padronizar o uso.
        bases_ativas = {
            nome: Path(caminho)
            for nome, caminho in bases.items()
            if caminho is not None
        }

        # Exibe apenas informacoes gerais da execucao.
        print("\n" + "=" * 80)
        print("DATA AUGMENTATION")
        print("=" * 80)

        nomes_bases = ", ".join(bases_ativas) if bases_ativas else "nenhuma"

        print(f"\nBases selecionadas : {nomes_bases}")
        print(f"Seed               : {seed}")

        # Sem bases ativas nao existe pasta que possa ser modificada.
        # O retorno ocorre antes de qualquer validacao ou limpeza.
        if not bases_ativas:
            print("\nNenhuma base foi selecionada.")
            print("Nenhum arquivo foi removido ou criado.")
            return {}

        # Sem classes ativas nao ha novos arquivos para gerar.
        # O encerramento ocorre antes da remocao dos augmentations existentes.
        if not cls._alguma_classe_ativa(config_classes):
            print("\nNenhuma classe possui gerar=True.")
            print("Nenhum arquivo foi removido ou criado.")
            return {}

        # Armazena separadamente o resultado produzido por cada base.
        resultados = {}

        # Cada base e processada por uma instancia independente.
        # Isso mantem caminho, validacao, seed, contagens e resultado isolados.
        for nome_base, train_dir in bases_ativas.items():
            print("\n" + "=" * 80)
            print(f"DATA AUGMENTATION - BASE {nome_base}")
            print("=" * 80)
            print(f"\nTrain selecionado: {train_dir}")

            # Cria o processador da base atual.
            processador = cls(
                train_dir=train_dir,
                config_classes=config_classes,
                parametros=parametros,
                seed=seed,
            )

            # Executa o fluxo completo e guarda o resultado usando o nome
            # da base em minusculo.
            resultados[nome_base.lower()] = processador.executar()

        print("\n" + "=" * 80)
        print("AUGMENTATION FINALIZADO EM TODAS AS BASES")
        print("=" * 80)

        return resultados

    # =========================================================================
    # INICIALIZACAO
    # =========================================================================

    def __init__(self, train_dir, config_classes, parametros, seed=42):
        """
        Inicializa o processador de uma unica pasta train.

        Parameters
        ----------
        train_dir : str ou Path
            Caminho direto para a pasta train que sera processada.

        config_classes : dict
            Configuracao de augmentation por classe.

        parametros : dict
            Parametros numericos compartilhados pelas transformacoes.

        seed : int
            Seed utilizada pelas operacoes aleatorias.
        """

        # Padroniza o caminho recebido como objeto Path.
        self.train_dir = Path(train_dir)

        # Copia cada configuracao de classe para impedir alteracoes acidentais
        # no dicionario original definido pelo script de processing.
        self.config_classes = {
            classe: dict(config)
            for classe, config in config_classes.items()
        }

        # Copia os parametros numericos pelo mesmo motivo.
        self.parametros = dict(parametros)

        # Converte a seed para inteiro antes de utiliza-la nos geradores.
        self.seed = int(seed)

        # A lista real de classes e preenchida somente durante validar(),
        # depois da leitura das subpastas existentes em train.
        self.classes_dataset = []

    # =========================================================================
    # IDENTIFICACAO DE ARQUIVOS
    # =========================================================================

    @staticmethod
    def eh_augmentation(caminho):
        """
        Identifica se um arquivo possui nomenclatura de augmentation.

        Parameters
        ----------
        caminho : str ou Path
            Caminho do arquivo analisado.

        Returns
        -------
        bool
            True quando o stem termina com o padrao de augmentation.
        """

        # Somente o nome sem extensao e comparado com a expressao regular.
        stem = Path(caminho).stem

        return bool(
            PADRAO_AUGMENTATION.search(stem)
        )

    def listar_originais(self, classe):
        """
        Lista os NPY originais existentes em uma classe.

        Arquivos reconhecidos como augmentation sao excluidos da lista.

        Parameters
        ----------
        classe : str
            Nome da classe analisada.

        Returns
        -------
        list[Path]
            Arquivos originais ordenados pelo nome.
        """

        # Monta o caminho da classe dentro do train atual.
        pasta = self.train_dir / classe

        # Seleciona somente arquivos .npy que nao possuem sufixo de augmentation.
        return sorted(
            caminho
            for caminho in pasta.glob("*.npy")
            if not self.eh_augmentation(caminho)
        )

    def listar_augmentations(self, classe):
        """
        Lista os NPY de augmentation existentes em uma classe.

        Parameters
        ----------
        classe : str
            Nome da classe analisada.

        Returns
        -------
        list[Path]
            Arquivos de augmentation ordenados pelo nome.
        """

        # Monta o caminho da classe dentro do train atual.
        pasta = self.train_dir / classe

        # Seleciona somente arquivos reconhecidos pelo padrao de augmentation.
        return sorted(
            caminho
            for caminho in pasta.glob("*.npy")
            if self.eh_augmentation(caminho)
        )

    # =========================================================================
    # VALIDACAO DA ESTRUTURA
    # =========================================================================

    def validar(self):
        """
        Valida a estrutura do dataset e toda a configuracao da execucao.

        A validacao e concluida antes de qualquer arquivo ser removido.

        Verificacoes realizadas:
        - existencia de train;
        - train realmente ser um diretorio;
        - existencia de subpastas de classes;
        - configuracao para todas as classes existentes;
        - existencia de pelo menos uma classe ativa;
        - campos e tipos da configuracao de cada classe;
        - validade dos parametros numericos das transformacoes.

        Raises
        ------
        FileNotFoundError
            Quando train ou suas classes nao sao encontrados.

        NotADirectoryError
            Quando train existe, mas nao e um diretorio.

        ValueError, TypeError, KeyError
            Quando alguma configuracao e invalida.
        """

        # O caminho precisa existir antes da leitura das classes.
        if not self.train_dir.exists():
            raise FileNotFoundError(
                f"Pasta train nao encontrada:\n{self.train_dir}"
            )

        # Impede o uso acidental de um arquivo no lugar da pasta train.
        if not self.train_dir.is_dir():
            raise NotADirectoryError(
                f"O caminho de train nao e um diretorio:\n{self.train_dir}"
            )

        # Cada subdiretorio imediato de train e interpretado como uma classe.
        self.classes_dataset = sorted(
            pasta.name
            for pasta in self.train_dir.iterdir()
            if pasta.is_dir()
        )

        # Um train sem subpastas de classe nao pode ser processado.
        if not self.classes_dataset:
            raise FileNotFoundError(
                f"Nenhuma classe encontrada em:\n{self.train_dir}"
            )

        # Toda classe existente fisicamente precisa possuir configuracao.
        classes_sem_config = (
            set(self.classes_dataset)
            - set(self.config_classes)
        )

        if classes_sem_config:
            raise ValueError(
                "Classes do dataset sem configuracao de augmentation: "
                f"{sorted(classes_sem_config)}"
            )

        # Configuracoes extras sao permitidas.
        #
        # Isso permite manter no processing todas as classes do projeto mesmo
        # quando uma determinada base nao possuir imagens de uma delas.
        #
        # Exemplo:
        # CONFIG_AUGMENTATION possui "sem_falha", mas a base externa ainda nao.
        # Nesse caso a configuracao extra simplesmente nao e processada.

        # A existencia de pelo menos uma classe ativa e conferida novamente
        # dentro da instancia antes de permitir qualquer limpeza.
        if not any(
            bool(config.get("gerar", False))
            for config in self.config_classes.values()
        ):
            raise ValueError(
                "Nenhuma classe esta com gerar=True. "
                "A execucao foi interrompida antes da limpeza."
            )

        # Valida somente as classes realmente presentes nesta base.
        for classe in self.classes_dataset:
            config = self.config_classes[classe]
            self._validar_config_classe(classe, config)

        # Valida os parametros numericos usados pelas transformacoes.
        self._validar_parametros_transformacoes()

    # =========================================================================
    # VALIDACAO DA CONFIGURACAO POR CLASSE
    # =========================================================================

    def _validar_config_classe(self, classe, config):
        """
        Valida a configuracao de augmentation de uma classe.

        Campos obrigatorios:
        - gerar
        - quantidade
        - flip_horizontal
        - rotacao
        - brilho
        - contraste
        - ruido
        - blur

        Parameters
        ----------
        classe : str
            Nome da classe validada.

        config : dict
            Configuracao correspondente a classe.
        """

        # Todos os campos precisam existir mesmo quando gerar=False.
        # Isso mantem a configuracao uniforme entre todas as classes.
        campos_obrigatorios = {
            "gerar",
            "quantidade",
            *TRANSFORMACOES_DISPONIVEIS,
        }

        # Calcula quais campos obrigatorios nao foram informados.
        faltando = campos_obrigatorios - set(config)

        if faltando:
            raise KeyError(
                f"{classe}: campos ausentes na configuracao: "
                f"{sorted(faltando)}"
            )

        # "gerar" funciona exclusivamente como flag booleana.
        if not isinstance(config["gerar"], bool):
            raise TypeError(
                f"{classe}: 'gerar' deve ser True ou False."
            )

        # A quantidade representa o numero de copias por imagem original.
        # Valores negativos e valores nao inteiros sao invalidos.
        if (
            not isinstance(config["quantidade"], int)
            or config["quantidade"] < 0
        ):
            raise ValueError(
                f"{classe}: 'quantidade' deve ser inteiro >= 0."
            )

        # Uma classe ativa precisa produzir pelo menos uma copia.
        if config["gerar"] and config["quantidade"] == 0:
            raise ValueError(
                f"{classe}: gerar=True exige quantidade > 0."
            )

        # Cada transformacao precisa ser explicitamente True ou False.
        for nome in TRANSFORMACOES_DISPONIVEIS:
            if not isinstance(config[nome], bool):
                raise TypeError(
                    f"{classe}: '{nome}' deve ser True ou False."
                )

        # Uma classe ativa sem transformacoes apenas duplicaria a imagem original.
        # Essa configuracao e tratada como erro.
        if config["gerar"] and not any(
            config[nome]
            for nome in TRANSFORMACOES_DISPONIVEIS
        ):
            raise ValueError(
                f"{classe}: gerar=True, mas nenhuma transformacao foi ativada."
            )

    # =========================================================================
    # VALIDACAO DOS PARAMETROS DAS TRANSFORMACOES
    # =========================================================================

    def _validar_parametros_transformacoes(self):
        """
        Valida os parametros numericos compartilhados pelas classes.

        Os parametros sao validados mesmo que uma determinada transformacao
        esteja desativada em algumas classes. Isso garante que a configuracao
        completa do processo esteja consistente antes da limpeza.
        """

        # Define todos os parametros exigidos pelo conjunto atual de transformacoes.
        obrigatorios = {
            "prob_flip_horizontal",
            "rotacao_graus",
            "brilho",
            "contraste",
            "ruido_std",
            "blur_kernel_size",
            "blur_sigma_min",
            "blur_sigma_max",
        }

        # Detecta parametros ausentes.
        faltando = obrigatorios - set(self.parametros)

        if faltando:
            raise KeyError(
                "Parametros de augmentation ausentes: "
                f"{sorted(faltando)}"
            )

        # Probabilidades validas devem pertencer ao intervalo fechado [0, 1].
        prob = self.parametros["prob_flip_horizontal"]

        if not 0.0 <= prob <= 1.0:
            raise ValueError(
                "prob_flip_horizontal deve estar entre 0 e 1."
            )

        # O limite de rotacao representa uma magnitude absoluta.
        if self.parametros["rotacao_graus"] < 0:
            raise ValueError(
                "rotacao_graus deve ser >= 0."
            )

        # A variacao de brilho nao pode ser negativa.
        if self.parametros["brilho"] < 0:
            raise ValueError(
                "brilho deve ser >= 0."
            )

        # A variacao de contraste nao pode ser negativa.
        if self.parametros["contraste"] < 0:
            raise ValueError(
                "contraste deve ser >= 0."
            )

        # O desvio padrao do ruido gaussiano nao pode ser negativo.
        if self.parametros["ruido_std"] < 0:
            raise ValueError(
                "ruido_std deve ser >= 0."
            )

        # O Gaussian Blur exige kernel positivo e impar.
        kernel = self.parametros["blur_kernel_size"]

        if (
            not isinstance(kernel, int)
            or kernel <= 0
            or kernel % 2 == 0
        ):
            raise ValueError(
                "blur_kernel_size deve ser inteiro positivo e impar."
            )

        # O sigma precisa ser positivo e o limite minimo nao pode ultrapassar
        # o limite maximo.
        sigma_min = self.parametros["blur_sigma_min"]
        sigma_max = self.parametros["blur_sigma_max"]

        if not 0 < sigma_min <= sigma_max:
            raise ValueError(
                "blur_sigma_min/max possuem valores invalidos."
            )

    # =========================================================================
    # ALEATORIEDADE
    # =========================================================================

    def fixar_seed(self):
        """
        Fixa as seeds utilizadas pelas operacoes aleatorias.

        Geradores utilizados:
        - random:
            flip, rotacao, brilho, contraste e sigma do blur.

        - numpy:
            mantido sincronizado para operacoes numericas que utilizem NumPy.

        - torch:
            geracao do ruido gaussiano.

        O mesmo conjunto de arquivos, parametros e seed produz a mesma sequencia
        de sorteios.
        """

        # Controla os sorteios realizados pelo modulo random.
        random.seed(self.seed)

        # Mantem o gerador NumPy sincronizado com a seed da execucao.
        np.random.seed(self.seed)

        # Controla os sorteios realizados pelo PyTorch.
        torch.manual_seed(self.seed)

    # =========================================================================
    # CONSULTA DAS TRANSFORMACOES
    # =========================================================================

    def transformacoes_ativas(self, classe):
        """
        Retorna as transformacoes habilitadas para uma classe.

        Parameters
        ----------
        classe : str
            Nome da classe consultada.

        Returns
        -------
        list[str]
            Nomes das transformacoes configuradas como True.
        """

        # Recupera a configuracao correspondente a classe.
        config = self.config_classes[classe]

        # Mantem somente os nomes cuja flag esta ativa.
        return [
            nome
            for nome in TRANSFORMACOES_DISPONIVEIS
            if config[nome]
        ]

    # =========================================================================
    # PLANO DE EXECUCAO
    # =========================================================================

    def mostrar_plano(self):
        """
        Analisa o estado atual do train e calcula a geracao prevista.

        Para cada classe sao contabilizados:
        - arquivos originais;
        - augmentations antigos;
        - copias configuradas por original;
        - quantidade prevista de novos arquivos;
        - transformacoes ativas.

        Returns
        -------
        dict
            Totais da base antes da limpeza.

            {
                "originais": quantidade de originais,
                "antigos": quantidade de augmentations antigos,
                "previstos": quantidade de novos augmentations,
            }
        """

        print("\n" + "=" * 80)
        print("PLANO DE DATA AUGMENTATION")
        print("=" * 80)
        print(f"\nTrain: {self.train_dir}")
        print(f"Seed : {self.seed}")
        print("\nAugmentations antigos serao removidos antes da nova geracao.")
        print("Somente a pasta train selecionada sera modificada.")

        # Acumuladores gerais da base.
        total_originais = 0
        total_antigos = 0
        total_previsto = 0

        # Analisa cada classe encontrada fisicamente em train.
        for classe in self.classes_dataset:
            # Recupera a configuracao da classe atual.
            config = self.config_classes[classe]

            # Separa arquivos originais e arquivos de augmentation existentes.
            originais = self.listar_originais(classe)
            antigos = self.listar_augmentations(classe)

            # Classes desativadas possuem previsao de zero novas copias.
            quantidade = (
                config["quantidade"]
                if config["gerar"]
                else 0
            )

            # Cada original gera exatamente "quantidade" novos arquivos.
            previsto = len(originais) * quantidade

            # Lista as transformacoes ativas apenas para exibicao do plano.
            ativas = self.transformacoes_ativas(classe)

            # Atualiza os totais da base.
            total_originais += len(originais)
            total_antigos += len(antigos)
            total_previsto += previsto

            # Exibe a composicao da classe antes da modificacao.
            print(f"\n{classe}")
            print(f"  originais             : {len(originais)}")
            print(f"  augmentations antigos : {len(antigos)}")
            print(f"  gerar                 : {config['gerar']}")
            print(f"  copias por original   : {quantidade}")
            print(f"  novos previstos       : {previsto}")

            print(
                "  transformacoes        : "
                + (
                    ", ".join(ativas)
                    if ativas
                    else "nenhuma"
                )
            )

        # Exibe os totais previstos para a base completa.
        print("\n" + "-" * 80)
        print(f"Originais totais       : {total_originais}")
        print(f"Augmentations antigos  : {total_antigos}")
        print(f"Novos previstos        : {total_previsto}")
        print(f"Total final previsto   : {total_originais + total_previsto}")

        # Esses valores sao reutilizados posteriormente para conferir a geracao.
        return {
            "originais": total_originais,
            "antigos": total_antigos,
            "previstos": total_previsto,
        }

    # =========================================================================
    # LIMPEZA
    # =========================================================================

    def remover_augmentations_antigos(self):
        """
        Remove todos os arquivos de augmentation existentes no train atual.

        A limpeza ocorre para todas as classes presentes no dataset, inclusive
        classes configuradas com gerar=False.

        Os arquivos originais nao correspondem ao padrao de augmentation e,
        portanto, nao sao removidos.

        Returns
        -------
        Counter
            Quantidade de arquivos removidos por classe.
        """

        # Counter facilita armazenar e somar as contagens por classe.
        removidos = Counter()

        # Percorre todas as classes fisicamente presentes na base.
        for classe in self.classes_dataset:
            # Lista somente arquivos reconhecidos como augmentation.
            arquivos = self.listar_augmentations(classe)

            # Remove cada arquivo encontrado.
            for caminho in arquivos:
                caminho.unlink()

            # Registra a quantidade removida nesta classe.
            removidos[classe] = len(arquivos)

        return removidos

    # =========================================================================
    # APLICACAO DAS TRANSFORMACOES
    # =========================================================================

    def aplicar_augmentation(self, tensor, classe):
        """
        Aplica ao tensor as transformacoes habilitadas para uma classe.

        As transformacoes sao avaliadas na seguinte ordem:
        1. flip horizontal;
        2. rotacao;
        3. brilho;
        4. contraste;
        5. ruido;
        6. blur.

        Uma transformacao configurada como False e ignorada.

        Parameters
        ----------
        tensor : torch.Tensor
            Imagem original no formato (1, H, W).

        classe : str
            Classe usada para consultar quais transformacoes estao habilitadas.

        Returns
        -------
        torch.Tensor
            Nova imagem float32 limitada ao intervalo [0, 1].
        """

        # A funcao opera diretamente com tensores PyTorch.
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "A entrada do augmentation deve ser torch.Tensor."
            )

        # As imagens do projeto utilizam um canal no formato (1,H,W).
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise ValueError(
                f"Formato esperado (1,H,W), recebido {tuple(tensor.shape)}."
            )

        # Recupera a configuracao especifica da classe.
        config = self.config_classes[classe]

        # Cria uma copia float do original.
        # O tensor de entrada permanece inalterado.
        saida = tensor.clone().float()

        # ---------------------------------------------------------------------
        # FLIP HORIZONTAL
        # ---------------------------------------------------------------------

        if config["flip_horizontal"]:
            # O flip possui uma probabilidade propria.
            # Quando o sorteio supera a probabilidade, a imagem segue sem flip.
            if random.random() < self.parametros["prob_flip_horizontal"]:
                saida = TF.hflip(saida)

        # ---------------------------------------------------------------------
        # ROTACAO
        # ---------------------------------------------------------------------

        if config["rotacao"]:
            # Define o maior modulo permitido para o angulo.
            graus = self.parametros["rotacao_graus"]

            # Sorteia um angulo uniforme entre -graus e +graus.
            angulo = random.uniform(
                -graus,
                graus,
            )

            # Usa interpolacao bilinear para evitar o aspecto serrilhado produzido
            # pela interpolacao por vizinho mais proximo.
            #
            # Pixels criados fora da imagem original recebem valor zero.
            saida = TF.rotate(
                saida,
                angle=angulo,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0.0,
            )

        # ---------------------------------------------------------------------
        # BRILHO
        # ---------------------------------------------------------------------

        if config["brilho"]:
            # "variacao" define o afastamento maximo em relacao ao fator neutro 1.
            variacao = self.parametros["brilho"]

            # Exemplo para variacao=0.15:
            # fator sorteado entre 0.85 e 1.15.
            fator = random.uniform(
                max(0.0, 1.0 - variacao),
                1.0 + variacao,
            )

            # Fator 1.0 mantem o brilho original.
            saida = TF.adjust_brightness(
                saida,
                fator,
            )

        # ---------------------------------------------------------------------
        # CONTRASTE
        # ---------------------------------------------------------------------

        if config["contraste"]:
            # Define o afastamento maximo do fator neutro de contraste.
            variacao = self.parametros["contraste"]

            # Exemplo para variacao=0.15:
            # fator sorteado entre 0.85 e 1.15.
            fator = random.uniform(
                max(0.0, 1.0 - variacao),
                1.0 + variacao,
            )

            # Fator 1.0 mantem o contraste original.
            saida = TF.adjust_contrast(
                saida,
                fator,
            )

        # ---------------------------------------------------------------------
        # RUIDO GAUSSIANO
        # ---------------------------------------------------------------------

        if config["ruido"]:
            # Cria um tensor de ruido com media zero e desvio padrao unitario,
            # mantendo exatamente o mesmo formato da imagem.
            ruido = torch.randn_like(saida)

            # Ajusta a intensidade pelo desvio padrao configurado.
            ruido *= self.parametros["ruido_std"]

            # Soma o ruido diretamente aos pixels normalizados.
            saida = saida + ruido

        # ---------------------------------------------------------------------
        # GAUSSIAN BLUR
        # ---------------------------------------------------------------------

        if config["blur"]:
            # Sorteia o sigma dentro do intervalo configurado.
            sigma = random.uniform(
                self.parametros["blur_sigma_min"],
                self.parametros["blur_sigma_max"],
            )

            # Recupera o tamanho do kernel previamente validado como impar.
            kernel = self.parametros["blur_kernel_size"]

            # Usa o mesmo kernel e sigma nos dois eixos espaciais.
            saida = TF.gaussian_blur(
                saida,
                kernel_size=[kernel, kernel],
                sigma=[sigma, sigma],
            )

        # Ruido, interpolacao e ajustes de intensidade podem produzir valores
        # ligeiramente abaixo de 0 ou acima de 1.
        #
        # clamp restaura o intervalo padrao do dataset.
        saida = saida.clamp(
            0.0,
            1.0,
        )

        # contiguous garante armazenamento continuo antes da conversao para NumPy.
        return saida.contiguous()

    # =========================================================================
    # NOMENCLATURA
    # =========================================================================

    @staticmethod
    def nome_augmentation(caminho_original, indice):
        """
        Gera o caminho de destino de uma copia de augmentation.

        Parameters
        ----------
        caminho_original : Path
            Arquivo original utilizado como fonte.

        indice : int
            Numero da copia de augmentation.

        Returns
        -------
        Path
            Caminho no formato:
            <stem_original>_aug01.npy
        """

        # :02d utiliza pelo menos dois digitos.
        # Exemplos: 01, 02, 10, 100.
        nome = (
            f"{caminho_original.stem}"
            f"_aug{indice:02d}"
            f".npy"
        )

        # O novo arquivo permanece na mesma pasta da classe do original.
        return caminho_original.parent / nome

    # =========================================================================
    # GERACAO DE UMA CLASSE
    # =========================================================================

    def gerar_classe(self, classe):
        """
        Gera todos os augmentations configurados para uma classe.

        Para cada imagem original:
        - carrega o NPY validado;
        - converte para tensor;
        - gera a quantidade configurada de copias;
        - realiza novos sorteios em cada copia;
        - salva cada resultado como novo NPY.

        Parameters
        ----------
        classe : str
            Classe que sera processada.

        Returns
        -------
        int
            Quantidade total de arquivos gerados.
        """

        # Recupera a configuracao da classe.
        config = self.config_classes[classe]

        # Classe desativada nao gera nenhum arquivo.
        if not config["gerar"]:
            return 0

        # Lista exclusivamente os originais.
        # Augmentations antigos nao podem entrar como entrada.
        originais = self.listar_originais(classe)

        # Quantidade de copias criada para cada original.
        quantidade = config["quantidade"]

        # Contador total da classe.
        gerados = 0

        # Processa cada arquivo original uma unica vez.
        for numero, caminho in enumerate(
            originais,
            start=1,
        ):
            # Carrega o NPY usando a validacao generica de utils.npy.
            array_original = carregar_imagem_npy(
                caminho
            )

            # Converte o array NumPy para tensor PyTorch sem alterar os dados.
            tensor_original = torch.from_numpy(
                array_original
            )

            # Cria as copias configuradas para este original.
            for indice_aug in range(
                1,
                quantidade + 1,
            ):
                # Cada chamada realiza novos sorteios para as transformacoes
                # aleatorias habilitadas.
                tensor_aug = self.aplicar_augmentation(
                    tensor_original,
                    classe,
                )

                # Move o tensor para CPU, converte para NumPy e garante float32.
                array_aug = (
                    tensor_aug
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )

                # Define o nome correspondente ao original e ao numero da copia.
                destino = self.nome_augmentation(
                    caminho,
                    indice_aug,
                )

                # Salva utilizando a rotina generica de utils.npy.
                salvar_imagem_npy(
                    destino,
                    array_aug,
                )

                # Atualiza a quantidade criada nesta classe.
                gerados += 1

            # Atualiza uma unica linha do terminal com o progresso da classe.
            print(
                f"\r  {classe:<15} "
                f"{numero:>4}/{len(originais):<4} "
                f"| gerados={gerados}",
                end="",
                flush=True,
            )

        # Finaliza a linha de progresso somente quando existiam originais.
        if originais:
            print()

        return gerados

    # =========================================================================
    # GERACAO DE TODAS AS CLASSES
    # =========================================================================

    def gerar_todas_classes(self):
        """
        Executa a geracao para todas as classes existentes no train.

        Returns
        -------
        Counter
            Quantidade de novos arquivos gerados por classe.
        """

        # Armazena a quantidade gerada individualmente por classe.
        gerados = Counter()

        print("\n" + "=" * 80)
        print("GERANDO AUGMENTATIONS")
        print("=" * 80)

        # Percorre somente as classes realmente existentes nesta base.
        for classe in self.classes_dataset:
            # Classes desativadas permanecem somente com seus arquivos originais.
            if not self.config_classes[classe]["gerar"]:
                print(f"\n{classe}: geracao desativada.")
                gerados[classe] = 0
                continue

            # Executa a geracao completa da classe ativa.
            print(f"\nClasse: {classe}")

            gerados[classe] = self.gerar_classe(
                classe
            )

        return gerados

    # =========================================================================
    # VERIFICACAO FINAL
    # =========================================================================

    def verificar_resultado(self):
        """
        Verifica a integridade e a quantidade dos arquivos depois da geracao.

        Para cada classe:
        - conta os originais;
        - conta os augmentations encontrados;
        - calcula a quantidade esperada;
        - compara quantidade encontrada e esperada;
        - reabre todos os augmentations;
        - valida formato, dtype, finitude e intervalo por meio de utils.npy.

        Returns
        -------
        dict
            Resumo final por classe.

        Raises
        ------
        RuntimeError
            Quando alguma quantidade esta incorreta ou um arquivo gerado falha
            na validacao.
        """

        print("\n" + "=" * 80)
        print("VERIFICACAO FINAL")
        print("=" * 80)

        # Armazena todas as falhas para apresentar um diagnostico conjunto.
        erros = []

        # Armazena a composicao final de cada classe.
        resumo = {}

        # Totais gerais da base.
        total_originais = 0
        total_aug = 0

        # Verifica cada classe individualmente.
        for classe in self.classes_dataset:
            # Recupera configuracao e arquivos atuais.
            config = self.config_classes[classe]
            originais = self.listar_originais(classe)
            augmentations = self.listar_augmentations(classe)

            # Classes desativadas devem terminar com zero augmentation.
            quantidade = (
                config["quantidade"]
                if config["gerar"]
                else 0
            )

            # Quantidade esperada:
            # numero de originais x numero de copias por original.
            esperado = (
                len(originais)
                * quantidade
            )

            # Atualiza os totais globais.
            total_originais += len(originais)
            total_aug += len(augmentations)

            # Registra divergencia entre quantidade esperada e encontrada.
            if len(augmentations) != esperado:
                erros.append(
                    f"{classe}: "
                    f"esperado={esperado}, "
                    f"encontrado={len(augmentations)}"
                )

            # Reabre todos os novos arquivos.
            #
            # carregar_imagem_npy valida o padrao exigido pelo projeto.
            for caminho in augmentations:
                try:
                    carregar_imagem_npy(
                        caminho
                    )

                except Exception as erro:
                    erros.append(
                        f"{classe}/{caminho.name}: {erro}"
                    )

            # Registra a composicao final da classe.
            resumo[classe] = {
                "originais": len(originais),
                "augmentations": len(augmentations),
                "esperado": esperado,
            }

            # Define o estado quantitativo da classe para exibicao.
            status = (
                "OK"
                if len(augmentations) == esperado
                else "ERRO"
            )

            print(
                f"{classe:<15} | "
                f"originais={len(originais):>4} | "
                f"aug={len(augmentations):>5} | "
                f"esperado={esperado:>5} | "
                f"{status}"
            )

        # Mostra a composicao final completa da base.
        print("-" * 80)
        print(f"Originais : {total_originais}")
        print(f"Augmented : {total_aug}")
        print(f"Total     : {total_originais + total_aug}")

        # Qualquer erro invalida a execucao.
        # Todas as falhas encontradas sao apresentadas juntas.
        if erros:
            raise RuntimeError(
                "Falha na verificacao final:\n- "
                + "\n- ".join(erros)
            )

        print("\nVerificacao concluida sem erros.")

        return resumo

    # =========================================================================
    # EXECUCAO COMPLETA DE UMA BASE
    # =========================================================================

    def executar(self):
        """
        Executa o fluxo completo de augmentation da base atual.

        Ordem:
        1. validar configuracao e estrutura;
        2. fixar seeds;
        3. calcular plano;
        4. remover augmentations antigos;
        5. gerar novos arquivos;
        6. conferir quantidade gerada;
        7. validar os arquivos finais.

        Returns
        -------
        dict
            Informacoes completas da execucao:

            {
                "plano": {...},
                "removidos": {...},
                "gerados": {...},
                "resultado": {...},
            }
        """

        # Valida todas as condicoes antes de modificar qualquer arquivo.
        self.validar()

        # Inicializa os geradores aleatorios com a seed configurada.
        self.fixar_seed()

        # Analisa a base antes da limpeza e calcula a quantidade prevista.
        plano = self.mostrar_plano()

        # ---------------------------------------------------------------------
        # LIMPEZA
        # ---------------------------------------------------------------------

        print("\n" + "=" * 80)
        print("REMOVENDO AUGMENTATIONS ANTIGOS")
        print("=" * 80)

        # Remove todos os arquivos antigos reconhecidos como augmentation.
        removidos = self.remover_augmentations_antigos()

        # Exibe a quantidade removida em cada classe.
        for classe in self.classes_dataset:
            print(
                f"{classe:<15}: "
                f"{removidos[classe]} removido(s)"
            )

        # Exibe a quantidade total removida da base.
        print(
            f"Total removido: "
            f"{sum(removidos.values())}"
        )

        # ---------------------------------------------------------------------
        # GERACAO
        # ---------------------------------------------------------------------

        # Gera os novos arquivos somente depois da limpeza completa.
        gerados = self.gerar_todas_classes()

        # Confere a quantidade total imediatamente apos a geracao.
        #
        # Esta verificacao detecta falhas antes da etapa de reabertura dos arquivos.
        if sum(gerados.values()) != plano["previstos"]:
            raise RuntimeError(
                "Quantidade total gerada difere da quantidade prevista."
            )

        # ---------------------------------------------------------------------
        # VERIFICACAO
        # ---------------------------------------------------------------------

        # Confere novamente as quantidades e valida fisicamente os arquivos.
        resumo_final = self.verificar_resultado()

        print("\n" + "=" * 80)
        print("AUGMENTATION FINALIZADO")
        print("=" * 80)

        print(f"\nTrain processado:\n{self.train_dir}")
        print("Val e test nao foram modificados.")

        # Retorna informacoes estruturadas para o processing e para o futuro main.
        return {
            "plano": plano,
            "removidos": dict(removidos),
            "gerados": dict(gerados),
            "resultado": resumo_final,
        }