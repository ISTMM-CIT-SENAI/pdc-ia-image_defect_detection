"""
split_class.py

Objetivo:
- Implementar toda a logica especifica da divisao dos datasets processados.
- Separar cada classe em train, val e test utilizando proporcoes configuraveis.
- Processar a base interna e a base externa de forma independente.
- Preservar os arquivos processados originais e criar copias fisicas nos splits.
- Validar completamente as entradas antes de remover qualquer split existente.

Entrada esperada:
- Uma pasta de dataset processado.
- Cada subpasta direta representa uma classe.
- Cada classe contem imagens no formato .npy padrao do projeto.

Saida gerada:
<dataset>_split/
    train/
        <classe>/
    val/
        <classe>/
    test/
        <classe>/

Divisao por classe:
1. Os arquivos da classe sao ordenados pelo nome.
2. A lista e embaralhada utilizando a seed configurada.
3. Train recebe round(total * proporcao_train).
4. Val recebe round(total * proporcao_val), limitado ao numero restante.
5. Test recebe todas as imagens restantes.
6. Cada arquivo pertence a exatamente um dos tres conjuntos.

Referencia de classes:
- Quando dataset_processado existe, suas subpastas definem as classes de
  referencia do projeto.
- A base externa pode possuir menos classes que a referencia.
- Classes ausentes na base externa sao mantidas como pastas vazias no split.
- A base externa nao pode introduzir uma classe que nao exista na referencia.
- Se a referencia interna nao estiver disponivel, as classes das bases ativas
  sao utilizadas sem essa comparacao adicional.

Seguranca:
- Todas as proporcoes sao validadas antes do processamento.
- Todas as bases selecionadas sao verificadas antes de qualquer limpeza.
- Todos os arquivos NPY de entrada sao abertos e validados antes da limpeza.
- Splits antigos sao removidos somente depois que todas as validacoes terminam.
- Os arquivos de entrada nunca sao movidos, renomeados ou removidos.

Arquitetura:
- Os parametros ajustaveis ficam em processing/split.py.
- A logica especifica do split fica neste arquivo.
- A validacao generica dos arquivos NPY fica em utils/npy.py.
"""

from pathlib import Path
import random
import shutil

from pdc_ia_image_defect_detection.utils.npy import carregar_imagem_npy


# =============================================================================
# PROCESSADOR DE SPLIT
# =============================================================================

class SplitProcessor:
    """
    Divide uma base processada em train, val e test.

    Cada instancia representa uma unica base de entrada e sua respectiva pasta
    de saida. O metodo executar_bases() coordena varias bases na mesma execucao.
    """

    # Nomes fixos das tres particoes produzidas pelo processo.
    CONJUNTOS = ("train", "val", "test")

    # =========================================================================
    # EXECUCAO DE MULTIPLAS BASES
    # =========================================================================

    @classmethod
    def executar_bases(cls, bases, proporcoes, seed=42, referencia_classes=None):
        """
        Valida e divide todas as bases habilitadas.

        Parameters
        ----------
        bases : dict
            Configuracao das bases no formato:
            {
                "INTERNA": {
                    "usar": bool,
                    "origem": Path,
                    "destino": Path,
                },
                "EXTERNA": {...},
            }

        proporcoes : dict
            Proporcoes destinadas a train, val e test.

        seed : int
            Seed utilizada no embaralhamento das imagens.

        referencia_classes : str ou Path ou None
            Pasta usada para obter a lista de classes de referencia do projeto.

        Returns
        -------
        dict
            Resumo separado por base processada.
        """

        # Mantem somente as bases explicitamente ativadas pelo processing.
        bases_ativas = {
            nome: config
            for nome, config in bases.items()
            if bool(config.get("usar", False))
        }

        print("\n" + "=" * 80)
        print("SPLIT DOS DATASETS")
        print("=" * 80)

        nomes = ", ".join(bases_ativas) if bases_ativas else "nenhuma"
        print(f"\nBases selecionadas : {nomes}")
        print(f"Seed               : {seed}")

        # Sem bases ativas nao existe nenhuma pasta que possa ser alterada.
        if not bases_ativas:
            print("\nNenhuma base foi selecionada.")
            print("Nenhum arquivo foi removido ou criado.")
            return {}

        # Valida as proporcoes uma unica vez antes de analisar os datasets.
        proporcoes_validadas = cls._validar_proporcoes(proporcoes)

        # Obtem a lista de classes utilizada como referencia entre as bases.
        classes_referencia = cls._obter_classes_referencia(
            referencia_classes=referencia_classes,
            bases_ativas=bases_ativas,
        )

        # Cria uma instancia independente para cada base selecionada.
        processadores = {}

        for nome, config in bases_ativas.items():
            processadores[nome] = cls(
                origem=config["origem"],
                destino=config["destino"],
                proporcoes=proporcoes_validadas,
                seed=seed,
                nome_base=nome,
                classes_referencia=classes_referencia,
            )

        # Todas as bases sao validadas antes que qualquer split antigo seja apagado.
        for processador in processadores.values():
            processador.validar()

        # Somente depois da validacao completa inicia a recriacao dos splits.
        resultados = {}

        for nome, processador in processadores.items():
            resultados[nome.lower()] = processador.executar_validado()

        print("\n" + "=" * 80)
        print("SPLIT FINALIZADO EM TODAS AS BASES")
        print("=" * 80)

        return resultados

    @staticmethod
    def _validar_proporcoes(proporcoes):
        """
        Valida as proporcoes de train, val e test.

        Returns
        -------
        dict
            Copia das proporcoes convertidas para float.
        """

        # Os tres conjuntos precisam estar explicitamente configurados.
        obrigatorios = set(SplitProcessor.CONJUNTOS)
        faltando = obrigatorios - set(proporcoes)

        if faltando:
            raise KeyError(
                "Proporcoes ausentes para: "
                f"{sorted(faltando)}"
            )

        # Converte os valores para float sem alterar o dicionario recebido.
        valores = {
            nome: float(proporcoes[nome])
            for nome in SplitProcessor.CONJUNTOS
        }

        # Cada proporcao precisa pertencer ao intervalo fechado [0,1].
        for nome, valor in valores.items():
            if not 0.0 <= valor <= 1.0:
                raise ValueError(
                    f"Proporcao de {nome} deve estar entre 0 e 1. "
                    f"Recebido: {valor}."
                )

        # A soma precisa representar exatamente 100% do conjunto.
        soma = sum(valores.values())

        if abs(soma - 1.0) > 1e-9:
            raise ValueError(
                "As proporcoes de train, val e test devem somar 1.0. "
                f"Soma recebida: {soma:.12f}."
            )

        return valores

    @staticmethod
    def _listar_classes(pasta):
        """
        Retorna as subpastas diretas utilizadas como classes.

        Parameters
        ----------
        pasta : str ou Path
            Pasta de dataset processado analisada.

        Returns
        -------
        list[str]
            Nomes das classes em ordem alfabetica.
        """

        # Padroniza o caminho recebido.
        pasta = Path(pasta)

        # Uma pasta inexistente nao possui classes validas para consulta.
        if not pasta.exists() or not pasta.is_dir():
            return []

        # Somente subdiretorios diretos representam classes.
        return sorted(
            item.name
            for item in pasta.iterdir()
            if item.is_dir()
        )

    @classmethod
    def _obter_classes_referencia(cls, referencia_classes, bases_ativas):
        """
        Determina a lista de classes usada para comparar as bases.

        A pasta interna processada e utilizada quando estiver disponivel.
        Caso contrario, a lista e formada pelas classes presentes nas bases
        selecionadas.
        """

        # Prioriza a pasta interna informada pelo processing.
        if referencia_classes is not None:
            classes = cls._listar_classes(referencia_classes)

            if classes:
                return classes

        # Sem referencia interna disponivel, usa a uniao das bases ativas.
        classes = set()

        for config in bases_ativas.values():
            classes.update(
                cls._listar_classes(config["origem"])
            )

        if not classes:
            raise FileNotFoundError(
                "Nenhuma classe foi encontrada nas bases selecionadas."
            )

        return sorted(classes)

    # =========================================================================
    # INICIALIZACAO
    # =========================================================================

    def __init__(
        self,
        origem,
        destino,
        proporcoes,
        seed,
        nome_base,
        classes_referencia,
    ):
        """
        Inicializa o processador de uma unica base.

        Parameters
        ----------
        origem : str ou Path
            Dataset processado que sera dividido.

        destino : str ou Path
            Pasta onde o novo split sera criado.

        proporcoes : dict
            Proporcoes validadas de train, val e test.

        seed : int
            Seed utilizada no embaralhamento.

        nome_base : str
            Nome exibido no terminal para identificar a base.

        classes_referencia : list[str]
            Classes utilizadas como referencia do projeto.
        """

        # Padroniza os caminhos utilizados durante toda a instancia.
        self.origem = Path(origem)
        self.destino = Path(destino)

        # Copia as proporcoes para evitar alteracao do dicionario original.
        self.proporcoes = dict(proporcoes)

        # Converte a seed para inteiro antes de criar o gerador local.
        self.seed = int(seed)

        # Armazena o nome da base apenas para mensagens e resumos.
        self.nome_base = str(nome_base)

        # Mantem a ordem alfabetica das classes de referencia.
        self.classes_referencia = sorted(classes_referencia)

        # A lista real de classes da base e preenchida durante validar().
        self.classes_origem = []

    # =========================================================================
    # CONSULTA DOS ARQUIVOS
    # =========================================================================

    def listar_arquivos_classe(self, classe):
        """
        Lista os arquivos NPY existentes em uma classe.

        Uma classe de referencia ausente fisicamente na base retorna lista vazia.
        Isso permite que a base externa nao possua imagens de todas as classes.
        """

        # Monta o caminho da classe dentro do dataset processado.
        pasta = self.origem / classe

        # Classe ausente nesta base e representada por uma lista vazia.
        if not pasta.exists():
            return []

        # Somente arquivos .npy diretamente dentro da classe participam do split.
        return sorted(
            caminho
            for caminho in pasta.glob("*.npy")
            if caminho.is_file()
        )

    # =========================================================================
    # VALIDACAO DA BASE
    # =========================================================================

    def validar(self):
        """
        Valida estrutura, classes e arquivos da base antes da limpeza.

        Verificacoes realizadas:
        - existencia da pasta de origem;
        - origem realmente ser um diretorio;
        - existencia de pelo menos uma classe;
        - ausencia de classes desconhecidas em relacao a referencia;
        - ausencia de arquivos nao NPY diretamente nas classes;
        - validade numerica de todos os arquivos NPY.
        """

        # O dataset processado precisa existir.
        if not self.origem.exists():
            raise FileNotFoundError(
                f"Base {self.nome_base}: dataset processado nao encontrado:\n"
                f"{self.origem}"
            )

        # Impede o uso acidental de um arquivo como pasta de dataset.
        if not self.origem.is_dir():
            raise NotADirectoryError(
                f"Base {self.nome_base}: origem nao e um diretorio:\n"
                f"{self.origem}"
            )

        # Descobre as classes realmente presentes nesta base.
        self.classes_origem = self._listar_classes(self.origem)

        if not self.classes_origem:
            raise FileNotFoundError(
                f"Base {self.nome_base}: nenhuma classe encontrada em:\n"
                f"{self.origem}"
            )

        # Classes extras indicam divergencia entre a base e a referencia do projeto.
        classes_desconhecidas = (
            set(self.classes_origem)
            - set(self.classes_referencia)
        )

        if classes_desconhecidas:
            raise ValueError(
                f"Base {self.nome_base}: classes nao reconhecidas: "
                f"{sorted(classes_desconhecidas)}.\n"
                f"Classes de referencia: {self.classes_referencia}"
            )

        # Valida os arquivos existentes em cada classe presente na origem.
        for classe in self.classes_origem:
            pasta = self.origem / classe

            # Arquivos diretamente dentro da classe precisam ser NPY.
            arquivos_nao_npy = sorted(
                caminho.name
                for caminho in pasta.iterdir()
                if caminho.is_file() and caminho.suffix.lower() != ".npy"
            )

            if arquivos_nao_npy:
                raise ValueError(
                    f"Base {self.nome_base}, classe {classe}: "
                    "foram encontrados arquivos que nao sao NPY: "
                    f"{arquivos_nao_npy}"
                )

            # Abre cada NPY usando a validacao generica do projeto.
            for caminho in self.listar_arquivos_classe(classe):
                carregar_imagem_npy(caminho)

    # =========================================================================
    # CALCULO DAS QUANTIDADES
    # =========================================================================

    def calcular_quantidades(self, total):
        """
        Calcula quantas imagens da classe pertencerao a cada conjunto.

        Train e val utilizam arredondamento. Test recebe o restante, garantindo
        que nenhuma imagem seja perdida ou duplicada.

        Parameters
        ----------
        total : int
            Numero de imagens disponiveis na classe.

        Returns
        -------
        dict
            Quantidades destinadas a train, val e test.
        """

        # Calcula o tamanho de train usando a proporcao configurada.
        quantidade_train = int(
            round(total * self.proporcoes["train"])
        )

        # Impede qualquer excesso causado por arredondamentos de configuracoes
        # diferentes das proporcoes padrao.
        quantidade_train = min(
            quantidade_train,
            total,
        )

        # Calcula val e limita sua quantidade ao numero ainda disponivel.
        quantidade_val = int(
            round(total * self.proporcoes["val"])
        )

        quantidade_val = min(
            quantidade_val,
            total - quantidade_train,
        )

        # Test recebe exatamente todas as imagens que restaram.
        quantidade_test = (
            total
            - quantidade_train
            - quantidade_val
        )

        return {
            "train": quantidade_train,
            "val": quantidade_val,
            "test": quantidade_test,
        }

    # =========================================================================
    # PLANO DO SPLIT
    # =========================================================================

    def mostrar_plano(self):
        """
        Calcula e exibe a divisao prevista de todas as classes.

        Returns
        -------
        dict
            Quantidades previstas por classe e por conjunto.
        """

        print("\n" + "=" * 80)
        print(f"PLANO DO SPLIT - BASE {self.nome_base}")
        print("=" * 80)
        print(f"\nOrigem : {self.origem}")
        print(f"Destino: {self.destino}")
        print(f"Seed   : {self.seed}")

        resumo = {}
        totais = {nome: 0 for nome in self.CONJUNTOS}

        # Calcula a distribuicao individual de cada classe de referencia.
        for classe in self.classes_referencia:
            total = len(self.listar_arquivos_classe(classe))
            quantidades = self.calcular_quantidades(total)

            resumo[classe] = {
                "total": total,
                **quantidades,
            }

            # Atualiza os totais gerais da base.
            for conjunto in self.CONJUNTOS:
                totais[conjunto] += quantidades[conjunto]

            print(f"\n{classe}")
            print(f"  total : {total}")
            print(f"  train : {quantidades['train']}")
            print(f"  val   : {quantidades['val']}")
            print(f"  test  : {quantidades['test']}")

        # Exibe os totais previstos para a base completa.
        print("\n" + "-" * 80)
        print(f"Train total : {totais['train']}")
        print(f"Val total   : {totais['val']}")
        print(f"Test total  : {totais['test']}")
        print(f"Total geral : {sum(totais.values())}")

        return resumo

    # =========================================================================
    # PREPARACAO DA PASTA DE SAIDA
    # =========================================================================

    def preparar_destino(self):
        """
        Remove o split anterior e cria a nova estrutura de pastas.

        A funcao e chamada somente depois que todas as bases selecionadas foram
        validadas por executar_bases().
        """

        # Remove completamente o split antigo desta base quando ele existir.
        if self.destino.exists():
            shutil.rmtree(self.destino)

        # Cria train, val e test para todas as classes de referencia.
        for conjunto in self.CONJUNTOS:
            for classe in self.classes_referencia:
                (self.destino / conjunto / classe).mkdir(
                    parents=True,
                    exist_ok=True,
                )

    # =========================================================================
    # DIVISAO DE UMA CLASSE
    # =========================================================================

    def dividir_classe(self, classe, gerador):
        """
        Embaralha e copia os arquivos de uma classe para train, val e test.

        Parameters
        ----------
        classe : str
            Classe que sera dividida.

        gerador : random.Random
            Gerador aleatorio local da base.

        Returns
        -------
        dict
            Quantidades efetivamente copiadas para cada conjunto.
        """

        # Obtem uma lista independente dos arquivos para permitir embaralhamento.
        arquivos = list(
            self.listar_arquivos_classe(classe)
        )

        # Embaralha a ordem sem alterar nomes ou conteudo dos arquivos.
        gerador.shuffle(arquivos)

        # Calcula os limites da divisao para esta classe.
        quantidades = self.calcular_quantidades(
            len(arquivos)
        )

        fim_train = quantidades["train"]
        fim_val = fim_train + quantidades["val"]

        # Divide a lista embaralhada em tres partes sem sobreposicao.
        arquivos_por_conjunto = {
            "train": arquivos[:fim_train],
            "val": arquivos[fim_train:fim_val],
            "test": arquivos[fim_val:],
        }

        # Copia fisicamente os arquivos para a estrutura de saida.
        for conjunto, arquivos_conjunto in arquivos_por_conjunto.items():
            pasta_destino = self.destino / conjunto / classe

            for origem_arquivo in arquivos_conjunto:
                destino_arquivo = pasta_destino / origem_arquivo.name
                shutil.copy2(
                    origem_arquivo,
                    destino_arquivo,
                )

        return {
            conjunto: len(arquivos_por_conjunto[conjunto])
            for conjunto in self.CONJUNTOS
        }

    # =========================================================================
    # VERIFICACAO FINAL
    # =========================================================================

    def verificar_resultado(self, plano):
        """
        Confere a quantidade e a integridade dos arquivos copiados.

        Parameters
        ----------
        plano : dict
            Quantidades previstas antes da divisao.

        Returns
        -------
        dict
            Quantidades encontradas em cada classe e conjunto.

        Raises
        ------
        RuntimeError
            Quando a estrutura final diverge do plano ou algum NPY e invalido.
        """

        print("\n" + "=" * 80)
        print(f"VERIFICACAO FINAL - BASE {self.nome_base}")
        print("=" * 80)

        erros = []
        resumo = {}

        # Verifica cada classe em cada uma das tres particoes.
        for classe in self.classes_referencia:
            resumo[classe] = {}

            for conjunto in self.CONJUNTOS:
                pasta = self.destino / conjunto / classe
                arquivos = sorted(pasta.glob("*.npy"))
                esperado = plano[classe][conjunto]

                # Registra divergencias quantitativas.
                if len(arquivos) != esperado:
                    erros.append(
                        f"{classe}/{conjunto}: esperado={esperado}, "
                        f"encontrado={len(arquivos)}"
                    )

                # Reabre os arquivos copiados para confirmar sua integridade.
                for caminho in arquivos:
                    try:
                        carregar_imagem_npy(caminho)
                    except Exception as erro:
                        erros.append(
                            f"{classe}/{conjunto}/{caminho.name}: {erro}"
                        )

                resumo[classe][conjunto] = len(arquivos)

            print(
                f"{classe:<15} | "
                f"train={resumo[classe]['train']:>4} | "
                f"val={resumo[classe]['val']:>4} | "
                f"test={resumo[classe]['test']:>4}"
            )

        # Qualquer divergencia invalida o split produzido.
        if erros:
            raise RuntimeError(
                "Falha na verificacao final do split:\n- "
                + "\n- ".join(erros)
            )

        print("\nVerificacao concluida sem erros.")

        return resumo

    # =========================================================================
    # EXECUCAO COMPLETA DE UMA BASE JA VALIDADA
    # =========================================================================

    def executar_validado(self):
        """
        Recria e verifica o split de uma base previamente validada.

        Returns
        -------
        dict
            Plano calculado e composicao final do split.
        """

        # Calcula a distribuicao esperada antes de modificar o split antigo.
        plano = self.mostrar_plano()

        # Remove a saida anterior e cria toda a nova estrutura.
        self.preparar_destino()

        # O gerador local garante reproducibilidade sem alterar o estado global.
        gerador = random.Random(self.seed)

        print("\n" + "=" * 80)
        print(f"CRIANDO SPLIT - BASE {self.nome_base}")
        print("=" * 80)

        # Divide cada classe usando o mesmo gerador da base.
        for classe in self.classes_referencia:
            quantidades = self.dividir_classe(
                classe,
                gerador,
            )

            print(
                f"{classe:<15} | "
                f"train={quantidades['train']:>4} | "
                f"val={quantidades['val']:>4} | "
                f"test={quantidades['test']:>4}"
            )

        # Confere estrutura, quantidade e integridade dos arquivos copiados.
        resultado = self.verificar_resultado(plano)

        print("\n" + "=" * 80)
        print(f"SPLIT FINALIZADO - BASE {self.nome_base}")
        print("=" * 80)
        print(f"\nSaida:\n{self.destino}")

        return {
            "plano": plano,
            "resultado": resultado,
        }
