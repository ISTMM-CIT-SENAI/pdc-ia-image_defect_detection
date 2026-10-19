"""
prepara_img_class.py

Objetivo:
- Implementar toda a logica especifica da preparacao das bases interna e externa.
- Validar completamente as entradas antes de recriar qualquer pasta de saida.
- Converter as imagens das duas origens para um formato numerico comum.
- Manter as diferencas de tratamento entre a base interna e a base externa.
- Verificar quantidades e integridade dos NPY gerados antes de finalizar.

Base interna:
- Origem esperada: dataset_atual_crop.
- Estrutura esperada: material/condicao/imagem.
- Material, condicao e codigo RAC sao validados pela nomenclatura do projeto.
- O rotulo final e obtido a partir da condicao validada.
- As imagens ja passaram pelo crop do SAM e nao recebem novo crop nesta etapa.
- O nome final do NPY preserva o stem do arquivo original.

Base externa:
- Origem esperada: dataset_externo_test/imagens_selecionadas.
- Estrutura esperada: classe/imagem, permitindo subpastas dentro da classe.
- O rotulo final e obtido diretamente pela pasta da classe.
- A orientacao EXIF e corrigida antes do processamento.
- Imagens retangulares recebem crop quadrado central.
- Imagens quadradas permanecem sem crop.
- O nome final recebe o prefixo externo e a classe por extenso.

Formato numerico final comum:
- grayscale;
- resize LANCZOS;
- numpy.ndarray;
- dtype float32;
- shape (1,H,W);
- valores no intervalo [0,1];
- arquivo .npy.

Fluxo geral:
1. Selecionar as bases ativadas pelo processing.
2. Validar parametros e caminhos de todas as bases selecionadas.
3. Analisar todas as imagens de entrada.
4. Validar nomenclatura e nomes de saida.
5. Interromper a execucao se qualquer base possuir erro.
6. Recriar as pastas de saida das bases selecionadas.
7. Processar e salvar todos os arquivos.
8. Reabrir os NPY gerados e conferir as quantidades finais.
9. Retornar um resumo separado por base.

Regras de seguranca:
- Nenhuma saida e removida antes da validacao completa de todas as bases ativas.
- Bases desativadas nao sao analisadas nem modificadas.
- Se nenhuma base estiver ativa, a execucao termina sem alterar arquivos.
- Augmentation nao e aplicado neste modulo.
- Split nao e realizado neste modulo.

Arquitetura:
- Parametros ajustaveis e caminhos ficam em processing/prepara_img.py.
- A logica especifica desta etapa fica neste arquivo.
- Leitura e transformacoes genericas de imagem ficam em utils.imagens.
- Leitura, validacao e escrita genericas de NPY ficam em utils.npy.
- Regras de nomenclatura da base interna ficam em utils.nomenclatura.
"""

from collections import Counter
from pathlib import Path
import re
import shutil

from pdc_ia_image_defect_detection.utils.imagens import (
    carregar_imagem_pil,
    converter_imagem_para_npy,
    crop_quadrado_central,
    listar_imagens,
)
from pdc_ia_image_defect_detection.utils.npy import (
    carregar_imagem_npy,
    salvar_imagem_npy,
)
from pdc_ia_image_defect_detection.utils.nomenclatura import (
    ROTULOS,
    interpretar_nome,
)


# =============================================================================
# BASE COMUM DOS PREPARADORES
# =============================================================================

class _PreparadorBase:
    """
    Reune o comportamento comum das bases interna e externa.

    A classe nao define como os registros sao encontrados nem como cada imagem
    deve ser tratada. Essas regras ficam nas subclasses especificas.
    """

    def __init__(self, origem, destino, tamanho_final, extensoes_validas):
        """
        Inicializa os parametros comuns de uma base.

        Parameters
        ----------
        origem : str ou Path
            Pasta que contem as imagens de entrada.

        destino : str ou Path
            Pasta onde os NPY processados serao criados.

        tamanho_final : tuple[int, int]
            Resolucao final no formato (largura, altura).

        extensoes_validas : iterable[str]
            Extensoes de imagem aceitas na entrada.
        """

        # Padroniza os caminhos para Path.
        self.origem = Path(origem)
        self.destino = Path(destino)

        # Armazena a resolucao comum utilizada durante a conversao.
        self.tamanho_final = (
            int(tamanho_final[0]),
            int(tamanho_final[1]),
        )

        # Normaliza as extensoes para comparacoes consistentes.
        self.extensoes_validas = tuple(
            extensao.lower() if str(extensao).startswith(".") else f".{str(extensao).lower()}"
            for extensao in extensoes_validas
        )

        # Os registros validos sao preenchidos durante a analise da entrada.
        self.registros = []

        # A contagem esperada por rotulo e calculada antes do processamento.
        self.esperado_por_rotulo = Counter()

    @staticmethod
    def _formatar_counter(counter):
        """
        Formata um Counter para exibicao compacta no terminal.

        Parameters
        ----------
        counter : collections.Counter
            Contagem que sera convertida para texto.

        Returns
        -------
        str
            Texto no formato valor=quantidade, separado por virgulas.
            Retorna "nenhum" quando o Counter estiver vazio.
        """

        # Ordena as chaves pela representacao textual para manter a exibicao
        # previsivel entre execucoes.
        itens = sorted(
            counter.items(),
            key=lambda item: str(item[0]),
        )

        # Converte cada par chave/quantidade para uma representacao compacta.
        return ", ".join(
            f"{valor}={quantidade}"
            for valor, quantidade in itens
        ) or "nenhum"

    @staticmethod
    def _formatar_resolucoes(counter):
        """
        Formata contagens de resolucao para exibicao no terminal.

        Parameters
        ----------
        counter : collections.Counter
            Counter cujas chaves seguem o formato (largura, altura).

        Returns
        -------
        str
            Texto no formato LARGURAxALTURA=quantidade.
            Retorna "nenhuma" quando nao houver resolucoes registradas.
        """

        # Ordena numericamente os pares (largura, altura).
        itens = sorted(
            counter.items()
        )

        # Converte cada resolucao para uma representacao legivel.
        return ", ".join(
            f"{largura}x{altura}={quantidade}"
            for (largura, altura), quantidade in itens
        ) or "nenhuma"

    def validar_parametros_comuns(self):
        """
        Valida caminho de entrada, resolucao e extensoes da base.

        A funcao nao modifica arquivos.
        """

        # A origem precisa existir para que a base selecionada possa ser usada.
        if not self.origem.exists():
            raise FileNotFoundError(
                f"Dataset de entrada nao encontrado:\n{self.origem}"
            )

        # A origem precisa representar uma pasta.
        if not self.origem.is_dir():
            raise NotADirectoryError(
                f"O caminho de entrada nao e um diretorio:\n{self.origem}"
            )

        # Largura e altura precisam ser estritamente positivas.
        if self.tamanho_final[0] <= 0 or self.tamanho_final[1] <= 0:
            raise ValueError(
                f"TAMANHO_FINAL invalido: {self.tamanho_final}"
            )

        # Pelo menos uma extensao precisa estar habilitada.
        if not self.extensoes_validas:
            raise ValueError(
                "Nenhuma extensao de imagem foi configurada."
            )

    def recriar_saida(self):
        """
        Recria a pasta de saida e cria uma subpasta para cada rotulo do projeto.

        A remocao acontece somente depois da etapa global de validacao.
        """

        # Remove completamente a saida anterior para impedir mistura de execucoes.
        if self.destino.exists():
            shutil.rmtree(self.destino)

        # Cria a raiz da nova saida.
        self.destino.mkdir(
            parents=True,
            exist_ok=False,
        )

        # Todas as classes sao criadas mesmo quando uma delas nao possui imagens.
        for rotulo in ROTULOS:
            (self.destino / rotulo).mkdir(
                parents=False,
                exist_ok=False,
            )

    def verificar_saida(self):
        """
        Reabre todos os NPY gerados e confere a quantidade esperada por classe.

        Returns
        -------
        dict
            Resumo final com quantidade de arquivos por rotulo.

        Raises
        ------
        RuntimeError
            Quando a quantidade ou a integridade de algum arquivo esta incorreta.
        """

        # Acumula todas as falhas para apresentar um diagnostico conjunto.
        erros = []

        # Armazena a composicao final da base.
        resumo = {}

        # Verifica todas as classes, inclusive as que deveriam possuir zero arquivos.
        for rotulo in ROTULOS:
            pasta = self.destino / rotulo

            # Lista somente os NPY criados na classe atual.
            arquivos = sorted(
                caminho
                for caminho in pasta.glob("*.npy")
                if caminho.is_file()
            )

            # Recupera a quantidade calculada antes do processamento.
            esperado = int(self.esperado_por_rotulo[rotulo])

            # Registra divergencias de quantidade.
            if len(arquivos) != esperado:
                erros.append(
                    f"{rotulo}: esperado={esperado}, encontrado={len(arquivos)}"
                )

            # Cada arquivo e reaberto pela rotina generica de validacao NPY.
            for caminho in arquivos:
                try:
                    carregar_imagem_npy(caminho)
                except Exception as erro:
                    erros.append(
                        f"{rotulo}/{caminho.name}: {erro}"
                    )

            # Guarda a contagem final da classe.
            resumo[rotulo] = {
                "esperado": esperado,
                "encontrado": len(arquivos),
            }

            # Exibe um estado simples por classe.
            status = "OK" if len(arquivos) == esperado else "ERRO"
            print(
                f"{rotulo:<15} | "
                f"esperado={esperado:>5} | "
                f"encontrado={len(arquivos):>5} | "
                f"{status}"
            )

        # Qualquer divergencia invalida a preparacao da base.
        if erros:
            raise RuntimeError(
                "Falha na verificacao da saida:\n- "
                + "\n- ".join(erros)
            )

        return resumo


# =============================================================================
# PREPARADOR DA BASE INTERNA
# =============================================================================

class _PreparadorInterno(_PreparadorBase):
    """
    Implementa as regras especificas da base interna.

    A entrada ja foi recortada pelo SAM e utiliza a estrutura:
    material/condicao/imagem.

    As regras das pastas pertencem a esta classe porque descrevem a
    organizacao especifica do dataset interno. O util de nomenclatura conhece
    somente o nome MATERIAL_CP_RAC e nao depende da estrutura de diretorios.
    """

    # Relaciona cada pasta de material ao codigo esperado no nome do arquivo.
    MATERIAIS_POR_PASTA = {
        "aco_2500": "A36",
        "Aluminio_2500": "AL",
        "ferro_fundido_4500": "FoFo",
    }

    # Relaciona cada pasta de condicao ao RAC exato esperado no nome.
    # Ordem RAC: Risco-Amassamento-Corrosao.
    CONDICOES_POR_PASTA = {
        "sem_defeito": "000",
        "risco_leve": "100",
        "risco_moderado": "200",
        "risco_severo": "300",
        "amassamento_leve": "010",
        "amassamento_moderado": "020",
        "amassamento_severo": "030",
        "corrosao_leve": "001",
        "corrosao_moderado": "002",
        "corrosao_severo": "003",
    }

    def __init__(
        self,
        origem,
        destino,
        tamanho_final,
        extensoes_validas,
        pastas_ignoradas,
    ):
        """Inicializa a base interna e suas pastas ignoradas."""

        # Inicializa os parametros comuns.
        super().__init__(
            origem=origem,
            destino=destino,
            tamanho_final=tamanho_final,
            extensoes_validas=extensoes_validas,
        )

        # Converte a configuracao para set para consultas rapidas por nome.
        self.pastas_ignoradas = set(pastas_ignoradas)

        # Guarda informacoes estatisticas encontradas durante a analise.
        self.estatisticas = {}

        # Guarda os caminhos ignorados explicitamente pela configuracao.
        self.ignorados = []

    def analisar_entrada(self):
        """
        Analisa e valida todas as imagens da base interna.

        Validacoes realizadas:
        - estrutura material/condicao/imagem;
        - material permitido;
        - condicao permitida;
        - nomenclatura MATERIAL_CP_RAC;
        - consistencia do material entre pasta e nome;
        - consistencia do RAC entre pasta e nome;
        - abertura da imagem pelo Pillow.

        Returns
        -------
        dict
            Resumo estatistico da entrada.
        """

        # Valida primeiro os parametros que independem do conteudo das imagens.
        self.validar_parametros_comuns()

        # Lista todas as imagens reconhecidas abaixo da raiz interna.
        imagens = listar_imagens(
            self.origem,
            self.extensoes_validas,
        )

        # Contadores usados no resumo da base.
        formatos = Counter()
        modos = Counter()
        canais = Counter()
        resolucoes = Counter()
        materiais = Counter()
        condicoes = Counter()
        rotulos = Counter()

        # Erros sao acumulados para que a analise apresente todos de uma vez.
        erros = []

        # Reinicia os registros caso o mesmo objeto seja analisado novamente.
        self.registros = []
        self.ignorados = []
        self.esperado_por_rotulo = Counter()

        # Avalia cada arquivo encontrado.
        for caminho in imagens:
            # Calcula o caminho relativo para validar a estrutura das pastas.
            relativo = caminho.relative_to(self.origem)

            # Qualquer arquivo dentro de uma pasta ignorada e excluido da base.
            if any(
                parte in self.pastas_ignoradas
                for parte in relativo.parts
            ):
                self.ignorados.append(caminho)
                continue

            # A estrutura interna precisa possuir exatamente tres niveis.
            if len(relativo.parts) != 3:
                erros.append(
                    (
                        caminho,
                        "Estrutura invalida. Esperado: material/condicao/imagem.",
                    )
                )
                continue

            # Extrai os nomes das pastas que definem material e condicao.
            pasta_material, pasta_condicao, _ = relativo.parts

            # Valida a pasta de material segundo a estrutura desta base.
            if pasta_material not in self.MATERIAIS_POR_PASTA:
                erros.append(
                    (caminho, f"Material desconhecido: {pasta_material}")
                )
                continue

            # Valida a pasta de condicao segundo a estrutura desta base.
            if pasta_condicao not in self.CONDICOES_POR_PASTA:
                erros.append(
                    (caminho, f"Condicao desconhecida: {pasta_condicao}")
                )
                continue

            try:
                # Interpreta somente a nomenclatura generica MATERIAL_CP_RAC.
                dados = interpretar_nome(caminho)

                # Confere se o codigo de material do nome corresponde a pasta.
                material_esperado = self.MATERIAIS_POR_PASTA[pasta_material]
                if dados["material"] != material_esperado:
                    raise ValueError(
                        f"Material do nome ({dados['material']}) nao corresponde "
                        f"a pasta {pasta_material} ({material_esperado})."
                    )

                # Confere se o RAC do nome corresponde exatamente a condicao.
                rac_esperado = self.CONDICOES_POR_PASTA[pasta_condicao]
                if dados["rac"] != rac_esperado:
                    raise ValueError(
                        f"RAC do nome ({dados['rac']}) nao corresponde "
                        f"a pasta {pasta_condicao} ({rac_esperado})."
                    )

                # Abre a imagem sem alterar orientacao EXIF para preservar o fluxo
                # aplicado anteriormente as imagens ja recortadas pelo SAM.
                imagem, metadata = carregar_imagem_pil(
                    caminho,
                    corrigir_exif=False,
                )

                # Coleta propriedades do arquivo antes da conversao numerica.
                formatos[caminho.suffix.lower()] += 1
                modos[metadata["modo_original"]] += 1
                canais[len(imagem.getbands())] += 1
                resolucoes[imagem.size] += 1
                materiais[dados["material"]] += 1
                condicoes[pasta_condicao] += 1
                rotulos[dados["rotulo"]] += 1

                # O registro contem tudo que sera necessario na etapa de gravacao.
                registro = {
                    "caminho": caminho,
                    "rotulo": dados["rotulo"],
                    "material": dados["material"],
                    "cp": dados["cp"],
                    "rac": dados["rac"],
                    "pasta_material": pasta_material,
                    "pasta_condicao": pasta_condicao,
                    "nome_saida": f"{caminho.stem}.npy",
                }

                self.registros.append(registro)
                self.esperado_por_rotulo[dados["rotulo"]] += 1

            except Exception as erro:
                erros.append((caminho, str(erro)))

        # Uma base selecionada precisa possuir pelo menos uma imagem valida.
        if not self.registros and not erros:
            erros.append(
                (self.origem, "Nenhuma imagem valida foi encontrada na base interna.")
            )

        # Verifica se dois arquivos diferentes produziriam o mesmo destino.
        conflitos = self._verificar_conflitos_nomes()

        # Conflitos sao tratados como erros antes da limpeza da saida.
        for rotulo, primeiro, segundo, nome_saida in conflitos:
            erros.append(
                (
                    segundo,
                    f"Conflito de nome em {rotulo}: {nome_saida} ja seria gerado por {primeiro}.",
                )
            )

        # Guarda o resumo para exibicao e retorno posterior.
        self.estatisticas = {
            "imagens_encontradas": len(imagens),
            "imagens_validas": len(self.registros),
            "ignoradas": len(self.ignorados),
            "formatos": formatos,
            "modos": modos,
            "canais": canais,
            "resolucoes": resolucoes,
            "materiais": materiais,
            "condicoes": condicoes,
            "rotulos": rotulos,
            "erros": erros,
        }

        return self.estatisticas

    def _verificar_conflitos_nomes(self):
        """
        Detecta sobrescritas potenciais dentro de cada rotulo final.

        Returns
        -------
        list[tuple]
            Conflitos encontrados entre arquivos de origem.
        """

        conflitos = []

        # Cada rotulo possui seu proprio espaco de nomes na pasta de saida.
        for rotulo in ROTULOS:
            nomes = {}

            for registro in self.registros:
                if registro["rotulo"] != rotulo:
                    continue

                nome_saida = registro["nome_saida"]

                if nome_saida in nomes:
                    conflitos.append(
                        (
                            rotulo,
                            nomes[nome_saida],
                            registro["caminho"],
                            nome_saida,
                        )
                    )
                else:
                    nomes[nome_saida] = registro["caminho"]

        return conflitos

    def mostrar_plano(self):
        """Exibe o estado validado da base interna antes da recriacao da saida."""

        print("\n" + "=" * 90)
        print("PREPARACAO - BASE INTERNA")
        print("=" * 90)
        print(f"\nOrigem              : {self.origem}")
        print(f"Destino             : {self.destino}")
        print(f"Imagens encontradas : {self.estatisticas['imagens_encontradas']}")
        print(f"Imagens validas     : {self.estatisticas['imagens_validas']}")
        print(f"Imagens ignoradas   : {self.estatisticas['ignoradas']}")
        print(f"Resolucao final     : {self.tamanho_final[0]}x{self.tamanho_final[1]}")
        print("Crop                : nao aplicado; entrada ja recortada pelo SAM")
        print("Canal final         : grayscale")
        print("Dtype final         : float32")
        print("Intervalo final     : [0,1]")
        print("Formato final       : NPY")
        print(f"Formatos de entrada : {self._formatar_counter(self.estatisticas['formatos'])}")
        print(f"Modos de entrada    : {self._formatar_counter(self.estatisticas['modos'])}")
        print(f"Canais de entrada   : {self._formatar_counter(self.estatisticas['canais'])}")
        print(f"Resolucoes          : {self._formatar_resolucoes(self.estatisticas['resolucoes'])}")

        print("\nROTULOS FINAIS")
        for rotulo in ROTULOS:
            print(f"  {rotulo:<15}: {self.esperado_por_rotulo[rotulo]}")

        print("\nMATERIAIS")
        for material, quantidade in sorted(self.estatisticas["materiais"].items()):
            print(f"  {material:<15}: {quantidade}")

        print("\nCONDICOES")
        for condicao, quantidade in sorted(self.estatisticas["condicoes"].items()):
            print(f"  {condicao:<25}: {quantidade}")

    def processar(self):
        """
        Converte e salva todas as imagens validas da base interna.

        Returns
        -------
        Counter
            Quantidade de NPY gerados por rotulo.
        """

        gerados = Counter()
        total = len(self.registros)

        print("\n" + "=" * 90)
        print("PROCESSANDO BASE INTERNA")
        print("=" * 90)

        # Processa somente os registros aprovados durante a analise completa.
        for indice, registro in enumerate(self.registros, start=1):
            caminho = registro["caminho"]
            rotulo = registro["rotulo"]

            # A base interna nao recebe correcao EXIF nem crop nesta etapa.
            imagem, _ = carregar_imagem_pil(
                caminho,
                corrigir_exif=False,
            )

            # A conversao generica aplica grayscale, resize, normalizacao e canal.
            array = converter_imagem_para_npy(
                imagem,
                self.tamanho_final,
            )

            # O destino utiliza o rotulo final validado e preserva o nome original.
            destino = self.destino / rotulo / registro["nome_saida"]

            # A rotina generica valida novamente o array antes da gravacao.
            salvar_imagem_npy(
                destino,
                array,
            )

            gerados[rotulo] += 1

            # Atualiza uma unica linha com o progresso total da base.
            print(
                f"\r  {indice:>5}/{total:<5} | "
                f"classe={rotulo:<15} | "
                f"arquivo={caminho.name}",
                end="",
                flush=True,
            )

        if total:
            print()

        return gerados


# =============================================================================
# PREPARADOR DA BASE EXTERNA
# =============================================================================

class _PreparadorExterno(_PreparadorBase):
    """
    Implementa as regras especificas da base externa.

    A entrada utiliza pastas de classe e nao depende da nomenclatura RAC da base
    interna.
    """

    # Expressao usada para substituir caracteres inadequados nos nomes externos.
    PADRAO_CARACTERE_INVALIDO = re.compile(
        r"[^A-Za-z0-9._-]+"
    )

    def __init__(
        self,
        origem,
        destino,
        tamanho_final,
        extensoes_validas,
        prefixo_externo,
    ):
        """Inicializa a base externa e o prefixo dos nomes de saida."""

        # Inicializa os parametros comuns.
        super().__init__(
            origem=origem,
            destino=destino,
            tamanho_final=tamanho_final,
            extensoes_validas=extensoes_validas,
        )

        # Remove espacos laterais do prefixo configurado.
        self.prefixo_externo = str(prefixo_externo).strip()

        # O prefixo vazio dificultaria identificar a origem dos NPY externos.
        if not self.prefixo_externo:
            raise ValueError(
                "PREFIXO_EXTERNO nao pode ser vazio."
            )

        # Guarda estatisticas calculadas durante a analise.
        self.estatisticas = {}

    @classmethod
    def sanitizar_nome(cls, texto):
        """
        Converte um nome de arquivo para uma forma segura e previsivel.

        O texto preserva letras, numeros, ponto, hifen e sublinhado. Outros
        caracteres sao substituidos por sublinhado.
        """

        # Remove espacos laterais antes das substituicoes.
        texto = str(texto).strip()

        # Substitui grupos de caracteres inadequados por um unico sublinhado.
        texto = cls.PADRAO_CARACTERE_INVALIDO.sub("_", texto)

        # Elimina repeticoes produzidas pela normalizacao.
        while "__" in texto:
            texto = texto.replace("__", "_")

        # Remove sublinhados extras nas extremidades.
        return texto.strip("_")

    def nome_saida(self, classe, caminho):
        """
        Gera o nome NPY externo no formato ext__CLASSE__NOME.npy.

        Parameters
        ----------
        classe : str
            Rotulo obtido pela pasta da classe.

        caminho : Path
            Arquivo original.

        Returns
        -------
        str
            Nome completo do arquivo NPY de destino.
        """

        # Sanitiza somente o stem; a extensao original nao participa do nome final.
        stem = self.sanitizar_nome(caminho.stem)

        # Um stem completamente removido pela sanitizacao nao e aceito.
        if not stem:
            raise ValueError(
                f"Nome externo invalido apos sanitizacao: {caminho.name}"
            )

        return f"{self.prefixo_externo}__{classe}__{stem}.npy"

    def analisar_entrada(self):
        """
        Analisa e valida todas as imagens da base externa.

        Validacoes realizadas:
        - existencia das pastas;
        - nomes permitidos de classes;
        - abertura das imagens;
        - primeiro frame quando houver varios;
        - orientacao EXIF;
        - dimensoes e modo;
        - conflitos de nomes de saida.

        Classes ausentes sao permitidas e terminam com zero arquivos.

        Returns
        -------
        dict
            Resumo estatistico da entrada.
        """

        # Valida os parametros comuns da base.
        self.validar_parametros_comuns()

        # Identifica todas as subpastas diretas existentes na raiz externa.
        pastas_encontradas = sorted(
            pasta.name
            for pasta in self.origem.iterdir()
            if pasta.is_dir()
        )

        # Pastas que nao correspondem a classes conhecidas indicam estrutura invalida.
        pastas_invalidas = [
            pasta
            for pasta in pastas_encontradas
            if pasta not in ROTULOS
        ]

        if pastas_invalidas:
            raise ValueError(
                "Pastas externas que nao correspondem as classes do projeto: "
                f"{pastas_invalidas}"
            )

        # Pelo menos uma pasta de classe precisa existir na base selecionada.
        if not pastas_encontradas:
            raise FileNotFoundError(
                f"Nenhuma pasta de classe encontrada em:\n{self.origem}"
            )

        # Contadores usados no resumo da entrada.
        formatos = Counter()
        modos = Counter()
        canais = Counter()
        resolucoes = Counter()
        quadradas = 0
        retangulares = 0
        convertidas_grayscale = 0
        rotulos = Counter()

        # Erros sao acumulados para impedir limpeza parcial da saida.
        erros = []

        # Reinicia os dados calculados.
        self.registros = []
        self.esperado_por_rotulo = Counter()

        # Analisa cada classe prevista pelo projeto.
        for classe in ROTULOS:
            pasta_classe = self.origem / classe

            # Classes ausentes sao permitidas.
            if not pasta_classe.exists():
                continue

            # Lista recursivamente as imagens presentes dentro da classe.
            imagens = listar_imagens(
                pasta_classe,
                self.extensoes_validas,
            )

            for caminho in imagens:
                try:
                    # A base externa corrige EXIF antes de analisar as dimensoes.
                    imagem, metadata = carregar_imagem_pil(
                        caminho,
                        corrigir_exif=True,
                    )

                    largura, altura = imagem.size

                    # Registra propriedades encontradas na entrada.
                    formatos[caminho.suffix.lower()] += 1
                    modos[metadata["modo_original"]] += 1
                    canais[len(imagem.getbands())] += 1
                    resolucoes[imagem.size] += 1
                    rotulos[classe] += 1

                    # Classifica a geometria utilizada para decidir o crop futuro.
                    if largura == altura:
                        quadradas += 1
                    else:
                        retangulares += 1

                    # Conta imagens que precisarao ser convertidas para grayscale.
                    if imagem.mode != "L":
                        convertidas_grayscale += 1

                    # Calcula o nome final ainda durante a validacao para detectar
                    # conflitos antes de recriar a pasta de saida.
                    nome_saida = self.nome_saida(
                        classe,
                        caminho,
                    )

                    self.registros.append(
                        {
                            "caminho": caminho,
                            "rotulo": classe,
                            "nome_saida": nome_saida,
                        }
                    )

                    self.esperado_por_rotulo[classe] += 1

                except Exception as erro:
                    erros.append((caminho, str(erro)))

        # Uma base externa selecionada precisa conter ao menos uma imagem valida.
        if not self.registros and not erros:
            erros.append(
                (self.origem, "Nenhuma imagem valida foi encontrada na base externa.")
            )

        # Detecta nomes finais repetidos dentro de uma mesma classe.
        conflitos = self._verificar_conflitos_nomes()

        for classe, primeiro, segundo, nome_saida in conflitos:
            erros.append(
                (
                    segundo,
                    f"Conflito de nome em {classe}: {nome_saida} ja seria gerado por {primeiro}.",
                )
            )

        # Guarda o resumo completo da analise.
        self.estatisticas = {
            "pastas_encontradas": pastas_encontradas,
            "imagens_validas": len(self.registros),
            "formatos": formatos,
            "modos": modos,
            "canais": canais,
            "resolucoes": resolucoes,
            "quadradas": quadradas,
            "retangulares": retangulares,
            "convertidas_grayscale": convertidas_grayscale,
            "rotulos": rotulos,
            "erros": erros,
        }

        return self.estatisticas

    def _verificar_conflitos_nomes(self):
        """Detecta arquivos externos que produziriam o mesmo nome de destino."""

        conflitos = []

        # A verificacao e independente para cada pasta de classe.
        for rotulo in ROTULOS:
            nomes = {}

            for registro in self.registros:
                if registro["rotulo"] != rotulo:
                    continue

                nome_saida = registro["nome_saida"]

                if nome_saida in nomes:
                    conflitos.append(
                        (
                            rotulo,
                            nomes[nome_saida],
                            registro["caminho"],
                            nome_saida,
                        )
                    )
                else:
                    nomes[nome_saida] = registro["caminho"]

        return conflitos

    def mostrar_plano(self):
        """Exibe o estado validado da base externa antes da recriacao da saida."""

        print("\n" + "=" * 90)
        print("PREPARACAO - BASE EXTERNA")
        print("=" * 90)
        print(f"\nOrigem                    : {self.origem}")
        print(f"Destino                   : {self.destino}")
        print(f"Imagens validas           : {self.estatisticas['imagens_validas']}")
        print(f"Resolucao final           : {self.tamanho_final[0]}x{self.tamanho_final[1]}")
        print("Orientacao EXIF           : corrigida")
        print("Crop                      : quadrado central somente em retangulares")
        print(f"Imagens quadradas         : {self.estatisticas['quadradas']}")
        print(f"Imagens retangulares      : {self.estatisticas['retangulares']}")
        print(f"Conversoes para grayscale : {self.estatisticas['convertidas_grayscale']}")
        print("Canal final               : grayscale")
        print("Dtype final               : float32")
        print("Intervalo final           : [0,1]")
        print("Formato final             : NPY")
        print(f"Formatos de entrada       : {self._formatar_counter(self.estatisticas['formatos'])}")
        print(f"Modos de entrada          : {self._formatar_counter(self.estatisticas['modos'])}")
        print(f"Canais de entrada         : {self._formatar_counter(self.estatisticas['canais'])}")
        print(f"Resolucoes                : {self._formatar_resolucoes(self.estatisticas['resolucoes'])}")

        print("\nROTULOS FINAIS")
        for rotulo in ROTULOS:
            print(f"  {rotulo:<15}: {self.esperado_por_rotulo[rotulo]}")

    def processar(self):
        """
        Converte e salva todas as imagens validadas da base externa.

        Returns
        -------
        Counter
            Quantidade de NPY gerados por rotulo.
        """

        gerados = Counter()
        total = len(self.registros)

        print("\n" + "=" * 90)
        print("PROCESSANDO BASE EXTERNA")
        print("=" * 90)

        # Processa somente registros aprovados durante a analise previa.
        for indice, registro in enumerate(self.registros, start=1):
            caminho = registro["caminho"]
            rotulo = registro["rotulo"]

            # Corrige EXIF antes de qualquer decisao geometrica.
            imagem, _ = carregar_imagem_pil(
                caminho,
                corrigir_exif=True,
            )

            # Imagens quadradas sao copiadas sem corte; retangulares recebem crop
            # central usando a menor dimensao como lado.
            imagem, _, _ = crop_quadrado_central(
                imagem
            )

            # Aplica grayscale, resize, normalizacao e formato (1,H,W).
            array = converter_imagem_para_npy(
                imagem,
                self.tamanho_final,
            )

            # O nome externo ja foi validado durante a analise da entrada.
            destino = self.destino / rotulo / registro["nome_saida"]

            # Salva somente depois da validacao numerica final.
            salvar_imagem_npy(
                destino,
                array,
            )

            gerados[rotulo] += 1

            # Atualiza uma unica linha com o progresso total da base.
            print(
                f"\r  {indice:>5}/{total:<5} | "
                f"classe={rotulo:<15} | "
                f"arquivo={caminho.name}",
                end="",
                flush=True,
            )

        if total:
            print()

        return gerados


# =============================================================================
# ORQUESTRADOR DA PREPARACAO
# =============================================================================

class PreparaImagensProcessor:
    """
    Coordena a preparacao das bases ativadas pelo script de processing.

    A classe garante que todas as entradas selecionadas sejam analisadas antes
    que qualquer pasta de saida seja removida.
    """

    @staticmethod
    def _criar_preparador(
        configuracao,
        tamanho_final,
        extensoes_validas,
        pastas_ignoradas_interno,
        prefixo_externo,
    ):
        """
        Cria o preparador especializado conforme o tipo configurado da base.

        Returns
        -------
        _PreparadorInterno ou _PreparadorExterno
            Instancia pronta para validacao e processamento.
        """

        tipo = str(configuracao["tipo"]).lower()

        # A base interna utiliza validacao de material, condicao e RAC.
        if tipo == "interna":
            return _PreparadorInterno(
                origem=configuracao["origem"],
                destino=configuracao["destino"],
                tamanho_final=tamanho_final,
                extensoes_validas=extensoes_validas,
                pastas_ignoradas=pastas_ignoradas_interno,
            )

        # A base externa utiliza classe por pasta, EXIF e crop quadrado central.
        if tipo == "externa":
            return _PreparadorExterno(
                origem=configuracao["origem"],
                destino=configuracao["destino"],
                tamanho_final=tamanho_final,
                extensoes_validas=extensoes_validas,
                prefixo_externo=prefixo_externo,
            )

        raise ValueError(
            f"Tipo de base desconhecido: {configuracao['tipo']}"
        )

    @classmethod
    def executar_bases(
        cls,
        bases,
        tamanho_final,
        extensoes_validas,
        pastas_ignoradas_interno,
        prefixo_externo,
    ):
        """
        Executa a preparacao completa de todas as bases habilitadas.

        Parameters
        ----------
        bases : dict
            Configuracao das bases interna e externa.

        tamanho_final : tuple[int, int]
            Resolucao final comum a todas as imagens.

        extensoes_validas : iterable[str]
            Extensoes aceitas nas entradas.

        pastas_ignoradas_interno : iterable[str]
            Pastas da base interna que nao participam do processamento.

        prefixo_externo : str
            Prefixo utilizado nos nomes dos NPY externos.

        Returns
        -------
        dict
            Resultado separado por base processada.
        """

        # Mantem apenas as bases explicitamente ativadas pelo processing.
        bases_ativas = {
            nome: configuracao
            for nome, configuracao in bases.items()
            if bool(configuracao.get("ativa", False))
        }

        print("\n" + "=" * 90)
        print("PREPARACAO DE IMAGENS")
        print("=" * 90)

        nomes = ", ".join(bases_ativas) if bases_ativas else "nenhuma"
        print(f"\nBases selecionadas : {nomes}")
        print(f"Resolucao final    : {tamanho_final[0]}x{tamanho_final[1]}")

        # Nenhuma base ativa significa encerramento sem leitura ou alteracao.
        if not bases_ativas:
            print("\nNenhuma base foi selecionada.")
            print("Nenhuma pasta foi removida ou criada.")
            return {}

        # Cria todos os preparadores antes de iniciar a validacao.
        preparadores = {
            nome: cls._criar_preparador(
                configuracao=configuracao,
                tamanho_final=tamanho_final,
                extensoes_validas=extensoes_validas,
                pastas_ignoradas_interno=pastas_ignoradas_interno,
                prefixo_externo=prefixo_externo,
            )
            for nome, configuracao in bases_ativas.items()
        }

        # Analisa todas as bases antes de permitir a limpeza de qualquer saida.
        erros_globais = []

        for nome, preparador in preparadores.items():
            print("\n" + "-" * 90)
            print(f"VALIDANDO BASE {nome}")
            print("-" * 90)

            try:
                resultado_analise = preparador.analisar_entrada()

                # Erros de arquivos individuais sao acumulados pela propria analise.
                for caminho, erro in resultado_analise.get("erros", []):
                    erros_globais.append(
                        f"[{nome}] {caminho}: {erro}"
                    )

            except Exception as erro:
                erros_globais.append(
                    f"[{nome}] {erro}"
                )

        # Nenhuma pasta de destino e tocada quando qualquer base falha na validacao.
        if erros_globais:
            raise RuntimeError(
                "A preparacao foi interrompida antes de modificar as saidas:\n- "
                + "\n- ".join(erros_globais)
            )

        # Exibe o plano somente depois que todas as bases foram aprovadas.
        for preparador in preparadores.values():
            preparador.mostrar_plano()

        # Armazena o retorno estruturado de cada base.
        resultados = {}

        # A partir deste ponto todas as entradas foram validadas.
        for nome, preparador in preparadores.items():
            print("\n" + "=" * 90)
            print(f"RECRIANDO SAIDA - BASE {nome}")
            print("=" * 90)
            print(f"\nDestino: {preparador.destino}")

            # Remove a saida antiga e cria a estrutura limpa de classes.
            preparador.recriar_saida()

            # Converte e salva os arquivos da base atual.
            gerados = preparador.processar()

            print("\n" + "=" * 90)
            print(f"VERIFICACAO FINAL - BASE {nome}")
            print("=" * 90)

            # Reabre todos os NPY e compara as quantidades com o plano validado.
            verificacao = preparador.verificar_saida()

            # Guarda informacoes suficientes para o futuro main.py e para logs.
            resultados[nome.lower()] = {
                "origem": str(preparador.origem),
                "destino": str(preparador.destino),
                "gerados": dict(gerados),
                "verificacao": verificacao,
            }

        print("\n" + "=" * 90)
        print("PREPARACAO DE IMAGENS FINALIZADA")
        print("=" * 90)

        for nome, resultado in resultados.items():
            total = sum(resultado["gerados"].values())
            print(f"{nome:<10}: {total} arquivo(s) gerado(s)")

        return resultados
