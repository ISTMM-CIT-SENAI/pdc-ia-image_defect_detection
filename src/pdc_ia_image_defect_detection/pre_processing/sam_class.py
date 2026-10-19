"""
sam_class.py

Objetivo:
- Implementar toda a logica especifica do recorte automatico com SAM 2.1.
- Localizar imagens recursivamente em uma pasta de entrada configurada.
- Segmentar o objeto central de cada imagem utilizando um ponto positivo no centro.
- Avaliar as mascaras candidatas por criterios geometricos e pelo score do SAM.
- Selecionar a mascara mais adequada e gerar um crop quadrado com margem.
- Preservar nomes e estrutura relativa de pastas na pasta de saida.
- Identificar resultados que precisam de revisao manual.
- Gerar CSV, resumo textual e diagnosticos dos casos marcados como REVISAR.

Entrada esperada:
- Uma pasta contendo imagens diretamente ou distribuídas em subpastas.
- A estrutura interna da pasta nao possui significado para a segmentacao.
- As extensoes processadas sao recebidas pelo script processing/sam.py.

Fluxo de cada imagem:
1. Carregar a imagem e corrigir a orientacao EXIF.
2. Converter uma copia para RGB para uso pelo SAM.
3. Definir um ponto positivo no centro geometrico da imagem.
4. Solicitar multiplas mascaras candidatas ao SAM.
5. Calcular as caracteristicas geometricas de cada mascara.
6. Rejeitar candidatas que ultrapassem os limites definidos.
7. Selecionar a melhor candidata valida pelo criterio ponderado.
8. Usar fallback quando nenhuma candidata passar pelos filtros.
9. Criar crop quadrado com margem ao redor da bounding box escolhida.
10. Avaliar se o resultado deve ser marcado como OK ou REVISAR.
11. Salvar o crop preservando o caminho relativo da imagem de origem.
12. Salvar diagnostico visual quando o resultado for REVISAR e a opcao estiver ativa.

Criterios de uma candidata:
- Deve conter o ponto central usado como prompt.
- A area relativa da mascara deve permanecer dentro dos limites configurados.
- A mascara deve preencher uma fracao minima de sua bounding box.
- A bounding box nao pode ocupar area excessiva da imagem.
- O crop estimado nao pode ultrapassar o limite relativo configurado.

Selecao entre candidatas validas:
- O criterio combina preenchimento da bounding box, centralidade e score do SAM.
- Os pesos sao definidos em processing/sam.py.
- Quando candidatas ficam dentro da tolerancia de empate, a de maior area e escolhida.
- A preferencia pela maior area reduz o risco de selecionar apenas um defeito interno
  em vez da superficie completa da peca.

Marcacao REVISAR:
- Fallback por ausencia de candidata valida.
- Valores proximos dos limites geometricos definidos para revisao.
- Crop limitado pelas bordas da imagem.
- Crop que nao contenha completamente a bounding box.
- O score baixo do SAM e registrado separadamente e nao marca REVISAR sozinho.

Saida:
- Os crops sao gravados na pasta de saida mantendo nomes, extensoes e subpastas.
- Cada execucao cria uma nova pasta temporal em PASTA_RESULTADOS_RAIZ.
- resultados.csv registra a candidata escolhida e dados das tres mascaras retornadas.
- resumo.txt registra totais, parametros principais, motivos de revisao e erros.
- revisar/ recebe diagnosticos graficos apenas quando habilitado.

Observacoes:
- A pasta de entrada nunca e alterada.
- Toda a logica especifica do SAM permanece nesta classe.
- Listagem e carregamento generico de imagens sao reutilizados de utils.imagens.
"""

import os

# Desativa apenas o aviso de links simbolicos do cache do Hugging Face.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from collections import Counter
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
import csv
import shutil

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from sam2.sam2_image_predictor import SAM2ImagePredictor

from pdc_ia_image_defect_detection.utils.imagens import (
    carregar_imagem_pil,
    listar_imagens,
)


# =============================================================================
# PROCESSADOR SAM
# =============================================================================

class SamProcessor:
    """
    Executa segmentacao, selecao de mascara, crop e diagnostico com SAM 2.1.

    A instancia representa uma unica execucao sobre uma pasta de entrada e utiliza
    somente os parametros recebidos pelo script processing/sam.py.
    """

    # =========================================================================
    # INICIALIZACAO
    # =========================================================================

    def __init__(
        self,
        pasta_entrada,
        pasta_saida,
        pasta_resultados_raiz,
        modelo_sam,
        extensoes_imagem,
        parametros,
        limpar_pasta_saida=True,
        salvar_diagnosticos_revisar=True,
        dpi_graficos=160,
    ):
        """
        Inicializa os caminhos e parametros da execucao.

        Parameters
        ----------
        pasta_entrada : str ou Path
            Pasta pesquisada recursivamente por imagens.

        pasta_saida : str ou Path
            Pasta que recebera os crops preservando a estrutura relativa.

        pasta_resultados_raiz : str ou Path
            Pasta principal onde sera criada a pasta temporal da execucao.

        modelo_sam : str
            Identificador do modelo usado por SAM2ImagePredictor.from_pretrained().

        extensoes_imagem : iterable[str]
            Extensoes aceitas durante a busca recursiva.

        parametros : dict
            Limites geometricos, faixas de revisao e pesos da selecao.

        limpar_pasta_saida : bool
            Quando True, remove a pasta de saida existente antes do processamento.

        salvar_diagnosticos_revisar : bool
            Quando True, salva graficos apenas para resultados REVISAR.

        dpi_graficos : int
            Resolucao dos diagnosticos graficos.
        """

        # Padroniza todos os caminhos recebidos como Path absolutos.
        self.pasta_entrada = Path(pasta_entrada).resolve()
        self.pasta_saida = Path(pasta_saida).resolve()
        self.pasta_resultados_raiz = Path(pasta_resultados_raiz).resolve()

        # Armazena a identificacao do modelo SAM utilizado na execucao.
        self.modelo_sam = str(modelo_sam)

        # Mantem uma copia imutavel por convencao das extensoes aceitas.
        self.extensoes_imagem = tuple(extensoes_imagem)

        # Copia os parametros para evitar alterar o dicionario definido no processing.
        self.parametros = dict(parametros)

        # Padroniza as opcoes de execucao nos tipos esperados.
        self.limpar_pasta_saida = bool(limpar_pasta_saida)
        self.salvar_diagnosticos_revisar = bool(salvar_diagnosticos_revisar)
        self.dpi_graficos = int(dpi_graficos)

        # Seleciona GPU quando CUDA estiver disponivel.
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # O predictor e carregado somente depois de todas as validacoes.
        self.predictor = None

        # Os caminhos da execucao sao criados em _preparar_pastas().
        self.inicio = None
        self.pasta_resultados = None
        self.pasta_diagnosticos = None

    # =========================================================================
    # VALIDACAO
    # =========================================================================

    def _validar(self):
        """
        Valida caminhos e parametros antes de carregar o modelo ou remover arquivos.

        Raises
        ------
        FileNotFoundError
            Quando a pasta de entrada nao existe.

        NotADirectoryError
            Quando entrada ou pasta de resultados existente nao sao diretorios.

        ValueError, TypeError, KeyError
            Quando caminhos ou parametros possuem configuracao invalida.
        """

        # A pasta de entrada precisa existir antes da busca de imagens.
        if not self.pasta_entrada.exists():
            raise FileNotFoundError(
                f"Pasta de entrada nao encontrada:\n{self.pasta_entrada}"
            )

        # A entrada precisa ser um diretorio porque a busca sera recursiva.
        if not self.pasta_entrada.is_dir():
            raise NotADirectoryError(
                f"A pasta de entrada nao e um diretorio:\n{self.pasta_entrada}"
            )

        # Entrada e saida iguais causariam sobrescrita direta das imagens originais.
        if self.pasta_entrada == self.pasta_saida:
            raise ValueError(
                "PASTA_ENTRADA e PASTA_SAIDA nao podem ser iguais."
            )

        # Impede criar a saida dentro da entrada, o que poderia fazer crops antigos
        # aparecerem em buscas recursivas de execucoes futuras.
        try:
            self.pasta_saida.relative_to(self.pasta_entrada)
            saida_dentro_entrada = True
        except ValueError:
            saida_dentro_entrada = False

        if saida_dentro_entrada:
            raise ValueError(
                "PASTA_SAIDA nao pode ficar dentro de PASTA_ENTRADA."
            )

        # A pasta principal de resultados pode nao existir ainda, mas quando existe
        # precisa ser um diretorio valido.
        if self.pasta_resultados_raiz.exists() and not self.pasta_resultados_raiz.is_dir():
            raise NotADirectoryError(
                "PASTA_RESULTADOS_RAIZ existe, mas nao e um diretorio:\n"
                f"{self.pasta_resultados_raiz}"
            )

        # O identificador do modelo nao pode ser vazio.
        if not self.modelo_sam.strip():
            raise ValueError("MODELO_SAM nao pode ser vazio.")

        # Pelo menos uma extensao precisa estar configurada para a busca.
        if not self.extensoes_imagem:
            raise ValueError("EXTENSOES_IMAGEM nao pode ficar vazio.")

        # A resolucao dos graficos precisa ser um inteiro positivo.
        if not isinstance(self.dpi_graficos, int) or self.dpi_graficos <= 0:
            raise ValueError("DPI_GRAFICOS deve ser inteiro positivo.")

        # Valida os campos numericos usados pelas demais etapas do algoritmo.
        self._validar_parametros()

    def _validar_parametros(self):
        """Valida os limites geometricos e os pesos recebidos do processing."""

        # Define todos os parametros exigidos pelo algoritmo atual.
        obrigatorios = {
            "margem_crop",
            "area_minima_candidata",
            "area_maxima_candidata",
            "preenchimento_bbox_minimo",
            "area_bbox_maxima_candidata",
            "crop_relativo_maximo_candidata",
            "area_revisar_minima",
            "area_revisar_maxima",
            "preenchimento_revisar_minimo",
            "area_bbox_revisar_maxima",
            "crop_relativo_revisar_maximo",
            "centralidade_revisar_minima",
            "score_sam_baixo",
            "peso_preenchimento",
            "peso_centralidade",
            "peso_score",
            "tolerancia_empate",
        }

        # Detecta configuracoes incompletas antes do processamento.
        faltando = obrigatorios - set(self.parametros)
        if faltando:
            raise KeyError(
                f"Parametros SAM ausentes: {sorted(faltando)}"
            )

        # Todos os parametros atuais precisam ser valores numericos reais.
        for nome in obrigatorios:
            valor = self.parametros[nome]
            if not isinstance(valor, (int, float)):
                raise TypeError(
                    f"{nome} deve ser numerico. Recebido: {type(valor).__name__}."
                )

        # A margem nao pode remover area ao redor da bounding box.
        if self.parametros["margem_crop"] < 0:
            raise ValueError("margem_crop deve ser >= 0.")

        # Limites percentuais e scores devem permanecer entre zero e um.
        parametros_unitarios = (
            "area_minima_candidata",
            "area_maxima_candidata",
            "preenchimento_bbox_minimo",
            "area_bbox_maxima_candidata",
            "crop_relativo_maximo_candidata",
            "area_revisar_minima",
            "area_revisar_maxima",
            "preenchimento_revisar_minimo",
            "area_bbox_revisar_maxima",
            "crop_relativo_revisar_maximo",
            "centralidade_revisar_minima",
            "score_sam_baixo",
        )

        for nome in parametros_unitarios:
            valor = float(self.parametros[nome])
            if not 0.0 <= valor <= 1.0:
                raise ValueError(
                    f"{nome} deve estar entre 0 e 1. Recebido: {valor}."
                )

        # O limite inferior de area precisa ser menor que o limite superior.
        if not (
            self.parametros["area_minima_candidata"]
            < self.parametros["area_maxima_candidata"]
        ):
            raise ValueError(
                "area_minima_candidata deve ser menor que area_maxima_candidata."
            )

        # A faixa conservadora de revisao tambem precisa possuir ordem valida.
        if not (
            self.parametros["area_revisar_minima"]
            < self.parametros["area_revisar_maxima"]
        ):
            raise ValueError(
                "area_revisar_minima deve ser menor que area_revisar_maxima."
            )

        # Os pesos nao podem ser negativos.
        for nome in (
            "peso_preenchimento",
            "peso_centralidade",
            "peso_score",
        ):
            if self.parametros[nome] < 0:
                raise ValueError(f"{nome} deve ser >= 0.")

        # Pelo menos um componente precisa participar do criterio combinado.
        soma_pesos = (
            self.parametros["peso_preenchimento"]
            + self.parametros["peso_centralidade"]
            + self.parametros["peso_score"]
        )

        if soma_pesos <= 0:
            raise ValueError("A soma dos pesos de selecao deve ser maior que zero.")

        # A tolerancia representa uma diferenca absoluta de criterio.
        if self.parametros["tolerancia_empate"] < 0:
            raise ValueError("tolerancia_empate deve ser >= 0.")

    # =========================================================================
    # PREPARACAO DE PASTAS
    # =========================================================================

    def _preparar_pastas(self):
        """Prepara a pasta de saida e cria a pasta exclusiva desta execucao."""

        # Remove a saida anterior somente quando a opcao estiver habilitada.
        if self.limpar_pasta_saida and self.pasta_saida.exists():
            shutil.rmtree(self.pasta_saida)

        # Cria a pasta que recebera os novos crops.
        self.pasta_saida.mkdir(parents=True, exist_ok=True)

        # Registra o inicio somente no momento em que a execucao sera realmente criada.
        self.inicio = datetime.now()

        # Cada execucao possui uma pasta propria para evitar sobrescrever relatorios.
        self.pasta_resultados = (
            self.pasta_resultados_raiz
            / f"execucao_{self.inicio:%Y-%m-%d_%H-%M-%S}"
        )
        self.pasta_resultados.mkdir(parents=True, exist_ok=False)

        # Os diagnosticos REVISAR ficam isolados em uma subpasta propria.
        self.pasta_diagnosticos = self.pasta_resultados / "revisar"

        if self.salvar_diagnosticos_revisar:
            self.pasta_diagnosticos.mkdir(parents=True, exist_ok=False)

    # =========================================================================
    # MODELO E PRECISAO
    # =========================================================================

    def _contexto_autocast(self):
        """Retorna autocast BF16 em CUDA compativel ou contexto neutro nos demais casos."""

        # BF16 reduz memoria e pode acelerar inferencia em GPUs compativeis.
        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )

        # CPU ou GPU sem BF16 executa sem autocast adicional.
        return nullcontext()

    def _carregar_sam(self):
        """Carrega o predictor SAM 2.1 configurado para o dispositivo disponivel."""

        print("\nCarregando SAM 2.1...")
        print(f"Modelo      : {self.modelo_sam}")
        print(f"Dispositivo : {self.device}")

        # from_pretrained baixa ou reutiliza o checkpoint configurado.
        self.predictor = SAM2ImagePredictor.from_pretrained(
            self.modelo_sam,
            device=self.device,
        )

        print("SAM 2.1 carregado.")

    # =========================================================================
    # GEOMETRIA DA MASCARA
    # =========================================================================

    @staticmethod
    def _obter_bbox(mask):
        """
        Calcula a menor bounding box que contem todos os pixels True da mascara.

        Returns
        -------
        tuple[int, int, int, int] ou None
            Caixa no formato (x1, y1, x2, y2), com x2 e y2 exclusivos.
            Retorna None quando a mascara nao possui pixels ativos.
        """

        # Localiza as coordenadas de todos os pixels pertencentes a mascara.
        ys, xs = np.where(mask)

        # Uma mascara vazia nao possui bounding box valida.
        if len(xs) == 0:
            return None

        # Soma um aos limites maximos para seguir a convencao de caixas exclusivas.
        return (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        )

    @staticmethod
    def _calcular_centralidade(bbox, largura, altura):
        """
        Mede o quanto o centro da bounding box esta proximo do centro da imagem.

        O valor resultante fica em [0,1], onde 1 representa coincidencia dos centros.
        """

        # Sem bounding box nao existe centralidade mensuravel.
        if bbox is None:
            return 0.0

        x1, y1, x2, y2 = bbox

        # Calcula o centro geometrico da bounding box.
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Normaliza a distancia em cada eixo pelo semitamanho da imagem.
        dx = (cx - largura / 2.0) / (largura / 2.0)
        dy = (cy - altura / 2.0) / (altura / 2.0)

        # A distancia normalizada maxima entre centro e canto e sqrt(2).
        distancia = np.sqrt(dx ** 2 + dy ** 2)

        # Converte distancia em uma medida de proximidade limitada a zero.
        return float(
            max(0.0, 1.0 - distancia / np.sqrt(2.0))
        )

    @staticmethod
    def _calcular_preenchimento_bbox(mask, bbox):
        """Calcula a fracao da bounding box efetivamente preenchida pela mascara."""

        # Sem bounding box o preenchimento e definido como zero.
        if bbox is None:
            return 0.0

        x1, y1, x2, y2 = bbox

        # Calcula a area total da caixa em pixels.
        area_bbox = (x2 - x1) * (y2 - y1)

        # Protege contra caixas degeneradas.
        if area_bbox <= 0:
            return 0.0

        # mask.sum() corresponde ao numero de pixels True da mascara.
        return float(mask.sum() / area_bbox)

    def _calcular_crop_relativo_bbox(self, bbox, largura, altura):
        """Estima o lado do crop quadrado em relacao ao menor lado da imagem."""

        # Mascara sem bounding box recebe o pior valor possivel.
        if bbox is None:
            return 1.0

        x1, y1, x2, y2 = bbox

        # O maior lado da bounding box define o quadrado necessario para conter o objeto.
        largura_bbox = x2 - x1
        altura_bbox = y2 - y1
        lado_objeto = max(largura_bbox, altura_bbox)

        # Acrescenta a margem configurada nos dois lados do objeto.
        lado_desejado = lado_objeto * (
            1.0 + 2.0 * self.parametros["margem_crop"]
        )

        # Um crop quadrado nao pode exceder o menor lado da imagem sem criar pixels novos.
        menor_dimensao = min(largura, altura)
        lado_estimado = min(lado_desejado, menor_dimensao)

        return float(lado_estimado / menor_dimensao)

    def _motivos_invalidade_candidata(self, candidata):
        """Retorna os filtros geometricos que uma candidata nao conseguiu atender."""

        motivos = []

        # A mascara deve incluir o ponto positivo fornecido ao SAM.
        if not candidata["contem_centro"]:
            motivos.append("nao contem centro")

        # Area muito pequena pode representar apenas detalhe ou defeito local.
        if candidata["area"] < self.parametros["area_minima_candidata"]:
            motivos.append("area muito pequena")

        # Area muito grande pode representar fundo ou uma regiao incorreta.
        if candidata["area"] > self.parametros["area_maxima_candidata"]:
            motivos.append("area muito grande")

        # Fill baixo indica mascara esparsa dentro de uma bounding box grande.
        if (
            candidata["preenchimento_bbox"]
            < self.parametros["preenchimento_bbox_minimo"]
        ):
            motivos.append("fill baixo")

        # Bounding box excessiva indica uma regiao ampla demais da imagem.
        if (
            candidata["bbox_area_relativa"]
            > self.parametros["area_bbox_maxima_candidata"]
        ):
            motivos.append("bbox muito grande")

        # Crop estimado muito amplo tende a preservar excesso de fundo.
        if (
            candidata["crop_relativo_estimado"]
            > self.parametros["crop_relativo_maximo_candidata"]
        ):
            motivos.append("crop estimado muito grande")

        return motivos

    def _analisar_candidata(self, mask, score, largura, altura):
        """Calcula todas as caracteristicas usadas para avaliar uma mascara candidata."""

        # Area relativa corresponde a proporcao de pixels True na imagem completa.
        area = float(mask.mean())

        # Bounding box representa a menor caixa que contem a mascara.
        bbox = self._obter_bbox(mask)

        # Mascara vazia recebe valores neutros e e automaticamente invalida.
        if bbox is None:
            return {
                "score": float(score),
                "area": 0.0,
                "centralidade": 0.0,
                "preenchimento_bbox": 0.0,
                "bbox_area_relativa": 0.0,
                "crop_relativo_estimado": 1.0,
                "contem_centro": False,
                "criterio": 0.0,
                "valida": False,
                "motivos_invalidade": "mascara vazia",
                "bbox": None,
            }

        x1, y1, x2, y2 = bbox

        # Calcula a area ocupada pela bounding box em relacao a imagem completa.
        bbox_area = (x2 - x1) * (y2 - y1)
        bbox_area_relativa = bbox_area / (largura * altura)

        # Usa um pixel inteiro proximo do centro geometrico como referencia.
        xc = min(int(largura / 2), largura - 1)
        yc = min(int(altura / 2), altura - 1)

        # Confere se a mascara realmente inclui o ponto positivo central.
        contem_centro = bool(mask[yc, xc])

        # Calcula as medidas geometricas complementares da candidata.
        preenchimento = self._calcular_preenchimento_bbox(mask, bbox)
        centralidade = self._calcular_centralidade(bbox, largura, altura)
        crop_relativo = self._calcular_crop_relativo_bbox(
            bbox,
            largura,
            altura,
        )

        # Limita apenas a parcela do score usada no criterio ponderado.
        score_normalizado = float(np.clip(score, 0.0, 1.0))

        # Combina os indicadores usando os pesos configurados.
        criterio = (
            self.parametros["peso_preenchimento"] * preenchimento
            + self.parametros["peso_centralidade"] * centralidade
            + self.parametros["peso_score"] * score_normalizado
        )

        # Reune todos os dados calculados para as etapas seguintes.
        candidata = {
            "score": float(score),
            "area": area,
            "centralidade": centralidade,
            "preenchimento_bbox": preenchimento,
            "bbox_area_relativa": float(bbox_area_relativa),
            "crop_relativo_estimado": crop_relativo,
            "contem_centro": contem_centro,
            "criterio": float(criterio),
            "bbox": bbox,
        }

        # Aplica os limites duros que definem se a candidata e valida.
        motivos = self._motivos_invalidade_candidata(candidata)

        candidata["valida"] = len(motivos) == 0
        candidata["motivos_invalidade"] = ", ".join(motivos)

        return candidata

    # =========================================================================
    # SELECAO DA MASCARA
    # =========================================================================

    def _selecionar_mascara(self, masks, scores, largura, altura):
        """
        Seleciona a melhor mascara candidata.

        Returns
        -------
        tuple[int, list[dict], bool]
            Indice escolhido, dados de todas as candidatas e indicador de fallback.
        """

        # Analisa cada par mascara/score retornado pelo SAM.
        candidatas = [
            self._analisar_candidata(mask, score, largura, altura)
            for mask, score in zip(masks, scores)
        ]

        # Guarda os indices que passaram por todos os filtros geometricos.
        validas = [
            indice
            for indice, candidata in enumerate(candidatas)
            if candidata["valida"]
        ]

        # Sem candidata valida, usa o maior criterio como fallback.
        # O resultado sera posteriormente marcado para revisao.
        if not validas:
            indice = int(
                np.argmax([
                    candidata["criterio"]
                    for candidata in candidatas
                ])
            )

            return indice, candidatas, True

        # Encontra o melhor criterio apenas entre candidatas validas.
        melhor_criterio = max(
            candidatas[indice]["criterio"]
            for indice in validas
        )

        # Considera empatadas as candidatas suficientemente proximas do melhor valor.
        proximas = [
            indice
            for indice in validas
            if (
                melhor_criterio - candidatas[indice]["criterio"]
                <= self.parametros["tolerancia_empate"]
            )
        ]

        # Entre candidatas equivalentes, prefere a de maior area segmentada.
        indice = max(
            proximas,
            key=lambda i: candidatas[i]["area"],
        )

        return indice, candidatas, False

    # =========================================================================
    # SEGMENTACAO
    # =========================================================================

    def _segmentar(self, imagem):
        """Executa SAM com um ponto positivo no centro e retorna as mascaras candidatas."""

        # SAM recebe uma representacao RGB independentemente do modo original.
        imagem_rgb = imagem.convert("RGB")
        imagem_np = np.asarray(imagem_rgb).copy()

        # Extrai altura e largura da representacao NumPy.
        altura, largura = imagem_np.shape[:2]

        # Define o prompt positivo exatamente no centro geometrico da imagem.
        ponto = np.array(
            [[largura / 2.0, altura / 2.0]],
            dtype=np.float32,
        )

        # Rotulo 1 informa ao SAM que o ponto pertence ao objeto desejado.
        rotulo = np.array([1], dtype=np.int32)

        # inference_mode desativa gradientes porque o modelo esta apenas em inferencia.
        with torch.inference_mode():
            # Usa BF16 automaticamente quando a GPU oferece suporte.
            with self._contexto_autocast():
                # Registra a imagem atual no predictor.
                self.predictor.set_image(imagem_np)

                # multimask_output=True solicita as mascaras alternativas do SAM.
                masks, scores, _ = self.predictor.predict(
                    point_coords=ponto,
                    point_labels=rotulo,
                    multimask_output=True,
                )

        # Padroniza os tipos usados pelas analises geometricas.
        masks = np.asarray(masks, dtype=bool)
        scores = np.asarray(scores, dtype=np.float32)

        # Escolhe a mascara com base nos criterios configurados.
        indice, candidatas, fallback = self._selecionar_mascara(
            masks,
            scores,
            largura,
            altura,
        )

        return masks, scores, indice, candidatas, fallback, ponto[0]

    # =========================================================================
    # CROP
    # =========================================================================

    def _criar_crop_quadrado(self, imagem, bbox):
        """
        Cria um crop quadrado centralizado na bounding box selecionada.

        A margem e aplicada aos dois lados do maior eixo da bounding box. Quando o
        tamanho desejado excede a menor dimensao da imagem, o crop e limitado ao
        maior quadrado que cabe integralmente na imagem.
        """

        x1, y1, x2, y2 = bbox

        # Calcula largura e altura da bounding box selecionada.
        largura_bbox = x2 - x1
        altura_bbox = y2 - y1

        # O maior eixo define o lado minimo do quadrado que contem toda a caixa.
        lado_objeto = max(largura_bbox, altura_bbox)

        # Converte a margem relativa configurada para pixels.
        margem = round(
            lado_objeto * self.parametros["margem_crop"]
        )

        # A margem e adicionada dos dois lados do quadrado.
        lado_desejado = lado_objeto + 2 * margem

        # O crop nao pode ser maior que o menor lado da imagem.
        largura_img, altura_img = imagem.size
        menor_dimensao = min(largura_img, altura_img)
        lado_crop = min(lado_desejado, menor_dimensao)

        # Registra se a margem precisou ser limitada pelas dimensoes da imagem.
        limitado = lado_crop < lado_desejado

        # Calcula o centro geometrico da bounding box.
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Posiciona inicialmente o quadrado ao redor do centro da bounding box.
        esquerda = int(round(cx - lado_crop / 2.0))
        superior = int(round(cy - lado_crop / 2.0))

        # Desloca o quadrado quando necessario para mantê-lo dentro da imagem.
        esquerda = max(
            0,
            min(esquerda, largura_img - lado_crop),
        )
        superior = max(
            0,
            min(superior, altura_img - lado_crop),
        )

        # Calcula os limites finais da caixa de crop.
        direita = esquerda + lado_crop
        inferior = superior + lado_crop

        # Confere se o crop final preserva integralmente a bounding box original.
        contem_bbox = (
            esquerda <= x1
            and superior <= y1
            and direita >= x2
            and inferior >= y2
        )

        # Executa o recorte na imagem original carregada.
        crop = imagem.crop(
            (esquerda, superior, direita, inferior)
        )

        # O resultado precisa ser rigorosamente quadrado.
        if crop.width != crop.height:
            raise ValueError(
                f"Crop nao ficou quadrado: {crop.width}x{crop.height}"
            )

        # Retorna o crop e todas as informacoes geometricas usadas no relatorio.
        return crop, {
            "lado_desejado": lado_desejado,
            "lado_crop": lado_crop,
            "margem_px": margem,
            "crop_relativo": lado_crop / menor_dimensao,
            "caixa_crop": (
                esquerda,
                superior,
                direita,
                inferior,
            ),
            "limitado": limitado,
            "contem_bbox": contem_bbox,
        }

    # =========================================================================
    # AVALIACAO FINAL
    # =========================================================================

    def _avaliar_resultado(self, candidata, fallback, info_crop):
        """Classifica o resultado como OK ou REVISAR e registra os motivos encontrados."""

        motivos = []

        # Fallback indica que nenhuma das mascaras passou pelos filtros obrigatorios.
        if fallback:
            motivos.append(
                "nenhuma candidata passou pelos filtros geometricos"
            )

        # Repete os limites absolutos para registrar explicitamente qualquer falha.
        if not candidata["contem_centro"]:
            motivos.append("mascara nao contem ponto central")

        if candidata["area"] < self.parametros["area_minima_candidata"]:
            motivos.append("area abaixo do limite da candidata")

        if candidata["area"] > self.parametros["area_maxima_candidata"]:
            motivos.append("area acima do limite da candidata")

        if (
            candidata["preenchimento_bbox"]
            < self.parametros["preenchimento_bbox_minimo"]
        ):
            motivos.append("fill abaixo do limite da candidata")

        if (
            candidata["bbox_area_relativa"]
            > self.parametros["area_bbox_maxima_candidata"]
        ):
            motivos.append("bounding box excessivamente grande")

        if (
            candidata["crop_relativo_estimado"]
            > self.parametros["crop_relativo_maximo_candidata"]
        ):
            motivos.append("crop estimado excessivamente grande")

        # As faixas conservadoras nao invalidam a candidata, mas solicitam revisao.
        if candidata["area"] < self.parametros["area_revisar_minima"]:
            motivos.append("area proxima do limite inferior")

        if candidata["area"] > self.parametros["area_revisar_maxima"]:
            motivos.append("area proxima do limite superior")

        if (
            candidata["preenchimento_bbox"]
            < self.parametros["preenchimento_revisar_minimo"]
        ):
            motivos.append("fill baixo para o padrao do dataset")

        if (
            candidata["bbox_area_relativa"]
            > self.parametros["area_bbox_revisar_maxima"]
        ):
            motivos.append("bbox grande para o padrao do dataset")

        if (
            candidata["centralidade"]
            < self.parametros["centralidade_revisar_minima"]
        ):
            motivos.append("centralidade abaixo do padrao esperado")

        if (
            info_crop["crop_relativo"]
            > self.parametros["crop_relativo_revisar_maximo"]
        ):
            motivos.append("crop final grande para o padrao do dataset")

        # Registra quando a imagem limitou a margem desejada.
        if info_crop["limitado"]:
            motivos.append("margem do crop limitada pela imagem")

        # O crop precisa conter toda a bounding box selecionada.
        if not info_crop["contem_bbox"]:
            motivos.append("crop nao contem toda a bounding box")

        # Remove motivos duplicados sem alterar a ordem de aparecimento.
        motivos = list(dict.fromkeys(motivos))

        # Ausencia de motivos representa resultado aprovado automaticamente.
        status = "OK" if not motivos else "REVISAR"

        return status, motivos

    # =========================================================================
    # SALVAMENTO DO CROP
    # =========================================================================

    def _salvar_crop(self, crop, caminho_origem):
        """Salva o crop mantendo caminho relativo, nome e extensao da imagem original."""

        # Calcula o caminho original relativo a raiz pesquisada.
        relativo = caminho_origem.relative_to(self.pasta_entrada)

        # Reproduz a mesma estrutura de subpastas na pasta de saida.
        destino = self.pasta_saida / relativo
        destino.parent.mkdir(parents=True, exist_ok=True)

        # Mantem o crop original sempre que o formato de destino aceitar seu modo.
        imagem_salvar = crop

        # JPEG suporta os modos usuais RGB e L; outros modos sao convertidos para RGB.
        if (
            destino.suffix.lower() in {".jpg", ".jpeg"}
            and imagem_salvar.mode not in {"RGB", "L"}
        ):
            imagem_salvar = imagem_salvar.convert("RGB")

        # Salva usando a extensao original para preservar a estrutura da fonte.
        imagem_salvar.save(destino)

        return destino

    # =========================================================================
    # DIAGNOSTICO GRAFICO
    # =========================================================================

    def _nome_diagnostico(self, caminho):
        """Converte o caminho relativo da imagem em um nome unico para o diagnostico."""

        # Remove apenas a extensao da imagem original.
        relativo = caminho.relative_to(self.pasta_entrada).with_suffix("")

        # Substitui separadores de pasta por '__' para produzir um unico arquivo.
        return "__".join(relativo.parts)

    def _salvar_diagnostico(
        self,
        caminho,
        imagem,
        masks,
        candidatas,
        indice,
        ponto,
        crop,
        info_crop,
        motivos,
    ):
        """Salva um painel visual com original, candidatas, escolha e crop final."""

        # Garante representacao RGB consistente para visualizacao no Matplotlib.
        imagem_np = np.asarray(
            imagem.convert("RGB")
        )

        # O painel utiliza seis regioes: original, tres mascaras, escolha e crop.
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(18, 11),
        )
        axes = axes.ravel()

        # Exibe a imagem original e o ponto positivo fornecido ao SAM.
        axes[0].imshow(imagem_np)
        axes[0].scatter(
            ponto[0],
            ponto[1],
            s=100,
            marker="*",
        )
        axes[0].set_title(
            f"Original + ponto central\n{imagem.width}x{imagem.height}"
        )
        axes[0].axis("off")

        # Exibe ate tres candidatas retornadas pelo modo multimask do SAM.
        for i in range(min(3, len(masks))):
            candidata = candidatas[i]

            axes[i + 1].imshow(imagem_np)
            axes[i + 1].imshow(
                masks[i],
                alpha=0.45,
            )

            # Identifica visualmente a mascara escolhida.
            selecionada = " - ESCOLHIDA" if i == indice else ""
            validade = "VALIDA" if candidata["valida"] else "INVALIDA"

            axes[i + 1].set_title(
                f"Mask {i + 1}{selecionada}\n"
                f"SAM={candidata['score']:.4f} | "
                f"Area={candidata['area']:.2%}\n"
                f"Fill={candidata['preenchimento_bbox']:.2%} | "
                f"BBox={candidata['bbox_area_relativa']:.2%}\n"
                f"Crop={candidata['crop_relativo_estimado']:.2%} | "
                f"Centro={candidata['centralidade']:.3f}\n"
                f"Criterio={candidata['criterio']:.3f} | {validade}"
            )
            axes[i + 1].axis("off")

        # Recupera a candidata selecionada para desenhar sua geometria.
        candidata = candidatas[indice]

        axes[4].imshow(imagem_np)
        axes[4].imshow(
            masks[indice],
            alpha=0.35,
        )

        # Desenha a bounding box da mascara quando ela existe.
        if candidata["bbox"] is not None:
            x1, y1, x2, y2 = candidata["bbox"]
            axes[4].add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    linewidth=2,
                )
            )

        # Desenha a caixa quadrada efetivamente usada no crop.
        esquerda, superior, direita, inferior = info_crop["caixa_crop"]
        axes[4].add_patch(
            Rectangle(
                (esquerda, superior),
                direita - esquerda,
                inferior - superior,
                fill=False,
                linewidth=2,
                linestyle="--",
            )
        )

        axes[4].set_title(
            "Selecionada\n"
            f"Area={candidata['area']:.2%} | "
            f"Fill={candidata['preenchimento_bbox']:.2%}\n"
            f"BBox={candidata['bbox_area_relativa']:.2%} | "
            f"Crop={info_crop['crop_relativo']:.2%}"
        )
        axes[4].axis("off")

        # Exibe o crop final salvo para permitir avaliacao visual direta.
        axes[5].imshow(
            crop.convert("RGB")
        )
        axes[5].set_title(
            f"REVISAR\n{crop.width}x{crop.height}"
        )
        axes[5].axis("off")

        # O titulo geral identifica a imagem e os motivos de revisao.
        fig.suptitle(
            f"{caminho.relative_to(self.pasta_entrada)}\n"
            + " | ".join(motivos),
            fontsize=11,
        )

        fig.tight_layout()

        # O caminho relativo e convertido para um nome unico de diagnostico.
        destino = (
            self.pasta_diagnosticos
            / f"{self._nome_diagnostico(caminho)}_diagnostico.png"
        )

        fig.savefig(
            destino,
            dpi=self.dpi_graficos,
        )

        # Fecha explicitamente a figura para liberar memoria durante lotes grandes.
        plt.close(fig)

        return destino

    # =========================================================================
    # PROCESSAMENTO DE UMA IMAGEM
    # =========================================================================

    def _processar_imagem(self, caminho):
        """Executa o fluxo completo do SAM para uma unica imagem."""

        # Corrige a orientacao EXIF antes de qualquer medida geometrica.
        imagem, _ = carregar_imagem_pil(
            caminho,
            corrigir_exif=True,
        )

        # Executa a segmentacao e recebe todas as candidatas analisadas.
        (
            masks,
            scores,
            indice,
            candidatas,
            fallback,
            ponto,
        ) = self._segmentar(imagem)

        # Recupera a candidata selecionada pelo criterio atual.
        candidata = candidatas[indice]

        # Uma mascara vazia nao permite gerar crop geometricamente valido.
        if candidata["bbox"] is None:
            raise ValueError("Mascara selecionada esta vazia.")

        # Cria o crop quadrado usando a bounding box da mascara escolhida.
        crop, info_crop = self._criar_crop_quadrado(
            imagem,
            candidata["bbox"],
        )

        # Classifica o resultado e registra todos os motivos de revisao.
        status, motivos = self._avaliar_resultado(
            candidata,
            fallback,
            info_crop,
        )

        # Salva o crop mantendo o caminho relativo da imagem de origem.
        destino = self._salvar_crop(
            crop,
            caminho,
        )

        # Diagnosticos graficos sao criados apenas para resultados REVISAR.
        if status == "REVISAR" and self.salvar_diagnosticos_revisar:
            self._salvar_diagnostico(
                caminho,
                imagem,
                masks,
                candidatas,
                indice,
                ponto,
                crop,
                info_crop,
                motivos,
            )

        # Desempacota a bounding box para registrar medidas detalhadas no CSV.
        x1, y1, x2, y2 = candidata["bbox"]

        # Reune todos os dados principais da imagem processada.
        resultado = {
            "imagem": str(
                caminho.relative_to(self.pasta_entrada)
            ),
            "saida": str(
                destino.relative_to(self.pasta_saida)
            ),
            "largura_original": imagem.width,
            "altura_original": imagem.height,
            "mascara_escolhida": indice + 1,
            "score_sam": candidata["score"],
            "score_sam_baixo": (
                candidata["score"]
                < self.parametros["score_sam_baixo"]
            ),
            "area_mascara": candidata["area"],
            "fill_bbox": candidata["preenchimento_bbox"],
            "area_bbox": candidata["bbox_area_relativa"],
            "crop_relativo_estimado": candidata["crop_relativo_estimado"],
            "crop_relativo_final": info_crop["crop_relativo"],
            "centralidade": candidata["centralidade"],
            "criterio": candidata["criterio"],
            "fallback": fallback,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "bbox_largura": x2 - x1,
            "bbox_altura": y2 - y1,
            "margem_px": info_crop["margem_px"],
            "lado_crop": info_crop["lado_crop"],
            "status": status,
            "motivos": ", ".join(motivos),
        }

        # Registra separadamente as tres candidatas esperadas do modo multimask.
        for i in range(3):
            if i < len(candidatas):
                c = candidatas[i]
                resultado[f"score_{i + 1}"] = c["score"]
                resultado[f"area_{i + 1}"] = c["area"]
                resultado[f"fill_{i + 1}"] = c["preenchimento_bbox"]
                resultado[f"bbox_area_{i + 1}"] = c["bbox_area_relativa"]
                resultado[f"crop_rel_{i + 1}"] = c["crop_relativo_estimado"]
                resultado[f"centro_{i + 1}"] = c["centralidade"]
                resultado[f"criterio_{i + 1}"] = c["criterio"]
                resultado[f"valida_{i + 1}"] = c["valida"]
                resultado[f"invalidade_{i + 1}"] = c["motivos_invalidade"]
            else:
                resultado[f"score_{i + 1}"] = ""
                resultado[f"area_{i + 1}"] = ""
                resultado[f"fill_{i + 1}"] = ""
                resultado[f"bbox_area_{i + 1}"] = ""
                resultado[f"crop_rel_{i + 1}"] = ""
                resultado[f"centro_{i + 1}"] = ""
                resultado[f"criterio_{i + 1}"] = ""
                resultado[f"valida_{i + 1}"] = ""
                resultado[f"invalidade_{i + 1}"] = ""

        return resultado

    # =========================================================================
    # CSV
    # =========================================================================

    def _salvar_csv(self, resultados):
        """Salva os resultados detalhados de todas as imagens processadas."""

        # Define o arquivo dentro da pasta exclusiva da execucao.
        caminho = self.pasta_resultados / "resultados.csv"

        # A ordem fixa das colunas facilita comparacao entre execucoes.
        campos = [
            "imagem",
            "saida",
            "largura_original",
            "altura_original",
            "mascara_escolhida",
            "score_1",
            "area_1",
            "fill_1",
            "bbox_area_1",
            "crop_rel_1",
            "centro_1",
            "criterio_1",
            "valida_1",
            "invalidade_1",
            "score_2",
            "area_2",
            "fill_2",
            "bbox_area_2",
            "crop_rel_2",
            "centro_2",
            "criterio_2",
            "valida_2",
            "invalidade_2",
            "score_3",
            "area_3",
            "fill_3",
            "bbox_area_3",
            "crop_rel_3",
            "centro_3",
            "criterio_3",
            "valida_3",
            "invalidade_3",
            "score_sam",
            "score_sam_baixo",
            "area_mascara",
            "fill_bbox",
            "area_bbox",
            "crop_relativo_estimado",
            "crop_relativo_final",
            "centralidade",
            "criterio",
            "fallback",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "bbox_largura",
            "bbox_altura",
            "margem_px",
            "lado_crop",
            "status",
            "motivos",
        ]

        # utf-8-sig facilita abertura correta dos textos em planilhas no Windows.
        with open(
            caminho,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as arquivo:
            writer = csv.DictWriter(
                arquivo,
                fieldnames=campos,
                delimiter=";",
            )
            writer.writeheader()
            writer.writerows(resultados)

        return caminho

    # =========================================================================
    # RESUMO
    # =========================================================================

    def _salvar_resumo(self, resultados, erros):
        """Salva um resumo textual da execucao e dos motivos de revisao."""

        # Define o arquivo dentro da pasta exclusiva da execucao.
        caminho = self.pasta_resultados / "resumo.txt"

        # Conta os dois estados produzidos pelo avaliador final.
        ok = sum(
            resultado["status"] == "OK"
            for resultado in resultados
        )
        revisar = sum(
            resultado["status"] == "REVISAR"
            for resultado in resultados
        )

        # Consolida a frequencia de cada motivo de revisao.
        contador_motivos = Counter()

        for resultado in resultados:
            if resultado["status"] == "REVISAR":
                for motivo in resultado["motivos"].split(", "):
                    if motivo:
                        contador_motivos[motivo] += 1

        # Grava caminhos, contagens, parametros e erros da execucao.
        with open(
            caminho,
            "w",
            encoding="utf-8",
        ) as arquivo:
            arquivo.write("CROP DE IMAGENS COM SAM 2.1\n")
            arquivo.write("=" * 70 + "\n\n")

            arquivo.write(f"Entrada    : {self.pasta_entrada}\n")
            arquivo.write(f"Saida      : {self.pasta_saida}\n")
            arquivo.write(f"Modelo     : {self.modelo_sam}\n")
            arquivo.write(f"Processadas: {len(resultados)}\n")
            arquivo.write(f"OK         : {ok}\n")
            arquivo.write(f"REVISAR    : {revisar}\n")
            arquivo.write(f"ERROS      : {len(erros)}\n\n")

            arquivo.write("LIMITES DAS CANDIDATAS\n")
            arquivo.write(
                "Area       : "
                f"{self.parametros['area_minima_candidata']:.2%} a "
                f"{self.parametros['area_maxima_candidata']:.2%}\n"
            )
            arquivo.write(
                "Fill minimo: "
                f"{self.parametros['preenchimento_bbox_minimo']:.2%}\n"
            )
            arquivo.write(
                "BBox maxima: "
                f"{self.parametros['area_bbox_maxima_candidata']:.2%}\n"
            )
            arquivo.write(
                "Crop maximo: "
                f"{self.parametros['crop_relativo_maximo_candidata']:.2%}\n\n"
            )

            # Lista os motivos de revisao do mais frequente para o menos frequente.
            if contador_motivos:
                arquivo.write("MOTIVOS DE REVISAO\n")
                for motivo, quantidade in contador_motivos.most_common():
                    arquivo.write(f"{quantidade:3d} - {motivo}\n")

            # Registra imagens que falharam sem interromper o restante do lote.
            if erros:
                arquivo.write("\nERROS\n")
                for erro in erros:
                    arquivo.write(f"- {erro}\n")

        return caminho

    # =========================================================================
    # EXECUCAO COMPLETA
    # =========================================================================

    def executar(self):
        """
        Executa o processamento completo da pasta configurada.

        Returns
        -------
        dict
            Resumo estruturado com contagens e caminhos dos resultados gerados.
        """

        # Valida toda a configuracao antes de criar, remover ou alterar arquivos.
        self._validar()

        # Procura imagens em toda a arvore da pasta de entrada.
        imagens = listar_imagens(
            self.pasta_entrada,
            self.extensoes_imagem,
        )

        # Uma pasta valida sem imagens compativeis nao possui trabalho a executar.
        if not imagens:
            raise FileNotFoundError(
                "Nenhuma imagem com extensao permitida foi encontrada em:\n"
                f"{self.pasta_entrada}"
            )

        # Mostra a configuracao efetiva antes de modificar a pasta de saida.
        print("\n" + "=" * 80)
        print("CROP AUTOMATICO COM SAM 2.1")
        print("=" * 80)
        print(f"\nEntrada                  : {self.pasta_entrada}")
        print(f"Saida                    : {self.pasta_saida}")
        print(f"Imagens encontradas      : {len(imagens)}")
        print(f"Margem                   : {self.parametros['margem_crop']:.0%}")
        print(
            "Area candidata           : "
            f"{self.parametros['area_minima_candidata']:.1%} a "
            f"{self.parametros['area_maxima_candidata']:.1%}"
        )
        print(
            "Fill minimo candidata    : "
            f"{self.parametros['preenchimento_bbox_minimo']:.0%}"
        )
        print(
            "BBox maxima candidata    : "
            f"{self.parametros['area_bbox_maxima_candidata']:.0%}"
        )
        print(
            "Crop maximo candidata    : "
            f"{self.parametros['crop_relativo_maximo_candidata']:.0%}"
        )
        print(f"Dispositivo              : {self.device}")

        # Exibe detalhes adicionais quando a inferencia sera executada em CUDA.
        if self.device.type == "cuda":
            print(f"GPU                      : {torch.cuda.get_device_name(0)}")
            print(f"CUDA                     : {torch.version.cuda}")
            print(f"BF16                     : {torch.cuda.is_bf16_supported()}")

        # Somente depois das validacoes e da listagem a saida e preparada.
        self._preparar_pastas()

        # O modelo e carregado uma unica vez para todo o lote.
        self._carregar_sam()

        # Resultados validos e erros sao mantidos separadamente.
        resultados = []
        erros = []

        print("\n" + "=" * 80)
        print("PROCESSANDO")
        print("=" * 80)

        # Processa todas as imagens encontradas em ordem alfabetica de caminho.
        for numero, caminho in enumerate(imagens, start=1):
            relativo = caminho.relative_to(self.pasta_entrada)

            try:
                # Executa segmentacao, crop, avaliacao e salvamento da imagem atual.
                resultado = self._processar_imagem(caminho)
                resultados.append(resultado)

                # Exibe as principais medidas da mascara escolhida.
                print(
                    f"[{numero:03d}/{len(imagens):03d}] "
                    f"{resultado['status']:<7} | "
                    f"{str(relativo):<48} | "
                    f"Area {resultado['area_mascara']:.1%} | "
                    f"Fill {resultado['fill_bbox']:.1%} | "
                    f"BBox {resultado['area_bbox']:.1%} | "
                    f"Crop {resultado['crop_relativo_final']:.1%}"
                )

            except Exception as erro:
                # Uma imagem com erro nao interrompe o processamento das demais.
                mensagem = f"{relativo}: {erro}"
                erros.append(mensagem)

                print(
                    f"[{numero:03d}/{len(imagens):03d}] "
                    f"ERRO    | {mensagem}"
                )

        # Salva os relatorios depois que todas as imagens forem tentadas.
        caminho_csv = self._salvar_csv(resultados)
        caminho_resumo = self._salvar_resumo(resultados, erros)

        # Calcula os totais finais exibidos e retornados.
        ok = sum(
            resultado["status"] == "OK"
            for resultado in resultados
        )
        revisar = sum(
            resultado["status"] == "REVISAR"
            for resultado in resultados
        )

        print("\n" + "=" * 80)
        print("RESULTADO FINAL")
        print("=" * 80)
        print(f"\nImagens encontradas : {len(imagens)}")
        print(f"Processadas         : {len(resultados)}")
        print(f"OK                  : {ok}")
        print(f"REVISAR             : {revisar}")
        print(f"ERROS               : {len(erros)}")
        print(f"\nCrops:\n{self.pasta_saida}")
        print(f"\nCSV:\n{caminho_csv}")
        print(f"\nResumo:\n{caminho_resumo}")

        if self.salvar_diagnosticos_revisar:
            print(f"\nDiagnosticos REVISAR:\n{self.pasta_diagnosticos}")

        print("\nA pasta de entrada nao foi modificada.")

        # Retorna dados estruturados para integracao com o futuro main.py.
        return {
            "entrada": self.pasta_entrada,
            "saida": self.pasta_saida,
            "resultados": self.pasta_resultados,
            "csv": caminho_csv,
            "resumo": caminho_resumo,
            "diagnosticos": (
                self.pasta_diagnosticos
                if self.salvar_diagnosticos_revisar
                else None
            ),
            "imagens_encontradas": len(imagens),
            "processadas": len(resultados),
            "ok": ok,
            "revisar": revisar,
            "erros": len(erros),
        }
