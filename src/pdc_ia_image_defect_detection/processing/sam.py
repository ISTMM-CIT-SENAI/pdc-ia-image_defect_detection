"""
sam.py

Objetivo:
- Configurar e executar o recorte automatico de imagens utilizando SAM 2.1.
- Permitir ativar ou desativar completamente a etapa por uma unica flag.
- Permitir escolher manualmente qualquer pasta de entrada e pasta de saida.
- Procurar imagens recursivamente na pasta de entrada e em todas as subpastas.
- Manter na saida a mesma estrutura relativa de pastas e nomes dos arquivos.
- Concentrar neste arquivo somente parametros, caminhos e chamada do processamento.

Funcionamento:
- EXECUTAR_SAM controla toda a etapa.
- Quando EXECUTAR_SAM=False, a execucao termina antes de importar a classe do SAM,
  carregar PyTorch/SAM2, consultar CUDA, carregar o modelo ou modificar arquivos.
- Quando EXECUTAR_SAM=True, todas as imagens encontradas nas extensoes configuradas
  sao processadas.
- A imagem e corrigida pela orientacao EXIF antes da segmentacao.
- O SAM recebe a imagem em RGB e um ponto positivo localizado no centro da imagem.
- O modelo retorna multiplas mascaras candidatas.
- As candidatas sao analisadas por area, centralidade, preenchimento da bounding box,
  tamanho da bounding box, tamanho estimado do crop e score do SAM.
- A melhor candidata valida e selecionada pelo criterio combinado configurado.
- Quando nenhuma candidata atende aos limites, a melhor candidata disponivel e usada
  como fallback e o resultado e marcado para revisao.
- O crop final e quadrado e recebe margem ao redor da bounding box selecionada.
- Casos suspeitos sao marcados como REVISAR e podem gerar diagnosticos visuais.
- Um CSV e um resumo textual sao salvos para cada execucao.

Pastas:
- PASTA_ENTRADA pode apontar para qualquer diretorio de imagens.
- PASTA_SAIDA recebe os crops mantendo a estrutura relativa da entrada.
- PASTA_RESULTADOS_RAIZ recebe CSV, resumo e diagnosticos da execucao.

Configuracao:
- Todos os parametros sao definidos manualmente neste arquivo.
- O terminal e utilizado somente para exibir andamento e resultados.
- Nao ha selecao de parametros pelo terminal.
- A logica especifica do SAM fica em pre_processing/sam_class.py.
- Funcoes genericas de caminhos e imagens permanecem em utils.

Observacoes:
- A pasta de entrada nunca e modificada.
- Se LIMPAR_PASTA_SAIDA=True, a pasta de saida existente e removida antes do processo.
- Se LIMPAR_PASTA_SAIDA=False, arquivos com o mesmo caminho podem ser sobrescritos e
  arquivos antigos sem correspondencia podem permanecer na saida.
- A pasta de saida nao pode ser igual nem estar dentro da pasta de entrada.
- A escrita de bytecode e desativada para evitar arquivos .pyc e pastas __pycache__.
"""

import sys

# Desativa a escrita de bytecode Python durante a execucao.
sys.dont_write_bytecode = True

# =============================================================================
# PARAMETROS AJUSTAVEIS
# =============================================================================

# Ativa ou desativa completamente a etapa de recorte com SAM.
EXECUTAR_SAM = False

# Define a pasta de entrada em relacao a raiz do projeto.
PASTA_ENTRADA = "dataset_rgb"

# Define a pasta que recebera os crops em relacao a raiz do projeto.
PASTA_SAIDA = "dataset_rgb_crop"

# Define a pasta principal dos relatorios em relacao a raiz do projeto.
PASTA_RESULTADOS_RAIZ = "resultados/sam"

# Define o checkpoint SAM 2.1 carregado pelo predictor.
MODELO_SAM = "facebook/sam2.1-hiera-small"

# Remove a pasta de saida existente antes de salvar os novos crops.
LIMPAR_PASTA_SAIDA = True

# Salva diagnosticos graficos somente para imagens classificadas como REVISAR.
SALVAR_DIAGNOSTICOS_REVISAR = True

# Define a resolucao dos diagnosticos graficos salvos.
DPI_GRAFICOS = 160

# Define as extensoes de imagem aceitas durante a busca recursiva.
EXTENSOES_IMAGEM = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)

# Define os limites geometricos, criterios de revisao e pesos de selecao.
PARAMETROS_SAM = {
    # Margem adicionada ao redor da bounding box selecionada.
    "margem_crop": 0.10,

    # Limites de area relativa para uma mascara ser candidata valida.
    "area_minima_candidata": 0.015,
    "area_maxima_candidata": 0.12,

    # Preenchimento minimo da bounding box pela mascara.
    "preenchimento_bbox_minimo": 0.50,

    # Area relativa maxima permitida para a bounding box.
    "area_bbox_maxima_candidata": 0.16,

    # Tamanho maximo estimado do crop em relacao ao menor lado da imagem.
    "crop_relativo_maximo_candidata": 0.65,

    # Faixas conservadoras usadas para marcar resultados como REVISAR.
    "area_revisar_minima": 0.020,
    "area_revisar_maxima": 0.11,
    "preenchimento_revisar_minimo": 0.60,
    "area_bbox_revisar_maxima": 0.14,
    "crop_relativo_revisar_maximo": 0.60,
    "centralidade_revisar_minima": 0.80,

    # Score abaixo deste valor e registrado como baixo no relatorio.
    "score_sam_baixo": 0.70,

    # Pesos do criterio usado para comparar candidatas validas.
    "peso_preenchimento": 0.55,
    "peso_centralidade": 0.35,
    "peso_score": 0.10,

    # Diferenca maxima de criterio para considerar duas candidatas empatadas.
    "tolerancia_empate": 0.015,
}


# =============================================================================
# EXECUCAO
# =============================================================================

def executar():
    """Executa o SAM quando a etapa estiver ativada."""

    # Encerra antes de importar PyTorch, SAM2 ou consultar CUDA.
    if not EXECUTAR_SAM:
        print("\nSAM desativado: EXECUTAR_SAM=False.")
        return {}

    # Os imports internos ocorrem somente quando o SAM realmente sera executado.
    from pathlib import Path

    from pdc_ia_image_defect_detection.pre_processing.sam_class import SamProcessor
    from pdc_ia_image_defect_detection.utils.paths import encontrar_raiz_projeto

    # Localiza a raiz do projeto somente depois da verificacao da flag geral.
    base_dir = encontrar_raiz_projeto(__file__)

    # Permite configurar caminhos relativos a raiz ou caminhos absolutos.
    pasta_entrada = Path(PASTA_ENTRADA)
    pasta_saida = Path(PASTA_SAIDA)
    pasta_resultados_raiz = Path(PASTA_RESULTADOS_RAIZ)

    if not pasta_entrada.is_absolute():
        pasta_entrada = base_dir / pasta_entrada
    if not pasta_saida.is_absolute():
        pasta_saida = base_dir / pasta_saida
    if not pasta_resultados_raiz.is_absolute():
        pasta_resultados_raiz = base_dir / pasta_resultados_raiz

    # Cria o processador com os caminhos e parametros configurados.
    processador = SamProcessor(
        pasta_entrada=pasta_entrada,
        pasta_saida=pasta_saida,
        pasta_resultados_raiz=pasta_resultados_raiz,
        modelo_sam=MODELO_SAM,
        extensoes_imagem=EXTENSOES_IMAGEM,
        parametros=PARAMETROS_SAM,
        limpar_pasta_saida=LIMPAR_PASTA_SAIDA,
        salvar_diagnosticos_revisar=SALVAR_DIAGNOSTICOS_REVISAR,
        dpi_graficos=DPI_GRAFICOS,
    )

    # Retorna o resumo estruturado da execucao para uso pelo futuro main.py.
    return processador.executar()


# Executa o processo quando este arquivo for iniciado diretamente.
if __name__ == "__main__":
    executar()
