"""
prepara_img.py

Objetivo:
- Configurar e executar a preparacao das imagens das bases interna e externa.
- Permitir ativar independentemente o processamento de cada base.
- Aplicar a mesma resolucao final e o mesmo formato numerico nas duas bases.
- Manter neste arquivo apenas parametros, caminhos e chamada do processamento.

Base interna:
- Entrada: dataset_atual_crop
- Saida: dataset_processado
- As imagens ja foram recortadas anteriormente pelo SAM.
- A estrutura material/condicao/imagem e validada pela nomenclatura interna.
- Nenhum crop adicional e aplicado nesta etapa.

Base externa:
- Entrada: dataset_externo_test/imagens_selecionadas
- Saida: dataset_externo_processado
- O rotulo e obtido pela pasta da classe.
- A orientacao EXIF e corrigida.
- Imagens retangulares recebem crop quadrado central.
- Imagens quadradas nao recebem crop.

Processamento comum:
- conversao para grayscale;
- resize LANCZOS;
- conversao para float32;
- normalizacao de 0-255 para 0-1;
- adicao do canal, resultando em (1,H,W);
- salvamento em NPY;
- verificacao final dos arquivos gerados.

Funcionamento:
- PREPARAR_DATASET_INTERNO controla a base interna.
- PREPARAR_DATASET_EXTERNO controla a base externa.
- As duas bases podem ser preparadas na mesma execucao.
- EXECUTAR_PREPARA_IMG=False encerra a etapa antes dos imports internos.
- Se ambas as bases estiverem desativadas, nenhuma pasta e modificada.
- Todas as bases selecionadas sao validadas antes da limpeza das saidas.
- A pasta de saida de cada base selecionada e recriada para evitar arquivos antigos.
- O split nao e realizado neste processo; sera uma etapa propria do pipeline.
- Augmentation nao e aplicado neste processo.

Configuracao:
- Todos os parametros sao definidos manualmente neste arquivo.
- O terminal e utilizado somente para exibir andamento e resultados.
- A logica especifica fica em pre_processing/prepara_img_class.py.
- Funcoes genericas de imagem, NPY e caminhos ficam em utils.

Observacao:
- A escrita de bytecode e desativada para evitar arquivos .pyc e pastas __pycache__.
"""

import sys

# Desativa a escrita de bytecode Python durante a execucao.
sys.dont_write_bytecode = True

# =============================================================================
# PARAMETROS AJUSTAVEIS
# =============================================================================

# Ativa ou desativa completamente esta etapa do pipeline.
EXECUTAR_PREPARA_IMG = False

# Ativa a preparacao da base interna.
PREPARAR_DATASET_INTERNO = True

# Ativa a preparacao da base externa.
PREPARAR_DATASET_EXTERNO = True

# Define a resolucao final comum as duas bases no formato (largura, altura).
TAMANHO_FINAL = (224, 224)

# Define os formatos de imagem aceitos nas bases de entrada.
EXTENSOES_VALIDAS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
)

# Define pastas da base interna que nao participam do processamento.
PASTAS_IGNORADAS_INTERNO = {
    "fotos_antigas",
}

# Prefixo que identifica os NPY provenientes da base externa.
PREFIXO_EXTERNO = "ext"


# =============================================================================
# EXECUCAO
# =============================================================================

def executar():
    """Executa a preparacao somente quando a etapa estiver ativada."""

    # Encerra antes de importar a classe especifica ou analisar qualquer base.
    if not EXECUTAR_PREPARA_IMG:
        print("\nPREPARA_IMG desativado: EXECUTAR_PREPARA_IMG=False.")
        return {}

    # Imports internos ocorrem somente quando esta etapa realmente sera executada.
    from pdc_ia_image_defect_detection.pre_processing.prepara_img_class import (
        PreparaImagensProcessor,
    )
    from pdc_ia_image_defect_detection.utils.paths import encontrar_raiz_projeto

    # Localiza a raiz do projeto somente apos a verificacao da flag geral.
    base_dir = encontrar_raiz_projeto(__file__)

    # Define as entradas e saidas das duas bases.
    dataset_interno_original = base_dir / "dataset_atual_crop"
    dataset_interno_processado = base_dir / "dataset_processado"
    dataset_externo_original = base_dir / "dataset_externo_test" / "imagens_selecionadas"
    dataset_externo_processado = base_dir / "dataset_externo_processado"

    # Reune a configuracao de cada base para o processador.
    bases_preparacao = {
        "INTERNA": {
            "ativa": PREPARAR_DATASET_INTERNO,
            "tipo": "interna",
            "origem": dataset_interno_original,
            "destino": dataset_interno_processado,
        },
        "EXTERNA": {
            "ativa": PREPARAR_DATASET_EXTERNO,
            "tipo": "externa",
            "origem": dataset_externo_original,
            "destino": dataset_externo_processado,
        },
    }

    # Executa a logica especifica mantendo todos os parametros neste processing.
    return PreparaImagensProcessor.executar_bases(
        bases=bases_preparacao,
        tamanho_final=TAMANHO_FINAL,
        extensoes_validas=EXTENSOES_VALIDAS,
        pastas_ignoradas_interno=PASTAS_IGNORADAS_INTERNO,
        prefixo_externo=PREFIXO_EXTERNO,
    )


# Executa o processo quando este arquivo for iniciado diretamente.
if __name__ == "__main__":
    executar()
