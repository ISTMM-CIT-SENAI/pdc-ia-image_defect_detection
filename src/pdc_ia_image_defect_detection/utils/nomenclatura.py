"""
nomenclatura.py

Objetivo:
- Centraliza as regras de nomenclatura e rotulagem do dataset principal.
- Relaciona as pastas dos materiais aos codigos usados nos nomes das imagens.
- Relaciona cada condicao ao rotulo da rede e ao codigo RAC esperado.
- Interpreta e valida nomes no formato MATERIAL_CP_RAC.

Usado por:
- pre_processing_data.py: valida material, condicao, nome e RAC.
- split.py: valida os nomes e identifica os rotulos.
- augmentation_train.py e images_dataset.py: utilizam ROTULOS.
- predict_imagens.py: interpreta imagens que seguem o padrao principal.

Parametros ajustaveis:
- MATERIAIS: pastas dos materiais e seus codigos.
- CONDICOES: pastas das condicoes, rotulos e RAC esperado.
- ROTULOS: classes utilizadas pelo modelo.
- PADRAO_NOME: expressao regular da nomenclatura principal.

Padrao principal:
- MATERIAL_CP_RAC
- MATERIAL: AL, A36 ou FoFo.
- CP: numero do corpo de prova ou XNN para CP extra.
- RAC: Risco-Amassamento-Corrosao, com valores de 0 a 3.
- Ex.: AL_10_001, A36_42_020, FoFo_X01_000.

Observacoes:
- Alteracoes neste arquivo podem afetar todo o pipeline.
- O codigo X identifica um CP extra; nao representa uma classe ou defeito.
- Os aliases "corrosa_*" existem por causa da nomenclatura atual de algumas pastas.
- A nomenclatura externa MATERIAL_CP_CLASSE e tratada separadamente em predict_imagens.py.
"""

from pathlib import Path
import re


# =============================================================================
# PARAMETROS AJUSTAVEIS - REGRAS CENTRAIS DE NOMENCLATURA
# =============================================================================

# Pasta do material -> codigo usado no nome da imagem
MATERIAIS = {
    "aco_2500": "A36",
    "Aluminio_2500": "AL",
    "ferro_fundido_4500": "FoFo",
}

# Pasta da condicao -> (rotulo da rede, RAC esperado)
CONDICOES = {
    "sem_defeito": ("sem_falha", "000"),

    "risco_leve": ("risco", "100"),
    "risco_moderado": ("risco", "200"),
    "risco_severo": ("risco", "300"),

    "amassamento_leve": ("amassamento", "010"),
    "amassamento_moderado": ("amassamento", "020"),
    "amassamento_severo": ("amassamento", "030"),

    "corrosao_leve": ("corrosao", "001"),
    "corrosao_moderado": ("corrosao", "002"),
    "corrosao_severo": ("corrosao", "003"),

    # Aliases existentes no dataset atual
    "corrosa_leve": ("corrosao", "001"),
    "corrosa_moderado": ("corrosao", "002"),
    "corrosa_severo": ("corrosao", "003"),
}

# Classes finais utilizadas pela rede
ROTULOS = ["amassamento", "corrosao", "risco", "sem_falha"]

# MATERIAL_CP_RAC | CP numerico ou XNN
PADRAO_NOME = re.compile(r"^(AL|A36|FoFo)_((?:X\d{2})|(?:\d+))_([0-3]{3})$")


def interpretar_nome(caminho):
    # Extrai material, CP e RAC do nome
    caminho = Path(caminho)
    correspondencia = PADRAO_NOME.fullmatch(caminho.stem)

    if correspondencia is None: raise ValueError(f"Nome fora do padrao MATERIAL_CP_RAC: {caminho.name}")

    material, cp, rac = correspondencia.groups()
    return {"material": material, "cp": cp, "rac": rac}


def validar_imagem(caminho, pasta_material, pasta_condicao):
    # Confere pasta, material e RAC antes do pre-processamento
    if pasta_material not in MATERIAIS: raise ValueError(f"Pasta de material desconhecida: {pasta_material}")
    if pasta_condicao not in CONDICOES: raise ValueError(f"Pasta de condicao desconhecida: {pasta_condicao}")

    dados_nome = interpretar_nome(caminho)
    material_esperado = MATERIAIS[pasta_material]
    rotulo, rac_esperado = CONDICOES[pasta_condicao]

    if dados_nome["material"] != material_esperado:
        raise ValueError(
            f"Material inconsistente em {Path(caminho).name}: "
            f"pasta={material_esperado}, nome={dados_nome['material']}"
        )

    if dados_nome["rac"] != rac_esperado:
        raise ValueError(
            f"RAC inconsistente em {Path(caminho).name}: "
            f"pasta={rac_esperado}, nome={dados_nome['rac']}"
        )

    return {
        "rotulo": rotulo, "material": dados_nome["material"], "cp": dados_nome["cp"], "rac": dados_nome["rac"],
        "pasta_material": pasta_material, "pasta_condicao": pasta_condicao,
    }