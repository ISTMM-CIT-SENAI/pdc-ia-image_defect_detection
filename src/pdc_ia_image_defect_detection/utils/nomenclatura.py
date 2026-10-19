"""
nomenclatura.py

Objetivo:
- Reunir apenas regras genericas de nomenclatura dos arquivos internos.
- Interpretar nomes no formato MATERIAL_CP_RAC.
- Validar os codigos de material, corpo de prova e RAC.
- Interpretar separadamente os tres componentes do codigo RAC.
- Permitir reutilizacao por qualquer etapa do projeto sem conhecer datasets,
  pastas de classes ou caminhos de entrada e saida.

Formato oficial:
- <MATERIAL>_<CP>_<RAC>.<extensao>

Materiais:
- AL   -> aluminio
- A36  -> aco A36
- FoFo -> ferro fundido

Corpo de prova:
- CP numerico original, por exemplo: 10, 42, 55.
- CP adicional no formato XNN, por exemplo: X01, X03, X15.

Codigo RAC:
- Possui exatamente tres digitos.
- Ordem: Risco, Amassamento, Corrosao.
- Cada digito aceita valores de 0 a 3:
    0 -> ausente
    1 -> leve
    2 -> moderado
    3 -> severo
- 000 representa ausencia de risco, amassamento e corrosao.
- Combinacoes entre defeitos sao permitidas pelo padrao de nomenclatura.

Exemplos validos:
- AL_10_001.jpg
- A36_42_020.png
- FoFo_55_300.jpg
- AL_X01_000.png
- AL_X01_102.png

Observacoes:
- Este modulo nao conhece a estrutura material/condicao/imagem do dataset.
- Este modulo nao decide em qual pasta ou classe uma imagem deve ficar.
- A coerencia entre nome do arquivo e pasta de origem pertence a classe
  responsavel pela preparacao do dataset.
- A extensao do arquivo nao participa da interpretacao do nome.
"""

from pathlib import Path
import re


# =============================================================================
# CONSTANTES DE NOMENCLATURA
# =============================================================================

# Codigos de material aceitos no primeiro campo do nome.
MATERIAIS = {
    "AL": "aluminio",
    "A36": "aco_A36",
    "FoFo": "ferro_fundido",
}

# Classes finais atualmente utilizadas pelo classificador.
ROTULOS = (
    "amassamento",
    "corrosao",
    "risco",
    "sem_falha",
)

# Ordem fixa dos tres digitos do codigo RAC.
ORDEM_RAC = (
    "risco",
    "amassamento",
    "corrosao",
)

# Significado das intensidades permitidas em cada posicao do RAC.
INTENSIDADES = {
    0: "ausente",
    1: "leve",
    2: "moderado",
    3: "severo",
}

# Formato oficial MATERIAL_CP_RAC.
# CP aceita numero original ou identificador adicional XNN.
PADRAO_NOME = re.compile(
    r"^(?P<material>AL|A36|FoFo)_"
    r"(?P<cp>(?:\d+|X\d{2}))_"
    r"(?P<rac>\d{3})$"
)


# =============================================================================
# INTERPRETACAO DO CODIGO RAC
# =============================================================================

def interpretar_rac(rac):
    """
    Interpreta e valida um codigo RAC.

    Parameters
    ----------
    rac : str
        Codigo com exatamente tres digitos na ordem
        Risco-Amassamento-Corrosao.

    Returns
    -------
    dict
        Valores individuais, niveis e defeitos presentes.

        rotulo_unico recebe:
        - "sem_falha" quando RAC == "000";
        - o nome do defeito quando somente um defeito esta presente;
        - None quando existe combinacao de dois ou tres defeitos.

    Raises
    ------
    TypeError
        Quando rac nao e uma string.

    ValueError
        Quando o RAC nao possui exatamente tres digitos ou algum digito
        esta fora do intervalo permitido de 0 a 3.
    """

    # Mantem o codigo textual exatamente como aparece no nome do arquivo.
    if not isinstance(rac, str):
        raise TypeError(
            f"RAC deve ser string, recebido {type(rac).__name__}."
        )

    # O campo RAC precisa possuir exatamente tres caracteres numericos.
    if len(rac) != 3 or not rac.isdigit():
        raise ValueError(
            f"RAC invalido: '{rac}'. Esperado exatamente tres digitos."
        )

    # Converte as tres posicoes para inteiros mantendo a ordem oficial.
    valores = tuple(int(digito) for digito in rac)

    # Cada posicao aceita somente intensidades entre zero e tres.
    if any(valor not in INTENSIDADES for valor in valores):
        raise ValueError(
            f"RAC invalido: '{rac}'. Cada digito deve estar entre 0 e 3."
        )

    # Associa explicitamente cada posicao ao respectivo defeito.
    risco, amassamento, corrosao = valores

    # Identifica todos os defeitos presentes. Combinacoes sao permitidas.
    defeitos_presentes = tuple(
        nome
        for nome, valor in zip(ORDEM_RAC, valores)
        if valor > 0
    )

    # O rotulo unico so existe quando a interpretacao e inequivoca.
    if not defeitos_presentes:
        rotulo_unico = "sem_falha"
    elif len(defeitos_presentes) == 1:
        rotulo_unico = defeitos_presentes[0]
    else:
        rotulo_unico = None

    return {
        "rac": rac,
        "risco": risco,
        "amassamento": amassamento,
        "corrosao": corrosao,
        "nivel_risco": INTENSIDADES[risco],
        "nivel_amassamento": INTENSIDADES[amassamento],
        "nivel_corrosao": INTENSIDADES[corrosao],
        "defeitos_presentes": defeitos_presentes,
        "rotulo_unico": rotulo_unico,
    }


# =============================================================================
# INTERPRETACAO DO NOME
# =============================================================================

def interpretar_nome(caminho_ou_nome):
    """
    Interpreta um nome no formato MATERIAL_CP_RAC.

    Parameters
    ----------
    caminho_ou_nome : str ou Path
        Nome simples ou caminho completo do arquivo.
        A extensao e ignorada durante a interpretacao.

    Returns
    -------
    dict
        Campos extraidos e validados.

    Raises
    ------
    ValueError
        Quando o nome nao segue o padrao oficial ou algum campo e invalido.
    """

    # Path permite receber tanto um nome simples quanto um caminho completo.
    caminho = Path(caminho_ou_nome)

    # Somente o stem participa da nomenclatura.
    stem = caminho.stem

    # Separa MATERIAL, CP e RAC conforme o padrao oficial.
    correspondencia = PADRAO_NOME.fullmatch(stem)
    if correspondencia is None:
        raise ValueError(
            f"Nome invalido: '{caminho.name}'. "
            "Esperado MATERIAL_CP_RAC, com CP numerico ou XNN."
        )

    material = correspondencia.group("material")
    cp = correspondencia.group("cp")
    rac = correspondencia.group("rac")

    # Mantem a constante MATERIAIS como fonte central do significado do codigo.
    if material not in MATERIAIS:
        raise ValueError(
            f"Material desconhecido no arquivo '{caminho.name}': {material}."
        )

    # Interpreta e valida separadamente o codigo RAC.
    dados_rac = interpretar_rac(rac)

    return {
        "material": material,
        "material_nome": MATERIAIS[material],
        "cp": cp,
        **dados_rac,
    }


# =============================================================================
# VALIDACAO SIMPLES
# =============================================================================

def validar_nome(caminho_ou_nome):
    """
    Valida um nome interno e retorna sua interpretacao completa.

    Parameters
    ----------
    caminho_ou_nome : str ou Path
        Nome simples ou caminho completo do arquivo.

    Returns
    -------
    dict
        Mesmo resultado produzido por interpretar_nome().
    """

    # interpretar_nome concentra todas as validacoes genericas do nome.
    return interpretar_nome(caminho_ou_nome)
