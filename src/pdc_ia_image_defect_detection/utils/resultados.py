"""
resultados.py

Objetivo:
- Padronizar nomes de pastas de resultados em todo o projeto.
- Registrar tipo da execucao, detalhes permanentes, indicadores booleanos ativos,
  horario de inicio e horario de fim.
- Incluir no nome somente indicadores cuja flag esteja True.
- Manter uma pasta com sufixo _em_andamento enquanto a execucao nao termina.

Exemplo:
    treino_lenet5_r256_int_aug_inicio_2026-10-23_12-47-03_em_andamento
    treino_lenet5_r256_int_aug_inicio_2026-10-23_12-47-03_fim_2026-10-23_13-05-11

Este modulo e generico e nao conhece treino, teste, datasets ou modelos especificos.
"""

from pathlib import Path
import re


_PADRAO_INVALIDO = re.compile(r"[^a-zA-Z0-9.-]+")


def sanitizar_token_resultado(valor):
    """Converte um valor em token curto e seguro para nome de pasta."""

    token = str(valor).strip().lower()
    token = _PADRAO_INVALIDO.sub("-", token)
    token = re.sub(r"-+", "-", token).strip("-.")

    if not token:
        raise ValueError(f"Token de resultado invalido: {valor!r}")

    return token


def montar_nome_pasta_resultados(tipo, inicio, fim=None, detalhes=(), indicadores=()):
    """
    Monta o nome padronizado de uma pasta de resultados.

    Parameters
    ----------
    tipo : str
        Tipo principal da execucao, por exemplo "treino" ou "test".

    inicio, fim : datetime
        Horarios da execucao. Quando fim=None, o nome termina em
        ``_em_andamento``.

    detalhes : iterable
        Informacoes nao booleanas que devem aparecer sempre, por exemplo nome
        do modelo e resolucao.

    indicadores : iterable[tuple[str, bool]]
        Pares ``(nome, ativo)``. O nome so entra na pasta quando ativo=True.

    Returns
    -------
    str
        Nome final da pasta.
    """

    partes = [sanitizar_token_resultado(tipo)]

    for detalhe in detalhes:
        if detalhe is None:
            continue
        partes.append(sanitizar_token_resultado(detalhe))

    for nome, ativo in indicadores:
        if bool(ativo):
            partes.append(sanitizar_token_resultado(nome))

    partes.extend(
        [
            "inicio",
            inicio.strftime("%Y-%m-%d_%H-%M-%S"),
        ]
    )

    if fim is None:
        partes.append("em_andamento")
    else:
        partes.extend(
            [
                "fim",
                fim.strftime("%Y-%m-%d_%H-%M-%S"),
            ]
        )

    return "_".join(partes)


def criar_pasta_resultados(raiz, tipo, inicio, detalhes=(), indicadores=()):
    """Cria uma pasta de execucao usando o padrao central do projeto."""

    raiz = Path(raiz)
    raiz.mkdir(parents=True, exist_ok=True)

    nome = montar_nome_pasta_resultados(
        tipo=tipo,
        inicio=inicio,
        fim=None,
        detalhes=detalhes,
        indicadores=indicadores,
    )

    pasta = raiz / nome
    pasta.mkdir(parents=False, exist_ok=False)
    return pasta


def finalizar_pasta_resultados(pasta_atual, tipo, inicio, fim, detalhes=(), indicadores=()):
    """Renomeia a pasta em andamento incluindo o horario final da execucao."""

    pasta_atual = Path(pasta_atual)

    nome_final = montar_nome_pasta_resultados(
        tipo=tipo,
        inicio=inicio,
        fim=fim,
        detalhes=detalhes,
        indicadores=indicadores,
    )

    destino = pasta_atual.parent / nome_final

    if destino.exists():
        raise FileExistsError(f"Pasta final de resultados ja existe: {destino}")

    pasta_atual.rename(destino)
    return destino
