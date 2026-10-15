"""
pre_processing_data.py

Objetivo:
- Valida o dataset recortado pelo SAM e gera o dataset processado usado
  pelo restante do pipeline.
- Confere estrutura de pastas, nomenclatura, material, condicao e codigo RAC.
- Converte as imagens para grayscale.
- Redimensiona para a resolucao final escolhida.
- Converte para float32.
- Normaliza os pixels de 0-255 para 0-1.
- Adiciona a dimensao de canal, resultando em (1,H,W).
- Salva os resultados em arquivos NPY.
- Organiza o resultado nas quatro classes finais:
  amassamento, corrosao, risco e sem_falha.

Usa:
- utils/nomenclatura.py:
  materiais, condicoes, rotulos e validacao dos nomes.

Entrada:
- dataset_atual_crop

Saida:
- dataset_processado

Usado antes de:
- split.py: separa dataset_processado em train, val e test.

Parametros ajustaveis:
- TAMANHO_FINAL: resolucao final das imagens.
- EXTENSOES_VALIDAS: formatos de imagem aceitos.
- PASTAS_IGNORADAS: pastas que nao participam do processamento.
- DATASET_ORIGINAL e DATASET_PROCESSADO: caminhos de entrada e saida.

Fluxo:
dataset_atual_crop
    -> validacao
    -> grayscale
    -> resize
    -> float32
    -> /255
    -> (1,H,W)
    -> NPY

Observacoes:
- O crop NAO e realizado neste arquivo.
- As imagens de entrada ja devem estar recortadas pelo SAM.
- Nao aplica augmentation.
- Augmentation e realizado posteriormente somente no conjunto train.
- dataset_processado e recriado a cada execucao para evitar mistura
  com arquivos antigos.
"""

from pathlib import Path
from collections import Counter
import shutil

import numpy as np
from PIL import Image

from pdc_ia_image_defect_detection.utils.nomenclatura import (
    MATERIAIS,
    CONDICOES,
    ROTULOS,
    validar_imagem,
)


# =============================================================================
# PARAMETROS AJUSTAVEIS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[3]

# Dataset ja recortado pelo SAM
DATASET_ORIGINAL = BASE_DIR / "dataset_atual_crop"

# Dataset NPY de saida
DATASET_PROCESSADO = BASE_DIR / "dataset_processado"

# Resolucao final
TAMANHO_FINAL = (256, 256)

# Formatos aceitos
EXTENSOES_VALIDAS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
)

# Pastas ignoradas
PASTAS_IGNORADAS = {
    "fotos_antigas"
}


# =============================================================================
# LISTAGEM DE ARQUIVOS
# =============================================================================

def obter_imagens(root):
    return sorted([
        arquivo
        for arquivo in root.rglob("*")
        if arquivo.is_file()
        and arquivo.suffix.lower() in EXTENSOES_VALIDAS
    ])


def obter_arrays(pasta):
    return sorted([
        arquivo
        for arquivo in pasta.rglob("*.npy")
        if arquivo.is_file()
    ])


# =============================================================================
# ANALISE DO DATASET DE ENTRADA
# =============================================================================

def analisar_dataset_original(root):
    registros = []
    erros = []
    ignorados = []

    formatos = Counter()
    modos = Counter()
    canais = Counter()
    resolucoes = Counter()

    materiais = Counter()
    condicoes = Counter()
    rotulos = Counter()

    for caminho in obter_imagens(root):
        relativo = caminho.relative_to(root)

        # Ignora pastas que nao fazem parte do dataset atual
        if any(
            parte in PASTAS_IGNORADAS
            for parte in relativo.parts
        ):
            ignorados.append(
                caminho
            )
            continue

        # Estrutura esperada:
        # material/condicao/imagem
        if len(relativo.parts) != 3:
            erros.append((
                caminho,
                "Estrutura invalida. "
                "Esperado: material/condicao/imagem."
            ))
            continue

        pasta_material, pasta_condicao, _ = relativo.parts

        if pasta_material not in MATERIAIS:
            erros.append((
                caminho,
                f"Material desconhecido: {pasta_material}"
            ))
            continue

        if pasta_condicao not in CONDICOES:
            erros.append((
                caminho,
                f"Condicao desconhecida: {pasta_condicao}"
            ))
            continue

        try:
            dados = validar_imagem(
                caminho,
                pasta_material,
                pasta_condicao
            )

            with Image.open(caminho) as imagem:
                formatos[
                    caminho.suffix.lower()
                ] += 1

                modos[
                    imagem.mode
                ] += 1

                canais[
                    len(imagem.getbands())
                ] += 1

                resolucoes[
                    imagem.size
                ] += 1

            materiais[
                dados["material"]
            ] += 1

            condicoes[
                pasta_condicao
            ] += 1

            rotulos[
                dados["rotulo"]
            ] += 1

            registros.append({
                "caminho": caminho,
                "rotulo": dados["rotulo"],
                "material": dados["material"],
                "cp": dados["cp"],
                "rac": dados["rac"],
                "pasta_material": pasta_material,
                "pasta_condicao": pasta_condicao,
            })

        except Exception as erro:
            erros.append((
                caminho,
                str(erro)
            ))

    return {
        "registros": registros,
        "erros": erros,
        "ignorados": ignorados,
        "formatos": formatos,
        "modos": modos,
        "canais": canais,
        "resolucoes": resolucoes,
        "materiais": materiais,
        "condicoes": condicoes,
        "rotulos": rotulos,
    }


# =============================================================================
# ANALISE DO DATASET PROCESSADO
# =============================================================================

def analisar_dataset_processado(root):
    resultados = {}

    for rotulo in ROTULOS:
        pasta = root / rotulo

        arquivos = (
            obter_arrays(pasta)
            if pasta.exists()
            else []
        )

        shapes = Counter()
        dtypes = Counter()

        minimo = float("inf")
        maximo = float("-inf")

        erros = []

        for caminho in arquivos:
            try:
                array = np.load(
                    caminho,
                    mmap_mode="r",
                    allow_pickle=False
                )

                shapes[
                    array.shape
                ] += 1

                dtypes[
                    str(array.dtype)
                ] += 1

                minimo = min(
                    minimo,
                    float(array.min())
                )

                maximo = max(
                    maximo,
                    float(array.max())
                )

                if not np.isfinite(array).all():
                    erros.append((
                        caminho,
                        "Valores NaN ou infinitos encontrados."
                    ))

            except Exception as erro:
                erros.append((
                    caminho,
                    str(erro)
                ))

        resultados[
            rotulo
        ] = {
            "quantidade": len(arquivos),
            "shapes": shapes,
            "dtypes": dtypes,
            "minimo": (
                minimo
                if arquivos
                else None
            ),
            "maximo": (
                maximo
                if arquivos
                else None
            ),
            "erros": erros,
        }

    return resultados


# =============================================================================
# FORMATACAO
# =============================================================================

def formatar_counter(counter):
    return ", ".join(
        f"{valor}={quantidade}"
        for valor, quantidade
        in counter.items()
    )


def formatar_resolucoes(counter):
    return ", ".join(
        f"{largura}x{altura}={quantidade}"
        for (largura, altura), quantidade
        in counter.items()
    )


# =============================================================================
# RESUMO DO DATASET DE ENTRADA
# =============================================================================

def mostrar_resumo_original(resultado):
    corretas = len(
        resultado["registros"]
    )

    erradas = len(
        resultado["erros"]
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "ANALISE DO DATASET RECORTADO"
    )

    print(
        "=" * 90
    )

    print(
        f"\nImagens analisadas : "
        f"{corretas + erradas}"
    )

    print(
        f"Imagens corretas   : "
        f"{corretas}"
    )

    print(
        f"Imagens com erro   : "
        f"{erradas}"
    )

    print(
        f"Imagens ignoradas  : "
        f"{len(resultado['ignorados'])}"
    )

    print(
        "\nROTULOS FINAIS"
    )

    for rotulo in ROTULOS:
        print(
            f"  {rotulo}: "
            f"{resultado['rotulos'][rotulo]}"
        )

    print(
        "\nMATERIAIS"
    )

    for material, quantidade in resultado[
        "materiais"
    ].items():
        print(
            f"  {material}: "
            f"{quantidade}"
        )

    print(
        "\nCONDICOES"
    )

    for condicao, quantidade in resultado[
        "condicoes"
    ].items():
        print(
            f"  {condicao}: "
            f"{quantidade}"
        )

    print(
        "\nPROPRIEDADES DAS IMAGENS"
    )

    print(
        f"  Formatos   : "
        f"{formatar_counter(resultado['formatos'])}"
    )

    print(
        f"  Modos      : "
        f"{formatar_counter(resultado['modos'])}"
    )

    print(
        f"  Canais     : "
        f"{formatar_counter(resultado['canais'])}"
    )

    print(
        f"  Resolucoes : "
        f"{formatar_resolucoes(resultado['resolucoes'])}"
    )

    if resultado["erros"]:
        print(
            "\nERROS ENCONTRADOS"
        )

        for caminho, erro in resultado[
            "erros"
        ]:
            print(
                f"\n  Arquivo: {caminho}"
            )

            print(
                f"  Erro   : {erro}"
            )


# =============================================================================
# RESUMO DO DATASET PROCESSADO
# =============================================================================

def mostrar_resumo_processado(resultados):
    total = sum(
        resultado["quantidade"]
        for resultado
        in resultados.values()
    )

    total_erros = sum(
        len(resultado["erros"])
        for resultado
        in resultados.values()
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "VERIFICACAO DO DATASET PROCESSADO"
    )

    print(
        "=" * 90
    )

    for rotulo, resultado in resultados.items():
        print(
            f"\n{rotulo}"
        )

        print(
            f"  Arquivos: "
            f"{resultado['quantidade']} | "
            f"Shape: "
            f"{formatar_counter(resultado['shapes'])}"
        )

        if resultado[
            "quantidade"
        ]:
            print(
                f"  Dtype: "
                f"{formatar_counter(resultado['dtypes'])} | "
                f"Intervalo: "
                f"[{resultado['minimo']:.6f}, "
                f"{resultado['maximo']:.6f}]"
            )

    print(
        "\n"
        + "-" * 90
    )

    print(
        f"Total: {total} arquivos | "
        f"Rotulos: {len(resultados)} | "
        f"Arquivos com erro: {total_erros}"
    )

    print(
        "-" * 90
    )


# =============================================================================
# VERIFICACAO DE NOMES
# =============================================================================

def verificar_nomes_saida(registros):
    """
    Evita sobrescrita caso dois arquivos diferentes gerem
    o mesmo nome NPY dentro de uma mesma classe.
    """

    conflitos = []

    for rotulo in ROTULOS:
        nomes = {}

        for registro in registros:
            if registro[
                "rotulo"
            ] != rotulo:
                continue

            caminho = registro[
                "caminho"
            ]

            nome_saida = (
                caminho.stem
                + ".npy"
            )

            if nome_saida in nomes:
                conflitos.append((
                    rotulo,
                    nomes[nome_saida],
                    caminho,
                    nome_saida
                ))

            else:
                nomes[
                    nome_saida
                ] = caminho

    return conflitos


# =============================================================================
# PLANO DE PROCESSAMENTO
# =============================================================================

def mostrar_plano_processamento(resultado):
    print(
        "\n"
        + "=" * 90
    )

    print(
        "PLANO DE PRE-PROCESSAMENTO"
    )

    print(
        "=" * 90
    )

    print(
        f"\nOrigem : "
        f"{DATASET_ORIGINAL}"
    )

    print(
        f"Destino: "
        f"{DATASET_PROCESSADO}"
    )

    print(
        f"\nImagens a processar : "
        f"{len(resultado['registros'])}"
    )

    print(
        f"Rotulos finais      : "
        f"{len(ROTULOS)}"
    )

    print(
        f"Rotulos             : "
        f"{', '.join(ROTULOS)}"
    )

    print(
        "Canal final         : "
        "grayscale - 1 canal"
    )

    print(
        "Crop                : "
        "ja realizado anteriormente pelo SAM"
    )

    print(
        "Resize              : "
        "direto da imagem recortada"
    )

    print(
        f"Resolucao final     : "
        f"{TAMANHO_FINAL[0]}x"
        f"{TAMANHO_FINAL[1]}"
    )

    print(
        "Interpolacao        : "
        "LANCZOS"
    )

    print(
        f"Shape final         : "
        f"(1, {TAMANHO_FINAL[1]}, "
        f"{TAMANHO_FINAL[0]})"
    )

    print(
        "Dtype final         : "
        "float32"
    )

    print(
        "Intervalo final     : "
        "0.0 a 1.0"
    )

    print(
        "Formato final       : "
        "NPY"
    )

    print(
        "\nAntes do processamento, "
        "cada imagem e validada por:"
        "\n  - material da pasta x material do nome"
        "\n  - condicao da pasta x codigo RAC do nome"
    )

    print(
        "\nAs intensidades e materiais sao usados "
        "para validacao, mas os rotulos finais continuam sendo:"
        f"\n  {', '.join(ROTULOS)}"
    )

    print(
        "\nA unica normalizacao aplicada sera "
        "a divisao dos pixels por 255."
    )

    if DATASET_PROCESSADO.exists():
        print(
            "\nATENCAO: dataset_processado ja existe."
        )

        print(
            "A pasta sera recriada para evitar "
            "mistura com arquivos antigos."
        )


# =============================================================================
# CONFIRMACAO
# =============================================================================

def confirmar_processamento():
    while True:
        resposta = input(
            "\nDeseja continuar com estes parametros? [Y/N]: "
        ).strip().upper()

        if resposta == "Y":
            return True

        if resposta == "N":
            return False

        print(
            "Opcao invalida. Digite Y ou N."
        )


# =============================================================================
# PRE-PROCESSAMENTO DA IMAGEM
# =============================================================================

def processar_imagem(
    caminho_entrada,
    caminho_saida
):
    """
    Fluxo:
    grayscale
        -> resize
        -> float32
        -> /255
        -> (1,H,W)
        -> NPY
    """

    with Image.open(
        caminho_entrada
    ) as imagem:

        # 1 canal
        imagem = imagem.convert(
            "L"
        )

        # Resolucao padronizada
        imagem = imagem.resize(
            TAMANHO_FINAL,
            Image.Resampling.LANCZOS
        )

        # float32 + normalizacao 0-1
        array = np.asarray(
            imagem,
            dtype=np.float32
        ) / 255.0

        # (H,W) -> (1,H,W)
        array = np.expand_dims(
            array,
            axis=0
        )

    np.save(
        caminho_saida,
        array
    )


# =============================================================================
# PROCESSAMENTO DO DATASET
# =============================================================================

def processar_dataset(registros):
    """
    Recria dataset_processado e processa somente
    imagens previamente validadas.
    """

    if DATASET_PROCESSADO.exists():
        shutil.rmtree(
            DATASET_PROCESSADO
        )

    DATASET_PROCESSADO.mkdir(
        parents=True,
        exist_ok=True
    )

    for rotulo in ROTULOS:
        (
            DATASET_PROCESSADO
            / rotulo
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    total_processado = 0
    erros = []

    for rotulo in ROTULOS:
        registros_rotulo = [
            registro
            for registro in registros
            if registro["rotulo"] == rotulo
        ]

        print(
            f"\n{rotulo}:"
        )

        for indice, registro in enumerate(
            registros_rotulo,
            start=1
        ):
            caminho_entrada = registro[
                "caminho"
            ]

            caminho_saida = (
                DATASET_PROCESSADO
                / rotulo
                / f"{caminho_entrada.stem}.npy"
            )

            try:
                processar_imagem(
                    caminho_entrada,
                    caminho_saida
                )

                total_processado += 1

            except Exception as erro:
                erros.append((
                    caminho_entrada,
                    str(erro)
                ))

            print(
                f"\r  Processando "
                f"{indice}/"
                f"{len(registros_rotulo)}",
                end="",
                flush=True
            )

        print()

    return (
        total_processado,
        erros
    )


# =============================================================================
# VERIFICACAO FINAL
# =============================================================================

def verificar_dataset_processado(
    resultado_original,
    resultados_processados
):
    problemas = []

    esperado_por_rotulo = Counter(
        registro["rotulo"]
        for registro
        in resultado_original["registros"]
    )

    total_original = len(
        resultado_original["registros"]
    )

    total_processado = sum(
        resultado["quantidade"]
        for resultado
        in resultados_processados.values()
    )

    for rotulo in ROTULOS:
        processado = resultados_processados[
            rotulo
        ]

        esperado = esperado_por_rotulo[
            rotulo
        ]

        quantidade = processado[
            "quantidade"
        ]

        if quantidade != esperado:
            problemas.append(
                f"{rotulo}: esperado "
                f"{esperado} arquivo(s), "
                f"encontrado(s) {quantidade}."
            )

        shape_esperado = Counter({
            (
                1,
                TAMANHO_FINAL[1],
                TAMANHO_FINAL[0]
            ): quantidade
        })

        if processado[
            "shapes"
        ] != shape_esperado:
            problemas.append(
                f"{rotulo}: foram encontrados "
                f"arrays com shape incorreto."
            )

        dtype_esperado = Counter({
            "float32": quantidade
        })

        if processado[
            "dtypes"
        ] != dtype_esperado:
            problemas.append(
                f"{rotulo}: foram encontrados arrays "
                f"com dtype diferente de float32."
            )

        if (
            processado["minimo"] is not None
            and processado["minimo"] < 0.0
        ):
            problemas.append(
                f"{rotulo}: foram encontrados "
                f"valores menores que 0."
            )

        if (
            processado["maximo"] is not None
            and processado["maximo"] > 1.0
        ):
            problemas.append(
                f"{rotulo}: foram encontrados "
                f"valores maiores que 1."
            )

        if processado[
            "erros"
        ]:
            problemas.append(
                f"{rotulo}: "
                f"{len(processado['erros'])} "
                f"arquivo(s) apresentaram erro."
            )

    if total_original != total_processado:
        problemas.append(
            f"Total esperado ({total_original}) "
            f"diferente do total processado "
            f"({total_processado})."
        )

    return problemas


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not DATASET_ORIGINAL.exists():
        raise FileNotFoundError(
            f"Dataset recortado nao encontrado:\n"
            f"{DATASET_ORIGINAL}"
        )

    # -------------------------------------------------------------------------
    # Validacao do dataset de entrada
    # -------------------------------------------------------------------------

    resultado_original = analisar_dataset_original(
        DATASET_ORIGINAL
    )

    mostrar_resumo_original(
        resultado_original
    )

    if resultado_original[
        "erros"
    ]:
        print(
            "\n"
            + "=" * 90
        )

        print(
            "PROCESSAMENTO CANCELADO"
        )

        print(
            "=" * 90
        )

        print(
            "\nExistem imagens com estrutura "
            "ou nomenclatura inconsistente."
        )

        print(
            "Corrija os erros apresentados antes "
            "de gerar o dataset processado."
        )

        return

    if not resultado_original[
        "registros"
    ]:
        print(
            "\nNenhuma imagem valida foi encontrada."
        )

        return

    # -------------------------------------------------------------------------
    # Verificacao de conflitos de nome
    # -------------------------------------------------------------------------

    conflitos = verificar_nomes_saida(
        resultado_original[
            "registros"
        ]
    )

    if conflitos:
        print(
            "\nERRO: foram encontrados "
            "nomes conflitantes."
        )

        for (
            rotulo,
            arquivo_1,
            arquivo_2,
            nome_saida
        ) in conflitos:

            print(
                f"  {rotulo}: "
                f"{arquivo_1} e "
                f"{arquivo_2} gerariam "
                f"{nome_saida}"
            )

        print(
            "\nO processamento foi cancelado "
            "para evitar sobrescrita."
        )

        return

    # -------------------------------------------------------------------------
    # Plano
    # -------------------------------------------------------------------------

    mostrar_plano_processamento(
        resultado_original
    )

    if not confirmar_processamento():
        print(
            "\nProcessamento cancelado pelo usuario."
        )

        return

    # -------------------------------------------------------------------------
    # Processamento
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "PROCESSANDO DATASET"
    )

    print(
        "=" * 90
    )

    (
        total_processado,
        erros_processamento
    ) = processar_dataset(
        resultado_original[
            "registros"
        ]
    )

    print(
        f"\nProcessamento concluido: "
        f"{total_processado} arquivos criados."
    )

    if erros_processamento:
        print(
            f"\nErros durante o processamento: "
            f"{len(erros_processamento)}"
        )

        for caminho, erro in erros_processamento:
            print(
                f"  {caminho}"
            )

            print(
                f"    -> {erro}"
            )

    # -------------------------------------------------------------------------
    # Verificacao dos NPY
    # -------------------------------------------------------------------------

    resultados_processados = analisar_dataset_processado(
        DATASET_PROCESSADO
    )

    mostrar_resumo_processado(
        resultados_processados
    )

    problemas = verificar_dataset_processado(
        resultado_original,
        resultados_processados
    )

    # -------------------------------------------------------------------------
    # Resultado
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "RESULTADO FINAL"
    )

    print(
        "=" * 90
    )

    if problemas:
        print(
            "\nATENCAO - Foram encontrados problemas:"
        )

        for problema in problemas:
            print(
                f"  - {problema}"
            )

        return

    total_final = sum(
        resultado["quantidade"]
        for resultado
        in resultados_processados.values()
    )

    print(
        "\nOK - O pre-processamento "
        "foi concluido corretamente."
    )

    print(
        "\nFluxo aplicado:"
        "\n  imagem recortada pelo SAM"
        "\n  -> grayscale"
        "\n  -> resize"
        "\n  -> float32"
        "\n  -> /255"
        "\n  -> (1,H,W)"
        "\n  -> NPY"
    )

    print(
        "\nTodos os arquivos processados possuem:"
        "\n  - formato NPY"
        f"\n  - shape "
        f"(1, {TAMANHO_FINAL[1]}, "
        f"{TAMANHO_FINAL[0]})"
        "\n  - dtype float32"
        "\n  - valores entre 0.0 e 1.0"
    )

    print(
        f"\nQuantidade original valida: "
        f"{len(resultado_original['registros'])}"
    )

    print(
        f"Quantidade final          : "
        f"{total_final}"
    )

    print(
        "\nDistribuicao final:"
    )

    for rotulo in ROTULOS:
        print(
            f"  {rotulo}: "
            f"{resultados_processados[rotulo]['quantidade']}"
        )

    print(
        f"\nDataset salvo em:\n"
        f"{DATASET_PROCESSADO}"
    )


if __name__ == "__main__":
    main()