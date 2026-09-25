from pathlib import Path
from collections import Counter
import shutil

import numpy as np
from PIL import Image


# Caminhos do projeto
BASE_DIR = Path(__file__).resolve().parents[3]
DATASET_ORIGINAL = BASE_DIR / "dataset_falhas"
DATASET_PROCESSADO = BASE_DIR / "dataset_processado"

# Configuracoes do pre-processamento
TAMANHO_FINAL = (1024, 1024)
EXTENSOES_VALIDAS = (".png", ".jpg", ".jpeg", ".bmp")


def obter_classes(root):
    """Retorna as pastas de classes encontradas no dataset"""
    return sorted([pasta for pasta in root.iterdir() if pasta.is_dir()])


def obter_imagens(pasta):
    """Retorna os caminhos das imagens validas"""
    return sorted([
        arquivo for arquivo in pasta.rglob("*")
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_VALIDAS
    ])


def obter_arrays(pasta):
    """Retorna os arquivos NPY encontrados"""
    return sorted([arquivo for arquivo in pasta.rglob("*.npy") if arquivo.is_file()])


def analisar_dataset_original(root):
    """Analisa formato, modo, canais, resolucao e erros das imagens"""

    resultados = {}

    for pasta_classe in obter_classes(root):
        imagens = obter_imagens(pasta_classe)
        formatos, modos, canais, resolucoes = Counter(), Counter(), Counter(), Counter()
        erros = []

        for caminho_img in imagens:
            try:
                formatos[caminho_img.suffix.lower()] += 1

                with Image.open(caminho_img) as imagem:
                    modos[imagem.mode] += 1
                    canais[len(imagem.getbands())] += 1
                    resolucoes[imagem.size] += 1

            except Exception as erro:
                erros.append((caminho_img, str(erro)))

        resultados[pasta_classe.name] = {
            "quantidade": len(imagens),
            "formatos": formatos,
            "modos": modos,
            "canais": canais,
            "resolucoes": resolucoes,
            "erros": erros,
        }

    return resultados


def analisar_dataset_processado(root):
    """Analisa shape, dtype, intervalo e erros dos arquivos NPY"""

    resultados = {}

    for pasta_classe in obter_classes(root):
        arquivos = obter_arrays(pasta_classe)
        shapes, dtypes = Counter(), Counter()
        minimo, maximo = float("inf"), float("-inf")
        erros = []

        for caminho in arquivos:
            try:
                array = np.load(caminho, mmap_mode="r")
                shapes[array.shape] += 1
                dtypes[str(array.dtype)] += 1
                minimo = min(minimo, float(array.min()))
                maximo = max(maximo, float(array.max()))

                if not np.isfinite(array).all():
                    erros.append((caminho, "Valores NaN ou infinitos encontrados."))

            except Exception as erro:
                erros.append((caminho, str(erro)))

        resultados[pasta_classe.name] = {
            "quantidade": len(arquivos),
            "shapes": shapes,
            "dtypes": dtypes,
            "minimo": minimo if arquivos else None,
            "maximo": maximo if arquivos else None,
            "erros": erros,
        }

    return resultados


def formatar_counter(counter):
    """Transforma um Counter em texto compacto"""
    return ", ".join(f"{valor}={quantidade}" for valor, quantidade in counter.items())


def formatar_resolucoes(counter):
    """Transforma as resolucoes em texto compacto"""
    return ", ".join(f"{largura}x{altura}={quantidade}" for (largura, altura), quantidade in counter.items())


def mostrar_resumo_original(resultados):
    """Mostra um resumo compacto do dataset original"""

    print("\n" + "=" * 90)
    print("ANALISE DO DATASET ORIGINAL")
    print("=" * 90)

    total = sum(resultado["quantidade"] for resultado in resultados.values())
    total_erros = sum(len(resultado["erros"]) for resultado in resultados.values())

    for classe, resultado in resultados.items():
        print(f"\n{classe}")
        print(f"  Imagens: {resultado['quantidade']} | Formatos: {formatar_counter(resultado['formatos'])}")
        print(
            f"  Modo: {formatar_counter(resultado['modos'])} | "
            f"Canais: {formatar_counter(resultado['canais'])} | "
            f"Resolucao: {formatar_resolucoes(resultado['resolucoes'])}"
        )

    print("\n" + "-" * 90)
    print(f"Total: {total} imagens | Classes: {len(resultados)} | Arquivos com erro: {total_erros}")
    print("-" * 90)


def mostrar_resumo_processado(resultados):
    """Mostra um resumo dos arquivos NPY gerados"""

    print("\n" + "=" * 90)
    print("VERIFICACAO DO DATASET PROCESSADO")
    print("=" * 90)

    total = sum(resultado["quantidade"] for resultado in resultados.values())
    total_erros = sum(len(resultado["erros"]) for resultado in resultados.values())

    for classe, resultado in resultados.items():
        print(f"\n{classe}")
        print(f"  Arquivos: {resultado['quantidade']} | Shape: {formatar_counter(resultado['shapes'])}")
        print(
            f"  Dtype: {formatar_counter(resultado['dtypes'])} | "
            f"Intervalo: [{resultado['minimo']:.6f}, {resultado['maximo']:.6f}]"
        )

    print("\n" + "-" * 90)
    print(f"Total: {total} arquivos | Classes: {len(resultados)} | Arquivos com erro: {total_erros}")
    print("-" * 90)


def verificar_nomes_saida():
    """Verifica se imagens diferentes gerariam o mesmo nome NPY"""

    conflitos = []

    for pasta_classe in obter_classes(DATASET_ORIGINAL):
        nomes_saida = {}

        for caminho_img in obter_imagens(pasta_classe):
            nome_saida = caminho_img.stem + ".npy"

            if nome_saida in nomes_saida:
                conflitos.append((pasta_classe.name, nomes_saida[nome_saida], caminho_img, nome_saida))
            else:
                nomes_saida[nome_saida] = caminho_img

    return conflitos


def mostrar_plano_processamento(resultados):
    """Mostra os parametros do pre-processamento"""

    total = sum(resultado["quantidade"] for resultado in resultados.values())

    print("\n" + "=" * 90)
    print("PLANO DE PRE-PROCESSAMENTO")
    print("=" * 90)

    print(f"\nOrigem : {DATASET_ORIGINAL}")
    print(f"Destino: {DATASET_PROCESSADO}")

    print(
        f"\nImagens a processar : {total}\n"
        f"Classes             : {len(resultados)}\n"
        f"Canal final         : grayscale - 1 canal\n"
        f"Recorte             : maior quadrado central\n"
        f"Resolucao final     : {TAMANHO_FINAL[0]}x{TAMANHO_FINAL[1]}\n"
        f"Interpolacao        : LANCZOS\n"
        f"Shape final         : (1, {TAMANHO_FINAL[1]}, {TAMANHO_FINAL[0]})\n"
        f"Dtype final         : float32\n"
        f"Intervalo final     : 0.0 a 1.0\n"
        f"Formato final       : NPY"
    )

    print("\nO centro geometrico da imagem sera preservado durante o recorte.")
    print("A unica normalizacao aplicada sera a divisao dos pixels por 255.")

    if DATASET_PROCESSADO.exists():
        print(
            "\nATENCAO: a pasta dataset_processado ja existe.\n"
            "Ela sera recriada para evitar arquivos antigos misturados ao novo processamento."
        )


def confirmar_processamento():
    """Pergunta ao usuario se deseja iniciar o processamento"""

    while True:
        resposta = input("\nDeseja continuar com estes parametros? [Y/N]: ").strip().upper()

        if resposta == "Y":
            return True

        if resposta == "N":
            return False

        print("Opcao invalida. Digite Y ou N.")


def recortar_quadrado_central(imagem):
    """Recorta o maior quadrado possivel mantendo o centro geometrico"""

    largura, altura = imagem.size
    lado = min(largura, altura)
    esquerda = (largura - lado) // 2
    topo = (altura - lado) // 2

    return imagem.crop((esquerda, topo, esquerda + lado, topo + lado))


def processar_imagem(caminho_entrada, caminho_saida):
    """Pre-processa a imagem e salva como array float32 entre 0 e 1"""

    with Image.open(caminho_entrada) as imagem:
        imagem = imagem.convert("L")
        imagem = recortar_quadrado_central(imagem)
        imagem = imagem.resize(TAMANHO_FINAL, Image.Resampling.LANCZOS)
        array = np.asarray(imagem, dtype=np.float32) / 255.0
        array = np.expand_dims(array, axis=0)

    np.save(caminho_saida, array)


def processar_dataset():
    """Processa todas as imagens e cria o novo dataset"""

    if DATASET_PROCESSADO.exists():
        shutil.rmtree(DATASET_PROCESSADO)

    DATASET_PROCESSADO.mkdir(parents=True, exist_ok=True)

    total_processado = 0
    erros = []

    for pasta_classe in obter_classes(DATASET_ORIGINAL):
        pasta_saida = DATASET_PROCESSADO / pasta_classe.name
        pasta_saida.mkdir(parents=True, exist_ok=True)

        imagens = obter_imagens(pasta_classe)
        print(f"\n{pasta_classe.name}:")

        for indice, caminho_entrada in enumerate(imagens, start=1):
            caminho_saida = pasta_saida / f"{caminho_entrada.stem}.npy"

            try:
                processar_imagem(caminho_entrada, caminho_saida)
                total_processado += 1

            except Exception as erro:
                erros.append((caminho_entrada, str(erro)))

            print(f"\r  Processando {indice}/{len(imagens)}", end="", flush=True)

        print()

    return total_processado, erros


def verificar_dataset_processado(resultados_originais, resultados_processados):
    """Verifica se os arquivos processados possuem os parametros esperados"""

    problemas = []

    if set(resultados_originais) != set(resultados_processados):
        problemas.append("As classes do dataset processado sao diferentes das classes originais.")

    total_original = sum(resultado["quantidade"] for resultado in resultados_originais.values())
    total_processado = sum(resultado["quantidade"] for resultado in resultados_processados.values())

    for classe, original in resultados_originais.items():
        if classe not in resultados_processados:
            continue

        processado = resultados_processados[classe]
        quantidade = processado["quantidade"]

        if original["quantidade"] != quantidade:
            problemas.append(
                f"{classe}: quantidade original ({original['quantidade']}) diferente da processada ({quantidade})."
            )

        if processado["shapes"] != Counter({(1, TAMANHO_FINAL[1], TAMANHO_FINAL[0]): quantidade}):
            problemas.append(f"{classe}: foram encontrados arrays com shape diferente do esperado.")

        if processado["dtypes"] != Counter({"float32": quantidade}):
            problemas.append(f"{classe}: foram encontrados arrays com dtype diferente de float32.")

        if processado["minimo"] is not None and processado["minimo"] < 0.0:
            problemas.append(f"{classe}: foram encontrados valores menores que 0.")

        if processado["maximo"] is not None and processado["maximo"] > 1.0:
            problemas.append(f"{classe}: foram encontrados valores maiores que 1.")

        if processado["erros"]:
            problemas.append(f"{classe}: {len(processado['erros'])} arquivo(s) apresentaram erro.")

    if total_original != total_processado:
        problemas.append(f"Total original ({total_original}) diferente do total processado ({total_processado}).")

    return problemas


def main():

    if not DATASET_ORIGINAL.exists():
        raise FileNotFoundError(f"Dataset original nao encontrado:\n{DATASET_ORIGINAL}")

    # Analise inicial
    resultados_originais = analisar_dataset_original(DATASET_ORIGINAL)
    mostrar_resumo_original(resultados_originais)

    # Verificacao de nomes conflitantes
    conflitos = verificar_nomes_saida()

    if conflitos:
        print("\nERRO: foram encontrados nomes conflitantes.")

        for classe, arquivo_1, arquivo_2, nome_saida in conflitos:
            print(f"  {classe}: {arquivo_1.name} e {arquivo_2.name} gerariam {nome_saida}")

        print("\nO processamento foi cancelado para evitar sobrescrita de arquivos.")
        return

    # Confirmacao do processamento
    mostrar_plano_processamento(resultados_originais)

    if not confirmar_processamento():
        print("\nProcessamento cancelado pelo usuario.")
        return

    print("\n" + "=" * 90)
    print("PROCESSANDO DATASET")
    print("=" * 90)

    total_processado, erros_processamento = processar_dataset()

    print(f"\nProcessamento concluido: {total_processado} arquivos criados.")

    if erros_processamento:
        print(f"Erros durante o processamento: {len(erros_processamento)}")

    # Verificacao do resultado
    resultados_processados = analisar_dataset_processado(DATASET_PROCESSADO)
    mostrar_resumo_processado(resultados_processados)

    problemas = verificar_dataset_processado(resultados_originais, resultados_processados)

    print("\n" + "=" * 90)
    print("RESULTADO FINAL")
    print("=" * 90)

    if problemas:
        print("\nATENCAO - Foram encontrados problemas na verificacao:")

        for problema in problemas:
            print(f"  - {problema}")

        return

    total_original = sum(resultado["quantidade"] for resultado in resultados_originais.values())
    total_final = sum(resultado["quantidade"] for resultado in resultados_processados.values())

    print("\nOK - O pre-processamento foi concluido corretamente.")
    print(
        "Todos os arquivos foram verificados e possuem:\n"
        f"  - formato NPY\n"
        f"  - shape (1, {TAMANHO_FINAL[1]}, {TAMANHO_FINAL[0]})\n"
        f"  - dtype float32\n"
        f"  - valores entre 0.0 e 1.0"
    )

    print(f"\nQuantidade original : {total_original}")
    print(f"Quantidade final    : {total_final}")
    print(f"\nDataset salvo em:\n{DATASET_PROCESSADO}")


if __name__ == "__main__":
    main()