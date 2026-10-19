"""
paths.py

Objetivo:
- Reunir funcoes genericas relacionadas a localizacao de caminhos do projeto.
- Permitir que scripts executados em diferentes subpastas encontrem a raiz
  do repositorio sem depender de uma quantidade fixa de parents[].
- Centralizar a identificacao da raiz para evitar caminhos relativos
  inconsistentes entre os diferentes processos.

Identificacao da raiz:
- A raiz do projeto e definida como o primeiro diretorio ancestral que contem
  o arquivo pyproject.toml.
- A busca parte do arquivo utilizado como referencia e sobe pela arvore de
  diretorios ate encontrar esse marcador.

Uso esperado:
- Scripts de processing podem passar __file__ para localizar automaticamente
  a raiz do projeto.

Exemplo:
    BASE_DIR = encontrar_raiz_projeto(__file__)

Observacoes:
- A funcao nao depende da pasta atual do terminal.
- Caminhos relativos sao convertidos para caminhos absolutos antes da busca.
- Links simbolicos sao resolvidos antes da verificacao.
- Se pyproject.toml nao for encontrado, a funcao gera um erro explicito.
- Este modulo nao conhece datasets, modelos ou etapas especificas do pipeline.
"""

from pathlib import Path


# =============================================================================
# LOCALIZACAO DA RAIZ DO PROJETO
# =============================================================================

def encontrar_raiz_projeto(arquivo_referencia):
    """
    Localiza a raiz do projeto a partir de um arquivo conhecido.

    A funcao sobe pelos diretorios ancestrais ate encontrar uma pasta contendo
    pyproject.toml, utilizado como marcador da raiz do repositorio.

    Parameters
    ----------
    arquivo_referencia : str ou Path
        Caminho de um arquivo localizado dentro da estrutura do projeto.
        Normalmente recebe __file__ do script que chama esta funcao.

    Returns
    -------
    Path
        Caminho absoluto da raiz do projeto.

    Raises
    ------
    FileNotFoundError
        Quando nenhum pyproject.toml e encontrado entre os diretorios
        ancestrais da referencia fornecida.
    """

    # Converte a referencia recebida para Path, transforma o caminho em absoluto
    # e resolve componentes como ".." e links simbolicos.
    referencia = Path(
        arquivo_referencia
    ).resolve()

    # A busca deve comecar na pasta que contem o arquivo de referencia.
    pasta_inicial = referencia.parent

    # Testa a pasta inicial e, em seguida, todos os seus diretorios ancestrais.
    #
    # Exemplo:
    # C:/projeto/src/pacote/processing/augmentation.py
    #
    # A busca percorre:
    # C:/projeto/src/pacote/processing
    # C:/projeto/src/pacote
    # C:/projeto/src
    # C:/projeto
    # ...
    for pasta in (
        pasta_inicial,
        *pasta_inicial.parents,
    ):
        # A existencia de pyproject.toml identifica a raiz do projeto.
        marcador = (
            pasta
            / "pyproject.toml"
        )

        # Retorna imediatamente o primeiro diretorio que contem o marcador.
        if marcador.is_file():
            return pasta

    # Nenhum caminho e retornado quando o marcador nao existe na arvore
    # analisada, portanto a execucao e interrompida com uma mensagem clara.
    raise FileNotFoundError(
        "Nao foi possivel localizar a raiz do projeto. "
        "pyproject.toml nao foi encontrado nos diretorios ancestrais de:\n"
        f"{referencia}"
    )