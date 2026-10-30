# PDC-IA - Estrutura do `src`

Este diretório concentra a nova organização do pipeline de classificação de defeitos por imagem.

## Estrutura principal

```text
src/
├── main.py
├── README.md
└── pdc_ia_image_defect_detection/
    ├── models/
    ├── pre_processing/
    ├── processing/
    └── utils/
```

### `main.py`

O `main.py` apenas executa as etapas na ordem correta:

```text
SAM
→ PREPARA_IMG
→ SPLIT
→ AUGMENTATION
→ TREINO
→ TEST
```

Cada etapa possui sua própria flag `EXECUTAR_...`.

Se a flag estiver `False`, a etapa é ignorada sem modificar arquivos.

## `processing/`

Contém os scripts principais configurados manualmente:

```text
processing/
├── sam.py
├── prepara_img.py
├── split.py
├── augmentation.py
├── treino.py
└── test.py
```

É aqui que devem ser alterados os parâmetros de cada etapa, por exemplo:

- ativar ou desativar a execução;
- escolher bases interna e externa;
- definir caminhos e resolução;
- configurar augmentation;
- configurar treino;
- escolher pasta de teste;
- escolher modelo melhor ou final.

Exemplo:

```python
EXECUTAR_TREINO = True
USAR_DATASET_INTERNO = True
USAR_DATASET_EXTERNO = False
```

## `pre_processing/`

Contém a implementação específica de cada processo:

```text
pre_processing/
├── sam_class.py
├── prepara_img_class.py
├── split_class.py
├── augmentation_class.py
├── treino_class.py
└── test_class.py
```

Os arquivos de `processing/` configuram e chamam essas classes. A lógica principal de cada processo deve permanecer aqui, e não no `main.py`.

## `models/`

Contém somente arquiteturas de redes neurais.

```text
models/
└── lenet5.py
```

O treino e o teste importam o modelo desta pasta.

## `utils/`

Contém somente funções genéricas e reutilizáveis.

```text
utils/
├── dataset.py
├── imagens.py
├── metrics.py
├── nomenclatura.py
├── npy.py
└── paths.py
```

Responsabilidades principais:

- `dataset.py`: carregamento de datasets;
- `imagens.py`: operações genéricas com imagens;
- `metrics.py`: métricas, matriz de confusão e gráficos;
- `nomenclatura.py`: interpretação de `MATERIAL_CP_RAC`;
- `npy.py`: leitura, validação e escrita de `.npy`;
- `paths.py`: localização da raiz do projeto.

# Pastas de dados

## Base interna original

```text
dataset_atual_crop/
└── material/
    └── condicao/
        └── imagem
```

Exemplo:

```text
dataset_atual_crop/
└── aco_2500/
    └── risco_severo/
        └── A36_81_300.png
```

As imagens internas seguem a nomenclatura `MATERIAL_CP_RAC`.

## Base externa original

```text
dataset_externo_test/
└── imagens_selecionadas/
    ├── amassamento/
    ├── corrosao/
    ├── risco/
    └── sem_falha/
```

Na base externa, o rótulo vem da pasta da classe.

## Bases processadas

```text
dataset_processado/
dataset_externo_processado/
```

Contêm imagens convertidas para `.npy`.

## Bases após o split

```text
dataset_processado_split/
├── train/
├── val/
└── test/

dataset_externo_processado_split/
├── train/
├── val/
└── test/
```

Cada divisão mantém uma pasta por classe.

# Resultados

Os resultados de treino e teste são salvos dentro de:

```text
resultados/
```

Os scripts criam suas próprias pastas de execução para evitar mistura entre experimentos.

# Uso recomendado

1. Ajuste os parâmetros nos scripts em `processing/`.
2. Ative somente as etapas desejadas com `EXECUTAR_... = True`.
3. Deixe como `False` as etapas já executadas e com resultados válidos.
4. Execute:

```bash
python src/main.py
```

O `main.py` executará somente as etapas ativadas, respeitando a ordem do pipeline.

# Padrao de pastas de resultados

Toda etapa que criar uma pasta de resultados deve usar `utils/resultados.py`.

O nome registra:

```text
tipo + modelo + resolucao + indicadores True + inicio + fim
```

Somente flags ativas recebem indicador no nome. Flags `False` nao geram texto.

Exemplos:

```text
treino_lenet5_r256_int_aug_inicio_2026-10-23_12-47-03_fim_2026-10-23_13-05-11
test_lenet5_r256_split_int_melhor_inicio_2026-10-23_13-10-00_fim_2026-10-23_13-11-24
```

Indicadores atuais:

- treino: `int`, `ext`, `aug`, `bal`;
- test: `split`, `int`, `ext`, `melhor`.

Enquanto uma execucao ainda nao terminou, a pasta termina em `_em_andamento`. Ao final, ela e renomeada e recebe o horario de fim.
