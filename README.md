# Configuração do Ambiente Python (Windows) — Troubleshooting

Este documento reúne problemas comuns ao configurar Python + Poetry no Windows, encontrados durante a configuração deste projeto.

## Problema 1: Erro ao usar caminhos do Windows em strings Python

**Sintoma:**
```
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 2-3: truncated \UXXXXXXXX escape
```

**Causa:** a barra invertida (`\`) é um caractere de escape em strings Python. Um caminho como `"C:\Users\DELL\Documents\..."` faz o Python interpretar `\U` como início de um escape Unicode inválido.

**Solução:** use uma das opções abaixo:
```python
# Opção 1 - string raw (recomendado)
root = r"C:\Users\DELL\Documents\dataset_falhas"

# Opção 2 - barras normais (funcionam no Windows também)
root = "C:/Users/DELL/Documents/dataset_falhas"

# Opção 3 - duplicar as barras
root = "C:\\Users\\DELL\\Documents\\dataset_falhas"
```

## Problema 2: `FileNotFoundError` ao rodar notebook no Google Colab

**Sintoma:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\DELL\\...'
```

**Causa:** caminhos como `C:\Users\...` são do disco local do Windows. O Google Colab roda em uma máquina virtual na nuvem, sem acesso ao HD do computador local — os dois ambientes não compartilham arquivos.

**Solução (upload manual, para dados sensíveis/da empresa):**
1. Compactar a pasta localmente (`.zip`)
2. Fazer upload pelo painel de arquivos do Colab (ícone de pasta → upload)
3. Descompactar dentro do notebook:
   ```python
   import zipfile
   with zipfile.ZipFile("dataset_falhas.zip", "r") as zip_ref:
       zip_ref.extractall("./dataset_falhas")
   ```
4. Apontar o `ImageFolder`/`Dataset` para `./dataset_falhas`

> ⚠️ Arquivos enviados assim são apagados quando a sessão do Colab reinicia — é preciso repetir o upload em cada nova sessão.
>
> ⚠️ Atenção com dados sensíveis/da empresa: o Colab também roda em servidores do Google (não é "local"), só não persiste os arquivos entre sessões. Confirme com TI/compliance antes de subir dados sensíveis, mesmo via upload direto.

## Problema 3: Poetry falha com `exit status 9009` / `python` aponta para a Microsoft Store

**Sintoma:**
```
Command '[...\WindowsApps\python.EXE', ...]' returned non-zero exit status 9009.
```

**Causa:** o Windows cria "aliases de execução" (stubs fantasmas) em `AppData\Local\Microsoft\WindowsApps\python.exe`. Se a instalação real do Python não estiver corretamente priorizada no PATH, o Windows usa esse stub, que não funciona como um Python de verdade.

**Diagnóstico:**
```powershell
where.exe python          # 'where' sozinho NÃO funciona no PowerShell (é alias de Where-Object)
Get-Command python -All   # lista todas as instalações encontradas, em ordem de prioridade
```

Se o primeiro resultado apontar para `WindowsApps\python.exe`, é o stub.

**Solução:**

1. **Desativar os aliases** (evita que o stub tenha prioridade):
   - `Configurações → Aplicativos → Aliases de execução de aplicativos`
   - Desligar `python.exe` e `python3.exe`

2. **Adicionar a instalação real do Python ao PATH:**
   - `Configurações → Variáveis de ambiente do sistema → Variáveis de Ambiente → Path → Editar → Novo`
   - Adicionar (ajustando a versão instalada):
     ```
     C:\Users\<usuario>\AppData\Local\Programs\Python\Python313\
     C:\Users\<usuario>\AppData\Local\Programs\Python\Python313\Scripts\
     ```

3. **Fechar e abrir um novo terminal** (o PATH só é recarregado em sessões novas)

4. **Confirmar:**
   ```powershell
   python --version
   Get-Command python -All
   ```
   O primeiro resultado deve ser o caminho real da instalação (`...\Python313\python.exe`), não o `WindowsApps`.

Depois disso, os comandos do Poetry (`poetry install`, `poetry env use`, etc.) devem funcionar normalmente.

## Problema 4: PyCharm mostra "No Python interpreter configured" mesmo após configurar o ambiente Poetry

**Sintoma:** ao clicar em OK na janela "Add Python Interpreter" (Type: Poetry), a janela fecha sem erro visível, mas o aviso amarelo continua. No painel "Python Process Output" aparecem várias linhas `poetry.exe check --lock` com ícone vermelho — mas essa mensagem (`Error: poetry.lock was not found`) é esperada em um projeto novo e **não é a causa real** do problema.

**Solução:** rodar o `poetry install` manualmente no terminal (dentro da pasta do projeto), em vez de depender só da interface gráfica:
```powershell
cd C:\caminho\do\projeto
poetry --version
poetry install
```
Isso cria o ambiente virtual e o lockfile diretamente, mostrando qualquer erro real de forma completa. Depois, no PyCharm, em vez de "Generate new", use **"Select existing"** — o ambiente recém-criado pelo Poetry deve aparecer na lista automaticamente. Se não aparecer, confirme o caminho exato com:
```powershell
poetry env info --path
```

## Problema 5: Instalando PyTorch com suporte a GPU (CUDA)

PyTorch com CUDA não vem do PyPI padrão — vem de um índice próprio do PyTorch, específico por versão de CUDA. Antes de instalar, confira a versão máxima de CUDA suportada pelo driver:
```powershell
nvidia-smi
```
O campo "CUDA Version" no topo é o teto suportado (retrocompatível com versões mais antigas do CUDA). Verifique também em https://pytorch.org/get-started/locally/ qual é a build de CUDA mais recente disponível no momento (ex.: cu128).

**Comandos (Poetry 2.x):**
```powershell
# 1. Adiciona o índice do PyTorch como fonte "explícita" (só usada quando pedida via --source)
poetry source add pytorch-cu128 --priority=explicit https://download.pytorch.org/whl/cu128

# 2. Instala torch e torchvision vindos dessa fonte específica
poetry add torch torchvision --source pytorch-cu128

# 3. Instala o resto normalmente, do PyPI padrão
poetry add matplotlib scikit-learn pillow numpy jupyter ipykernel
```

**Verificação:**
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### Erro relacionado: `version solving failed` por conflito de `requires-python`

**Sintoma:**
```
The current project's supported Python range (>=3.13) is not compatible with some of the required packages
- torchvision requires Python >=3.10,!=3.14.1, so it will not be installable for Python 3.14.1
```

**Causa:** `requires-python = ">=3.13"` no `pyproject.toml` é uma faixa aberta, que inclui versões futuras (3.14, 3.14.1, etc.). O `torchvision` exclui explicitamente a versão `3.14.1`. Mesmo usando Python 3.13 de fato, o Poetry precisa garantir compatibilidade com toda a faixa declarada, e trava por causa disso.

**Solução:** restringir a faixa à versão realmente usada, em `pyproject.toml`:
```toml
requires-python = ">=3.13,<3.14"
```
Depois, rodar novamente o `poetry add` que havia falhado.

## Problema 6: `ImportError: cannot load module more than once per process` ao rodar pelo botão "Run" do PyCharm

**Sintoma:** ao rodar `main.py` pelo botão "Run" do PyCharm, o próprio `import torch` falha com esse erro, envolvendo `numpy` no traceback. Rodando o mesmo script direto no terminal (`poetry run python src\main.py`), funciona normalmente.

**Causa:** o PyCharm tem um recurso ("Python Console" → "Show plots in tool window") que pré-carrega bibliotecas como matplotlib/numpy no mesmo processo antes do script do usuário rodar. Versões recentes do numpy (2.4+) passaram a bloquear ser inicializado duas vezes no mesmo processo, o que antes era só ignorado.

**Solução:**
1. Testar primeiro direto no terminal para confirmar que é isso: `poetry run python <caminho_do_script>`
2. Se funcionar no terminal, desativar no PyCharm: `File → Settings → Build, Execution, Deployment → Console → Python Console` → desmarcar **"Show plots in tool window"**
3. Alternativa, se persistir: fixar uma versão do numpy anterior à 2.4: `poetry add "numpy<2.4"`

## Problema 7: `ImportError: DLL load failed... Uma política de Controle de Aplicativos bloqueou este arquivo`

**Sintoma:** ao importar `matplotlib` (que importa `kiwisolver` internamente), o Windows bloqueia a execução do arquivo `.pyd` nativo do `kiwisolver`, mesmo `numpy`/`torch` funcionando normalmente.

**Causa:** em instalações novas do Windows 11, o **Smart App Control** vem ativado por padrão e bloqueia automaticamente binários sem assinatura digital reconhecida ou sem "reputação" nos serviços da Microsoft — mesmo sendo bibliotecas legítimas (comum com pacotes Python compilados menos populares, como o `kiwisolver`). Isso não tem relação com localização de pasta, política de rede corporativa, ou permissões de arquivo — trocar o ambiente de local, ou tentar `Unblock-File`, não resolve.

**Diagnóstico:** confirmar que não é rede/TI corporativa (a mensagem de erro é idêntica à de políticas WDAC empresariais, mas o Smart App Control é um recurso padrão do Windows 11, mesmo em máquinas não gerenciadas).

**Solução:**
1. Abrir **Windows Security → App & Browser Control → Smart App Control settings**
2. Mudar de "On"/"Evaluation" para **"Off"**
3. Reiniciar o computador
4. Testar de novo: `poetry run python src\main.py`

> ⚠️ Em algumas versões do Windows 11, desativar o Smart App Control é **permanente** (só reativa reinstalando o Windows do zero) — versões mais recentes já permitem ligar/desligar livremente. Verifique a versão antes de desativar, ciente de que essa camada específica de proteção é removida (o Defender comum continua ativo).#   p d c - i a - i m a g e _ d e f e c t _ d e t e c t i o n  
 #   p d c - i a - i m a g e _ d e f e c t _ d e t e c t i o n  
 