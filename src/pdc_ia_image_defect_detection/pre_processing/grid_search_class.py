"""
grid_search_class.py

Objetivo:
- Executar combinacoes LeNet5 ou VGG16 em subprocessos independentes.
- Persistir configuracoes, estados, logs, metricas e rankings.
- Continuar apos falhas individuais e retomar execucoes interrompidas.

Regras:
- Todos os experimentos usam a seed e as configuracoes comuns definidas.
- Apenas treino e validacao participam da busca.
- Treinamentos finalizados nao sao repetidos na retomada.
- Falhas sao registradas e ignoradas no ranking.
"""

import argparse
import csv
from datetime import datetime
from itertools import product
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import shutil
import zipfile

sys.dont_write_bytecode = True


COLUNAS_RESUMO = (
    "id", "status", "tentativa", "modelo", "kernel", "dropout", "neuronios_fc1", "neuronios_fc2",
    "learning_rate", "seed", "epocas_executadas", "melhor_epoca",
    "melhor_val_loss", "maior_val_acc", "epoca_maior_val_acc",
    "val_acc_na_melhor_loss", "train_acc_final", "val_acc_final",
    "val_loss_final", "balanced_accuracy_val_final", "f1_macro_val_final",
    "mcc_val_final", "early_stopping", "validacoes_sem_melhoria",
    "tempo_segundos", "pasta_resultados", "erro", "codigo_saida",
    "situacao_execucao", "pasta_origem", "epocas_origem",
    "epocas_adicionais", "estado_otimizador_retomado",
    "paciencia_atingida", "limite_epocas_atingido", "limite_epocas", "motivo_parada",
    "numero_continuacoes", "tempo_total_segundos",
)


def _agora():
    return datetime.now().isoformat(timespec="seconds")


def _gravar_json(caminho, dados):
    """Atualiza um JSON sem deixar arquivos parcialmente gravados."""

    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(caminho.name + ".tmp")
    with temporario.open("w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporario, caminho)


def _gravar_csv(caminho, registros, colunas):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(caminho.name + ".tmp")
    with temporario.open("w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=colunas, extrasaction="ignore")
        escritor.writeheader()
        for registro in registros:
            escritor.writerow(registro)
    os.replace(temporario, caminho)


def _rotulo_float(valor):
    return format(float(valor), ".0e").replace("+", "")


def _identificadores(experimento, tentativa):
    return (f"id{experimento['id']:03d}", f"t{tentativa:02d}")


def _localizar_pasta(raiz, experimento, tentativa):
    """Encontra a pasta criada pelo TreinoProcessor para esta tentativa."""

    identificador, tag_tentativa = _identificadores(experimento, tentativa)
    candidatas = [
        pasta for pasta in Path(raiz).iterdir()
        if pasta.is_dir() and f"_{identificador}_{tag_tentativa}_" in pasta.name
    ]
    return max(candidatas, key=lambda p: p.stat().st_mtime) if candidatas else None


def _garantir_pasta_falha(raiz, experimento, tentativa):
    pasta = _localizar_pasta(raiz, experimento, tentativa)
    if pasta:
        return pasta

    from pdc_ia_image_defect_detection.utils.resultados import criar_pasta_resultados
    id_experimento, tag_tentativa = _identificadores(experimento, tentativa)
    return criar_pasta_resultados(
        raiz=raiz,
        tipo="treino",
        inicio=datetime.now(),
        # Nome curto evita MAX_PATH no Windows; parametros ficam nos JSON/CSV.
        detalhes=(id_experimento, tag_tentativa),
    )


def _resumo_historico(historico):
    validacoes = [
        linha for linha in historico
        if math.isfinite(float(linha["val_loss"]))
        and math.isfinite(float(linha["val_acc"]))
    ]
    if not validacoes:
        return {}
    maior_acc = max(validacoes, key=lambda r: float(r["val_acc"]))
    melhor_loss = min(validacoes, key=lambda r: float(r["val_loss"]))
    ultima = validacoes[-1]
    return {
        "maior_val_acc": float(maior_acc["val_acc"]),
        "epoca_maior_val_acc": int(maior_acc["epoca"]),
        "val_acc_na_melhor_loss": float(melhor_loss["val_acc"]),
        "val_loss_final": float(ultima["val_loss"]),
    }


def _executar_trabalho(arquivo_configuracao):
    """Executa um treinamento e gera o relatorio individual do subprocesso."""

    config = json.loads(Path(arquivo_configuracao).read_text(encoding="utf-8"))
    experimento = config["experimento"]
    tentativa = config["tentativa"]
    raiz_resultados = Path(config["raiz_resultados"])
    relatorio_caminho = Path(config["relatorio_caminho"])
    inicio = time.perf_counter()
    modelo = str(config.get("modelo", "lenet5")).lower()
    parametros_modelo = {
        "neuronios_fc1": int(experimento["neuronios_fc1"]),
        "neuronios_fc2": int(experimento["neuronios_fc2"]),
    }
    if modelo == "lenet5":
        parametros_modelo["kernel_size"] = int(experimento["kernel"])
    elif modelo == "vgg16":
        parametros_modelo["dropout"] = float(experimento["dropout"])
    else:
        raise ValueError(f"Modelo nao suportado no Grid Search: {modelo}")
    ident, tag = _identificadores(experimento, tentativa)
    # Identificador unico por tentativa; hiperparametros registrados em JSON e CSV.
    detalhes = (ident, tag)
    relatorio = {
        **experimento,
        "modelo": modelo,
        "tentativa": tentativa,
        "seed": config["seed"],
        "status": "erro",
        "inicio": _agora(),
        "situacao_execucao": "continuado" if config.get("continuar_de") else "novo",
        "pasta_origem": config.get("continuar_de"),
    }

    try:
        from pdc_ia_image_defect_detection.pre_processing.treino_class import TreinoProcessor
        bases = {
            nome: {
                "ativa": dados["ativa"],
                "train": Path(dados["train"]),
                "val": Path(dados["val"]),
            }
            for nome, dados in config["bases"].items()
        }
        processador = TreinoProcessor(
            nome_modelo=modelo, bases=bases,
            pasta_resultados=raiz_resultados,
            num_epochs=config["num_epochs"],
            batch_size=config["batch_size"],
            learning_rate=experimento["learning_rate"],
            intervalo_validacao=config["intervalo_validacao"],
            paciencia=config["paciencia"],
            seed=config["seed"],
            num_workers=config["num_workers"],
            usar_cuda=config["usar_cuda"],
            balancear_origens=config["balancear_origens"],
            usar_augmentation=config["usar_augmentation"],
            parametros_modelo=parametros_modelo,
            detalhes_resultado=detalhes,
            salvar_historico_incremental=True,
            continuar_de=config.get("continuar_de"),
            nome_curto_grid=True,
        )
        resultado = processador.executar()
        pasta = Path(resultado["pasta_resultados"])
        metricas = resultado["metricas_finais"]
        validacao = metricas["validacao_final"]
        treino = metricas["treino_final"]
        relatorio.update({
            "status": "concluido",
            "pasta_resultados": str(pasta),
            "epocas_executadas": resultado["epocas_executadas"],
            "melhor_epoca": resultado["melhor_epoca"],
            "melhor_val_loss": resultado["melhor_val_loss"],
            "train_acc_final": treino["accuracy"],
            "val_acc_final": validacao["accuracy"],
            "balanced_accuracy_val_final": validacao["balanced_accuracy"],
            "f1_macro_val_final": validacao["f1_macro"],
            "mcc_val_final": validacao["mcc"],
            "early_stopping": resultado["interrompido_por_paciencia"],
            "paciencia_atingida": resultado["paciencia_atingida"],
            "limite_epocas_atingido": resultado["limite_epocas_atingido"],
            "limite_epocas": resultado["limite_epocas"],
            "motivo_parada": resultado["motivo_parada"],
            "validacoes_sem_melhoria": resultado["validacoes_sem_melhoria"],
            "situacao_execucao": "continuado" if config.get("continuar_de") else "novo",
            "pasta_origem": config.get("continuar_de"),
            "epocas_origem": int(processador.epocas_origem),
            "epocas_adicionais": resultado["epocas_executadas"] - processador.epocas_origem,
            "estado_otimizador_retomado": processador.estado_otimizador_retomado,
            **_resumo_historico(resultado["historico"]),
        })
        _gravar_json(pasta / "configuracao_experimento.json", config)
        _gravar_json(pasta / "estado_treinamento.json", relatorio)
    except BaseException as erro:
        relatorio["erro"] = f"{type(erro).__name__}: {erro}"
        relatorio["traceback"] = traceback.format_exc()
        # Exibe a falha no terminal, alem de registra-la na pasta do experimento.
        print(f"\nERRO EXPERIMENTO {experimento['id']}: {relatorio['erro']}", flush=True)
        print(relatorio["traceback"], flush=True)
        try:
            pasta = _garantir_pasta_falha(raiz_resultados, experimento, tentativa)
            relatorio["pasta_resultados"] = str(pasta)
            (pasta / "erro_treinamento.txt").write_text(
                relatorio["traceback"], encoding="utf-8"
            )
            _gravar_json(pasta / "configuracao_experimento.json", config)
        except Exception as erro_salvamento:
            relatorio["erro_salvamento"] = str(erro_salvamento)
    finally:
        relatorio["tempo_segundos"] = round(time.perf_counter() - inicio, 3)
        relatorio["fim"] = _agora()
        _gravar_json(relatorio_caminho, relatorio)

    return 0 if relatorio["status"] == "concluido" else 1


class GridSearchProcessor:
    """Executa buscas novas ou prolonga somente treinamentos sem early stopping."""

    def __init__(
        self, raiz_projeto, bases, learning_rates, kernels, pares_neuronios,
        num_epochs, batch_size, intervalo_validacao, paciencia, seed,
        num_workers, usar_cuda, balancear_origens, usar_augmentation,
        continuar_treinos=False, pasta_grid_continuar="", epocas_extras=100,
        retomar_execucao=False, pasta_execucao_retomar="", reexecutar_falhas=False,
        modelo="lenet5", dropouts=(),
    ):
        self.raiz = Path(raiz_projeto).resolve()
        self.modelo = str(modelo).strip().lower()
        if self.modelo not in ("lenet5", "vgg16"):
            raise ValueError("MODELO_GRID deve ser lenet5 ou vgg16.")
        self.raiz_pai = self.raiz / "resultados"
        self.continuar_treinos = bool(continuar_treinos)
        self.retomar_execucao = bool(retomar_execucao)
        self.pasta_grid_continuar = str(pasta_grid_continuar or "").strip()
        self.pasta_execucao_retomar = str(pasta_execucao_retomar or "").strip()
        self.epocas_extras = int(epocas_extras)
        self.reexecutar_falhas = bool(reexecutar_falhas)
        if self.continuar_treinos and self.retomar_execucao:
            raise ValueError("CONTINUAR_TREINOS e RETOMAR_EXECUCAO nao podem ser True juntos.")
        self.raiz_resultados = None
        self.comparados = None
        self.inicio_execucao = None
        self.detalhes_execucao = ()
        self.fonte_raiz = None
        self.fonte_estado = None
        self.fonte_limite = None
        self.ids_continuacao = set()
        self.limites_experimento = {}
        self.continuacao_ativa = None

        base_config = {
            "modelo": self.modelo,
            "learning_rates": list(learning_rates),
            "pares_neuronios": [list(par) for par in pares_neuronios],
            **({"kernels": list(kernels)} if self.modelo == "lenet5" else
               {"dropouts": list(dropouts)}),
            "bases": {
                nome: {
                    "ativa": bool(dados["ativa"]),
                    "train": str(Path(dados["train"]).resolve()),
                    "val": str(Path(dados["val"]).resolve()),
                } for nome, dados in bases.items()
            },
            "num_epochs": int(num_epochs),
            "batch_size": int(batch_size),
            "intervalo_validacao": int(intervalo_validacao),
            "paciencia": None if paciencia is None else int(paciencia),
            "seed": int(seed),
            "num_workers": int(num_workers),
            "usar_cuda": bool(usar_cuda),
            "balancear_origens": bool(balancear_origens),
            "usar_augmentation": bool(usar_augmentation),
        }

        if self.continuar_treinos:
            if self.epocas_extras <= 0:
                raise ValueError("EPOCAS_EXTRAS deve ser maior que zero.")
            self.fonte_raiz = self._resolver_pasta(self.pasta_grid_continuar)
            config_fonte = json.loads(
                (self.fonte_raiz / "resultados_comparados" /
                 "configuracoes_grid_search.json").read_text(encoding="utf-8")
            )
            self.fonte_estado = json.loads(
                (self.fonte_raiz / "resultados_comparados" /
                 "progresso_grid_search.json").read_text(encoding="utf-8")
            )
            if config_fonte.get("modelo", "lenet5") != self.modelo:
                raise ValueError("A pasta de origem pertence a outro modelo.")
            self.fonte_limite = int(config_fonte["num_epochs"])
            # A continuacao sempre utiliza todos os parametros efetivos da origem.
            self.config = copy.deepcopy(config_fonte)
            # Configuracao-base nao e alterada: cada experimento pode ter limite proprio.
        else:
            self.config = base_config
            self.config["modo_grid"] = "novo"

        if self.modelo == "lenet5":
            self.experimentos = [
                {"id": i, "learning_rate": float(lr), "kernel": int(kernel),
                 "neuronios_fc1": int(par[0]), "neuronios_fc2": int(par[1])}
                for i, (kernel, par, lr) in enumerate(
                    product(self.config["kernels"], self.config["pares_neuronios"],
                            self.config["learning_rates"]), start=1
                )
            ]
        else:
            self.experimentos = [
                {"id": i, "learning_rate": float(lr), "dropout": float(dropout),
                 "neuronios_fc1": int(par[0]), "neuronios_fc2": int(par[1])}
                for i, (par, dropout, lr) in enumerate(
                    product(self.config["pares_neuronios"], self.config["dropouts"],
                            self.config["learning_rates"]), start=1
                )
            ]
        self._validar()
        if self.continuar_treinos:
            self._selecionar_continuacoes()

    def _resolver_pasta(self, nome):
        if not nome:
            raise ValueError("Informe o nome/caminho da pasta de origem.")
        caminho = Path(nome)
        if not caminho.is_absolute():
            caminho = self.raiz_pai / caminho
        caminho = caminho.resolve()
        if not caminho.is_dir():
            raise FileNotFoundError(f"Pasta de resultados nao encontrada: {caminho}")
        return caminho

    def _validar(self):
        if not self.experimentos:
            raise ValueError("O Grid Search nao possui combinacoes.")
        if self.modelo == "lenet5":
            chaves = [(e["kernel"], e["neuronios_fc1"], e["neuronios_fc2"],
                       e["learning_rate"]) for e in self.experimentos]
            if any(min(*valores) <= 0 for valores in chaves):
                raise ValueError("Kernel, neuronios e learning rate devem ser positivos.")
        else:
            chaves = [(e["neuronios_fc1"], e["neuronios_fc2"], e["dropout"],
                       e["learning_rate"]) for e in self.experimentos]
            if any(e["neuronios_fc1"] <= 0 or e["neuronios_fc2"] <= 0
                   or e["learning_rate"] <= 0 or not 0 <= e["dropout"] < 1
                   for e in self.experimentos):
                raise ValueError("Neuronios/LR devem ser positivos e dropout em [0, 1).")
        if len(set(chaves)) != len(chaves):
            raise ValueError("O Grid Search possui combinacoes duplicadas.")
        if self.config['num_epochs'] <= 0 or self.config['batch_size'] <= 0:
            raise ValueError("Epocas e batch size devem ser positivos.")
        if self.config['intervalo_validacao'] <= 0:
            raise ValueError("Intervalo de validacao deve ser positivo.")
        if self.config['paciencia'] is not None and self.config['paciencia'] <= 0:
            raise ValueError("Paciencia deve ser positiva ou None.")
        if self.config['num_workers'] < 0:
            raise ValueError("NUM_WORKERS nao pode ser negativo.")
        if not any(base['ativa'] for base in self.config['bases'].values()):
            raise ValueError("Nenhuma base foi ativada.")
        for nome, base in self.config['bases'].items():
            if base['ativa']:
                for particao in ('train', 'val'):
                    if not Path(base[particao]).is_dir():
                        raise FileNotFoundError(f"{nome}/{particao}: {base[particao]}")

    def _selecionar_continuacoes(self):
        """Seleciona pelo estado real de cada treino, nao pelo limite global."""
        ativa = self.fonte_estado.get("continuacao_ativa")
        if ativa and ativa.get("status") == "em_andamento":
            self.continuacao_ativa = ativa
            self.limites_experimento = {
                int(k): int(v) for k, v in ativa["alvos"].items()
            }
            self.ids_continuacao = set(self.limites_experimento)
            return

        for experimento in self.experimentos:
            id_ = str(experimento["id"])
            anterior = self.fonte_estado.get("resultados", {}).get(id_, {})
            if not all(anterior.get(k) == v for k, v in experimento.items()):
                continue
            if anterior.get("status") != "concluido":
                continue
            pasta = Path(anterior.get("pasta_resultados") or "/caminho/ausente")
            if not all((pasta / nome).is_file() for nome in (
                "modelo_final.pt", "modelo_melhor.pt", "historico_treinamento.csv"
            )):
                continue
            # Estado individual prevalece sobre metadados de rodadas antigas.
            arquivo = pasta / "estado_treinamento.json"
            individual = json.loads(arquivo.read_text(encoding="utf-8")) if arquivo.exists() else anterior
            if bool(individual.get("paciencia_atingida", individual.get("early_stopping"))):
                continue
            epocas = int(individual.get("epocas_executadas") or 0)
            limite = int(individual.get("limite_epocas") or self.fonte_limite)
            if epocas <= 0 or epocas != limite:
                continue
            self.ids_continuacao.add(experimento["id"])
            self.limites_experimento[experimento["id"]] = epocas + self.epocas_extras

        # Nao ha modificacao das pastas ao simplesmente selecionar experimentos.

    def _novo_estado(self):
        return {
            "inicio": _agora(), "total": len(self.experimentos),
            "resultados": {}, "continuacoes": [], "continuacao_ativa": None,
        }

    def _preparar_pastas(self):
        """Busca nova cria pasta; continuacao modifica somente a pasta indicada."""
        from pdc_ia_image_defect_detection.utils.resultados import criar_pasta_resultados
        if self.continuar_treinos:
            self.raiz_resultados = self.fonte_raiz
            self.comparados = self.raiz_resultados / "resultados_comparados"
            self.arquivo_estado = self.comparados / "progresso_grid_search.json"
            self.arquivo_config = self.comparados / "configuracoes_grid_search.json"
            estado = json.loads(self.arquivo_estado.read_text(encoding="utf-8"))
            if self.continuacao_ativa:
                # Os dois objetos devem apontar para o mesmo registro mutavel.
                self.continuacao_ativa = estado["continuacao_ativa"]
            if not self.continuacao_ativa and self.ids_continuacao:
                numero = len(estado.get("continuacoes", [])) + 1
                self.continuacao_ativa = {
                    "numero": numero, "inicio": _agora(),
                    "epocas_extras": self.epocas_extras,
                    "alvos": {str(k): v for k, v in self.limites_experimento.items()},
                    "status": "em_andamento", "resultados": {},
                }
                estado["continuacao_ativa"] = self.continuacao_ativa
                _gravar_json(self.arquivo_estado, estado)
            return estado

        if self.retomar_execucao:
            pasta = self._resolver_pasta(self.pasta_execucao_retomar)
            arq = pasta / "resultados_comparados" / "configuracoes_grid_search.json"
            anterior = json.loads(arq.read_text(encoding="utf-8"))
            if anterior != self.config:
                raise ValueError("Configuracao nao coincide com a execucao interrompida.")
            self.raiz_resultados = pasta
            self.inicio_execucao = datetime.fromisoformat(
                json.loads((pasta / "resultados_comparados" /
                            "progresso_grid_search.json").read_text(encoding="utf-8"))["inicio"]
            )
            nome = pasta.name
            sufixo = nome.split("_inicio_")[0]
            prefixo = f"resultados_grid_{'lenet' if self.modelo == 'lenet5' else 'vgg16'}"
            token = sufixo.removeprefix(prefixo)
            self.detalhes_execucao = tuple(filter(None, token.split("_")))
        else:
            self.inicio_execucao = datetime.now()
            for indice in range(1, 1000):
                detalhes = () if indice == 1 else (f"r{indice:03d}",)
                try:
                    self.raiz_resultados = criar_pasta_resultados(
                        raiz=self.raiz_pai, tipo="resultados",
                        inicio=self.inicio_execucao, detalhes=("grid", "lenet" if self.modelo == "lenet5" else "vgg16", *detalhes),
                    )
                    self.detalhes_execucao = detalhes
                    break
                except FileExistsError:
                    continue
            if self.raiz_resultados is None:
                raise FileExistsError("Nao foi possivel criar pasta exclusiva para a busca.")
        self.comparados = self.raiz_resultados / "resultados_comparados"
        self.comparados.mkdir(parents=True, exist_ok=True)
        self.arquivo_estado = self.comparados / "progresso_grid_search.json"
        self.arquivo_config = self.comparados / "configuracoes_grid_search.json"
        if self.retomar_execucao:
            return json.loads(self.arquivo_estado.read_text(encoding="utf-8"))
        estado = self._novo_estado()
        estado["inicio"] = self.inicio_execucao.isoformat(timespec="seconds")
        _gravar_json(self.arquivo_config, self.config)
        _gravar_json(self.arquivo_estado, estado)
        return estado

    def _comparar(self, estado, gerar_graficos=False):
        linhas = []
        for experimento in self.experimentos:
            registro = estado['resultados'].get(str(experimento['id']), {})
            linhas.append({**experimento, "modelo": self.modelo,
                           "status": "pendente", "seed": self.config['seed'], **registro})
        _gravar_csv(self.comparados / 'resumo_geral.csv', linhas, COLUNAS_RESUMO)
        validos = [linha for linha in linhas if linha['status'] == 'concluido'
                   and linha.get('melhor_val_loss') is not None]
        ranking = sorted(validos, key=lambda r: (float(r['melhor_val_loss']),
                            -float(r.get('maior_val_acc', 0))))
        _gravar_csv(self.comparados / 'ranking.csv',
                    [dict(linha, ranking=i) for i, linha in enumerate(ranking, 1)],
                    ('ranking', *COLUNAS_RESUMO))
        _gravar_csv(self.comparados / 'ranking_por_accuracy.csv',
                    [dict(linha, ranking=i) for i, linha in enumerate(
                        sorted(validos, key=lambda r: (-float(r.get('maior_val_acc', 0)),
                              float(r['melhor_val_loss']))), 1)],
                    ('ranking', *COLUNAS_RESUMO))
        self._consolidar_historicos(linhas)
        if gerar_graficos and validos:
            self._graficos(ranking)

    def _consolidar_historicos(self, linhas):
        registros = []
        for linha in linhas:
            pasta = linha.get('pasta_resultados')
            if not pasta:
                continue
            historico = Path(pasta) / 'historico_treinamento.csv'
            if not historico.is_file():
                continue
            with historico.open('r', encoding='utf-8-sig', newline='') as arquivo:
                for registro in csv.DictReader(arquivo):
                    registros.append({"id": linha['id'], "tentativa": linha.get('tentativa'),
                                      "modelo": self.modelo,
                                      "kernel": linha.get('kernel', ''),
                                      "dropout": linha.get('dropout', ''),
                                      "neuronios_fc1": linha['neuronios_fc1'],
                                      "neuronios_fc2": linha['neuronios_fc2'],
                                      "learning_rate": linha['learning_rate'], **registro})
        _gravar_csv(self.comparados / 'historicos_consolidados.csv', registros,
                    ('id', 'tentativa', 'modelo', 'kernel', 'dropout', 'neuronios_fc1', 'neuronios_fc2',
                     'learning_rate', 'epoca', 'train_loss', 'train_acc', 'val_loss', 'val_acc'))

    def _graficos(self, ranking):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        top = ranking[:min(20, len(ranking))]
        itens = (
            ('melhor_val_loss', 'Menor val_loss', 'comparacao_losses.png'),
            ('maior_val_acc', 'Maior val_accuracy', 'comparacao_acuracias.png'),
            ('epocas_executadas', 'Epocas executadas', 'comparacao_epocas.png'),
        )
        for campo, titulo, arquivo in itens:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar([f"{int(r['id']):03d}" for r in top], [float(r[campo]) for r in top])
            ax.set_title(f"{titulo} - 20 melhores por val_loss")
            ax.set_xlabel('Experimento')
            ax.set_ylabel(titulo)
            ax.grid(axis='y', alpha=0.25)
            fig.tight_layout()
            fig.savefig(self.comparados / arquivo, dpi=200)
            plt.close(fig)

    def _subprocesso(self, config_trabalho, log):
        modulo = 'pdc_ia_image_defect_detection.pre_processing.grid_search_class'
        ambiente = os.environ.copy()
        src = str(Path(__file__).resolve().parents[2])
        ambiente['PYTHONPATH'] = os.pathsep.join(filter(None, (src, ambiente.get('PYTHONPATH'))))
        ambiente['PYTHONDONTWRITEBYTECODE'] = '1'
        comando = [sys.executable, '-u', '-m', modulo, '--worker', str(config_trabalho)]
        with Path(log).open('w', encoding='utf-8') as arquivo:
            processo = subprocess.Popen(comando, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True,
                                        encoding='utf-8', errors='replace',
                                        bufsize=1, cwd=self.raiz, env=ambiente)
            for linha in processo.stdout:
                print(linha, end='', flush=True)
                arquivo.write(linha)
                arquivo.flush()
            return processo.wait()

    @staticmethod
    def _trocar_prefixo(dado, origem, destino):
        if isinstance(dado, str):
            return destino + dado[len(origem):] if dado.startswith(origem) else dado
        if isinstance(dado, dict):
            return {k: GridSearchProcessor._trocar_prefixo(v, origem, destino)
                    for k, v in dado.items()}
        if isinstance(dado, list):
            return [GridSearchProcessor._trocar_prefixo(v, origem, destino) for v in dado]
        return dado

    def _recuperar_troca(self, pasta_original):
        """Recupera pasta antiga apos interrupcao durante promocao."""
        backup = pasta_original.parent / (".backup_" + pasta_original.name)
        if backup.exists() and not pasta_original.exists():
            backup.rename(pasta_original)
        elif backup.exists() and pasta_original.exists():
            # Pasta atual ja promovida; arquivamento anterior deve existir.
            historicos = pasta_original / "historico_continuacoes"
            if historicos.is_dir() and any(historicos.glob("*.zip")):
                shutil.rmtree(backup)
            else:
                raise RuntimeError(f"Backup residual sem arquivo historico: {backup}")

    def _promover_continuacao(self, pasta_nova, pasta_original):
        """Atualiza a pasta original apos arquivar o resultado anterior."""
        pasta_nova = Path(pasta_nova).resolve()
        pasta_original = Path(pasta_original).resolve()
        if pasta_nova == pasta_original or pasta_nova.parent != pasta_original.parent:
            raise ValueError("Pastas de continuacao incompativeis.")
        self._recuperar_troca(pasta_original)
        if not pasta_original.is_dir() or not pasta_nova.is_dir():
            raise FileNotFoundError("Pasta anterior ou nova pasta de resultados ausente.")
        for nome in ("modelo_final.pt", "modelo_melhor.pt", "historico_treinamento.csv",
                     "estado_treinamento.json"):
            if not (pasta_nova / nome).is_file():
                raise FileNotFoundError(f"Resultado novo incompleto: {nome}")

        historico = pasta_nova / "historico_continuacoes"
        historico.mkdir(exist_ok=True)
        anterior = pasta_original / "historico_continuacoes"
        if anterior.is_dir():
            shutil.copytree(anterior, historico, dirs_exist_ok=True)
        estado_anterior = pasta_original / "estado_treinamento.json"
        dados_antigos = json.loads(estado_anterior.read_text(encoding="utf-8")) if estado_anterior.exists() else {}
        epoca = int(dados_antigos.get("epocas_executadas") or 0)
        # Nome curto: o padrao longo atingia 260 caracteres no Windows.
        n = len(list(historico.glob("*.zip"))) + 1
        arquivo_backup = historico / f"c{n:03d}.zip"
        with zipfile.ZipFile(arquivo_backup, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for caminho in pasta_original.rglob("*"):
                if caminho.is_file() and anterior not in caminho.parents:
                    z.write(caminho, arcname=caminho.relative_to(pasta_original).as_posix())
        with zipfile.ZipFile(arquivo_backup) as z:
            if z.testzip() is not None:
                raise RuntimeError("Falha de integridade no backup da continuacao.")

        backup = pasta_original.parent / (".backup_" + pasta_original.name)
        if backup.exists():
            raise FileExistsError(f"Backup temporario pendente: {backup}")
        pasta_original.rename(backup)
        try:
            pasta_nova.rename(pasta_original)
        except BaseException:
            if not pasta_original.exists():
                backup.rename(pasta_original)
            raise
        try:
            shutil.rmtree(backup)
        except OSError as erro:
            print(f"AVISO: backup residual em {backup}: {erro}", flush=True)
        return pasta_original

    def _localizar_resultado_pendente(self, experimento, pasta_original, alvo):
        """Encontra treino concluido cuja promocao para a pasta original falhou."""
        id_tag = f"_id{experimento['id']:03d}_t"
        pasta_original = Path(pasta_original).resolve()
        epoca_original = 0
        estado_original = pasta_original / "estado_treinamento.json"
        if estado_original.is_file():
            epoca_original = int(json.loads(estado_original.read_text(encoding="utf-8"))
                                   .get("epocas_executadas") or 0)
        candidatos = []
        for pasta in self.raiz_resultados.iterdir():
            if not pasta.is_dir() or pasta.resolve() == pasta_original or id_tag not in pasta.name:
                continue
            arquivo_config = pasta / "configuracao_experimento.json"
            arquivo_estado = pasta / "estado_treinamento.json"
            if not arquivo_config.is_file() or not arquivo_estado.is_file():
                continue
            if not all((pasta / arquivo).is_file() for arquivo in (
                "modelo_final.pt", "modelo_melhor.pt", "historico_treinamento.csv"
            )):
                continue
            try:
                config = json.loads(arquivo_config.read_text(encoding="utf-8"))
                estado = json.loads(arquivo_estado.read_text(encoding="utf-8"))
                origem_config = Path(config.get("continuar_de") or "").resolve()
                if (origem_config != pasta_original
                        or int(config["num_epochs"]) != int(alvo)
                        or config.get("experimento") != experimento
                        or estado.get("status") != "concluido"):
                    continue
                epocas = int(estado.get("epocas_executadas") or 0)
                if epocas <= epoca_original or epocas > int(alvo):
                    continue
                if not (bool(estado.get("paciencia_atingida")) or epocas == int(alvo)):
                    continue
                candidatos.append((epocas, pasta.stat().st_mtime, pasta, estado))
            except (OSError, KeyError, ValueError, TypeError):
                continue
        if not candidatos:
            return None
        # Prefere o treino mais avancado; evita reaproveitar uma tentativa incompleta.
        _, _, pasta, estado = max(candidatos, key=lambda item: (item[0], item[1]))
        return pasta, estado

    def _concluir_promocao(self, relatorio, origem, anterior, numero, arquivo_relatorio=None):
        """Promove e registra uma continuacao mantendo a pasta original."""
        pasta_atual = self._promover_continuacao(relatorio["pasta_resultados"], origem)
        atualizado = dict(relatorio)
        atualizado.update({
            "status": "concluido",
            "pasta_resultados": str(pasta_atual),
            "pasta_origem": str(pasta_atual),
            "numero_continuacoes": numero,
            "tempo_total_segundos": round(
                float(anterior.get("tempo_total_segundos") or
                      anterior.get("tempo_segundos") or 0)
                + float(relatorio.get("tempo_segundos") or 0), 3
            ),
        })
        _gravar_json(pasta_atual / "estado_treinamento.json", atualizado)
        if arquivo_relatorio is not None:
            _gravar_json(arquivo_relatorio, atualizado)
        return atualizado

    def _finalizar_execucao(self, estado):
        """Finaliza busca nova; continuacao preserva a pasta da origem."""
        if self.continuar_treinos:
            ativa = estado.get("continuacao_ativa")
            if ativa:
                ativa["status"] = "finalizado"
                ativa["fim"] = _agora()
                estado.setdefault("continuacoes", []).append(copy.deepcopy(ativa))
                estado["continuacao_ativa"] = None
            estado["atualizado_em"] = _agora()
            _gravar_json(self.arquivo_estado, estado)
            self._comparar(estado, gerar_graficos=True)
            return estado
        from pdc_ia_image_defect_detection.utils.resultados import finalizar_pasta_resultados
        anterior = self.raiz_resultados
        fim = datetime.now()
        nova = finalizar_pasta_resultados(
            pasta_atual=anterior, tipo='resultados',
            inicio=self.inicio_execucao, fim=fim, detalhes=('grid', 'lenet' if self.modelo == 'lenet5' else 'vgg16', *self.detalhes_execucao),
        )
        self.raiz_resultados = nova
        self.comparados = nova / 'resultados_comparados'
        self.arquivo_estado = self.comparados / 'progresso_grid_search.json'
        self.arquivo_config = self.comparados / 'configuracoes_grid_search.json'
        estado['fim'] = fim.isoformat(timespec='seconds')
        antigo = str(anterior)
        novo = str(nova)
        estado = self._trocar_prefixo(estado, antigo, novo)
        avisos = []
        for caminho in nova.rglob('*.json'):
            if caminho == self.arquivo_estado:
                continue
            try:
                dados = json.loads(caminho.read_text(encoding='utf-8'))
                dados_corrigidos = self._trocar_prefixo(dados, antigo, novo)
                if dados != dados_corrigidos:
                    _gravar_json(caminho, dados_corrigidos)
            except (OSError, ValueError) as erro:
                # Um JSON auxiliar inacessivel nao deve anular todos os treinos.
                avisos.append(f"{caminho}: {type(erro).__name__}: {erro}")
        if avisos:
            arquivo_avisos = self.comparados / 'avisos_finalizacao.txt'
            arquivo_avisos.write_text('\n'.join(avisos), encoding='utf-8')
            print(f"AVISO: {len(avisos)} arquivo(s) JSON auxiliar(es) nao atualizado(s). "
                  f"Detalhes: {arquivo_avisos}", flush=True)
        _gravar_json(self.arquivo_estado, estado)
        self._comparar(estado)
        return estado

    def executar(self):
        estado = self._preparar_pastas()
        pasta_configs = self.comparados / "configuracoes_experimentos"
        pasta_logs = self.comparados / "logs"
        pasta_relatorios = self.comparados / "relatorios_individuais"
        for pasta in (pasta_configs, pasta_logs, pasta_relatorios):
            pasta.mkdir(parents=True, exist_ok=True)
        print(f"\nGRID {self.modelo.upper()}: {len(self.experimentos)} combinacoes")
        print(f"Modo: {'continuacao' if self.continuar_treinos else 'novo'}")
        print(f"Resultados: {self.raiz_resultados}")
        if self.continuar_treinos:
            print(f"Elegiveis: {len(self.ids_continuacao)} | epocas extras: {self.epocas_extras}")
        self._comparar(estado)

        for experimento in self.experimentos:
            indice = str(experimento["id"])
            anterior = estado["resultados"].get(indice, {})
            if self.continuar_treinos:
                if experimento["id"] not in self.ids_continuacao:
                    continue
                origem = Path(anterior.get("pasta_resultados") or "")
                self._recuperar_troca(origem)
                alvo = self.limites_experimento[experimento["id"]]
                estado_individual = origem / "estado_treinamento.json"
                if estado_individual.exists():
                    atual = json.loads(estado_individual.read_text(encoding="utf-8"))
                    if (atual.get("status") == "concluido"
                            and int(atual.get("epocas_executadas") or 0) >= alvo):
                        # Reconciliacao de execucao interrompida apos promocao.
                        atual["pasta_resultados"] = str(origem)
                        atual["pasta_origem"] = str(origem)
                        _gravar_json(estado_individual, atual)
                        estado["resultados"][indice] = atual
                        if self.continuacao_ativa:
                            self.continuacao_ativa["resultados"][indice] = "concluido"
                        _gravar_json(self.arquivo_estado, estado)
                        self._comparar(estado)
                        continue
                # O treino pode ter terminado, mas a promocao anterior falhado.
                # Recupera o checkpoint novo ANTES de iniciar outro treinamento.
                pendente = self._localizar_resultado_pendente(experimento, origem, alvo)
                if pendente is not None:
                    pasta_pendente, relatorio_pendente = pendente
                    try:
                        relatorio_recuperado = self._concluir_promocao(
                            {**relatorio_pendente, "pasta_resultados": str(pasta_pendente)},
                            origem, anterior, self.continuacao_ativa["numero"],
                        )
                        estado["resultados"][indice] = relatorio_recuperado
                        self.continuacao_ativa["resultados"][indice] = "concluido"
                        self.continuacao_ativa.setdefault("falhas", {}).pop(indice, None)
                        _gravar_json(self.arquivo_estado, estado)
                        self._comparar(estado)
                        print(f"Experimento {indice}: resultado recuperado da epoca "
                              f"{relatorio_recuperado['epocas_executadas']} "
                              "sem repetir treinamento.", flush=True)
                        continue
                    except Exception as erro:
                        print(f"AVISO: promocao pendente do experimento {indice} "
                              f"ainda falhou: {type(erro).__name__}: {erro}", flush=True)
                        # Nao repete um treino concluido por falha de movimentacao.
                        self.continuacao_ativa["resultados"][indice] = "erro"
                        self.continuacao_ativa.setdefault("falhas", {})[indice] = str(erro)
                        _gravar_json(self.arquivo_estado, estado)
                        continue
                if (self.continuacao_ativa
                        and self.continuacao_ativa["resultados"].get(indice) == "concluido"):
                    continue
                if (self.continuacao_ativa
                        and self.continuacao_ativa["resultados"].get(indice) == "erro"
                        and not self.reexecutar_falhas):
                    continue
            else:
                if anterior.get("status") == "concluido":
                    pasta = Path(anterior.get("pasta_resultados") or "/caminho/ausente")
                    if (pasta / "modelo_melhor.pt").exists() and (pasta / "modelo_final.pt").exists():
                        print(f"Experimento {indice}: ja concluido.")
                        continue
                if anterior.get("status") == "erro" and not self.reexecutar_falhas:
                    continue

            tentativa = int(anterior.get("tentativa") or 0) + 1
            ident, tag = _identificadores(experimento, tentativa)
            nome = f"{ident}_{tag}"
            if self.continuar_treinos:
                nome = f"c{self.continuacao_ativa['numero']:03d}_{nome}"
            arquivo_config = pasta_configs / f"{nome}.json"
            arquivo_relatorio = pasta_relatorios / f"{nome}.json"
            arquivo_log = pasta_logs / f"{nome}.log"
            trabalho = {
                **self.config, "experimento": experimento, "tentativa": tentativa,
                "raiz_resultados": str(self.raiz_resultados),
                "relatorio_caminho": str(arquivo_relatorio),
            }
            if self.continuar_treinos:
                trabalho["continuar_de"] = str(origem)
                trabalho["num_epochs"] = alvo
            _gravar_json(arquivo_config, trabalho)
            if not self.continuar_treinos:
                estado["resultados"][indice] = {
                    **experimento, "status": "executando", "tentativa": tentativa,
                    "seed": self.config["seed"], "inicio": _agora(),
                    "situacao_execucao": "novo",
                }
            if self.continuacao_ativa:
                self.continuacao_ativa["resultados"][indice] = "executando"
            _gravar_json(self.arquivo_estado, estado)
            variavel = (f"kernel={experimento['kernel']}" if self.modelo == "lenet5"
                        else f"dropout={experimento['dropout']}")
            print(f"\nEXPERIMENTO {indice}/{len(self.experimentos)} - tentativa {tentativa} "
                  f"{self.modelo} {variavel} "
                  f"fc={experimento['neuronios_fc1']}/{experimento['neuronios_fc2']} "
                  f"lr={experimento['learning_rate']}", flush=True)
            inicio = time.perf_counter()
            try:
                codigo = self._subprocesso(arquivo_config, arquivo_log)
            except Exception as erro:
                codigo = -1
                falha_subprocesso = f"{type(erro).__name__}: {erro}"
            else:
                falha_subprocesso = None
            if arquivo_relatorio.is_file():
                relatorio = json.loads(arquivo_relatorio.read_text(encoding="utf-8"))
            else:
                pasta = _garantir_pasta_falha(self.raiz_resultados, experimento, tentativa)
                erro = falha_subprocesso or f"Subprocesso sem relatorio (codigo {codigo})."
                (pasta / "erro_treinamento.txt").write_text(erro, encoding="utf-8")
                relatorio = {**experimento, "modelo": self.modelo,
                             "status": "erro", "tentativa": tentativa,
                             "seed": self.config["seed"], "pasta_resultados": str(pasta),
                             "erro": erro, "tempo_segundos": round(time.perf_counter()-inicio, 3)}
                _gravar_json(arquivo_relatorio, relatorio)
            if codigo != 0 and relatorio.get("status") == "concluido":
                relatorio["status"] = "erro"
                relatorio["erro"] = f"Codigo de saida anormal: {codigo}"
            relatorio["codigo_saida"] = codigo
            if self.continuar_treinos:
                if relatorio["status"] == "concluido":
                    try:
                        relatorio = self._concluir_promocao(
                            relatorio, origem, anterior,
                            self.continuacao_ativa["numero"], arquivo_relatorio,
                        )
                    except BaseException as erro:
                        relatorio["status"] = "erro"
                        relatorio["erro"] = f"Promocao: {type(erro).__name__}: {erro}"
                        print(f"ERRO DE PROMOCAO {indice}: {relatorio['erro']}", flush=True)
                if relatorio["status"] == "concluido":
                    estado["resultados"][indice] = relatorio
                    self.continuacao_ativa["resultados"][indice] = "concluido"
                else:
                    # Treino anterior continua integro; erro restrito a esta rodada.
                    self.continuacao_ativa["resultados"][indice] = "erro"
                    falhas = self.continuacao_ativa.setdefault("falhas", {})
                    falhas[indice] = relatorio.get("erro", "Erro sem detalhes")
            else:
                estado["resultados"][indice] = relatorio
            _gravar_json(self.arquivo_estado, estado)
            self._comparar(estado)
            print(f"Experimento {indice}: {relatorio['status']}", flush=True)

        self._comparar(estado, gerar_graficos=True)
        estado = self._finalizar_execucao(estado)
        statuses = [estado["resultados"].get(str(e["id"]), {}).get("status", "pendente")
                    for e in self.experimentos]
        resumo = {tipo: statuses.count(tipo) for tipo in ("concluido", "erro", "pendente")}
        print(f"GRID FINALIZADO: {resumo}\nResultados: {self.raiz_resultados}")
        return {**resumo, "total": len(self.experimentos),
                "pasta_resultados": self.raiz_resultados,
                "pasta_comparacoes": self.comparados}



def _principal():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path, required=True)
    argumentos = parser.parse_args()
    raise SystemExit(_executar_trabalho(argumentos.worker))


if __name__ == '__main__':
    _principal()
