"""
grid_test_class.py

Objetivo:
- Localizar e avaliar os dois checkpoints de cada treinamento do Grid Search.
- Reconstruir a LeNet5 ou VGG16 diretamente dos metadados do checkpoint.
- Calcular metricas globais e por classe, matrizes, predicoes e rankings.
- Listar os parametros dos melhores experimentos sem duplicar checkpoints.
- Preservar integralmente as pastas e os resultados dos treinamentos.

Regras:
- Avaliar somente arquivos .npy do test do split, sem augmentation.
- Testar todos os checkpoints contra as mesmas amostras e ordem de classes.
- Registrar falhas individuais e continuar com os demais checkpoints.
- Nao usar metricas do test para alterar ou treinar os modelos.
"""

import csv
from datetime import datetime
import gc
import json
import math
from pathlib import Path
import random
import re
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pdc_ia_image_defect_detection.models.modelos import criar_modelo
from pdc_ia_image_defect_detection.pre_processing.test_class import _DatasetTeste
from pdc_ia_image_defect_detection.utils.metrics import (
    calcular_metricas_classificacao,
    plotar_matrizes_confusao,
    salvar_metricas_classificacao,
)
from pdc_ia_image_defect_detection.utils.nomenclatura import ROTULOS
from pdc_ia_image_defect_detection.utils.npy import carregar_imagem_npy
from pdc_ia_image_defect_detection.utils.resultados import (
    criar_pasta_resultados,
    finalizar_pasta_resultados,
)


# Acuracia por classe = acertos da classe / amostras reais da classe (recall).
CAMPOS_CLASSES = tuple(
    campo
    for classe in ROTULOS
    for campo in (f"amostras_{classe}", f"acertos_{classe}", f"acuracia_{classe}")
)

CAMPOS_AGREGADOS = (
    "acertos_totais", "erros_totais", "acertos_media_classes",
    "acuracia_media_classes", "classes_com_amostras",
)

CAMPOS_TESTE = (
    "id", "checkpoint", "status", "modelo", "kernel", "dropout", "neuronios_fc1",
    "neuronios_fc2", "learning_rate", "seed_treino", "epocas_executadas",
    "melhor_epoca", "melhor_val_loss", "maior_val_acc", "val_acc_final",
    "val_acc_na_melhor_loss", "val_acc_checkpoint", "gap_val_menos_test",
    "paciencia_atingida", "motivo_parada", "test_loss", "accuracy",
    "balanced_accuracy", *CAMPOS_AGREGADOS, *CAMPOS_CLASSES,
    "precision_macro", "recall_macro", "f1_macro", "precision_weighted",
    "recall_weighted", "f1_weighted", "mcc", "cohen_kappa",
    "num_amostras", "tempo_segundos", "arquivo_checkpoint", "pasta_teste", "erro",
)

CAMPOS_TOP_PARAMETROS = (
    "ranking", "id", "checkpoint", "modelo", "kernel", "dropout", "neuronios_fc1",
    "neuronios_fc2", "learning_rate", "seed_treino", "epocas_executadas",
    "melhor_epoca", "melhor_val_loss", "val_acc_checkpoint",
    "test_loss", "accuracy", "acuracia_media_classes", "num_amostras",
    "acertos_totais", "acertos_media_classes", "f1_macro",
    "balanced_accuracy", *CAMPOS_CLASSES,
)


class GridTestProcessor:
    """Testa cada checkpoint em um unico split fixo e gera comparativos."""

    def __init__(
        self, pasta_grid, raiz_resultados, bases, batch_size, num_workers,
        usar_cuda, seed, salvar_matrizes_confusao, top_graficos,
        top_parametros=20, modelo="lenet5",
    ):
        self.modelo = str(modelo).strip().lower()
        if self.modelo not in ("lenet5", "vgg16"):
            raise ValueError("MODELO_GRID_TEST deve ser lenet5 ou vgg16.")
        self.pasta_grid = Path(pasta_grid).resolve()
        self.raiz_resultados = Path(raiz_resultados).resolve()
        self.bases = dict(bases)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.usar_cuda = bool(usar_cuda)
        self.seed = int(seed)
        self.salvar_matrizes_confusao = bool(salvar_matrizes_confusao)
        self.top_graficos = int(top_graficos)
        self.top_parametros = int(top_parametros)
        self.classes = tuple(ROTULOS)
        self.registros = []
        self.experimentos = []
        self.inicio = None
        self.pasta_saida = None
        self.detalhes_saida = None
        self.device = None
        self.input_shape = None
        self.loader = None
        self.resultados = []

    @staticmethod
    def _ler_csv(caminho):
        with Path(caminho).open("r", newline="", encoding="utf-8-sig") as arquivo:
            return list(csv.DictReader(arquivo))

    @staticmethod
    def _salvar_csv(caminho, registros, colunas):
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        with caminho.open("w", newline="", encoding="utf-8-sig") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=colunas, extrasaction="ignore", delimiter=";")
            escritor.writeheader()
            escritor.writerows(registros)

    @staticmethod
    def _valor_numero(valor):
        try:
            numero = float(valor)
            return numero if math.isfinite(numero) else None
        except (TypeError, ValueError):
            return None

    def _validar(self):
        if not self.pasta_grid.is_dir():
            raise FileNotFoundError(f"Pasta do Grid Search nao encontrada: {self.pasta_grid}")
        if (self.batch_size <= 0 or self.num_workers < 0
                or self.top_graficos < 1 or self.top_parametros < 1):
            raise ValueError("BATCH_SIZE, NUM_WORKERS, TOP_GRAFICOS ou TOP_PARAMETROS invalidos.")
        if not any(info["ativa"] for info in self.bases.values()):
            raise ValueError("Ative pelo menos uma base de teste.")
        for origem, info in self.bases.items():
            if info["ativa"] and not Path(info["test"]).is_dir():
                raise FileNotFoundError(f"Conjunto test {origem} nao encontrado: {info['test']}")
        resumo = self.pasta_grid / "resultados_comparados" / "resumo_geral.csv"
        if not resumo.is_file():
            raise FileNotFoundError(f"Resumo original do Grid Search nao encontrado: {resumo}")
        config_grid = self.pasta_grid / "resultados_comparados" / "configuracoes_grid_search.json"
        if config_grid.exists():
            origem = json.loads(config_grid.read_text(encoding="utf-8"))
            modelo_origem = str(origem.get("modelo", "lenet5")).lower()
            if modelo_origem != self.modelo:
                raise ValueError(f"Busca de {modelo_origem}; MODELO_GRID_TEST={self.modelo}.")
        self.experimentos = self._ler_csv(resumo)
        if not self.experimentos:
            raise ValueError("O resumo da busca nao possui experimentos.")
        ids = [e["id"] for e in self.experimentos]
        if len(set(ids)) != len(ids):
            raise ValueError("O resumo da busca possui IDs duplicados.")

    def _coletar_dados(self):
        classes = {nome: indice for indice, nome in enumerate(self.classes)}
        for origem, base in self.bases.items():
            if not base["ativa"]:
                continue
            raiz = Path(base["test"])
            desconhecidas = [p.name for p in raiz.iterdir() if p.is_dir() and p.name not in classes]
            if desconhecidas:
                raise ValueError(f"Pastas desconhecidas no test {origem}: {desconhecidas}")
            for nome, indice in classes.items():
                pasta_classe = raiz / nome
                if not pasta_classe.is_dir():
                    continue
                for caminho in sorted(pasta_classe.rglob("*.npy")):
                    if re.search(r"_aug_?\d+$", caminho.stem, re.IGNORECASE):
                        raise ValueError(f"Augmentation encontrado no conjunto test: {caminho}")
                    self.registros.append({
                        "caminho": caminho, "indice": indice, "rotulo": nome,
                        "origem": origem, "tipo": "npy",
                    })
        if not self.registros:
            raise RuntimeError("Nenhuma imagem .npy encontrada nos conjuntos test.")
        self.input_shape = tuple(carregar_imagem_npy(self.registros[0]["caminho"]).shape)
        for registro in self.registros:
            array = carregar_imagem_npy(registro["caminho"])
            if tuple(array.shape) != self.input_shape:
                raise ValueError(f"Shape inconsistente no test: {registro['caminho']}")
        self.loader = DataLoader(
            _DatasetTeste(self.registros, self.input_shape),
            batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )

    def _pasta_experimento(self, experimento):
        """Prioriza o caminho registrado; nao pesquisa backups nem subpastas."""
        id_ = int(experimento["id"])
        caminho_texto = experimento.get("pasta_resultados")
        if caminho_texto:
            salvo = Path(caminho_texto)
            candidatas = [salvo, self.pasta_grid / salvo.name]
            for pasta in candidatas:
                if pasta.is_dir() and any((pasta / nome).is_file() for nome in (
                    "modelo_melhor.pt", "modelo_final.pt",
                )):
                    return pasta
        padrao = re.compile(rf"^treino_id{id_:03d}_t\d+_")
        candidatas = [p for p in self.pasta_grid.iterdir()
                      if p.is_dir() and padrao.match(p.name)
                      and any((p / nome).is_file() for nome in (
                          "modelo_melhor.pt", "modelo_final.pt",
                      ))]
        if len(candidatas) == 1:
            return candidatas[0]
        if not candidatas:
            raise FileNotFoundError(f"Checkpoints nao encontrados para experimento {id_:03d}.")
        raise RuntimeError(f"Multiplas pastas para experimento {id_:03d}; verifique resumo_geral.csv.")

    def _validar_checkpoint(self, checkpoint, caminho):
        essenciais = ("model_state_dict", "classes", "input_shape", "num_classes")
        faltantes = [campo for campo in essenciais if campo not in checkpoint]
        if faltantes:
            raise KeyError(f"{caminho.name}: campos ausentes: {faltantes}")
        if str(checkpoint.get("model_name", "lenet5")).strip().lower() != self.modelo:
            raise ValueError(f"Checkpoint nao corresponde a {self.modelo}: {caminho}")
        if tuple(checkpoint["classes"]) != self.classes:
            raise ValueError(f"Classes divergentes: {caminho}")
        if int(checkpoint["num_classes"]) != len(self.classes):
            raise ValueError(f"Quantidade de classes divergente: {caminho}")
        if tuple(checkpoint["input_shape"]) != self.input_shape:
            raise ValueError(f"Input shape do checkpoint diverge do test: {caminho}")

    def _inferir(self, modelo):
        criterio = nn.CrossEntropyLoss(reduction="sum")
        soma_loss = 0.0
        predicoes = []
        reais = []
        linhas = []
        modelo.eval()
        with torch.inference_mode():
            for imagens, rotulos, caminhos, origens, _ in self.loader:
                imagens = imagens.to(self.device, non_blocking=self.device.type == "cuda")
                rotulos = rotulos.to(self.device, non_blocking=self.device.type == "cuda")
                logits = modelo(imagens)
                soma_loss += criterio(logits, rotulos).item()
                probabilidades = torch.softmax(logits, dim=1).cpu().numpy()
                previstos = logits.argmax(1).cpu().tolist()
                reais_lote = rotulos.cpu().tolist()
                for indice, caminho in enumerate(caminhos):
                    real = int(reais_lote[indice])
                    previsto = int(previstos[indice])
                    reais.append(real)
                    predicoes.append(previsto)
                    linha = {
                        "arquivo": str(caminho), "origem": origens[indice],
                        "real": self.classes[real], "previsto": self.classes[previsto],
                        "acerto": int(real == previsto),
                        "confianca": float(probabilidades[indice, previsto]),
                    }
                    for j, classe in enumerate(self.classes):
                        linha[f"prob_{classe}"] = float(probabilidades[indice, j])
                    linhas.append(linha)
        return soma_loss / len(reais), reais, predicoes, linhas

    def _metricas_classes(self, reais, previstos):
        """Conta acertos e calcula a taxa de acerto para cada classe real."""
        verdade = np.asarray(reais, dtype=np.int64)
        previsao = np.asarray(previstos, dtype=np.int64)
        resultado = {}
        acertos = []
        acuracias = []
        for indice, classe in enumerate(self.classes):
            mascara = verdade == indice
            suporte = int(mascara.sum())
            corretos = int((previsao[mascara] == indice).sum())
            resultado[f"amostras_{classe}"] = suporte
            resultado[f"acertos_{classe}"] = corretos
            resultado[f"acuracia_{classe}"] = corretos / suporte if suporte else ""
            if suporte:
                acertos.append(corretos)
                acuracias.append(corretos / suporte)
        total_acertos = sum(acertos)
        resultado.update({
            "acertos_totais": total_acertos,
            "erros_totais": len(verdade) - total_acertos,
            "acertos_media_classes": sum(acertos) / len(acertos) if acertos else "",
            "acuracia_media_classes": sum(acuracias) / len(acuracias) if acuracias else "",
            "classes_com_amostras": len(acuracias),
        })
        return resultado

    def _testar_checkpoint(self, experimento, pasta_treino, tipo):
        inicio = time.perf_counter()
        id_ = int(experimento["id"])
        destino_relativo = f"resultados_testes/id{id_:03d}_{tipo}"
        destino = self.pasta_saida / destino_relativo
        destino.mkdir(parents=True, exist_ok=True)
        arquivo = pasta_treino / f"modelo_{tipo}.pt"
        if not arquivo.is_file():
            raise FileNotFoundError(f"Checkpoint ausente: {arquivo}")
        # Checkpoints locais e confiaveis podem incluir objetos NumPy e RNG.
        checkpoint = torch.load(arquivo, map_location="cpu", weights_only=False)
        self._validar_checkpoint(checkpoint, arquivo)
        parametros = dict(checkpoint.get("config_modelo") or {})
        if self.modelo == "lenet5":
            esperados = {
                "kernel_size": ("kernel", 5),
                "neuronios_fc1": ("neuronios_fc1", 120),
                "neuronios_fc2": ("neuronios_fc2", 84),
            }
        else:
            esperados = {
                "neuronios_fc1": ("neuronios_fc1", 4096),
                "neuronios_fc2": ("neuronios_fc2", 4096),
                "dropout": ("dropout", 0.5),
            }
        for campo_ckpt, (campo_grid, padrao) in esperados.items():
            valor_grid = experimento.get(campo_grid)
            if valor_grid in (None, ""):
                continue
            valor_ckpt = parametros.get(campo_ckpt, padrao)
            if not math.isclose(float(valor_ckpt), float(valor_grid), abs_tol=1e-9):
                raise ValueError(
                    f"Arquitetura divergente no checkpoint {arquivo.name}: "
                    f"{campo_ckpt} != {campo_grid} do Grid Search."
                )
        modelo = criar_modelo(self.modelo, num_classes=len(self.classes), **parametros)
        if self.modelo == "lenet5":
            # LazyLinear requer um primeiro forward para materializar o classificador.
            with torch.no_grad():
                modelo(torch.zeros((1, *self.input_shape), dtype=torch.float32))
        modelo.load_state_dict(checkpoint["model_state_dict"])
        modelo = modelo.to(self.device)
        try:
            loss, reais, previstos, predicoes = self._inferir(modelo)
        finally:
            del modelo
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        metricas, relatorio = calcular_metricas_classificacao(reais, previstos, self.classes)
        metricas_classes = self._metricas_classes(reais, previstos)
        salvar_metricas_classificacao(
            metricas, relatorio, loss, destino / "metricas_teste.txt", titulo="TEST",
        )
        if self.salvar_matrizes_confusao:
            plotar_matrizes_confusao(reais, previstos, self.classes, destino, prefixo="teste")
        self._salvar_csv(
            destino / "predicoes.csv", predicoes,
            ("arquivo", "origem", "real", "previsto", "acerto", "confianca",
             *(f"prob_{classe}" for classe in self.classes)),
        )
        info = {
            "id": id_, "checkpoint": tipo, "modelo": self.modelo,
            "checkpoint_origem": str(arquivo), "parametros_modelo": parametros,
            "epoca_checkpoint": checkpoint.get("epoca"),
            "test_loss": loss, "num_amostras": len(reais),
            **metricas, **metricas_classes,
        }
        (destino / "resumo.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
        )
        linha = {campo: experimento.get(campo, "") for campo in CAMPOS_TESTE}
        val_acc = self._valor_numero(
            experimento.get("val_acc_na_melhor_loss")
            if tipo == "melhor" else experimento.get("val_acc_final")
        )
        linha.update({
            "id": id_, "checkpoint": tipo, "status": "concluido", "modelo": self.modelo,
            "seed_treino": experimento.get("seed", ""),
            "val_acc_checkpoint": "" if val_acc is None else val_acc,
            "gap_val_menos_test": "" if val_acc is None else val_acc - metricas["accuracy"],
            "test_loss": loss, "num_amostras": len(reais),
            "tempo_segundos": round(time.perf_counter() - inicio, 3),
            "arquivo_checkpoint": str(arquivo), "pasta_teste": destino_relativo,
            **metricas, **metricas_classes,
        })
        return linha

    def _comparar(self):
        pasta = self.pasta_saida / "resultados_comparados"
        pasta.mkdir(exist_ok=True)
        self._salvar_csv(pasta / "resumo_geral.csv", self.resultados, CAMPOS_TESTE)
        validos = [r for r in self.resultados if r["status"] == "concluido"]
        criterios = {
            "accuracy": (lambda r: (-float(r["accuracy"]), -float(r["f1_macro"]), float(r["test_loss"]))),
            "f1_macro": (lambda r: (-float(r["f1_macro"]), -float(r["balanced_accuracy"]), -float(r["accuracy"]))),
            "loss": (lambda r: (float(r["test_loss"]), -float(r["accuracy"]))),
            "acuracia_media_classes": (lambda r: (
                -float(r["acuracia_media_classes"]), -float(r["f1_macro"]),
                -float(r["accuracy"]), float(r["test_loss"]),
            )),
            "acertos_totais": (lambda r: (
                -int(r["acertos_totais"]), -float(r["acuracia_media_classes"]),
                -float(r["f1_macro"]),
            )),
        }
        for nome, chave in criterios.items():
            ranking = sorted(validos, key=chave)
            self._salvar_csv(
                pasta / f"ranking_{nome}.csv",
                [dict(ranking=i, **registro) for i, registro in enumerate(ranking, 1)],
                ("ranking", *CAMPOS_TESTE),
            )
            for tipo in ("melhor", "final"):
                filtrados = [r for r in ranking if r["checkpoint"] == tipo]
                self._salvar_csv(
                    pasta / f"ranking_{nome}_{tipo}.csv",
                    [dict(ranking=i, **registro) for i, registro in enumerate(filtrados, 1)],
                    ("ranking", *CAMPOS_TESTE),
                )
            if nome in ("acuracia_media_classes", "accuracy"):
                # Um checkpoint por experimento evita duplicar a mesma configuracao.
                distintos = []
                vistos = set()
                for registro in ranking:
                    if registro["id"] in vistos:
                        continue
                    vistos.add(registro["id"])
                    distintos.append(registro)
                    if len(distintos) == self.top_parametros:
                        break
                top = [dict(ranking=i, **r) for i, r in enumerate(distintos, 1)]
                self._salvar_csv(
                    pasta / f"top{self.top_parametros}_parametros_{nome}.csv",
                    top, CAMPOS_TOP_PARAMETROS,
                )
                self._salvar_top_txt(
                    pasta / f"top{self.top_parametros}_parametros_{nome}.txt", top, nome,
                )
        resumo_classes = (
            "id", "checkpoint", "modelo", "kernel", "dropout", "neuronios_fc1", "neuronios_fc2",
            "learning_rate", "num_amostras", "accuracy", *CAMPOS_AGREGADOS,
            *CAMPOS_CLASSES,
        )
        self._salvar_csv(pasta / "comparacao_por_classe.csv", validos, resumo_classes)
        agrupados = {}
        for registro in validos:
            agrupados.setdefault(int(registro["id"]), {})[registro["checkpoint"]] = registro
        pares = []
        for id_, entradas in sorted(agrupados.items()):
            if "melhor" not in entradas or "final" not in entradas:
                continue
            melhor, final = entradas["melhor"], entradas["final"]
            pares.append({
                "id": id_, "modelo": self.modelo,
                "kernel": melhor.get("kernel", ""), "dropout": melhor.get("dropout", ""),
                "neuronios_fc1": melhor["neuronios_fc1"],
                "neuronios_fc2": melhor["neuronios_fc2"],
                "learning_rate": melhor["learning_rate"],
                "accuracy_melhor": melhor["accuracy"], "accuracy_final": final["accuracy"],
                "delta_accuracy_melhor_menos_final": float(melhor["accuracy"]) - float(final["accuracy"]),
                "f1_macro_melhor": melhor["f1_macro"], "f1_macro_final": final["f1_macro"],
                "delta_f1_melhor_menos_final": float(melhor["f1_macro"]) - float(final["f1_macro"]),
                "loss_melhor": melhor["test_loss"], "loss_final": final["test_loss"],
                "val_loss_melhor": melhor["melhor_val_loss"],
                "epocas_executadas": melhor["epocas_executadas"],
            })
        self._salvar_csv(
            pasta / "comparacao_validacao_teste.csv",
            [r for r in validos if r.get("val_acc_checkpoint") != ""],
            CAMPOS_TESTE,
        )
        self._salvar_csv(
            pasta / "comparacao_melhor_final.csv", pares,
            ("id", "modelo", "kernel", "dropout", "neuronios_fc1", "neuronios_fc2", "learning_rate",
             "accuracy_melhor", "accuracy_final", "delta_accuracy_melhor_menos_final",
             "f1_macro_melhor", "f1_macro_final", "delta_f1_melhor_menos_final",
             "loss_melhor", "loss_final", "val_loss_melhor", "epocas_executadas"),
        )

    def _salvar_top_txt(self, caminho, registros, criterio):
        """Lista os parametros dos melhores experimentos, sem duplicar IDs."""
        linhas = [
            f"TOP {len(registros)} - criterio: {criterio}",
            "Acuracia por classe = acertos / amostras reais da classe.",
            "Acuracia media = media simples das classes presentes no teste.",
            "",
        ]
        for r in registros:
            linhas.extend([
                f"{int(r['ranking']):02d}. ID {int(r['id']):03d} | checkpoint={r['checkpoint']}",
                (f"    Modelo={self.modelo} | "
                 + (f"Kernel={r['kernel']}" if self.modelo == "lenet5"
                    else f"Conv=3/Pool=2 | Dropout={r['dropout']}")
                 + f" | FC={r['neuronios_fc1']}/{r['neuronios_fc2']} "
                 f"| LR={r['learning_rate']} | Seed={r['seed_treino']} "
                 f"| Epocas={r['epocas_executadas']}"),
                (f"    Acertos={r['acertos_totais']}/{r['num_amostras']} "
                 f"| Acuracia global={float(r['accuracy']):.4f} "
                 f"| Acuracia media={float(r['acuracia_media_classes']):.4f} "
                 f"| F1-macro={float(r['f1_macro']):.4f}"),
                "    Classes: " + " | ".join(
                    f"{classe}={r[f'acertos_{classe}']}/{r[f'amostras_{classe}']} "
                    + (f"({float(r[f'acuracia_{classe}']):.4f})"
                       if r[f'acuracia_{classe}'] != "" else "(sem amostras)")
                    for classe in self.classes
                ),
                "",
            ])
        Path(caminho).write_text("\n".join(linhas), encoding="utf-8")

    def _graficos(self):
        pasta = self.pasta_saida / "resultados_comparados"
        validos = [r for r in self.resultados if r["status"] == "concluido"]
        if not validos:
            return
        campos_graficos = (
            ("accuracy", "acuracias"),
            ("f1_macro", "f1_macro"),
            ("test_loss", "losses"),
            ("acuracia_media_classes", "acuracia_media_classes"),
            ("acertos_totais", "acertos_totais"),
            ("acertos_media_classes", "acertos_medios_classes"),
        )
        for campo, nome in campos_graficos:
            reverso = campo != "test_loss"
            ordenados = sorted(validos, key=lambda r: float(r[campo]), reverse=reverso)
            top = ordenados[:self.top_graficos]
            rotulos = [f"{int(r['id']):03d}-{r['checkpoint'][0].upper()}" for r in top]
            valores = [float(r[campo]) for r in top]
            figura, eixo = plt.subplots(figsize=(max(10, len(top) * 0.52), 5.5))
            eixo.bar(rotulos, valores)
            eixo.set_title(f"Top {len(top)} - test {campo}")
            eixo.set_xlabel("Experimento-checkpoint (M=melhor, F=final)")
            eixo.set_ylabel(campo)
            eixo.tick_params(axis="x", labelrotation=65)
            eixo.grid(axis="y", alpha=0.25)
            figura.tight_layout()
            figura.savefig(pasta / f"comparacao_{nome}.png", dpi=200)
            plt.close(figura)

        # As barras por classe mostram ate TOP_GRAFICOS modelos, por acuracia media.
        top = sorted(
            validos,
            key=lambda r: (-float(r["acuracia_media_classes"]), -float(r["accuracy"])),
        )[:self.top_graficos]
        rotulos = [f"{int(r['id']):03d}-{r['checkpoint'][0].upper()}" for r in top]
        eixo_x = np.arange(len(top))
        n = len(self.classes)
        largura = 0.8 / n
        for nome, campo, eixo_y in (
            ("acuracias_por_classe", "acuracia", "Acuracia por classe"),
            ("acertos_por_classe", "acertos", "Quantidade de acertos"),
        ):
            figura, eixo = plt.subplots(figsize=(max(12, len(top) * 0.75), 6))
            for j, classe in enumerate(self.classes):
                valores = [
                    float(r[f"{campo}_{classe}"]) if r[f"{campo}_{classe}"] != "" else 0
                    for r in top
                ]
                posicoes = eixo_x + (j - (n - 1) / 2) * largura
                eixo.bar(posicoes, valores, width=largura, label=classe)
            eixo.set_xticks(eixo_x, rotulos, rotation=65)
            eixo.set_xlabel("Experimento-checkpoint (M=melhor, F=final)")
            eixo.set_ylabel(eixo_y)
            eixo.set_title(f"Top {len(top)} - {eixo_y.lower()}")
            if campo == "acuracia":
                eixo.set_ylim(0, 1.05)
            eixo.legend()
            eixo.grid(axis="y", alpha=0.25)
            figura.tight_layout()
            figura.savefig(pasta / f"comparacao_{nome}.png", dpi=200)
            plt.close(figura)

    def _criar_pasta(self):
        self.inicio = datetime.now()
        for numero in range(1, 100):
            nome = "lenet" if self.modelo == "lenet5" else "vgg16"
            self.detalhes_saida = ("grid", "test", nome) if numero == 1 else (
                "grid", "test", nome, f"r{numero:02d}",
            )
            try:
                self.pasta_saida = criar_pasta_resultados(
                    raiz=self.raiz_resultados, tipo="resultados",
                    inicio=self.inicio, detalhes=self.detalhes_saida,
                )
                return
            except FileExistsError:
                continue
        raise FileExistsError("Nao foi possivel criar pasta exclusiva de resultados.")

    def executar(self):
        """Avalia os checkpoints e atualiza os CSVs progressivamente."""
        self._validar()
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.usar_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.device = torch.device("cuda" if self.usar_cuda and torch.cuda.is_available() else "cpu")
        print(f"\nGRID TEST {self.modelo.upper()}: {len(self.experimentos)} experimentos")
        print(f"Origem: {self.pasta_grid}")
        print(f"Dispositivo: {self.device}")
        self._coletar_dados()
        print(f"Test: {len(self.registros)} imagens | formato: {self.input_shape}")
        self._criar_pasta()
        config = {
            "modelo": self.modelo, "pasta_grid_original": str(self.pasta_grid),
            "bases_test": {nome: str(info["test"]) for nome, info in self.bases.items() if info["ativa"]},
            "classes": list(self.classes), "input_shape": list(self.input_shape),
            "num_amostras_test": len(self.registros),
            "batch_size": self.batch_size, "num_workers": self.num_workers,
            "usar_cuda": self.usar_cuda, "seed_test": self.seed,
            "salvar_matrizes_confusao": self.salvar_matrizes_confusao,
            "top_graficos": self.top_graficos,
            "top_parametros": self.top_parametros,
            "definicao_acuracia_classe": "acertos da classe / amostras reais da classe",
            "definicao_acuracia_media_classes": "media simples das acuracias das classes com amostras",
            "definicao_acertos_media_classes": "total de acertos / numero de classes com amostras",
        }
        (self.pasta_saida / "configuracao_teste.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        for indice, experimento in enumerate(self.experimentos, 1):
            try:
                pasta_treino = self._pasta_experimento(experimento)
            except Exception as erro:
                pasta_treino = None
                erro_pasta = erro
            else:
                erro_pasta = None
            for tipo in ("melhor", "final"):
                print(f"[{indice}/{len(self.experimentos)}] ID {int(experimento['id']):03d} | {tipo}", flush=True)
                try:
                    if experimento.get("status") != "concluido":
                        raise RuntimeError("Experimento de treinamento nao concluido.")
                    if erro_pasta:
                        raise erro_pasta
                    linha = self._testar_checkpoint(experimento, pasta_treino, tipo)
                    print(f"  acc={float(linha['accuracy']):.4f} | F1={float(linha['f1_macro']):.4f} | loss={float(linha['test_loss']):.5f}", flush=True)
                except Exception as erro:
                    mensagem = f"{type(erro).__name__}: {erro}"
                    print(f"  ERRO: {mensagem}", flush=True)
                    pasta_erros = self.pasta_saida / "erros"
                    pasta_erros.mkdir(exist_ok=True)
                    (pasta_erros / f"id{int(experimento['id']):03d}_{tipo}.txt").write_text(
                        traceback.format_exc(), encoding="utf-8",
                    )
                    linha = {campo: experimento.get(campo, "") for campo in CAMPOS_TESTE}
                    linha.update({"id": int(experimento["id"]), "checkpoint": tipo,
                                  "seed_treino": experimento.get("seed", ""),
                                  "status": "erro", "modelo": self.modelo, "erro": mensagem})
                self.resultados.append(linha)
                self._comparar()
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        self._graficos()
        fim = datetime.now()
        self.pasta_saida = finalizar_pasta_resultados(
            pasta_atual=self.pasta_saida, tipo="resultados",
            inicio=self.inicio, fim=fim, detalhes=self.detalhes_saida,
        )
        ok = sum(r["status"] == "concluido" for r in self.resultados)
        erros = len(self.resultados) - ok
        print(f"\nGRID TEST FINALIZADO: {ok} checkpoints avaliados, {erros} falhas.")
        print(f"Resultados: {self.pasta_saida}")
        return {"total": len(self.resultados), "concluido": ok, "erro": erros,
                "pasta_resultados": str(self.pasta_saida)}
