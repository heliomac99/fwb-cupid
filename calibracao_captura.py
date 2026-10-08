# -*- coding: utf-8 -*-
"""
Calibracao do peso da interferencia assincrona (ASYNC_INTERF_WEIGHT, F28).

Alvo: o primeiro T do Crystal sobre o DW1000, medido por Lobba et al.
(EWSN 2020, sec. 5.2), com o mesmo radio do CLOVES (canal 4, 6,8 Mbps, PRF
64, preambulo de 64 us). Metrica da fonte: para cada no, a fracao dos floods
em que ele recebeu QUALQUER pacote, entre os floods em que ele escutava o
primeiro T; a media e entre os nos (Fig. 10). Pontos medidos com N = 2 e
quadros de 127 B, na variante GLOSSYTX (N retransmissoes em slots TX
consecutivos -- a de crystal_t_flood): U = 2, 5, 7, 10 (Fig. 10c) e o caso
extremo em que os 21 nao-sink de uma rede de 22 nos transmitem (~75%).

CALIBRACAO x EXTRAPOLACAO. So os U medidos na fonte (EWSN_T_SUCCESS) sao
calibracao, e U = 21 so e comparavel numa rede do MESMO tamanho: ele roda com
N_NODES = 22 (todos os nao-sink transmitindo, so o sink escuta). U = 35 de 36,
o caso do testbed do Cupid, e um experimento NOVO, de extrapolacao: a tabela
o marca assim e ele nunca entra como alvo. A geometria continua sintetica (o
testbed da fonte e um corredor de 23 nos com diametro de 4 hops).

Validacao (sem reajuste, tabela separada): o experimento de varias fontes do
SECON 2019 (Fig. 13), num receptor so, com as fontes somadas da mais proxima
para a mais distante: pacotes diferentes sincronizados caem para < 50% com 9
fontes; com atraso aleatorio de ate 20 us ficam acima de 85%.

Uso:
    py calibracao_captura.py                 # grade padrao de pesos
    py calibracao_captura.py --pesos 1 0.3 0.2 --tentativas 400

Resultados em outputs/calibracao/.
"""

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor

import cupid_sim_v2 as sim

# Pontos medidos do primeiro T: U -> (sucesso medio, fonte, N_NODES da rede
# em que o ponto e comparavel; None = a topologia do cenario). Leituras da
# Fig. 10c digitalizadas (+-0,001).
EWSN_T_SUCCESS = {
    2: (0.994, "EWSN 2020, Fig. 10c (N=2, 127 B, GLOSSYTX)", None),
    5: (0.989, "EWSN 2020, Fig. 10c (N=2, 127 B, GLOSSYTX)", None),
    7: (0.988, "EWSN 2020, Fig. 10c (N=2, 127 B, GLOSSYTX)", None),
    10: (0.985, "EWSN 2020, Fig. 10c (N=2, 127 B, GLOSSYTX)", None),
    21: (0.75, "EWSN 2020, sec. 5.2, 'Extreme case': os 21 nao-sink de 22 "
               "transmitindo, N=2, 127 B (~75%)", 22),
}
EWSN_MEASURED_U = frozenset(EWSN_T_SUCCESS)
EXTRAPOLATED_U = (35,)          # todos os nao-sink do cenario de 36 nos
US = tuple(sorted(EWSN_MEASURED_U | set(EXTRAPOLATED_U)))
KS = (2, 3, 5, 7, 9)
# padroes do processo (capturados no import, antes de qualquer tarefa)
DEFAULTS = {name: getattr(sim, name) for name in sim.PARAM_NAMES}


def task_config(cfg):
    """Configuracao completa da tarefa (workers sao reutilizados)."""
    return sim.fresh_config(cfg, DEFAULTS)


def config_for_u(cfg, u):
    """Configuracao do ponto U: completa, com a rede em que ele e
    comparavel (sem herdar N_NODES do ponto anterior)."""
    full = task_config(cfg)
    n_nodes = network_size_for(u)
    if n_nodes:
        full["N_NODES"] = n_nodes
    return full


def default_weights():
    """Grade padrao de pesos: sempre contem o peso que o preset legado usa
    (sem ele a tabela nao mostra o ponto escolhido) e os vizinhos da faixa
    fisica 0,03-0,1."""
    legacy = sim.PRESET_LEGACY_V2["ASYNC_INTERF_WEIGHT"]
    grid = {1.0, 0.5, 0.35, 0.25, 0.15, 0.1, 0.03, legacy}
    return sorted(grid, reverse=True)


def u_status(u):
    """'calibracao' se U foi medido na fonte; senao 'extrapolacao'."""
    return "calibracao" if u in EWSN_MEASURED_U else "extrapolacao"


def network_size_for(u):
    """N_NODES em que o ponto U e simulado (None = o do cenario)."""
    return EWSN_T_SUCCESS[u][2] if u in EWSN_T_SUCCESS else None


def _u_header(u, t_rows=()):
    """Cabecalho da coluna U. Com as linhas da tabela, a rede e a que
    RODOU (N_NODES efetivo registrado por ponto); sem elas, a declarada."""
    sizes = sorted({r["n_nodes"] for r in t_rows
                    if r["U"] == u and "n_nodes" in r})
    n = (sizes[0] if len(sizes) == 1 else
         "/".join(map(str, sizes)) if sizes else network_size_for(u))
    kind = "medido" if u in EWSN_MEASURED_U else "extrapolacao"
    return f"U={u} ({kind}, rede de {n})" if n else f"U={u} ({kind})"


def _t_cell(r, fmt):
    """Celula de sucesso; U efetivo menor que o pedido fica explicito."""
    cel = fmt.format(r["sucesso_nos"], r["sucesso_sink"])
    fontes = r.get("fontes_por_flood")
    if fontes is not None and fontes < r["U"]:
        cel += f" [U efetivo {fontes:.1f}]"
    return cel


def point_record(**fields):
    """Linha de um ponto com a configuracao efetiva em que ele rodou."""
    eff = sim.effective_config()
    return dict(fields, n_nodes=sim.N_NODES, config_hash=sim.config_hash(eff),
                config_efetiva=eff)


def artifact_json(a, t_rows, c_rows, git_info):
    """Registro verificavel da bancada: cada ponto aponta (config_hash) para
    a configuracao efetiva completa em que rodou. 'git_info' e o estado do
    repositorio lido no inicio da bancada."""
    configs = {}
    strip = []
    for r in t_rows + c_rows:
        configs[r["config_hash"]] = r["config_efetiva"]
        strip.append({k: v for k, v in r.items() if k != "config_efetiva"})
    commit, dirty, diff_hash = git_info
    return {"schema": "cupid-calibracao-captura/1",
            "model_version": sim.MODEL_VERSION, "git_commit": commit,
            "git_dirty": dirty, "git_diff_hash": diff_hash,
            "comando": {"topologias": a.topologias,
                        "tentativas": a.tentativas, "pesos": list(a.pesos),
                        "sem_modelos": a.sem_modelos},
            "u_medidos": sorted(EWSN_MEASURED_U),
            "u_extrapolados": list(EXTRAPOLATED_U),
            "linhas_t": [r for r in strip if "U" in r],
            "linhas_receptor": [r for r in strip if "k" in r],
            "configuracoes_efetivas": configs}


def render_tables(t_rows, c_rows, n_topo, tentativas, weight):
    """Markdown com a tabela de CALIBRACAO (T do Crystal) e, separada, a de
    VALIDACAO (SECON 2019). Colunas de U fora da fonte vem marcadas."""
    out = ["## Calibracao: primeiro T do Crystal -- sucesso por numero de "
           "fontes U", "",
           f"{n_topo} topologias x {tentativas} floods por U. Metrica: "
           "sucesso MEDIO por no ouvinte (sink entre parenteses). Alvos "
           "medidos (EWSN 2020, N = 2, 127 B, GLOSSYTX): "
           + "; ".join(f"{v:.3f} com U = {u}"
                       for u, (v, _, _) in sorted(EWSN_T_SUCCESS.items()))
           + ". Colunas marcadas 'extrapolacao' NAO foram medidas na fonte: "
             "sao experimentos novos, fora da calibracao. A 'rede' de cada "
             "coluna e o N_NODES efetivo do ponto; a configuracao efetiva "
             "completa de cada ponto esta em calibracao_captura.json "
             "(config_hash).", "",
           "| configuracao | " + " | ".join(_u_header(u, t_rows)
                                            for u in US) + " |",
           "|---|" + "---|" * len(US)]
    by_cfg = {}
    for r in t_rows:
        by_cfg.setdefault(r["configuracao"], {})[r["U"]] = r
    for nome, rows in by_cfg.items():
        # com todos transmitindo so o sink escuta: vale a taxa do sink
        cel = [_t_cell(rows[u], "{:.2f} (sink {:.2f})") for u in US]
        out.append(f"| {nome} | " + " | ".join(cel) + " |")
    out.append("| medido (EWSN 2020) | " + " | ".join(
        f"{EWSN_T_SUCCESS[u][0]:.3f}" if u in EWSN_T_SUCCESS else "--"
        for u in US) + " |")
    out += ["", f"## Validacao: receptor unico (SECON 2019, Fig. 13), "
            f"peso {weight:g}, sem reajuste", "",
            "SECON: mesmo pacote > 99%; diferentes sincronizados < 50% com 9 "
            "fontes; com atraso de ate 20 us > 85%.", "",
            "| caso | " + " | ".join(f"k={k}" for k in KS) + " |",
            "|---|" + "---|" * len(KS)]
    by_case = {}
    for r in c_rows:
        by_case.setdefault(r["configuracao"], {})[r["k"]] = r["prr"]
    for nome, rows in by_case.items():
        out.append(f"| {nome} | " +
                   " | ".join(f"{rows[k]:.2f}" for k in KS) + " |")
    return "\n".join(out) + "\n"


def t_success(args):
    """Sucesso do primeiro T do Crystal, por U, para uma configuracao, com a
    metrica da fonte: media entre os nos ouvintes da fracao de floods em que
    o no recebeu algum pacote. Cada U roda na rede em que e comparavel."""
    nome, cfg, n_topo, tentativas = args
    linhas = []
    for u in US:
        sim.apply_config(config_for_u(cfg, u))
        topos = sim.make_topologies(n_topo=n_topo, base_seed=sim.SEED)
        por_no = []                     # sucesso de cada (topologia, no)
        sink_ok = floods = n_fontes = 0
        for topo, seed in topos:
            valid = sim.valid_nodes(topo)
            rng = random.Random(seed * 31 + u)
            ouviu = [0] * topo.n
            recebeu = [0] * topo.n
            for _ in range(tentativas):
                fontes = sorted(rng.sample(valid, min(u, len(valid))))
                n_fontes += len(fontes)     # U efetivo (nos alcancaveis)
                clock = [rng.gauss(0.0, sim.CLOCK_JITTER_NS)
                         for _ in range(topo.n)]
                reached = set()
                pkt, _ = sim.crystal_t_flood(fontes, topo, rng, clock,
                                             reached=reached)
                for v in range(topo.n):
                    if v not in fontes and topo.hop[v] < 99:
                        ouviu[v] += 1
                        recebeu[v] += v in reached
                sink_ok += pkt is not None
                floods += 1
            por_no += [recebeu[v] / ouviu[v] for v in range(topo.n)
                       if ouviu[v]]
        linhas.append(point_record(
            configuracao=nome, U=u, status=u_status(u),
            fontes_por_flood=n_fontes / floods,
            sucesso_nos=sum(por_no) / len(por_no),
            sucesso_sink=sink_ok / floods))
    return linhas


def corridor(args):
    """Receptor unico com as k fontes mais proximas (SECON 2019, Fig. 13)."""
    nome, cfg, n_topo, tentativas, jitter_ns, same = args
    sim.apply_config(task_config(cfg))
    topos = sim.make_topologies(n_topo=n_topo, base_seed=sim.SEED)
    linhas = []
    for k in KS:
        ok = total = 0
        for topo, seed in topos:
            rng = random.Random(seed * 17 + k)
            valid = sim.valid_nodes(topo) + [0]
            for _ in range(tentativas):
                rx = rng.choice(valid)
                perto = sorted((v for v in topo.audible[rx] if v != rx),
                               key=lambda v: -topo.ps_lin[rx][v])[:k]
                if len(perto) < k:
                    continue
                clock = [rng.gauss(0.0, sim.CLOCK_JITTER_NS)
                         for _ in range(topo.n)]
                for v in perto:
                    clock[v] += rng.random() * jitter_ns
                # quadros CURTOS [hipotese]: o SECON nao informa o tamanho, e
                # os quadros do experimento (ranging/broadcast) sao curtos;
                # o modelo legado ignora o comprimento
                pkt_of = {v: ("secon", 1 if same else v) for v in perto}
                res = sim.decode_at_receiver(rx, perto, pkt_of, topo, clock,
                                             rng)
                ok += res is not None
                total += 1
        linhas.append(point_record(configuracao=nome, k=k, prr=ok / total))
    return linhas


KS_SECON = tuple(range(2, 10))          # fontes medidas na Fig. 13


def capture_models_to_compare():
    """Os dois modelos de captura sobre a configuracao PADRAO (radio CLOVES):
    o legado com o peso unico e o empirico nos tres limites da faixa."""
    base = {k: getattr(sim, k) for k in sim.PARAM_NAMES}
    legacy = dict(base, CAPTURE_MODEL="legacy_async_weight",
                  ASYNC_INTERF_WEIGHT=sim.PRESET_LEGACY_V2[
                      "ASYNC_INTERF_WEIGHT"])
    out = [("legado (peso %g)" % legacy["ASYNC_INTERF_WEIGHT"], legacy)]
    for mode in ("mid", "lower", "upper"):
        out.append((f"empirico (faixa: {mode})",
                    dict(base, CAPTURE_MODEL="empirical_uwb",
                         CAPTURE_EXTRAPOLATION=mode)))
    return out


def compare_models(a):
    """Tabelas de VALIDACAO dos modelos de captura, sem reajuste.

    T do Crystal (EWSN 2020): independente dos dois modelos. Receptor unico
    do SECON 2019: as curvas 'sincronizados' e 'mesmo pacote' sao validacao;
    a curva 'atraso ate 20 us' e a ENTRADA da tabela assincrona do modelo
    empirico -- para ele e consistencia com a origem, nao validacao."""
    import capture_models as cm
    modelos = capture_models_to_compare()
    tarefas = [(nome, cfg, a.topologias, a.tentativas)
               for nome, cfg in modelos]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        t_rows = [r for bloco in ex.map(t_success, tarefas) for r in bloco]
    casos = [("mesmo pacote, sincronizado", 0.0, True, cm.SAME_K,
              "validacao"),
             ("diferentes, sincronizados", 0.0, False, cm.DIFF_SYNC_K,
              "validacao"),
             ("diferentes, atraso ate 20 us", 20000.0, False,
              cm.DIFF_ASYNC_K, "entrada do empirico")]
    tarefas = [(f"{nm} | {caso}", cfg, a.topologias, a.tentativas, j, s)
               for nm, cfg in modelos for caso, j, s, _, _ in casos]
    global KS
    saved, KS = KS, KS_SECON
    try:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            c_rows = [r for bloco in ex.map(_corridor_all_k, tarefas)
                      for r in bloco]
    finally:
        KS = saved
    out = ["## Validacao dos modelos de captura no radio CLOVES (sem "
           "reajuste)", "",
           "Primeiro T do Crystal, metrica da fonte (media por no ouvinte; "
           "sink entre parenteses). Os dois modelos sao avaliados na "
           "configuracao padrao; a EWSN 2020 e independente de ambos.", "",
           "| modelo | " + " | ".join(_u_header(u, t_rows)
                                      for u in US) + " |",
           "|---|" + "---|" * len(US)]
    by_cfg = {}
    for r in t_rows:
        by_cfg.setdefault(r["configuracao"], {})[r["U"]] = r
    for nome, rows in by_cfg.items():
        out.append(f"| {nome} | " + " | ".join(
            _t_cell(rows[u], "{:.3f} (sink {:.2f})") for u in US) + " |")
    out.append("| medido (EWSN 2020) | " + " | ".join(
        f"{EWSN_T_SUCCESS[u][0]:.3f}" if u in EWSN_T_SUCCESS else "--"
        for u in US) + " |")
    out += ["", "Receptor unico com as k fontes mais fortes (SECON 2019, "
            "Fig. 13, valores do vetor do PDF). A coluna 'papel' diz se a "
            "curva medida e validacao ou entrada do modelo empirico.", "",
            "| modelo | caso | papel | " + " | ".join(
                f"k={k}" for k in KS_SECON) + " |",
            "|---|---|---|" + "---|" * len(KS_SECON)]
    by_case = {}
    for r in c_rows:
        by_case.setdefault(r["configuracao"], {})[r["k"]] = r["prr"]
    for (caso, _, _, tab, papel) in casos:
        for nm, _ in modelos:
            rows = by_case[f"{nm} | {caso}"]
            role = ("consistencia (entrada)" if (papel != "validacao"
                                                  and "empirico" in nm)
                    else "validacao")
            out.append(f"| {nm} | {caso} | {role} | " + " | ".join(
                f"{rows[k]:.2f}" for k in KS_SECON) + " |")
        raw = dict(tab.raw_points)          # o medido, sem o ajuste isotonico
        out.append(f"| medido | {caso} | -- | " + " | ".join(
            f"{raw[float(k)]:.2f}" for k in KS_SECON) + " |")
    return "\n".join(out) + "\n", t_rows, c_rows


def _corridor_all_k(args):
    global KS
    KS = KS_SECON
    return corridor(args)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pesos", type=float, nargs="+",
                   default=default_weights())
    p.add_argument("--topologias", type=int, default=5)
    p.add_argument("--tentativas", type=int, default=200)
    p.add_argument("--jobs", type=int,
                   default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--saida", default=os.path.join("outputs", "calibracao"))
    p.add_argument("--sem-modelos", action="store_true",
                   help="nao roda a comparacao dos modelos de captura no "
                        "radio CLOVES")
    a = p.parse_args()
    os.makedirs(a.saida, exist_ok=True)
    git_info = sim._git_info()          # o codigo que roda, lido no inicio

    # o peso assincrono pertence ao modelo de captura LEGADO (F28): a
    # calibracao roda sobre ele explicitamente, nao sobre o padrao do dia
    base = dict(sim.PRESET_LEGACY_V2)
    tarefas = [("T sincronizado (F26), peso 1",
                dict(base, CRYSTAL_T_OFFSET_NS=0.0, ASYNC_INTERF_WEIGHT=1.0),
                a.topologias, a.tentativas)]
    tarefas += [(f"T com 30,5 us, peso {w:g}",
                 dict(base, ASYNC_INTERF_WEIGHT=w), a.topologias,
                 a.tentativas) for w in a.pesos]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        t_rows = [r for bloco in ex.map(t_success, tarefas) for r in bloco]

    w_cal = sim.PRESET_LEGACY_V2["ASYNC_INTERF_WEIGHT"]
    corr = [("mesmo pacote, sincronizado", {}, 0.0, True),
            ("pacotes diferentes, sincronizados", {}, 0.0, False),
            ("pacotes diferentes, atraso ate 20 us", {}, 20000.0, False)]
    tarefas = [(n, dict(base, **c), a.topologias, a.tentativas, j, s)
               for n, c, j, s in corr]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        c_rows = [r for bloco in ex.map(corridor, tarefas) for r in bloco]

    md = render_tables(t_rows, c_rows, a.topologias, a.tentativas, w_cal)
    if not a.sem_modelos:
        md_models, t_models, c_models = compare_models(a)
        md += "\n" + md_models
        t_rows, c_rows = t_rows + t_models, c_rows + c_models
    print(md)
    with open(os.path.join(a.saida, "calibracao_captura.md"), "w",
              encoding="utf-8") as f:
        f.write(md)
    # a configuracao efetiva de CADA ponto (N_NODES incluido), por hash
    with open(os.path.join(a.saida, "calibracao_captura.json"), "w",
              encoding="utf-8") as f:
        json.dump(artifact_json(a, t_rows, c_rows, git_info), f, indent=2,
                  ensure_ascii=False, default=str)
        f.write("\n")


if __name__ == "__main__":
    main()
