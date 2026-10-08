# -*- coding: utf-8 -*-
"""
Bancada compacta para comparar configuracoes do simulador: efeito de cada
item de realismo e sensibilidade a parametros.

Uso:
    py sensibilidade.py progressao          # etapa 0 -> cada item -> atual
    py sensibilidade.py copias              # copias do mesmo pacote (P4)
    py sensibilidade.py ack captura txprob  # varios conjuntos de uma vez
    py sensibilidade.py captura limiar      # captura TEMPORAL x limiar/SINR
    py sensibilidade.py assincrono          # dessincronizacao e peso (P10)
    py sensibilidade.py cloves              # legado -> CLOVES, item a item
    py sensibilidade.py modelo_captura agrupamento
    py sensibilidade.py intra_aquisicao     # regime nao medido: pior/melhor
    py sensibilidade.py canal_memoria       # desvanecimento com memoria
    py sensibilidade.py progressao --topologias 5 --epocas 40 --jobs 8

Para cada configuracao e protocolo: fracao de epocas completas, latencia da
epoca (mediana censurada em MAX_SLOTS), latencia por relatorio (mediana dos
entregues), PDR sem prazo e PDR a 6 epocas/s (prazo de 205 slots). Resultados
em outputs/sensibilidade/<conjunto>.csv e .md.

E uma bancada de DIMENSIONAMENTO (poucas topologias e epocas): serve para ver
direcao e tamanho de efeitos, nao substitui a rodada de run_cenarios.py.
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
from concurrent.futures import ProcessPoolExecutor

import cupid_sim_v2 as sim
import experiment_stats as es

E = sim.PRESET_ETAPA0
LEGACY = sim.PRESET_LEGACY_V2
# padroes do processo (capturados no import, antes de qualquer tarefa)
DEFAULTS = {name: getattr(sim, name) for name in sim.PARAM_NAMES}


def task_config(cfg):
    """Configuracao completa da tarefa: nada vaza da tarefa anterior do
    mesmo worker."""
    return sim.fresh_config(cfg, DEFAULTS)


def _without(*keys):
    """Preset da etapa 0 sem as chaves dadas: o item volta ao valor do motor
    legado (F28), explicito em PRESET_LEGACY_V2 -- nunca ao padrao do dia."""
    return {k: (LEGACY[k] if k in LEGACY else v) if k in keys else v
            for k, v in E.items()}


def _progressao():
    """Cada passo liga um item a mais, na ordem PHY -> ACK -> retransmissao,
    sempre sobre o motor legado explicito (o padrao atual muda com o CLOVES)."""
    passos = [("etapa 0", dict(E))]
    ligados = []
    for rotulo, chaves in (
            ("+ curva de erro (F21)", ("PER_MODEL",)),
            ("+ copias do mesmo pacote (F22/F27)",
             ("SAME_PACKET_MODEL", "COMBINE_WINDOW_NS")),
            ("+ cabecalho no quadro (F27)", ("FRAME_HEADERS",)),
            ("+ ACK local do Weaver (F23)", ("LOCAL_ACK_MODEL",)),
            ("+ ACK global do Weaver (F24)", ("ACK_MODEL",)),
            ("+ mesmo hop (F25)", ("SAME_HOP_RELAY",)),
            ("+ Crystal TA e bootstrap (F26)",
             ("CRYSTAL_MODEL", "BOOTSTRAP_DELAY")),
            ("+ dessincronizacao e captura assincrona (F28) = legado v2",
             ("TX_DELAY_NS", "CRYSTAL_T_OFFSET_NS", "CAPTURE_TIME_NS",
              "PREAMBLE_ALIGN_NS", "ASYNC_INTERF_WEIGHT"))):
        ligados.extend(chaves)
        passos.append((rotulo, _without(*ligados)))
    assert passos[-1][1] == {k: LEGACY[k] for k in E}
    passos[-1] = (passos[-1][0], dict(LEGACY))
    return passos


def _varredura(param, valores, fmt="{:g}", unidade=""):
    """'(padrao)' e SEMPRE o valor em vigor, sem sobrescrita; os demais itens
    mudam so esse parametro e nunca repetem o padrao."""
    atual = getattr(sim, param)
    itens = [(f"{param} {fmt.format(atual)}{unidade} (padrao)", {})]
    for v in valores:
        if v != atual:
            itens.append((f"{param} {fmt.format(v)}{unidade}", {param: v}))
    return itens


def _cloves_progressao():
    """Do motor legado ao padrao CLOVES, ligando UMA mudanca por passo (cada
    passo acumula os anteriores). Os parametros que o perfil de radio deriva
    saem da configuracao quando o perfil muda, para nao virarem override."""
    derived = set(sim.RADIO_DERIVED_PARAMS)
    passos = [("legado v2 (5ef525b)", dict(LEGACY))]
    cfg = dict(LEGACY)
    for rotulo, mud in (
            ("+ radio DW1000 canal 4 (Fase 2)",
             {"RADIO_PROFILE": "cloves_dw1000_ch4"}),
            ("+ bootstrap fisico (Fase 3)", {"BOOTSTRAP_MODEL": "physical"}),
            ("+ agrupamento por ranging (Fase 4)",
             {"GROUPING_MODEL": "ranging"}),
            ("+ sorteios por chave (Fase 7)", {"RNG_MODEL": "keyed"}),
            ("+ grade da Fig. 5b do Cupid (Fase 5)",
             {"CUPID_SCHEDULE": "fig5b"}),
            ("+ TX_PROB 1 (Fase 5)", {"TX_PROB": 1.0}),
            ("+ fases do Crystal com airtime e A negativo (Fase 5)",
             {"CRYSTAL_PHASE_TIMING": "airtime",
              "CRYSTAL_A_MODEL": "physical"}),
            ("+ captura empirica = CLOVES (Fase 6)",
             {"CAPTURE_MODEL": "empirical_uwb"})):
        if "RADIO_PROFILE" in mud:
            cfg = {k: v for k, v in cfg.items() if k not in derived}
        cfg = dict(cfg, **mud)
        passos.append((rotulo, dict(cfg)))
    return passos


def conjuntos():
    """Conjuntos montados a partir dos valores EM VIGOR, para que o rotulo
    '(padrao)' nunca fique velho quando um padrao muda."""
    t_sym = sim.PREAMBLE_SYMBOL_NS
    pac = sim.CAPTURE_TIME_NS / t_sym
    captura = [(f"aquisicao {sim.CAPTURE_TIME_NS:g} ns = {pac:.0f} simbolos "
                f"de preambulo; roubo com {sim.PREAMBLE_ALIGN_NS:g} ns de "
                f"alinhamento (padrao)", {})]
    # [F28] janela de aquisicao em PACs de 4/16/32 simbolos, o F3 antigo
    # (first-path de 40 ns) e o roubo pos-aquisicao desligado ou dobrado
    for simbolos in (4, 16, 32):
        v = round(simbolos * t_sym, 1)
        if abs(v - sim.CAPTURE_TIME_NS) > 0.5:
            captura.append((f"aquisicao {v:g} ns = PAC {simbolos}",
                            {"CAPTURE_TIME_NS": v}))
    captura.append(("aquisicao first-path de 40 ns (F3), sem roubo",
                    {"CAPTURE_TIME_NS": 40.0, "PREAMBLE_ALIGN_NS": 0.0}))
    for v in (0.0, 2 * sim.PREAMBLE_ALIGN_NS):
        if v != sim.PREAMBLE_ALIGN_NS:
            captura.append((f"roubo pos-aquisicao {v:g} ns",
                            {"PREAMBLE_ALIGN_NS": v}))
    limiar = [(f"CAPTURE_SIR_DB {sim.CAPTURE_SIR_DB:g}, REQ_SNR_DB "
               f"{sim.REQ_SNR_DB:g}, PER_WIDTH_DB {sim.PER_WIDTH_DB:g} "
               f"(padrao)", {})]
    for param, valores in (("CAPTURE_SIR_DB", (1.0, 6.0)),
                           ("REQ_SNR_DB", (1.0, 5.0)),
                           ("PER_WIDTH_DB", (1.0, 4.0))):
        limiar += [item for item in _varredura(param, valores)[1:]]
    assincrono = [(f"atual (atraso {sim.TX_DELAY_NS / 1000:g} us, captura "
                   f"{sim.CAPTURE_MODEL}) (padrao)", {})]
    if sim.TX_DELAY_NS or sim.CRYSTAL_T_OFFSET_NS:
        assincrono.append(("sem atraso (CTX e T sincronizados)",
                           {"TX_DELAY_NS": 0.0, "CRYSTAL_T_OFFSET_NS": 0.0}))
    # o peso assincrono e parametro do modelo de captura LEGADO
    assincrono += [(f"captura legada, peso {w:g}",
                    {"CAPTURE_MODEL": "legacy_async_weight",
                     "ASYNC_INTERF_WEIGHT": w})
                   for w in (1.0, 0.1, 0.06, 0.03)]
    return {
        "progressao": _progressao(),
        "assincrono": assincrono,
        "copias": [(f"{sim.SAME_PACKET_MODEL} {sim.COMBINE_WINDOW_NS:g} ns "
                    f"(padrao)", {})]
        + [(f"nondestructive {w:g} ns",
            {"SAME_PACKET_MODEL": "nondestructive", "COMBINE_WINDOW_NS": w})
           for w in (128.0, 32.0, 8.0)
           if (w, "nondestructive") != (sim.COMBINE_WINDOW_NS,
                                        sim.SAME_PACKET_MODEL)]
        + [("interference (captura legada)",
            {"SAME_PACKET_MODEL": "interference",
             "CAPTURE_MODEL": "legacy_async_weight"}),
           ("additive 4 ns (antigo, captura legada)",
            {"SAME_PACKET_MODEL": "additive", "COMBINE_WINDOW_NS": 4.0,
             "CAPTURE_MODEL": "legacy_async_weight"})],
        "ack": [(f"{sim.ACK_MODEL} (padrao)", {})]
        + [(m, {"ACK_MODEL": m}) for m in ("weaver", "physical", "oracle",
                                            "none") if m != sim.ACK_MODEL],
        "captura": captura,
        "limiar": limiar,
        "txprob": _varredura("TX_PROB", (0.60, 0.75, 0.85, 1.0), "{:.2f}"),
        "cloves": _cloves_progressao(),
        "modelo_captura": [
            (f"{sim.CAPTURE_MODEL} ({sim.CAPTURE_EXTRAPOLATION}) (padrao)", {})]
        + [(f"empirico, extrapolacao {m}",
            {"CAPTURE_MODEL": "empirical_uwb", "CAPTURE_EXTRAPOLATION": m})
           for m in ("lower", "mid", "upper")
           if (m, "empirical_uwb") != (sim.CAPTURE_EXTRAPOLATION,
                                       sim.CAPTURE_MODEL)]
        + [("legado, peso 0,06", {"CAPTURE_MODEL": "legacy_async_weight",
                                  "ASYNC_INTERF_WEIGHT": 0.06})],
        "agrupamento": _varredura("RANGING_CLOCK_PPM", (0.0, 10.0, 20.0))
        + _varredura("GROUPING_ITERATIONS", (1, 20))[1:]
        + [("agrupamento estatico (oraculo de distancia, legado)",
            {"GROUPING_MODEL": "static"})],
        # regime NAO MEDIDO: concorrentes todos dentro da aquisicao (o
        # atraso <= 1 us dos CTX), entre o pior e o melhor caso medidos
        "intra_aquisicao": _intra_aquisicao(),
        # canal com memoria: parte lenta (fracao w) constante na epoca e
        # correlacionada entre epocas (rho); rho = 0 isola a memoria DENTRO
        # da epoca, w = rho = 1 e o ambiente estatico
        "canal_memoria": [
            (f"{sim.FADING_MODEL}, {sim.FAST_FADE_DB:g} dB por slot "
             f"(padrao)", {})]
        + [(f"duas escalas, w = {w:g}, rho entre epocas = {r:g}",
            {"FADING_MODEL": "two_scale", "FADING_SLOW_FRACTION": w,
             "FADING_EPOCH_CORR": r})
           for w, r in ((0.5, 0.9), (0.9, 0.0), (0.9, 0.9), (0.9, 0.99),
                        (1.0, 1.0))],
    }


def _intra_aquisicao():
    rotulo = {"async_table": "curva de ate 20 us, melhor caso medido",
              "sync_table": "curva sincronizada, pior caso medido",
              "no_penalty": "sem penalidade, limite teorico sem medicao"}
    out = [(f"{rotulo[sim.CAPTURE_INTRA_ACQUISITION]}, extrapolacao "
            f"{sim.CAPTURE_EXTRAPOLATION} (padrao)", {})]
    for mode, extra in (("sync_table", "mid"), ("sync_table", "lower"),
                        ("async_table", "upper"), ("no_penalty", "mid")):
        cfg = {"CAPTURE_INTRA_ACQUISITION": mode}
        if extra != sim.CAPTURE_EXTRAPOLATION:
            cfg["CAPTURE_EXTRAPOLATION"] = extra
        if cfg == {"CAPTURE_INTRA_ACQUISITION": sim.CAPTURE_INTRA_ACQUISITION}:
            continue                    # e o padrao, ja listado
        out.append((f"{rotulo[mode]}, extrapolacao {extra}", cfg))
    return out


CONJUNTOS = conjuntos()


def bancada(args):
    """Roda uma configuracao num processo proprio (o simulador usa globais).
    [Fase 7/8] Usa o mesmo gerador pareado dos experimentos
    (simulate_topology): no modo por chave todos os protocolos recebem o
    mesmo workload; no legado, as mesmas sementes de antes."""
    nome, cfg, n_topo, epocas, originadores = args
    sim.apply_config(task_config(cfg))
    topos = sim.make_topologies(n_topo=n_topo, base_seed=sim.SEED)
    budget_6 = int(1000.0 / 6 * 1000.0 / sim.SLOT_US)
    tau = sim.epoch_cap_ms()
    linhas = []
    for proto in sim.PROTOCOLS:
        sent = completas = entregues = sent6 = entregues6 = 0
        lats, rels, rmst = [], [], []
        por_topo = {}                   # RMST de cada topologia (pareamento)
        for topo, seed in topos:
            off = sim.PROTO_SEED_OFFSET[proto]
            run = sim.simulate_topology(proto, topo, seed, originadores,
                                        epocas, tag="sens",
                                        legacy_seed=seed * 7919 + off)
            lt = [r["latency_data_epoch_ms"] for r in run["records"]]
            rmst.append(es.summarize_censored_latency(lt, tau)["rmst"])
            por_topo[str(seed)] = rmst[-1]
            for r in run["records"]:
                o = len(r["originators"])
                sent += o
                entregues += r["delivered"]
                completas += r["complete"]
                lats.append(r["latency_data_epoch_ms"])
                rels.extend(sim.slots_to_ms(s)
                            for s in r["delivery_slots"].values())
            run6 = sim.simulate_topology(proto, topo, seed, originadores,
                                         epocas, slot_budget=budget_6,
                                         tag="sens6",
                                         legacy_seed=seed * 104729 + off)
            for r in run6["records"]:
                sent6 += len(r["originators"])
                entregues6 += r["delivered"]
        m, ci = sim.mean_ci(rmst)
        linhas.append({
            "configuracao": nome, "protocolo": proto,
            "epocas_completas": completas / len(lats),
            "lat_epoca_ms": sim.censored_median(lats),
            "rmst_ms": m, "rmst_ic_ms": ci,
            "lat_relatorio_ms": statistics.median(rels) if rels else math.nan,
            "pdr_sem_prazo": entregues / sent,
            "pdr_6eps": entregues6 / sent6,
            "rmst_por_topologia": por_topo,
        })
    return linhas


PARES = (("cupid", "weaver"), ("cupid", "crystal"), ("weaver", "crystal"))


def comparacoes_pareadas(linhas, pares=PARES):
    """Razao de RMST a/b topologia a topologia (a unidade independente):
    media, faixa e em quantas topologias 'a' foi mais rapido."""
    por_cfg = {}
    for r in linhas:
        por_cfg.setdefault(r["configuracao"], {})[r["protocolo"]] = (
            r["rmst_por_topologia"])
    out = []
    for cfg, protos in por_cfg.items():
        for a, b in pares:
            if a not in protos or b not in protos:
                continue
            seeds = sorted(set(protos[a]) & set(protos[b]))
            razoes = [protos[a][s] / protos[b][s] for s in seeds]
            out.append({"configuracao": cfg, "a": a, "b": b,
                        "razao_media": statistics.fmean(razoes),
                        "razao_min": min(razoes), "razao_max": max(razoes),
                        "a_mais_rapido": sum(x < 1.0 for x in razoes),
                        "n": len(razoes)})
    return out


def tabela_pareada(comps):
    out = ["| configuracao | comparacao | razao de RMST (media) | faixa entre "
           "topologias | mais rapido em |", "|---|---|---|---|---|"]
    for c in comps:
        out.append(f"| {c['configuracao']} | {sim.LABELS[c['a']]} / "
                   f"{sim.LABELS[c['b']]} | {c['razao_media']:.2f} | "
                   f"{c['razao_min']:.2f}-{c['razao_max']:.2f} | "
                   f"{c['a_mais_rapido']}/{c['n']} |")
    return "\n".join(out)


def _fmt(v, kind):
    if kind == "pct":
        return f"{100 * v:.1f}%"
    if math.isinf(v):
        return f">= {sim.epoch_cap_ms():.0f}"
    if math.isnan(v):
        return "n/d"
    return f"{v:.0f}"


def tabela(linhas):
    cab = ("| configuracao | protocolo | epocas completas | lat. epoca (ms) "
           "| RMST (ms, IC t) | lat. relatorio (ms) | PDR sem prazo | "
           "PDR 6 ep/s |")
    out = [cab, "|---|---|---|---|---|---|---|---|"]
    for r in linhas:
        rm = (f"{r['rmst_ms']:.0f}" + (f" +- {r['rmst_ic_ms']:.0f}"
                                        if not math.isnan(r['rmst_ic_ms'])
                                        else ""))
        out.append(
            f"| {r['configuracao']} | {sim.LABELS[r['protocolo']]} | "
            f"{_fmt(r['epocas_completas'], 'pct')} | "
            f"{_fmt(r['lat_epoca_ms'], 'ms')} | {rm} | "
            f"{_fmt(r['lat_relatorio_ms'], 'ms')} | "
            f"{_fmt(r['pdr_sem_prazo'], 'pct')} | "
            f"{_fmt(r['pdr_6eps'], 'pct')} |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("conjuntos", nargs="+", choices=sorted(CONJUNTOS))
    p.add_argument("--topologias", type=int, default=5)
    p.add_argument("--epocas", type=int, default=20)
    p.add_argument("--originadores", type=int, default=35)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)
                                                   // 2))
    p.add_argument("--saida", default=os.path.join("outputs",
                                                   "sensibilidade"))
    a = p.parse_args()
    os.makedirs(a.saida, exist_ok=True)
    for conjunto in a.conjuntos:
        tarefas = [(nome, cfg, a.topologias, a.epocas, a.originadores)
                   for nome, cfg in CONJUNTOS[conjunto]]
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            linhas = [r for bloco in ex.map(bancada, tarefas) for r in bloco]
        md = (f"## {conjunto} -- {a.topologias} topologias x {a.epocas} "
              f"epocas, O={a.originadores}\n\n{tabela(linhas)}\n\n"
              f"Comparacoes pareadas por topologia (razao < 1: o primeiro e "
              f"mais rapido):\n\n"
              f"{tabela_pareada(comparacoes_pareadas(linhas))}\n")
        print(md)
        with open(os.path.join(a.saida, f"{conjunto}.md"), "w",
                  encoding="utf-8") as f:
            f.write(md)
        with open(os.path.join(a.saida, f"{conjunto}.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0]))
            w.writeheader()
            w.writerows(dict(r, rmst_por_topologia=json.dumps(
                r["rmst_por_topologia"])) for r in linhas)


if __name__ == "__main__":
    main()
