# -*- coding: utf-8 -*-
"""
Microbenchmark do erro de ranging do agrupamento do Cupid.

Referencia EXTERNA, nao alvo: Park et al. (DCOSS-IoT 2025, sec. V-B): "After
taking 100 distance measurements, the average error is found to be 1.31 m",
contra a distancia verdadeira ao no mais proximo do hop de cima ou de baixo;
o artigo nao diz se e erro absoluto medio, vies ou RMSE. Nada aqui ajusta
parametro para chegar a esse numero: a tabela so mostra o erro que os
mecanismos modelados (truncamento de 8 ns do delayed TX, deriva de relogio,
ruido de borda, primeiro caminho de outro no, perdas) produzem, e como ele
varia com as hipoteses sem fonte (ppm do cristal, ruido de borda).

Uso:
    py microbenchmark_ranging.py
    py microbenchmark_ranging.py --topologias 5 --ppm 0 1.45 10 20
"""

import argparse
import math
import os
import random
import statistics

import cupid_sim_v2 as sim

PUBLISHED_AVG_ERROR_M = 1.31   # Cupid, sec. V-B (100 medicoes)


def error_stats(errors):
    """Estatisticas do erro (medido - verdadeiro, em m)."""
    if not errors:
        return {"n": 0, "mae": math.nan, "bias": math.nan, "rmse": math.nan,
                "p95_abs": math.nan}
    absolute = sorted(abs(e) for e in errors)
    return {
        "n": len(errors),
        "mae": statistics.fmean(absolute),
        "bias": statistics.fmean(errors),
        "rmse": math.sqrt(statistics.fmean(e * e for e in errors)),
        "p95_abs": absolute[min(len(absolute) - 1,
                                int(0.95 * len(absolute)))],
    }


def run_case(cfg, n_topo, base_seed):
    """Erros de todas as medicoes do periodo de agrupamento em n_topo
    topologias, com a configuracao do simulador + cfg."""
    sim.apply_config(cfg)
    errors, attempts, ok = [], 0, 0
    for topo, seed in sim.make_topologies(n_topo=n_topo, base_seed=base_seed):
        res = sim.run_grouping_period(topo, rng=random.Random(seed * 13 + 1))
        errors += list(res.measurement_errors)
        attempts += 2 * len(res.observations)
        ok += res.successful_upper + res.successful_lower
    stats = error_stats(errors)
    stats["success"] = ok / attempts if attempts else math.nan
    return stats


def default_grids():
    """Grades de ppm e de ruido de borda: sempre contem o valor em vigor
    (a linha '(padrao)'), 0 e a tolerancia sem trim do DW1000 (20 ppm)."""
    return (sorted({0.0, 1.45, sim.RANGING_CLOCK_PPM, 10.0, 20.0}),
            sorted({0.0, sim.RANGING_LDE_SIGMA_NS}))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--topologias", type=int, default=3)
    ppm_grid, lde_grid = default_grids()
    p.add_argument("--ppm", type=float, nargs="+", default=ppm_grid)
    p.add_argument("--lde", type=float, nargs="+", default=lde_grid)
    p.add_argument("--saida", default=os.path.join("outputs", "ranging"))
    a = p.parse_args()
    base = {k: getattr(sim, k) for k in sim.PARAM_NAMES}
    rows = []
    for ppm in a.ppm:
        for lde in a.lde:
            cfg = dict(base, RANGING_CLOCK_PPM=ppm, RANGING_LDE_SIGMA_NS=lde)
            st = run_case(cfg, a.topologias, sim.SEED)
            atual = (ppm == base["RANGING_CLOCK_PPM"]
                     and lde == base["RANGING_LDE_SIGMA_NS"])
            rows.append((ppm, lde, atual, st))
    out = ["## Erro de ranging do agrupamento (medido - verdadeiro)", "",
           f"Perfil {sim.RADIO_PROFILE}, {a.topologias} topologias, "
           f"M = {sim.GROUPING_ITERATIONS}. Referencia externa (nao alvo): "
           f"erro medio de {PUBLISHED_AVG_ERROR_M} m no testbed (Cupid, sec. "
           "V-B; metrica nao definida no artigo).", "",
           "| ppm do cristal | ruido de borda (ns) | medicoes | sucesso da "
           "janela | erro abs. medio (m) | vies (m) | RMSE (m) | p95 abs (m) |",
           "|---|---|---|---|---|---|---|---|"]
    for ppm, lde, atual, st in rows:
        tag = " (padrao)" if atual else ""
        out.append(f"| {ppm:g}{tag} | {lde:g} | {st['n']} | "
                   f"{st['success']:.2f} | {st['mae']:.2f} | "
                   f"{st['bias']:+.2f} | {st['rmse']:.2f} | "
                   f"{st['p95_abs']:.2f} |")
    md = "\n".join(out) + "\n"
    print(md)
    os.makedirs(a.saida, exist_ok=True)
    with open(os.path.join(a.saida, "erro_ranging.md"), "w",
              encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    main()
