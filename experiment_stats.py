# -*- coding: utf-8 -*-
"""
Workloads pareados, estimadores com censura, intervalos de confianca e
exportacao de amostras.

Regras que este modulo impoe:
  * A unidade independente e a TOPOLOGIA. Epocas da mesma topologia sao
    medidas repetidas: primeiro se resume cada topologia, depois se compara
    entre topologias (duplicar epocas nao cria topologias).
  * Comparacoes entre protocolos sao PAREADAS por topologia (e, dentro dela,
    pelo mesmo WorkloadPlan): IC t de Student com n - 1 graus de liberdade,
    ou bootstrap pareado reamostrando topologias. Nada de 1,96 fixo.
  * Epoca incompleta e CENSURADA no prazo, nunca descartada nem trocada pelo
    prazo "como se fosse" o valor: o que se reporta e a RMST ate o prazo
    (restricted mean survival time), a fracao completa e a mediana so quando
    ela e identificavel (senao, o limite inferior).
"""

import csv
import math
import random
from dataclasses import dataclass

import rng_streams as rs


# ---------------------------------------------------------------------------
# t de Student (sem SciPy): CDF pela beta incompleta regularizada
# ---------------------------------------------------------------------------

def _betacf(a, b, x, eps=3e-16, itmax=400):
    """Fracao continuada da beta incompleta (Numerical Recipes, Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    x = df / (df + t * t)
    tail = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_critical(conf, df):
    """Quantil bicaudal: P(|T| <= t) = conf, com df graus de liberdade."""
    if df < 1:
        raise ValueError("t_critical exige df >= 1")
    target = 0.5 + conf / 2.0
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < target:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mean_ci_t(values, conf=0.95):
    """(media, semi-largura do IC t, n). Com n < 2 a semi-largura e nan: um
    unico valor nao da incerteza, e fingir 0 seria afirmar precisao."""
    vals = [float(v) for v in values if not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return math.nan, math.nan, 0
    m = sum(vals) / n
    if n == 1:
        return m, math.nan, 1
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    return m, t_critical(conf, n - 1) * sd / math.sqrt(n), n


# ---------------------------------------------------------------------------
# Comparacao pareada entre topologias
# ---------------------------------------------------------------------------

def _paired_diffs(a_by_topo, b_by_topo):
    if set(a_by_topo) != set(b_by_topo):
        raise ValueError("comparacao pareada exige as mesmas topologias nos "
                         f"dois lados: {sorted(a_by_topo)} x "
                         f"{sorted(b_by_topo)}")
    keys = sorted(a_by_topo)
    return keys, [a_by_topo[k] - b_by_topo[k] for k in keys]


def paired_effect(a_by_topo, b_by_topo, conf=0.95):
    """Diferenca media a - b entre topologias, com IC t pareado."""
    keys, diffs = _paired_diffs(a_by_topo, b_by_topo)
    m, half, n = mean_ci_t(diffs, conf)
    return {"n": n, "keys": keys, "diffs": diffs, "mean_diff": m,
            "ci_low": m - half, "ci_high": m + half,
            "t_crit": t_critical(conf, n - 1) if n >= 2 else math.nan,
            "conf": conf, "method": "t pareado entre topologias"}


def paired_bootstrap_ci(a_by_topo, b_by_topo, n_boot=10000, conf=0.95,
                        seed=0):
    """IC percentil da diferenca media, reamostrando TOPOLOGIAS (pares)."""
    _, diffs = _paired_diffs(a_by_topo, b_by_topo)
    if len(diffs) < 2:
        return math.nan, math.nan
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n
                   for _ in range(n_boot))
    lo_i = int(math.floor((1.0 - conf) / 2.0 * n_boot))
    hi_i = int(math.ceil((1.0 + conf) / 2.0 * n_boot)) - 1
    return means[lo_i], means[min(hi_i, n_boot - 1)]


# ---------------------------------------------------------------------------
# Latencia censurada
# ---------------------------------------------------------------------------

def _as_observations(samples, tau):
    """Aceita latencias (inf/None = censurada no prazo) ou pares
    (tempo, evento). Devolve [(tempo, evento)] com tempo <= tau."""
    obs = []
    for s in samples:
        if isinstance(s, tuple):
            t, event = float(s[0]), bool(s[1])
        elif s is None or math.isinf(s):
            t, event = tau, False
        else:
            t, event = float(s), True
        if t > tau:
            t, event = tau, False
        obs.append((t, event))
    return obs


def kaplan_meier(samples, tau):
    """Pontos (t, S(t)) da curva de sobrevivencia ate tau."""
    obs = sorted(_as_observations(samples, tau))
    at_risk = len(obs)
    s = 1.0
    curve = [(0.0, 1.0)]
    i = 0
    while i < len(obs):
        t = obs[i][0]
        events = censored = 0
        while i < len(obs) and obs[i][0] == t:
            events += obs[i][1]
            censored += not obs[i][1]
            i += 1
        if events and at_risk:
            s *= 1.0 - events / at_risk
            curve.append((t, s))
        at_risk -= events + censored
    return curve


def restricted_mean_survival_time(samples, tau):
    """Area sob a curva de Kaplan-Meier de 0 a tau. Com censura so no prazo
    ela e exatamente a media de min(T, tau) -- que e o nome certo do
    'substituir infinito pelo prazo': RMST, nao media."""
    if not samples:
        return math.nan
    curve = kaplan_meier(samples, tau)
    area = 0.0
    for (t0, s0), (t1, _) in zip(curve, curve[1:] + [(tau, None)]):
        area += s0 * (min(t1, tau) - t0)
    return area


def summarize_censored_latency(latencies, tau):
    """Resumo de uma amostra de latencias com censura no prazo tau."""
    n = len(latencies)
    if n == 0:
        return {"n": 0, "n_complete": 0, "frac_complete": math.nan,
                "median": None, "median_lower_bound": None,
                "rmst": math.nan, "rmst_label": "RMST ate o prazo",
                "tau": tau, "status": "vazio"}
    finite = sorted(v for v in latencies
                    if v is not None and not math.isinf(v) and v <= tau)
    n_complete = len(finite)
    frac = n_complete / n
    median = lower = None
    if n_complete * 2 > n:
        # a mediana so e identificavel se MAIS da metade completou
        ordered = finite + [math.inf] * (n - n_complete)
        mid = n // 2
        median = (ordered[mid] if n % 2 else
                  0.5 * (ordered[mid - 1] + ordered[mid]))
    elif n % 2 == 0 and n_complete * 2 == n and n_complete:
        # exatamente metade: mediana = (ultimo completo + um censurado) / 2
        lower = 0.5 * (finite[-1] + tau)
    else:
        lower = tau
    status = ("todas censuradas" if n_complete == 0 else
              "mediana identificavel" if median is not None else
              "mediana nao identificavel")
    return {"n": n, "n_complete": n_complete, "frac_complete": frac,
            "median": median, "median_lower_bound": lower,
            "rmst": restricted_mean_survival_time(latencies, tau),
            "rmst_label": "RMST ate o prazo", "tau": tau, "status": status}


# ---------------------------------------------------------------------------
# Workload pareado
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpochWorkload:
    epoch: int
    originators: tuple
    generation_slot: int
    payload_bytes: int
    deadline_slots: int
    workload_id: str


@dataclass(frozen=True)
class WorkloadPlan:
    """Workload de uma topologia: o MESMO para todos os protocolos."""
    base_seed: int
    topology_seed: int
    tag: str
    n_originators: int
    epochs: tuple

    @property
    def plan_id(self):
        return (f"{self.base_seed}/{self.topology_seed}/{self.tag}/"
                f"O{self.n_originators}/E{len(self.epochs)}")


def make_workload_plan(base_seed, topology_seed, candidates, n_originators,
                       n_epochs, payload_bytes, deadline_slots, tag):
    """Originadores de cada epoca sorteados por chave (base, topologia, tag,
    epoca) sobre os candidatos em ordem canonica: nao depende de protocolo,
    da ordem dos candidatos nem de quantas epocas outro experimento rodou.
    Todos geram no inicio da epoca (slot 0), como no testbed do Cupid."""
    cand = sorted(set(candidates))
    o = min(int(n_originators), len(cand))
    epochs = []
    for e in range(int(n_epochs)):
        rng = rs.stream(base_seed, topology_seed, "workload", tag, o, e)
        origs = tuple(sorted(rng.sample(cand, o)))
        epochs.append(EpochWorkload(
            epoch=e, originators=origs, generation_slot=0,
            payload_bytes=int(payload_bytes),
            deadline_slots=int(deadline_slots),
            workload_id=f"{base_seed}/{topology_seed}/{tag}/O{o}/e{e}"))
    return WorkloadPlan(base_seed=base_seed, topology_seed=topology_seed,
                        tag=tag, n_originators=o, epochs=tuple(epochs))


# ---------------------------------------------------------------------------
# Exportacao de amostras brutas
# ---------------------------------------------------------------------------

def export_samples(path, rows, fields):
    """CSV com uma linha por (protocolo, topologia, epoca). Latencia
    censurada vai vazia com 'completa' = 0 -- nunca o prazo no lugar."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (v is None or (isinstance(v, float)
                                                 and math.isinf(v))) else v)
                        for k, v in r.items()})
