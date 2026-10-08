# -*- coding: utf-8 -*-
"""
Bootstrap fisico por epoca e agrupamento emitter/collector do Cupid.

Nada aqui tem fisica propria: toda recepcao passa pelas funcoes do motor
(decode_physical_channels, sample_slot_fading, sample_decode_draws), que sao
injetadas pelo argumento 'phy' -- o mesmo caminho de qualquer quadro de
dados. A topologia estatica (posicoes, perda, shadowing) nunca e alterada:
o estado da epoca vive em BootstrapState e numa EpochTopology que so troca
hop e grupo.

Fontes: Cupid (Park et al., DCOSS-IoT 2025, sec. IV-A/IV-B e Alg. 1);
Weaver (Trobinger et al., SenSys 2020, sec. 3.3 e Tab. 1).
"""

from collections import defaultdict
from dataclasses import dataclass

NO_HOP = 99          # hop de quem nao participa da epoca (como na BFS)


@dataclass(frozen=True)
class BootstrapState:
    """Resultado fisico do bootstrap de UMA epoca.

    hop_by_node      : so os nos sincronizados; relay count do primeiro
                       quadro de sync aceito + 1 (o sink tem hop 0).
    first_sync_slot  : slot (local ao bootstrap) do primeiro sync aceito; o
                       sink tem -1. O no pode transmitir a partir do slot
                       seguinte.
    parent           : no do primeiro caminho do sync aceito (o que
                       decode_at_receiver devolve como first_path).
    bootstrap_slots  : slots ate a ultima transmissao de sync.
    failure_reason   : no -> motivo de ficar fora da epoca ('no_sync').
    clock_ns         : offset de relogio de cada no nesta epoca: a
                       sincronizacao e o que o bootstrap estabelece, entao
                       todos os protocolos da epoca usam os mesmos offsets.
    """
    hop_by_node: dict
    synchronized_nodes: frozenset
    first_sync_slot: dict
    parent: dict
    bootstrap_slots: int
    failure_reason: dict
    clock_ns: tuple

    def start_slot(self, v):
        """Primeiro slot em que o no v participa; None se esta inativo."""
        if v not in self.synchronized_nodes:
            return None
        return self.first_sync_slot[v] + 1

    def last_sync_slot(self):
        return max(self.first_sync_slot.values())


class EpochTopology:
    """Topologia vista numa epoca: a fisica estatica e compartilhada (por
    delegacao, nunca copiada nem alterada); hop e grupo sao os da epoca."""

    def __init__(self, static, hop, group, bootstrap):
        self._static = static
        self.hop = hop
        self.group = group
        self.bootstrap = bootstrap

    def __getattr__(self, name):
        return getattr(self.__dict__["_static"], name)

    @property
    def static(self):
        return self._static


def epoch_view(topo, state, group=None):
    """EpochTopology com os hops do bootstrap; inativos ficam em NO_HOP."""
    static = topo.static if isinstance(topo, EpochTopology) else topo
    hop = {v: state.hop_by_node.get(v, NO_HOP) for v in range(static.n)}
    return EpochTopology(static, hop, dict(group if group is not None
                                          else static.group), state)


def run_bootstrap(topo, rng, *, phy, n_tx, relay_period, max_slots,
                  clock_jitter_ns, tx_delay_ns, phy_stats=None,
                  frame_kind="sync"):
    """Onda de sincronizacao iniciada pelo sink, simulada fisicamente.

    Cada no sincronizado retransmite o sync n_tx vezes, a primeira no slot
    seguinte ao do sync aceito e as demais a cada relay_period slots (Weaver:
    B copias em slots TX consecutivos do esquema TX-RX-RX; Glossy/Crystal:
    relay_period = 1). O quadro leva o relay count, entao copias com
    contadores diferentes sao quadros DIFERENTES e disputam por captura;
    copias com o mesmo contador seguem SAME_PACKET_MODEL. Ordem de consumo do
    rng: offsets de relogio, depois, por slot, fading, sorteios de decodificacao
    e atrasos de TX -- sempre em ordem canonica.
    """
    n = topo.n
    clock = tuple(rng.gauss(0.0, clock_jitter_ns) for _ in range(n))
    hop = {0: 0}
    first = {0: -1}
    parent = {}
    sent = defaultdict(int)
    last_tx = -1
    for slot in range(max_slots):
        txs = [v for v in sorted(hop)
               if sent[v] < n_tx and slot > first[v]
               and (slot - first[v] - 1) % relay_period == 0]
        if not txs:
            if all(sent[v] >= n_tx for v in hop):
                break
            continue
        for v in txs:
            sent[v] += 1
        last_tx = slot
        receivers = [v for v in range(n) if v not in hop]
        if not receivers:
            continue
        frame_of = {v: (frame_kind, hop[v]) for v in txs}
        fading = phy.sample_slot_fading(receivers, txs, topo, rng)
        draws = phy.sample_decode_draws(receivers, [0], rng)
        slot_clock = clock
        if tx_delay_ns > 0.0:
            slot_clock = list(clock)
            for v in txs:
                slot_clock[v] += rng.random() * tx_delay_ns
        accepted = {}
        for rx in receivers:
            (_, res), = phy.decode_physical_channels(
                rx, [(0, txs)], frame_of, topo, slot_clock, fading,
                max_channels=1, phy_stats=phy_stats, draws=draws)
            if res is not None:
                frame, src = res[0], res[1]
                accepted[rx] = (frame[1] + 1, src)
        for rx, (h, src) in accepted.items():
            hop[rx] = h
            first[rx] = slot
            parent[rx] = src
    return BootstrapState(
        hop_by_node=dict(hop),
        synchronized_nodes=frozenset(hop),
        first_sync_slot=dict(first),
        parent=dict(parent),
        bootstrap_slots=last_tx + 1,
        failure_reason={v: "no_sync" for v in range(n) if v not in hop},
        clock_ns=clock)


# ===========================================================================
# Periodo de agrupamento (Cupid, sec. IV-B, Alg. 1)
# ===========================================================================

@dataclass(frozen=True)
class GroupingConfig:
    """Parametros do periodo de agrupamento (classe de origem ao lado)."""
    iterations: int                 # M = 10 [medido: Cupid V-B]
    slot_us: float                  # 460 us [medido: Cupid V-B]
    bootstrap_slots: int            # 10 [hipotese: "bootstrapping iteration
                                    # is set to 10", lido como slots]
    wait_slots: int                 # dT_wait em slots [hipotese: 1; poll e
                                    # duas janelas sao os 3 slots por no]
    clock_ppm: float                # desvio maximo do cristal de cada no
                                    # [hipotese + sensibilidade]
    lde_sigma_ns: float             # ruido da deteccao do primeiro caminho
                                    # [hipotese]
    delayed_tx_granularity_ns: float  # [spec] UM 3.3: 8,013 ns
    sink_responds: bool = True      # [hipotese] o sink responde ao hop 1


@dataclass(frozen=True)
class RangingObservation:
    iteration: int
    node: int
    hop: int
    d_up_m: object                  # float ou None (medicao perdida)
    d_down_m: object
    up_first_node: object           # no do primeiro caminho medido
    down_first_node: object
    true_up_m: object               # distancia verdadeira ao no mais
    true_down_m: object             # proximo do hop alvo (so para o erro)
    value: float                    # hop + 0,25 / 0,5 / 0,75
    case: str                       # both | up_only | down_only | none


@dataclass(frozen=True)
class GroupingResult:
    virtual_hop: dict               # no -> media das iteracoes (None: nunca
                                    # participou)
    observations: tuple
    successful_upper: int
    successful_lower: int
    both_failed: int
    total_slots: int
    total_time_ms: float
    iterations: int
    measurement_errors: tuple       # medido - verdadeiro (m)
    operations: tuple               # (iteracao, tipo, slots)


def virtual_hop_value(hop, d_up, d_down):
    """Os quatro casos do Alg. 1 (linhas 7-13): (a) d_up < d_down ou so d_up
    -> hop + 0,25; (b) d_up > d_down ou so d_down -> hop + 0,75; (c) iguais
    ou as duas falharam -> hop + 0,5."""
    if d_up is not None and d_down is not None:
        if d_up < d_down:
            return hop + 0.25, "both"
        if d_up > d_down:
            return hop + 0.75, "both"
        return hop + 0.5, "both"
    if d_up is not None:
        return hop + 0.25, "up_only"
    if d_down is not None:
        return hop + 0.75, "down_only"
    return hop + 0.5, "none"


def assign_epoch_groups(grouping, state):
    """Grupo de cada no numa epoca de dados (Cupid, sec. IV-B): emitter se o
    virtual esta mais perto do hop atual h do que de h + 1; collector se
    esta mais perto de h + 1. Empate exato (h + 0,5) e virtual desconhecido
    -> emitter [hipotese; a Fig. 8 desenha o 2,5 como emitter]."""
    groups = {}
    for v in range(len(state.clock_ns)):
        h = state.hop_by_node.get(v)
        vh = grouping.virtual_hop.get(v)
        if h is None or h == 0 or vh is None:
            groups[v] = "emitter"
            continue
        groups[v] = "collector" if vh - (h + 0.5) > 1e-9 else "emitter"
    return groups


def _nearest(topo, node, candidates, dist):
    ds = [dist(topo.pos[node], topo.pos[v]) for v in candidates]
    return min(ds) if ds else None


def run_grouping_period(topo, config, rng, *, phy, dist, c_m_per_ns,
                        phy_stats=None):
    """M iteracoes de bootstrap + ranging concorrente (Cupid, Alg. 1).

    Cada no, em ordem de ID, ocupa 3 slots: poll (quadro curto com ID e
    hop), janela de cima (respondem os do hop h - 1 que DECODIFICARAM o
    poll, apos dT_wait) e janela de baixo (hop h + 1, apos 2 dT_wait). A
    janela so rende medicao se o poller decodificar alguma resposta -- as
    respostas sao quadros diferentes (IDs diferentes) e disputam por captura.
    T_RX1 e o primeiro caminho detectavel da janela (Cupid: "the packet from
    the closest node is received first"). ToF = (T_RX - T_TX - k dT)/2, com
    os erros que o DW1000 de fato tem: truncamento do delayed TX (UM 3.3),
    deriva de relogio 1/2 k dT (e_A - e_B) (UM 12.2) e ruido de borda. A
    distancia verdadeira (dist sobre topo.pos) so entra no erro.
    """
    n = topo.n
    ppm = [rng.uniform(-config.clock_ppm, config.clock_ppm) * 1e-6
           for _ in range(n)]
    wait_ns = config.wait_slots * config.slot_us * 1e3
    detect = phy.NOISE_EFF_LIN * 10.0 ** (phy.REQ_SNR_DB / 10.0)
    observations, operations, errors = [], [], []
    up_ok = down_ok = none = 0

    def measure(node, responders, k):
        if not responders:
            return None, None
        fading = phy.sample_slot_fading([node], responders, topo, rng)
        draws = phy.sample_decode_draws([node], [0], rng)
        clock = [0.0] * n
        for r in responders:
            trunc = rng.random() * config.delayed_tx_granularity_ns
            drift = k * wait_ns * (ppm[node] - ppm[r])
            clock[r] = topo.tprop_ns[node][r] - trunc + drift
        (_, res), = phy.decode_physical_channels(
            node, [(0, responders)], {r: ("resp", r) for r in responders},
            topo, clock, fading, max_channels=1, phy_stats=phy_stats,
            draws=draws)
        lde = rng.gauss(0.0, config.lde_sigma_ns)
        if res is None:
            return None, None
        plin = topo.ps_lin[node]
        seen = [r for r in responders
                if plin[r] * fading[(r, node)] >= detect] or [res[1]]
        first = min(seen, key=lambda r: (clock[r] + topo.tprop_ns[r][node],
                                         r))
        round_trip = clock[first] + topo.tprop_ns[first][node] + lde
        return c_m_per_ns * round_trip / 2.0, first

    for it in range(config.iterations):
        state = phy.run_bootstrap(topo, rng, max_slots=config.bootstrap_slots,
                                  phy_stats=phy_stats)
        operations.append((it, "bootstrap", config.bootstrap_slots))
        hop = state.hop_by_node
        synced = sorted(state.synchronized_nodes)
        for node in range(1, n):
            operations += [(it, "poll", 1), (it, "upper", 1),
                           (it, "lower", 1)]
            if node not in hop:
                continue
            h = hop[node]
            listeners = [v for v in synced if v != node]
            fading = phy.sample_slot_fading(listeners, [node], topo, rng)
            draws = phy.sample_decode_draws(listeners, [0], rng)
            heard = []
            for v in listeners:
                (_, res), = phy.decode_physical_channels(
                    v, [(0, [node])], {node: ("poll", node)}, topo,
                    list(state.clock_ns), fading, max_channels=1,
                    phy_stats=phy_stats, draws=draws)
                if res is not None:
                    heard.append(v)
            upper = [v for v in heard if hop[v] == h - 1
                     and (v != 0 or config.sink_responds)]
            lower = [v for v in heard if hop[v] == h + 1]
            d_up, f_up = measure(node, upper, 1)
            d_down, f_down = measure(node, lower, 2)
            value, case = virtual_hop_value(h, d_up, d_down)
            true_up = _nearest(topo, node, [v for v in synced
                                            if hop[v] == h - 1], dist)
            true_down = _nearest(topo, node, [v for v in synced
                                              if hop[v] == h + 1], dist)
            if d_up is not None:
                up_ok += 1
                errors.append(d_up - true_up)
            if d_down is not None:
                down_ok += 1
                errors.append(d_down - true_down)
            none += case == "none"
            observations.append(RangingObservation(
                it, node, h, d_up, d_down, f_up, f_down, true_up, true_down,
                value, case))
    values = defaultdict(list)
    for o in observations:
        values[o.node].append(o.value)
    virtual = {v: (sum(values[v]) / len(values[v]) if values[v] else None)
               for v in range(1, n)}
    total = sum(slots for _, _, slots in operations)
    return GroupingResult(
        virtual_hop=virtual, observations=tuple(observations),
        successful_upper=up_ok, successful_lower=down_ok, both_failed=none,
        total_slots=total, total_time_ms=total * config.slot_us / 1000.0,
        iterations=config.iterations, measurement_errors=tuple(errors),
        operations=tuple(operations))
