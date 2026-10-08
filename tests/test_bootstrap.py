"""Fase 3: bootstrap fisico por epoca e hops dinamicos.

Cupid (DCOSS-IoT 2025, sec. IV-A): "Nodes determine their logical hop from
the relay count of received synchronization packets"; o hop "can vary
between epochs". Weaver (SenSys 2020, sec. 3.3 e Tab. 1): o sink inicia o
bootstrap, os nos o repropagam B = 2 vezes em slots TX consecutivos do
esquema TX-RX-RX. Crystal: um flood S por epoca (N_S = 3).

Premissa pedida na especificacao (diverge do Weaver, que deixa o no reusar o
hop anterior): no que nao decodifica nenhum quadro de sincronizacao fica
INATIVO na epoca. A BFS da potencia media continua na topologia, so como
diagnostico e como modelo do preset legado.
"""

import copy
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import cupid_grouping as cg
import cupid_sim_v2 as sim


def make_topology(n, links, hop=None):
    """Enlaces reciprocos {(a, b): potencia_mW}; 'hop' e a BFS estatica."""
    power = np.zeros((n, n))
    for (a, b), p in links.items():
        power[a][b] = power[b][a] = p
    audible = [set(np.nonzero(power[r] > 0)[0].tolist()) for r in range(n)]
    hop = hop or {v: v for v in range(n)}
    return SimpleNamespace(
        n=n, ps_lin=power, tprop_ns=np.zeros((n, n)), audible=audible,
        reliable=[set(a) for a in audible], hop=hop,
        pos={v: (float(v), 0.0) for v in range(n)},
        group={v: "emitter" for v in range(n)})


def line(n):
    return make_topology(n, {(a, a + 1): 1e-6 for a in range(n - 1)})


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }
        sim.apply_config({"FAST_FADE_DB": 0.0, "CLOCK_JITTER_NS": 0.0,
                          "PER_MODEL": "threshold", "TX_PROB": 1.0,
                          "RANDOM_QUEUE_HEAD": False})

    def tearDown(self):
        sim.apply_config(self.original_config)

    def test_default_is_the_physical_bootstrap(self):
        self.assertEqual(sim.BOOTSTRAP_MODEL, "physical")
        self.assertEqual(sim.PRESET_LEGACY_V2["BOOTSTRAP_MODEL"], "bfs")
        with self.assertRaisesRegex(ValueError, "BOOTSTRAP_MODEL"):
            sim.apply_config({"BOOTSTRAP_MODEL": "magico"})

    def test_line_of_three_nodes_gets_hops_0_1_2(self):
        state = sim.run_bootstrap(line(3), random.Random(1))
        self.assertEqual(state.hop_by_node, {0: 0, 1: 1, 2: 2})
        self.assertEqual(state.synchronized_nodes, frozenset({0, 1, 2}))
        self.assertEqual(state.first_sync_slot, {0: -1, 1: 0, 2: 1})
        self.assertEqual(state.parent, {1: 0, 2: 1})
        self.assertEqual(state.failure_reason, {})
        # B = 2 copias a cada 3 slots: o ultimo quadro (no 2, 2a copia) sai
        # no slot 5
        self.assertEqual(state.bootstrap_slots, 6)

    def test_missing_the_bootstrap_makes_the_node_inactive(self):
        """O no 3 nao tem enlace: fica fora da epoca, nao transmite, nao
        recebe, e o relatorio dele conta como perdido."""
        topo = make_topology(4, {(0, 1): 1e-6, (1, 2): 1e-6},
                             hop={0: 0, 1: 1, 2: 2, 3: 99})
        state = sim.run_bootstrap(topo, random.Random(1))
        self.assertNotIn(3, state.synchronized_nodes)
        self.assertEqual(state.failure_reason, {3: "no_sync"})
        self.assertNotIn(3, state.hop_by_node)
        seen_tx = set()
        original = sim.decode_physical_channels

        def spy(rx, tx_sets, *args, **kwargs):
            seen_tx.update(v for _, nodes in tx_sets for v in nodes)
            self.assertNotEqual(rx, 3)
            return original(rx, tx_sets, *args, **kwargs)

        slots = {}
        with patch.object(sim, "decode_physical_channels", spy):
            result = sim.run_epoch("weaver", [2, 3], topo, random.Random(1),
                                   slot_budget=40, delivery_slots=slots,
                                   bootstrap=state)
        self.assertEqual(set(slots), {2})
        self.assertEqual(result, (40, 1))
        self.assertNotIn(3, seen_tx)

    def test_bootstrap_can_give_different_hops_in_different_epochs(self):
        """O no 2 ouve o sink no limite (sucesso ~50%) e o no 1 com folga:
        em algumas epocas ele pega o sync direto do sink (hop 1), em outras
        so o do no 1 (hop 2)."""
        sim.apply_config({"PER_MODEL": "logistic", "FAST_FADE_DB": 4.0})
        sens = sim.NOISE_EFF_LIN * 10 ** (sim.REQ_SNR_DB / 10)
        topo = make_topology(3, {(0, 1): 1e-6, (1, 2): 1e-6, (0, 2): sens})
        rng = random.Random(5)
        hops = {sim.run_bootstrap(topo, rng).hop_by_node[2]
                for _ in range(40)}
        self.assertEqual(hops, {1, 2})

    def test_geometry_and_static_shadowing_are_not_modified(self):
        sim.apply_config({"SEED": 42, "N_NODES": 16, "AREA_X": 60.0,
                          "AREA_Y": 12.0, "FAST_FADE_DB": 4.0,
                          "PER_MODEL": "logistic"})
        topo = sim.build_topology(random.Random(42))
        before = (copy.deepcopy(topo.pos), topo.ps_db.copy(),
                  dict(topo.hop), dict(topo.group))
        rng = random.Random(3)
        for _ in range(3):
            state = sim.run_bootstrap(topo, rng)
            sim.run_epoch("cupid", sim.valid_nodes(topo), topo, rng,
                          slot_budget=60, bootstrap=state)
        self.assertEqual(topo.pos, before[0])
        self.assertTrue(np.array_equal(topo.ps_db, before[1]))
        self.assertEqual(topo.hop, before[2])
        self.assertEqual(topo.group, before[3])

    def test_ctx_protocols_receive_the_same_bootstrap_state(self):
        """Mesmos sorteios fisicos -> mesmo BootstrapState para Weaver, Cupid
        e Cupid FWB (a onda de sync e a mesma: Cupid, sec. V-A)."""
        sim.apply_config({"SEED": 42, "N_NODES": 16, "AREA_X": 60.0,
                          "AREA_Y": 12.0, "FAST_FADE_DB": 4.0,
                          "PER_MODEL": "logistic"})
        topo = sim.build_topology(random.Random(42))
        states = {}
        for proto in ("weaver", "cupid", "cupid_fwb"):
            info = {}
            sim.run_epoch(proto, sim.valid_nodes(topo), topo,
                          random.Random(9), slot_budget=40, epoch_info=info)
            states[proto] = info["bootstrap"]
        self.assertEqual(states["weaver"], states["cupid"])
        self.assertEqual(states["weaver"], states["cupid_fwb"])

    def test_telemetry_does_not_change_the_rng_or_the_result(self):
        sim.apply_config({"FAST_FADE_DB": 4.0, "PER_MODEL": "logistic"})
        topo = make_topology(4, {(0, 1): 1e-6, (1, 2): 1e-6, (2, 3): 1e-8,
                                 (0, 2): 1e-9})
        plain, traced = random.Random(4), random.Random(4)
        stats = sim.new_phy_stats()
        a = sim.run_bootstrap(topo, plain)
        b = sim.run_bootstrap(topo, traced, phy_stats=stats)
        self.assertEqual(a, b)
        self.assertEqual(plain.random(), traced.random())
        self.assertGreater(stats["n_groups_singleton"]
                           + stats["n_groups_multi"], 0)

    def test_ideal_line_reproduces_the_legacy_bootstrap_delay(self):
        """Sem perdas, o bootstrap fisico da o mesmo atraso da regra antiga
        "o hop h espera h slots" (tests/test_crystal.py: entrega no slot 8)."""
        slots = {}
        sim.run_epoch("weaver", [3], line(4), random.Random(1),
                      slot_budget=60, delivery_slots=slots)
        self.assertEqual(slots, {3: 8})

    def test_crystal_does_not_assume_sync_when_s_fails(self):
        """Fonte que perdeu o S nao entra na fase T."""
        topo = line(3)
        lost = cg.BootstrapState(
            hop_by_node={0: 0, 2: 2}, synchronized_nodes=frozenset({0, 2}),
            first_sync_slot={0: -1, 2: 1}, parent={2: 0},
            bootstrap_slots=4, failure_reason={1: "no_sync"},
            clock_ns=(0.0, 0.0, 0.0))
        calls = []
        original = sim.crystal_t_flood

        def spy(sources, *args, **kwargs):
            calls.append(tuple(sources))
            return original(sources, *args, **kwargs)

        with patch.object(sim, "run_bootstrap", return_value=lost), \
                patch.object(sim, "crystal_t_flood", spy):
            result = sim.run_epoch("crystal", [1], topo, random.Random(1),
                                   slot_budget=100)
        self.assertEqual(result, (100, 0))
        self.assertTrue(all(1 not in c for c in calls))

    def test_crystal_s_phase_goes_through_the_physics(self):
        topo = make_topology(3, {(0, 1): 1e-6, (1, 2): 1e-6})
        with patch.object(sim, "run_bootstrap",
                          wraps=sim.run_bootstrap) as spy:
            sim.run_epoch("crystal", [2], topo, random.Random(1),
                          slot_budget=100)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.call_args.kwargs["n_tx"], sim.N_TX_SYNC_ACK)
        self.assertEqual(spy.call_args.kwargs["relay_period"], 1)

    def test_legacy_bfs_mode_ignores_the_bootstrap_physics(self):
        sim.apply_config({"BOOTSTRAP_MODEL": "bfs"})
        with patch.object(sim, "run_bootstrap") as spy:
            sim.run_epoch("weaver", [2], line(3), random.Random(1),
                          slot_budget=30)
            sim.run_epoch("crystal", [2], line(3), random.Random(1),
                          slot_budget=60)
        spy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
