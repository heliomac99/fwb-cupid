"""[F26] Crystal como no artigo e bootstrap dos protocolos CTX.

Crystal (Istomin et al., SenSys 2016, sec. 5 e Tab. 4): um flood S por
epoca, depois pares TA; na fase T todas as fontes iniciam juntas; o sink
encerra apos R pares silenciosos. Weaver (SenSys 2020, sec. 3.3): o no a h
hops espera h slots pelo bootstrap antes de transmitir.
"""

import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import cupid_sim_v2 as sim


def chain(n_nodes, extra_hops=None):
    """Sink 0 - 1 - 2 - ... em linha; extra_hops cria nos isolados."""
    hop = {v: v for v in range(n_nodes)}
    hop.update(extra_hops or {})
    n = len(hop)
    power = np.zeros((n, n))
    for a in range(n_nodes - 1):
        power[a][a + 1] = power[a + 1][a] = 1e-6
    audible = [set(np.nonzero(power[r] > 0)[0].tolist()) for r in range(n)]
    return SimpleNamespace(
        n=n, ps_lin=power, tprop_ns=np.zeros((n, n)), audible=audible,
        reliable=[set(a) for a in audible], hop=hop,
        group={v: "emitter" for v in range(n)})


class CrystalTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }
        sim.apply_config({
            "FAST_FADE_DB": 0.0, "CLOCK_JITTER_NS": 0.0,
            "PER_MODEL": "threshold", "SAME_PACKET_MODEL": "nondestructive",
            "CRYSTAL_MODEL": "tapairs", "N_TX_PER_FLOOD": 2,
            "N_TX_SYNC_ACK": 3, "CRYSTAL_R": 2,
            # [Fase 5] contagem em slots uniformes; o airtime por fase tem
            # teste proprio em test_protocol_timing
            "CRYSTAL_PHASE_TIMING": "uniform",
        })

    def tearDown(self):
        sim.apply_config(self.original_config)

    def test_one_sync_flood_then_ta_pairs(self):
        """H = 2: S = 4 slots, T = 3, A = 4. O relatorio 1 chega no 1o slot
        do primeiro T (slot 5); o 2 so no segundo T, depois que o ACK do 1
        calou a fonte 1 (slot 13). A latencia e a chegada do ultimo."""
        slots = {}
        result = sim.run_epoch("crystal", [1, 2], chain(3), random.Random(1),
                               slot_budget=100, delivery_slots=slots)
        self.assertEqual(slots, {1: 5, 2: 13})
        self.assertEqual(result, (13, 2))

    def test_sink_gives_up_after_r_silent_pairs(self):
        """A fonte 3 nao tem enlace: depois que o 1 chega, cada T fica mudo
        e o sink dorme apos CRYSTAL_R pares silenciosos seguidos. [P-boot]
        Regra R com uma fonte sem enlace: com o S fisico ela nem entraria na
        epoca (test_bootstrap), entao o teste fixa o bootstrap legado."""
        sim.apply_config({"BOOTSTRAP_MODEL": "bfs"})
        topo = chain(2, extra_hops={2: 99})
        for r, expected_pairs in ((2, 3), (5, 6)):
            with self.subTest(R=r):
                sim.apply_config({"CRYSTAL_R": r})
                with patch.object(sim, "crystal_t_flood",
                                  wraps=sim.crystal_t_flood) as spy:
                    result = sim.run_epoch("crystal", [1, 2], topo,
                                           random.Random(1), slot_budget=400)
                self.assertEqual(result, (400, 1))
                self.assertEqual(spy.call_count, expected_pairs)

    def test_ctx_nodes_wait_for_the_bootstrap(self):
        """No Weaver o hop 3 transmite no slot 0 (3 % 3); com o bootstrap so
        pode no slot 3, e a entrega atrasa 3 slots."""
        # [P-boot] regra do motor legado; o equivalente fisico esta em
        # test_bootstrap.test_ideal_line_reproduces_the_legacy_bootstrap_delay
        sim.apply_config({"TX_PROB": 1.0, "RANDOM_QUEUE_HEAD": False,
                          "BOOTSTRAP_MODEL": "bfs"})
        delivered = {}
        for boot in (False, True):
            sim.apply_config({"BOOTSTRAP_DELAY": boot})
            slots = {}
            sim.run_epoch("weaver", [3], chain(4), random.Random(1),
                          slot_budget=60, delivery_slots=slots)
            delivered[boot] = slots
        self.assertEqual(delivered[False], {3: 5})
        self.assertEqual(delivered[True], {3: 8})

    def test_unknown_crystal_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "CRYSTAL_MODEL"):
            sim.apply_config({"CRYSTAL_MODEL": "magico"})


if __name__ == "__main__":
    unittest.main()
