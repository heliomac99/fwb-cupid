"""[F25] Retransmissao dentro do mesmo hop.

Weaver: "a node relay[s] only packets from nodes at the same hop distance, or
higher"; "nodes without new data listen during TX slots to hear same-hop
neighbors and help them deliver their packets" (SenSys 2020). Cupid: o modo
silencioso existe para "allowing nodes within the same hop to receive data"
(DCOSS-IoT 2025, sec. IV-C).
"""

import random
import unittest
from types import SimpleNamespace

import numpy as np

import cupid_sim_v2 as sim


def make_topology(n, links, hop, group):
    power = np.zeros((n, n))
    for (a, b), p in links.items():
        power[a][b] = power[b][a] = p
    audible = [set(np.nonzero(power[r] > 0)[0].tolist()) for r in range(n)]
    return SimpleNamespace(
        n=n, ps_lin=power, tprop_ns=np.zeros((n, n)), audible=audible,
        reliable=[set(a) for a in audible], hop=hop, group=group)


class SameHopRelayTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }

    def tearDown(self):
        sim.apply_config(self.original_config)

    def relay_epoch(self, protocol, originators, topo, relay,
                    schedule="legacy"):
        # [Fase 5] slots exatos escritos para a grade legada do Cupid
        sim.apply_config({"CUPID_SCHEDULE": schedule,
            "FAST_FADE_DB": 0.0, "CLOCK_JITTER_NS": 0.0, "TX_PROB": 1.0,
            "RANDOM_QUEUE_HEAD": False, "PER_MODEL": "threshold",
            "SAME_PACKET_MODEL": "interference", "SAME_HOP_RELAY": relay,
            "CAPTURE_MODEL": "legacy_async_weight",  # [Fase 6] fisica legada de copias/peso; o empirico tem testes proprios
            "BOOTSTRAP_DELAY": False, "BOOTSTRAP_MODEL": "bfs",
        })
        slots = {}
        result = sim.run_epoch(protocol, originators, topo, random.Random(1),
                               slot_budget=40, delivery_slots=slots)
        return result, slots

    def test_cupid_collector_hands_the_packet_to_its_emitter(self):
        """O no 3 (hop 2) so alcanca o collector 2; o collector nao alcanca o
        sink. No slot 3 o collector transmite e o emitter 1, em silencio no
        mesmo hop, adota o pacote; no slot 5 o emitter entrega."""
        topo = make_topology(
            4, {(0, 1): 1e-6, (1, 2): 1e-6, (2, 3): 1e-6},
            {0: 0, 1: 1, 2: 1, 3: 2},
            {0: "emitter", 1: "emitter", 2: "collector", 3: "emitter"})
        self.assertEqual(self.relay_epoch("cupid", [3], topo, True),
                         ((6, 1), {3: 6}))
        self.assertEqual(self.relay_epoch("cupid", [3], topo, False)[1], {})
        # na grade da Fig. 5b o mecanismo e o mesmo: so o modo silencioso
        # (mesmo hop) leva o relatorio ate o sink
        self.assertEqual(set(self.relay_epoch("cupid", [3], topo, True,
                                              "fig5b")[1]), {3})
        self.assertEqual(self.relay_epoch("cupid", [3], topo, False,
                                          "fig5b")[1], {})

    def test_weaver_idle_node_helps_a_same_hop_neighbor(self):
        """O no 1 (hop 1) nao alcanca o sink; o no 2, ocioso no mesmo hop,
        escuta o slot do proprio hop, adota o pacote e o entrega."""
        topo = make_topology(
            3, {(0, 2): 1e-6, (1, 2): 1e-6}, {0: 0, 1: 1, 2: 1},
            {v: "emitter" for v in range(3)})
        self.assertEqual(self.relay_epoch("weaver", [1], topo, True),
                         ((5, 1), {1: 5}))
        self.assertEqual(self.relay_epoch("weaver", [1], topo, False)[1], {})


if __name__ == "__main__":
    unittest.main()
