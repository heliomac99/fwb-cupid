"""[F23] ACK local do Weaver: ID do ultimo pacote ouvido no cabecalho.

"Upon packet TX, nodes embed the initiator ID of their last heard packet in
the Weaver header" (Trobinger et al., SenSys 2020, sec. 4). Quem le esse ID
num quadro vindo de mais perto do sink suspende o pacote por L slots.
"""

import random
import unittest
from types import SimpleNamespace

import numpy as np

import cupid_sim_v2 as sim


def make_topology(n, links, hop):
    power = np.zeros((n, n))
    for (a, b), p in links.items():
        power[a][b] = power[b][a] = p
    audible = [set(np.nonzero(power[r] > 0)[0].tolist()) for r in range(n)]
    return SimpleNamespace(
        n=n, ps_lin=power, tprop_ns=np.zeros((n, n)), audible=audible,
        reliable=[set(a) for a in audible], hop=hop,
        group={v: "emitter" for v in range(n)})


class LocalAckTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }

    def tearDown(self):
        sim.apply_config(self.original_config)

    def test_suppression_period_follows_weaver_equation_4(self):
        """L = 2(h-2) + h + D, com D = Y oportunidades de TX (pior caso)."""
        sim.apply_config({"GACK_BATCH_Y": 4})
        self.assertEqual(sim.suppression_slots("weaver", 2), 0 + 2 + 12)
        self.assertEqual(sim.suppression_slots("weaver", 3), 2 + 3 + 12)
        self.assertEqual(sim.suppression_slots("cupid", 3), 2 + 3 + 16)
        self.assertEqual(sim.suppression_slots("weaver", 1), 0 + 1 + 12)

    def test_expired_suppression_puts_the_packet_back_on_air(self):
        stats = sim.new_holder_stats()
        sup = {5: 10}
        self.assertEqual(sim._sendable([5, 6], sup, 9, stats), [6])
        self.assertEqual(sim._sendable([5, 6], sup, 10, stats), [5, 6])
        self.assertEqual(sup, {})
        self.assertEqual(stats["local_ack_resumes"], 1)

    def test_header_ack_silences_holders_that_never_heard_their_packet(self):
        """O relatorio 4 sobe de 4 -> 3 -> 1. Os nos 2 e 3 (hop 2) o seguram.

        No slot 4 o no 1 transmite o PROPRIO relatorio (1), mas o cabecalho
        diz que o ultimo pacote que ele ouviu foi o 4: os dois detentores do
        hop 2 se calam sem nunca terem ouvido o 4 vindo de cima. No modo F11
        eles voltariam a transmitir o 4 no slot 5.
        """
        topo = make_topology(
            5, {(0, 1): 1e-6, (1, 2): 1e-6, (1, 3): 4e-6,
                (2, 4): 1e-6, (3, 4): 1e-6},
            {0: 0, 1: 1, 2: 2, 3: 2, 4: 3})
        base = {"FAST_FADE_DB": 0.0, "CLOCK_JITTER_NS": 0.0, "TX_PROB": 1.0,
                "RANDOM_QUEUE_HEAD": False, "PER_MODEL": "threshold",
                "SAME_PACKET_MODEL": "interference", "ACK_MODEL": "none",
                "CAPTURE_MODEL": "legacy_async_weight",  # [Fase 6] fisica legada de copias/peso; o empirico tem testes proprios
                "BOOTSTRAP_DELAY": False, "BOOTSTRAP_MODEL": "bfs", "TX_DELAY_NS": 0.0}
        results = {}
        for mode in ("weaver", "overhearing"):
            sim.apply_config(dict(base, LOCAL_ACK_MODEL=mode))
            stats = sim.new_holder_stats()
            slots = {}
            sim.run_epoch("weaver", [1, 4], topo, random.Random(1),
                          slot_budget=6, delivery_slots=slots,
                          holder_stats=stats)
            results[mode] = (slots, stats)

        slots, stats = results["weaver"]
        self.assertEqual(slots, {1: 2})
        self.assertEqual(stats["local_ack_suppressions"], 2)
        self.assertEqual(stats["overhearing_suppressions"], 0)
        self.assertEqual(stats["data_tx"], 6)

        _, stats = results["overhearing"]
        self.assertEqual(stats["local_ack_suppressions"], 0)
        self.assertEqual(stats["data_tx"], 8)

    def test_unknown_local_ack_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "LOCAL_ACK_MODEL"):
            sim.apply_config({"LOCAL_ACK_MODEL": "magico"})


if __name__ == "__main__":
    unittest.main()
