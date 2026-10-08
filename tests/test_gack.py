"""[F24] ACK global do Weaver: o bitmap desce tambem por nos sem dado.

"Nodes without data to send re-propagate the G-ACK as a no-payload packet
only if it contains new bits" (Trobinger et al., SenSys 2020, sec. 4); o
sink transmite o G-ACK "upon receiving a packet".
"""

import random
import unittest
from types import SimpleNamespace

import numpy as np

import cupid_sim_v2 as sim


def chain():
    """Sink 0 - 1 - 2 - 3 em linha; o no 4 nao tem enlace (mantem a epoca
    aberta depois que o relatorio 3 chega)."""
    n = 5
    power = np.zeros((n, n))
    for a, b in ((0, 1), (1, 2), (2, 3)):
        power[a][b] = power[b][a] = 1e-6
    audible = [set(np.nonzero(power[r] > 0)[0].tolist()) for r in range(n)]
    return SimpleNamespace(
        n=n, ps_lin=power, tprop_ns=np.zeros((n, n)), audible=audible,
        reliable=[set(a) for a in audible],
        hop={0: 0, 1: 1, 2: 2, 3: 3, 4: 99},
        group={v: "emitter" for v in range(n)})


class GlobalAckTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }

    def tearDown(self):
        sim.apply_config(self.original_config)

    def run_chain(self, ack_model):
        sim.apply_config({
            "FAST_FADE_DB": 0.0, "CLOCK_JITTER_NS": 0.0, "TX_PROB": 1.0,
            "RANDOM_QUEUE_HEAD": False, "PER_MODEL": "threshold",
            "SAME_PACKET_MODEL": "interference", "LOCAL_ACK_MODEL": "weaver",
            "CAPTURE_MODEL": "legacy_async_weight",  # [Fase 6] fisica legada de copias/peso; o empirico tem testes proprios
            "GACK_BATCH_Y": 4, "ACK_MODEL": ack_model,
            "BOOTSTRAP_DELAY": False, "BOOTSTRAP_MODEL": "bfs",
        })
        stats = sim.new_holder_stats()
        slots = {}
        sim.run_epoch("weaver", [3, 4], chain(), random.Random(1),
                      slot_budget=30, delivery_slots=slots,
                      holder_stats=stats)
        return slots, stats

    def test_bitmap_reaches_holders_through_idle_nodes(self):
        """O relatorio 3 chega no slot 5. No 6 o sink anuncia; no 7 o no 1
        (ja sem dado) repassa o bitmap; no 8 o no 2; os tres detentores
        apagam a copia sem nunca terem retransmitido o 3 depois da entrega."""
        slots, stats = self.run_chain("weaver")
        self.assertEqual(slots, {3: 5})
        self.assertEqual(stats["global_ack_suppressions"], 3)
        self.assertEqual(stats["node_gack_tx"], 3)
        self.assertEqual(stats["local_ack_resumes"], 0)
        # o sink so fala quando tem novidade
        self.assertEqual(stats["sink_ack_tx"], 1)

    def test_without_repropagation_deep_holders_never_learn(self):
        """No ACK da etapa 0 so o sink fala: o no 1 aprende, mas os nos 2 e
        3 ficam esperando, o prazo L vence e o 3 volta ao ar."""
        slots, stats = self.run_chain("physical")
        self.assertEqual(slots, {3: 5})
        self.assertEqual(stats["global_ack_suppressions"], 1)
        self.assertEqual(stats["node_gack_tx"], 0)
        self.assertGreaterEqual(stats["local_ack_resumes"], 1)
        self.assertGreaterEqual(stats["sink_ack_tx"], 5)


if __name__ == "__main__":
    unittest.main()
