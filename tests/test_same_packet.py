"""[F22] Copias do mesmo pacote vindas de transmissores diferentes.

Casos tirados de Lobba et al. (EWSN 2020, sec. 4.3), medidos no DW1000:
quadros longos com a mesma potencia se destroem; 5-7 dB de diferenca bastam;
quadros curtos desalinhados em ate 128 ns nao se destroem.
"""

import random
import unittest
from types import SimpleNamespace

import numpy as np

import cupid_sim_v2 as sim


def two_relays(p1, p2, delay2_ns=0.0):
    """Sink 0 ouve os nos 1 e 2 retransmitindo a mesma copia do pacote 7."""
    power = np.zeros((3, 3))
    power[0][1] = power[1][0] = p1
    power[0][2] = power[2][0] = p2
    tprop = np.zeros((3, 3))
    tprop[0][2] = delay2_ns
    return SimpleNamespace(ps_lin=power, tprop_ns=tprop,
                           audible=[{1, 2}, {0}, {0}])


class SamePacketTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }
        sim.apply_config({"PER_MODEL": "threshold", "REQ_SNR_DB": 3.0,
                          "CAPTURE_MODEL": "legacy_async_weight",  # [Fase 6] fisica legada de copias/peso; o empirico tem testes proprios
                          })

    def tearDown(self):
        sim.apply_config(self.original_config)

    @staticmethod
    def decode(topo):
        return sim.decode_at_receiver(
            0, [1, 2], {1: 7, 2: 7}, topo, [0.0, 0.0, 0.0],
            fading={(1, 0): 1.0, (2, 0): 1.0})

    def test_equal_power_copies_destroy_each_other_by_default(self):
        """Quadros longos, mesma potencia: PRR de 0-10% no DW1000."""
        sim.apply_config({"SAME_PACKET_MODEL": "interference"})
        self.assertIsNone(self.decode(two_relays(1e-7, 1e-7)))

    def test_a_6_db_stronger_copy_is_captured(self):
        """Com 5-7 dB de diferenca a copia mais forte e decodificada."""
        sim.apply_config({"SAME_PACKET_MODEL": "interference"})
        res = self.decode(two_relays(1e-7, 4e-7))
        self.assertIsNotNone(res)
        self.assertEqual((res[0], res[1]), (7, 2))

    def test_nondestructive_copies_inside_the_window_do_not_interfere(self):
        """Quadros curtos: desalinhados em ate 128 ns, sem perda."""
        sim.apply_config({"SAME_PACKET_MODEL": "nondestructive",
                          "COMBINE_WINDOW_NS": 128.0})
        self.assertIsNotNone(self.decode(two_relays(1e-7, 1e-7, 100.0)))
        # fora da janela a copia atrasada volta a ser interferencia (com peso
        # integral; o peso assincrono do F28 esta em test_async_capture)
        sim.apply_config({"ASYNC_INTERF_WEIGHT": 1.0})
        self.assertIsNone(self.decode(two_relays(1e-7, 1e-7, 200.0)))

    def test_additive_mode_keeps_the_old_4_ns_combining(self):
        sim.apply_config({"SAME_PACKET_MODEL": "additive",
                          "COMBINE_WINDOW_NS": 4.0})
        self.assertIsNotNone(self.decode(two_relays(1e-7, 1e-7, 3.0)))
        self.assertIsNone(self.decode(two_relays(1e-7, 1e-7, 10.0)))

    def test_copies_with_different_headers_are_different_frames(self):
        """[F27] P15: o cabecalho (ultimo ouvido, bitmap) faz parte do quadro.
        Mesmo relatorio 7, mesma potencia: com cabecalhos iguais as copias nao
        se destroem; com cabecalhos diferentes elas disputam e colidem."""
        sim.apply_config({"SAME_PACKET_MODEL": "nondestructive"})
        topo = two_relays(1e-7, 1e-7)
        fading = {(1, 0): 1.0, (2, 0): 1.0}
        same = {1: (7, 3, frozenset()), 2: (7, 3, frozenset())}
        differ = {1: (7, 3, frozenset()), 2: (7, 5, frozenset({4}))}
        self.assertIsNotNone(sim.decode_at_receiver(
            0, [1, 2], same, topo, [0.0] * 3, fading=fading))
        self.assertIsNone(sim.decode_at_receiver(
            0, [1, 2], differ, topo, [0.0] * 3, fading=fading))

    def test_twin_relays_with_identical_headers_reach_the_sink(self):
        """Os nos 1 e 2 (hop 1) pegam o relatorio 3 no mesmo slot, com o
        mesmo cabecalho, e o repetem juntos no slot 4 com a mesma potencia:
        quadros identicos nao se destroem, e o sink recebe no slot 5."""
        n = 4
        power = np.zeros((n, n))
        for a, b in ((0, 1), (0, 2), (1, 3), (2, 3)):
            power[a][b] = power[b][a] = 1e-6
        audible = [set(np.nonzero(power[r] > 0)[0].tolist())
                   for r in range(n)]
        topo = SimpleNamespace(
            n=n, ps_lin=power, tprop_ns=np.zeros((n, n)), audible=audible,
            reliable=[set(a) for a in audible],
            hop={0: 0, 1: 1, 2: 1, 3: 2},
            group={v: "emitter" for v in range(n)})
        base = {"FAST_FADE_DB": 0.0, "CLOCK_JITTER_NS": 0.0, "TX_PROB": 1.0,
                "RANDOM_QUEUE_HEAD": False, "BOOTSTRAP_DELAY": False, "BOOTSTRAP_MODEL": "bfs",
                "LOCAL_ACK_MODEL": "weaver", "SAME_HOP_RELAY": False,
                "CAPTURE_MODEL": "legacy_async_weight",  # [Fase 6] fisica legada de copias/peso; o empirico tem testes proprios
                "TX_DELAY_NS": 0.0}
        for model, expected in (("nondestructive", {3: 5}),
                                ("interference", {})):
            with self.subTest(model=model):
                sim.apply_config(dict(base, SAME_PACKET_MODEL=model))
                slots = {}
                sim.run_epoch("weaver", [3], topo, random.Random(1),
                              slot_budget=6, delivery_slots=slots)
                self.assertEqual(slots, expected)

    def test_unknown_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SAME_PACKET_MODEL"):
            sim.apply_config({"SAME_PACKET_MODEL": "magico"})


if __name__ == "__main__":
    unittest.main()
