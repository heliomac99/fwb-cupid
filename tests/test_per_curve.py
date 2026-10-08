"""[F21] Curva de erro suave no lugar do limiar duro de SINR."""

import math
import random
import unittest
from types import SimpleNamespace

import numpy as np

import cupid_sim_v2 as sim


class PerCurveTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }

    def tearDown(self):
        sim.apply_config(self.original_config)

    def test_logistic_curve_is_centered_and_has_the_declared_width(self):
        sim.apply_config({"PER_MODEL": "logistic", "REQ_SNR_DB": 3.0,
                          "PER_WIDTH_DB": 2.0})
        self.assertAlmostEqual(sim.decode_success_prob(3.0), 0.5)
        self.assertAlmostEqual(sim.decode_success_prob(4.0), 0.9)
        self.assertAlmostEqual(sim.decode_success_prob(2.0), 0.1)
        self.assertGreater(sim.decode_success_prob(7.0), 0.998)
        self.assertLess(sim.decode_success_prob(-3.0), 1e-5)

    def test_threshold_curve_is_the_old_hard_cut(self):
        sim.apply_config({"PER_MODEL": "threshold", "REQ_SNR_DB": 3.0})
        self.assertEqual(sim.decode_success_prob(3.0), 1.0)
        self.assertEqual(sim.decode_success_prob(2.999), 0.0)
        self.assertIsNone(sim.sample_decode_draws([1, 2], [0], random.Random(1)))

    @staticmethod
    def single_link():
        return SimpleNamespace(
            ps_lin=np.array([[0.0, 1e-9], [1e-9, 0.0]]),
            tprop_ns=np.zeros((2, 2)), audible=[{1}, {0}])

    def test_the_draw_decides_a_frame_at_the_midpoint(self):
        """No SINR de 50%, o sorteio abaixo decodifica e o acima perde."""
        topo = self.single_link()
        sim.apply_config({"PER_MODEL": "threshold", "REQ_SNR_DB": -100.0})
        sinr = sim.decode_at_receiver(0, [1], {1: 1}, topo, [0.0, 0.0],
                                      fading={(1, 0): 1.0})[4]
        sim.apply_config({"PER_MODEL": "logistic", "REQ_SNR_DB": sinr})
        kwargs = dict(fading={(1, 0): 1.0})
        self.assertIsNotNone(sim.decode_at_receiver(
            0, [1], {1: 1}, topo, [0.0, 0.0], draw=0.49, **kwargs))
        self.assertIsNone(sim.decode_at_receiver(
            0, [1], {1: 1}, topo, [0.0, 0.0], draw=0.51, **kwargs))
        with self.assertRaisesRegex(ValueError, "draw"):
            sim.decode_at_receiver(0, [1], {1: 1}, topo, [0.0, 0.0], **kwargs)

    def test_draws_are_canonical(self):
        """Como o fading: a ordem de entrada nao muda quem recebe qual sorteio."""
        sim.apply_config({"PER_MODEL": "logistic"})
        a = sim.sample_decode_draws([3, 1, 2], [1, 0], random.Random(9))
        b = sim.sample_decode_draws([2, 3, 1], [0, 1], random.Random(9))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 6)
        self.assertTrue(all(0.0 <= v < 1.0 for v in a.values()))

    def test_frames_below_the_old_threshold_now_sometimes_arrive(self):
        """O efeito que o limiar duro escondia: sucesso parcial perto de 3 dB."""
        sim.apply_config({"SEED": 42, "N_NODES": 24, "AREA_X": 60.0,
                          "AREA_Y": 20.0, "PER_MODEL": "logistic",
                          "CAPTURE_MODEL": "legacy_async_weight",  # [Fase 6] fisica legada de copias/peso; o empirico tem testes proprios
                          })
        topo = sim.build_topology(random.Random(42))
        events = []
        for seed in range(3):
            sim.run_epoch("cupid", sim.valid_nodes(topo), topo,
                          random.Random(seed), slot_budget=200,
                          sink_events=events)
        below = [e for e in events if e["sinr_db"] < sim.REQ_SNR_DB]
        self.assertTrue(events)
        self.assertTrue(below, "nenhuma recepcao abaixo de REQ_SNR_DB")
        self.assertTrue(all(math.isfinite(e["sinr_db"]) for e in events))


if __name__ == "__main__":
    unittest.main()
