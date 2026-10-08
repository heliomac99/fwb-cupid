import random
import unittest
from types import SimpleNamespace

import numpy as np

import cupid_sim_v2 as sim


class CombinationWindowTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }

    def tearDown(self):
        sim.apply_config(self.original_config)

    @staticmethod
    def controlled_topology():
        return SimpleNamespace(
            ps_lin=np.array([
                [0.0, 1e-7, 1e-7, 1e-7],
                [1e-7, 0.0, 1e-7, 1e-7],
                [1e-7, 1e-7, 0.0, 1e-7],
                [1e-7, 1e-7, 1e-7, 0.0],
            ]),
            tprop_ns=np.array([
                [0.0, 0.0, 3.0, 10.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]),
            audible=[{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}],
        )

    def test_instrumentation_counts_comb_and_resid_signals(self):
        sim.apply_config({
            "COMBINE_WINDOW_NS": 4.0,
            "COMBINE_EFF": 1.0,
            "REQ_SNR_DB": -100.0,
            "PER_MODEL": "threshold",
            "SAME_PACKET_MODEL": "additive",
            "CAPTURE_MODEL": "legacy_async_weight",  # [Fase 6] fisica legada de copias/peso; o empirico tem testes proprios
        })
        topo = self.controlled_topology()
        fading = {(1, 0): 1.0, (2, 0): 1.0, (3, 0): 1.0}
        stats = sim.new_phy_stats()

        sim.decode_at_receiver(
            0, [1, 2, 3], {1: 7, 2: 7, 3: 7}, topo,
            [0.0] * 4, fading=fading, phy_stats=stats)

        self.assertEqual(stats["comb_signals"], 2)
        self.assertEqual(stats["resid_signals"], 1)
        summary = sim.summarize_phy_stats(stats)
        self.assertEqual(summary["n_groups_singleton"], 0)
        self.assertEqual(summary["n_groups_multi"], 1)
        self.assertEqual(summary["comb_signals"], 2)
        self.assertEqual(summary["resid_signals"], 1)
        self.assertEqual(summary["total_signals"], 3)
        self.assertAlmostEqual(summary["comb_power"], 2e-7)
        self.assertAlmostEqual(summary["resid_power"], 1e-7)
        self.assertAlmostEqual(summary["comb_fraction"], 2 / 3)
        self.assertAlmostEqual(summary["resid_fraction"], 1 / 3)

    def test_instrumentation_does_not_change_results_or_rng(self):
        sim.apply_config({
            "SEED": 42,
            "N_NODES": 8,
            "AREA_X": 18.0,
            "AREA_Y": 7.0,
            "MAX_SLOTS": 100,
        })
        topo = sim.build_topology(random.Random(42))
        plain_rng = random.Random(2026)
        measured_rng = random.Random(2026)
        plain_slots = {}
        measured_slots = {}
        stats = sim.new_phy_stats()

        plain = sim.run_epoch(
            "cupid", list(range(1, 8)), topo, plain_rng,
            delivery_slots=plain_slots)
        measured = sim.run_epoch(
            "cupid", list(range(1, 8)), topo, measured_rng,
            delivery_slots=measured_slots, phy_stats=stats)

        self.assertEqual(measured, plain)
        self.assertEqual(measured_slots, plain_slots)
        self.assertEqual(measured_rng.random(), plain_rng.random())
        self.assertGreater(
            stats["n_groups_singleton"] + stats["n_groups_multi"], 0)


if __name__ == "__main__":
    unittest.main()
