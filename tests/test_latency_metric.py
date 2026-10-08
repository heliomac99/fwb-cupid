"""[F19] Latencia com epocas incompletas censuradas (auditoria Item 3).

Antes, _median_latency() so guardava as epocas em que TODOS os relatorios
chegaram. Um protocolo que perde relatorios tinha as epocas dificeis
descartadas e ficava com a CDF mais bonita exatamente por ser menos confiavel.
"""

import math
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cupid_sim_v2 as sim

INF = float("inf")


class LatencyMetricTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }

    def tearDown(self):
        sim.apply_config(self.original_config)

    def test_censored_median_is_exact_while_most_epochs_complete(self):
        self.assertEqual(sim.censored_median([10.0, 20.0, INF]), 20.0)
        self.assertEqual(sim.censored_median([10.0, 20.0, 30.0, INF]), 25.0)
        # metade ou mais incompleta: a mediana so pode ser um limite
        self.assertEqual(sim.censored_median([10.0, INF]), INF)
        self.assertEqual(sim.censored_median([10.0, INF, INF]), INF)
        self.assertTrue(math.isnan(sim.censored_median([])))

    def test_incomplete_epochs_are_censored_not_dropped(self):
        """A regra antiga daria 102 ms (mediana so das completas). [P-group]
        A topologia falsa nao tem fisica para o periodo de agrupamento: o
        teste fixa o grupo estatico (so a metrica esta em jogo aqui)."""
        sim.apply_config({"GROUPING_MODEL": "static"})
        topo = SimpleNamespace(n=4, hop={0: 0, 1: 1, 2: 1, 3: 1})
        script = iter([(100, 3), (200, 2), (150, 3)])

        def fake_run_epoch(protocol, origs, topo, rng, funnel=None,
                           delivery_slots=None, **kwargs):
            slots, delivered = next(script)
            for packet in origs[:delivered]:
                delivery_slots[packet] = slots
            return slots, delivered

        with patch.object(sim, "run_epoch", side_effect=fake_run_epoch):
            stats = sim._latency_stats("cupid", topo, 42, 3, epochs=3)

        ms = sim.slots_to_ms
        self.assertEqual(stats["lats"], [ms(100), INF, ms(150)])
        self.assertEqual(stats["complete"], 2)
        self.assertEqual(stats["epochs"], 3)
        self.assertEqual(stats["sent"], 9)
        self.assertEqual(len(stats["report_lats"]), 8)
        self.assertEqual(stats["median"], ms(150))

    def test_mean_over_topologies_flags_a_lower_bound(self):
        m, ci, censored = sim.mean_ci_censored([100.0, INF], 1000.0)
        self.assertEqual(m, 550.0)
        self.assertGreater(ci, 0.0)
        self.assertTrue(censored)
        m, _, censored = sim.mean_ci_censored([100.0, 300.0], 1000.0)
        self.assertEqual(m, 200.0)
        self.assertFalse(censored)

    def test_latency_text_marks_censored_values(self):
        sim.apply_config({"MAX_SLOTS": 1500})
        self.assertEqual(sim.fmt_latency(204.94), "204.9 ms")
        self.assertEqual(sim.fmt_latency(INF), ">= 1219.5 ms")
        self.assertEqual(sim.fmt_latency(700.0, censored=True), ">= 700.0 ms")
        self.assertEqual(sim.fmt_latency(float("nan")), "n/d")

    def test_cdf_experiment_keeps_every_epoch_and_every_report(self):
        """Todas as epocas entram na CDF; o PDR sai dos mesmos dados."""
        sim.apply_config({"SEED": 42, "N_NODES": 12, "AREA_X": 30.0,
                          "AREA_Y": 12.0, "EPOCHS": 3, "MAX_SLOTS": 60})
        topos = sim.make_topologies(n_topo=2, base_seed=42)
        cdf, stats = sim.experiment_latency_cdf(topos, O=11,
                                                protos=["cupid", "crystal"])
        for protocol in ("cupid", "crystal"):
            with self.subTest(protocol=protocol):
                st = stats[protocol]
                self.assertEqual(len(cdf[protocol]), st["epochs"])
                self.assertEqual(st["epochs"], 6)
                finite = [v for v in cdf[protocol] if math.isfinite(v)]
                self.assertEqual(len(finite), st["complete"])
                self.assertLessEqual(len(st["report_lats"]), st["sent"])


if __name__ == "__main__":
    unittest.main()
