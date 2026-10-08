"""Fase 8: estatistica com a topologia como unidade independente.

Os valores de referencia do t de Student sao os tabelados (bicaudal, 95%).
"""

import math
import unittest

import experiment_stats as es

INF = float("inf")


class StudentTTests(unittest.TestCase):
    def test_critical_t_for_small_n(self):
        for df, expected in ((1, 12.706), (2, 4.303), (4, 2.776),
                             (9, 2.262), (30, 2.042), (1000, 1.962)):
            with self.subTest(df=df):
                self.assertAlmostEqual(es.t_critical(0.95, df), expected,
                                       places=3)
        self.assertAlmostEqual(es.t_critical(0.99, 4), 4.604, places=3)
        with self.assertRaises(ValueError):
            es.t_critical(0.95, 0)

    def test_mean_ci_uses_t_not_1_96(self):
        m, half, n = es.mean_ci_t([10.0, 12.0, 14.0, 16.0, 18.0])
        sd = math.sqrt(10.0)
        self.assertEqual((m, n), (14.0, 5))
        self.assertAlmostEqual(half, 2.776445 * sd / math.sqrt(5), places=4)
        m, half, n = es.mean_ci_t([3.0])
        self.assertEqual((m, n), (3.0, 1))
        self.assertTrue(math.isnan(half))
        m, half, n = es.mean_ci_t([])
        self.assertTrue(math.isnan(m) and math.isnan(half) and n == 0)


class PairedTests(unittest.TestCase):
    def test_paired_difference_ci(self):
        a = {"t1": 10.0, "t2": 12.0, "t3": 14.0}
        b = {"t1": 9.0, "t2": 10.0, "t3": 13.0}
        eff = es.paired_effect(a, b)
        self.assertEqual(eff["n"], 3)
        self.assertEqual(eff["diffs"], [1.0, 2.0, 1.0])
        self.assertAlmostEqual(eff["mean_diff"], 4 / 3)
        half = 4.302653 * math.sqrt(1 / 3) / math.sqrt(3)
        self.assertAlmostEqual(eff["ci_low"], 4 / 3 - half, places=4)
        self.assertAlmostEqual(eff["ci_high"], 4 / 3 + half, places=4)
        self.assertEqual(eff["method"], "t pareado entre topologias")

    def test_unpaired_keys_are_refused(self):
        with self.assertRaisesRegex(ValueError, "pare"):
            es.paired_effect({"t1": 1.0}, {"t2": 1.0})

    def test_paired_bootstrap_is_seeded_and_brackets_the_mean(self):
        a = {f"t{i}": 10.0 + i for i in range(8)}
        b = {f"t{i}": 9.0 + 0.5 * i for i in range(8)}
        lo, hi = es.paired_bootstrap_ci(a, b, n_boot=2000, seed=1)
        self.assertEqual((lo, hi), es.paired_bootstrap_ci(a, b, n_boot=2000,
                                                          seed=1))
        mean = es.paired_effect(a, b)["mean_diff"]
        self.assertLess(lo, mean)
        self.assertGreater(hi, mean)

    def test_duplicating_epochs_does_not_shrink_the_topology_ci(self):
        """A unidade independente e a topologia: repetir epocas de uma
        topologia nao cria topologias novas."""
        epochs = {"t1": [100.0, 120.0], "t2": [90.0, 95.0],
                  "t3": [130.0, 110.0]}
        dup = dict(epochs, t1=epochs["t1"] * 50)
        tau = 1000.0
        one = {k: es.summarize_censored_latency(v, tau)["rmst"]
               for k, v in epochs.items()}
        many = {k: es.summarize_censored_latency(v, tau)["rmst"]
                for k, v in dup.items()}
        self.assertEqual(one, many)
        ref = {k: 50.0 for k in epochs}
        self.assertEqual(es.paired_effect(one, ref),
                         es.paired_effect(many, ref))


class CensoringTests(unittest.TestCase):
    def test_rmst_of_a_synthetic_sample(self):
        """Censura so no prazo: RMST = media de min(T, tau)."""
        self.assertAlmostEqual(
            es.restricted_mean_survival_time([1.0, 2.0, 3.0, INF], 4.0), 2.5)
        # censura antes do prazo (Kaplan-Meier): S = 1 ate 1; 2/3 ate 3; 0
        obs = [(1.0, True), (2.0, False), (3.0, True)]
        self.assertAlmostEqual(
            es.restricted_mean_survival_time(obs, 4.0), 1 + 2 * (2 / 3))

    def test_censoring_is_never_dropped(self):
        tau = 1219.5
        s = es.summarize_censored_latency([10.0, 20.0, INF, INF], tau)
        self.assertEqual((s["n"], s["n_complete"]), (4, 2))
        self.assertEqual(s["frac_complete"], 0.5)
        self.assertIsNone(s["median"])            # metade nao completou
        # n par, metade completa: mediana = (2a + 3a estatistica) / 2, e a
        # 3a so se sabe que passa de tau
        self.assertEqual(s["median_lower_bound"], (20.0 + tau) / 2)
        self.assertAlmostEqual(s["rmst"], (10 + 20 + 2 * tau) / 4)
        self.assertEqual(s["rmst_label"], "RMST ate o prazo")
        s = es.summarize_censored_latency([10.0, 20.0, 30.0, INF], tau)
        self.assertEqual(s["median"], 25.0)
        self.assertIsNone(s["median_lower_bound"])

    def test_empty_and_all_censored_are_explicit(self):
        tau = 100.0
        empty = es.summarize_censored_latency([], tau)
        self.assertEqual(empty["status"], "vazio")
        self.assertEqual(empty["n"], 0)
        self.assertTrue(math.isnan(empty["rmst"]))
        allc = es.summarize_censored_latency([INF, INF], tau)
        self.assertEqual(allc["status"], "todas censuradas")
        self.assertEqual(allc["rmst"], tau)
        self.assertIsNone(allc["median"])
        self.assertEqual(allc["median_lower_bound"], tau)


class WorkloadPlanTests(unittest.TestCase):
    def test_plan_is_protocol_free_stable_and_order_independent(self):
        cand = list(range(1, 36))
        a = es.make_workload_plan(base_seed=42, topology_seed=1042,
                                  candidates=cand, n_originators=35,
                                  n_epochs=5, payload_bytes=126,
                                  deadline_slots=1500, tag="cdf")
        b = es.make_workload_plan(base_seed=42, topology_seed=1042,
                                  candidates=list(reversed(cand)),
                                  n_originators=35, n_epochs=5,
                                  payload_bytes=126, deadline_slots=1500,
                                  tag="cdf")
        self.assertEqual(a, b)
        self.assertEqual(len(a.epochs), 5)
        for e, ep in enumerate(a.epochs):
            self.assertEqual(ep.epoch, e)
            self.assertEqual(len(set(ep.originators)), 35)
            self.assertEqual(ep.generation_slot, 0)
            self.assertEqual((ep.payload_bytes, ep.deadline_slots),
                             (126, 1500))
        self.assertEqual(a.plan_id, b.plan_id)
        c = es.make_workload_plan(base_seed=42, topology_seed=1042,
                                  candidates=cand, n_originators=10,
                                  n_epochs=5, payload_bytes=126,
                                  deadline_slots=1500, tag="cdf")
        self.assertNotEqual(c.plan_id, a.plan_id)
        self.assertNotEqual(c.epochs[0].originators,
                            c.epochs[1].originators)


if __name__ == "__main__":
    unittest.main()
