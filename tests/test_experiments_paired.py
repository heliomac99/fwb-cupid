"""Fases 5, 7 e 8 no nivel dos experimentos.

Todos os protocolos de um experimento recebem o MESMO WorkloadPlan e os
mesmos streams fisicos por chave; o Cupid usa grupos do periodo de
agrupamento (nunca o oraculo estatico); bootstrap e agrupamento sao medidos
a parte da latencia da coleta; e o caminho legado continua identico.
"""

import hashlib
import json
import math
import random
import unittest

import cupid_sim_v2 as sim
import experiment_stats as es


def small():
    sim.apply_config({"SEED": 42, "N_NODES": 14, "AREA_X": 50.0,
                      "AREA_Y": 12.0, "MAX_SLOTS": 300,
                      "GROUPING_ITERATIONS": 2})
    return sim.make_topologies(n_topo=2, base_seed=42)


class PairedExperimentTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }

    def tearDown(self):
        sim.apply_config(self.original_config)

    def test_originators_are_identical_across_protocols(self):
        topo, seed = small()[0]
        runs = {p: sim.simulate_topology(p, topo, seed, O=8, n_epochs=3)
                for p in sim.PROTOCOLS}
        ref = [r["originators"] for r in runs["cupid"]["records"]]
        for proto, run in runs.items():
            with self.subTest(protocol=proto):
                self.assertEqual([r["originators"] for r in run["records"]],
                                 ref)
                self.assertEqual(run["plan_id"], runs["cupid"]["plan_id"])

    def test_ctx_protocols_share_each_epoch_bootstrap(self):
        topo, seed = small()[0]
        boot = {p: [r["bootstrap_slots"] for r in sim.simulate_topology(
            p, topo, seed, O=8, n_epochs=4)["records"]]
            for p in ("weaver", "cupid", "cupid_fwb")}
        self.assertEqual(boot["weaver"], boot["cupid"])
        self.assertEqual(boot["weaver"], boot["cupid_fwb"])

    def test_cupid_groups_come_from_the_grouping_period(self):
        topo, seed = small()[0]
        run = sim.simulate_topology("cupid", topo, seed, O=8, n_epochs=2)
        self.assertTrue(all(r["group_source"] == "ranging"
                            for r in run["records"]))
        self.assertGreater(run["grouping_setup_ms"], 0.0)
        self.assertEqual(run["grouping_setup_ms"],
                         run["grouping"].total_time_ms)
        weaver = sim.simulate_topology("weaver", topo, seed, O=8, n_epochs=2)
        self.assertEqual(weaver["grouping_setup_ms"], 0.0)

    def test_grouping_period_does_not_depend_on_the_rng_model(self):
        """O agrupamento por ranging vale tambem com sorteio sequencial."""
        topo, seed = small()[0]
        sim.apply_config({"RNG_MODEL": "sequential"})
        run = sim.simulate_topology("cupid", topo, seed, O=8, n_epochs=2)
        self.assertTrue(all(r["group_source"] == "ranging"
                            for r in run["records"]))
        self.assertGreater(run["grouping_setup_ms"], 0.0)

    def test_setup_costs_are_reported_apart_from_the_data_latency(self):
        topo, seed = small()[0]
        for proto in sim.PROTOCOLS:
            with self.subTest(protocol=proto):
                run = sim.simulate_topology(proto, topo, seed, O=8,
                                            n_epochs=2)
                for r in run["records"]:
                    self.assertGreater(r["bootstrap_ms"], 0.0)
                    self.assertIn("latency_data_epoch_ms", r)
                    self.assertIn("last_report_ms", r)
                    self.assertGreaterEqual(r["tx_data"], r["delivered"])

    def test_last_report_latency_exists_even_for_incomplete_epochs(self):
        """Metrica do artigo do Cupid: do inicio da epoca ate o ultimo
        pacote recebido, mesmo se a epoca nao completou."""
        topo, seed = small()[0]
        run = sim.simulate_topology("crystal", topo, seed, O=13, n_epochs=2,
                                    slot_budget=40)
        for r in run["records"]:
            if r["delivered"]:
                self.assertEqual(r["last_report_ms"], sim.slots_to_ms(
                    max(r["delivery_slots"].values())))
            if r["delivered"] < len(r["originators"]):
                self.assertTrue(math.isinf(r["latency_data_epoch_ms"]))

    def test_confidence_interval_uses_student_t(self):
        m, half = sim.mean_ci([1.0, 2.0, 3.0])
        self.assertEqual(m, 2.0)
        self.assertAlmostEqual(half, es.t_critical(0.95, 2) * 1.0
                               / math.sqrt(3))
        m, half = sim.mean_ci([5.0])
        self.assertEqual(m, 5.0)
        self.assertTrue(math.isnan(half))

    def test_legacy_experiment_path_is_unchanged(self):
        """Valores medidos antes do refactor, no preset legado."""
        sim.apply_config(dict(sim.PRESET_LEGACY_V2, SEED=42, N_NODES=36,
                              AREA_X=83.0, AREA_Y=33.0, MAX_SLOTS=1500))
        topo, seed = sim.make_topologies(n_topo=1, base_seed=42)[0]
        expected = {
            "cupid": ([104.064, 97.56], "d7c14f1aa1bc55ab"),
            "weaver": ([140.649, 169.917], "1baa4b874ba51414"),
            "crystal": ([254.469, 254.469], "337776b0da71dd7d"),
        }
        for proto, (lats, digest) in expected.items():
            with self.subTest(protocol=proto):
                st = sim._latency_stats(proto, topo, seed, 35, epochs=2)
                self.assertEqual(st["lats"], lats)
                self.assertEqual((st["complete"], st["sent"]), (2, 70))
                self.assertEqual(hashlib.sha256(json.dumps(sorted(
                    st["report_lats"])).encode()).hexdigest()[:16], digest)


if __name__ == "__main__":
    unittest.main()
