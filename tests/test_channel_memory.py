"""Canal com memoria: desvanecimento em duas escalas de tempo.

Com FADING_MODEL = "iid" (padrao, bit a bit) o desvanecimento de cada enlace
e redesenhado a cada slot. Em "two_scale" uma fracao w da variancia
(FADING_SLOW_FRACTION) e LENTA: fica constante na epoca e segue um processo
AR(1) de uma epoca para a seguinte (FADING_EPOCH_CORR); o resto continua por
slot. A variancia total num instante nao muda. A parte lenta e reciproca e
chaveada por (SEED, topologia, epoca): igual para todos os protocolos. O
periodo de agrupamento do Cupid e a epoca -1.
"""

import math
import random
import statistics as st
import unittest
from unittest.mock import patch

import numpy as np

import cupid_sim_v2 as sim
import rng_streams as rs

TINY = {"N_NODES": 8, "AREA_X": 30.0, "AREA_Y": 8.0, "EPOCHS": 2,
        "N_TOPOLOGIES": 1, "MAX_SLOTS": 200, "GROUPING_ITERATIONS": 1}


class Base(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }
        self.topo, self.seed = sim.make_topologies(n_topo=1, base_seed=42)[0]

    def tearDown(self):
        sim.apply_config(self.original_config)

    def fading(self, slot=0, epoch=None, **cfg):
        sim.apply_config(cfg)
        draws = rs.SlotDraws(rs.key_seed("teste", slot), self.topo.n, 1)
        rx = list(range(self.topo.n))
        if epoch is None:
            return sim.sample_slot_fading(rx, rx, self.topo, draws)
        with sim.slow_fading_epoch(self.seed, epoch, self.topo.n):
            return sim.sample_slot_fading(rx, rx, self.topo, draws)


class SlowProcessTests(Base):
    def test_default_is_iid_and_a_zero_slow_fraction_changes_nothing(self):
        self.assertEqual(sim.FADING_MODEL, "iid")
        iid = self.fading(epoch=3)
        self.assertEqual(iid, self.fading())          # sem contexto
        self.assertEqual(iid, self.fading(epoch=3, FADING_MODEL="two_scale",
                                          FADING_SLOW_FRACTION=0.0))

    def test_all_slow_is_constant_in_the_epoch_and_reciprocal(self):
        cfg = dict(FADING_MODEL="two_scale", FADING_SLOW_FRACTION=1.0,
                   FADING_EPOCH_CORR=0.5)
        a = self.fading(slot=0, epoch=4, **cfg)
        self.assertEqual(a, self.fading(slot=17, epoch=4, **cfg))
        for (tx, rx), v in a.items():
            if (rx, tx) in a:
                self.assertAlmostEqual(v, a[(rx, tx)], places=12)
        self.assertNotEqual(a, self.fading(slot=0, epoch=5, **cfg))

    def test_a_static_environment_repeats_across_epochs(self):
        cfg = dict(FADING_MODEL="two_scale", FADING_SLOW_FRACTION=1.0,
                   FADING_EPOCH_CORR=1.0)
        self.assertEqual(self.fading(epoch=-1, **cfg),
                         self.fading(epoch=30, **cfg))

    def test_the_process_is_keyed_and_ar1_between_epochs(self):
        sim.apply_config({"FADING_MODEL": "two_scale",
                          "FADING_EPOCH_CORR": 0.8})
        n = self.topo.n
        late_first = sim.slow_fading_matrix(self.seed, 6, n).copy()
        sim.apply_config({"FADING_EPOCH_CORR": 0.8})   # limpa o cache
        for e in range(-1, 7):
            m = sim.slow_fading_matrix(self.seed, e, n)
        np.testing.assert_array_equal(m, late_first)   # ordem nao importa
        iu = np.triu_indices(n, 1)
        corr = [np.corrcoef(sim.slow_fading_matrix(self.seed, e, n)[iu],
                            sim.slow_fading_matrix(self.seed, e + 1, n)[iu])
                [0, 1] for e in range(-1, 20)]
        self.assertAlmostEqual(st.fmean(corr), 0.8, delta=0.05)
        allv = np.concatenate([sim.slow_fading_matrix(self.seed, e, n)[iu]
                               for e in range(0, 20)])
        self.assertAlmostEqual(float(np.std(allv)), 1.0, delta=0.05)

    def test_the_instant_variance_is_preserved(self):
        """Parte lenta + parte rapida: o desvio do log do desvanecimento
        continua FAST_FADE_DB."""
        sim.apply_config({"FADING_MODEL": "two_scale",
                          "FADING_SLOW_FRACTION": 0.7,
                          "FADING_EPOCH_CORR": 0.0})
        vals = []
        for e in range(12):
            with sim.slow_fading_epoch(self.seed, e, self.topo.n):
                for slot in range(4):
                    draws = rs.SlotDraws(rs.key_seed("v", e, slot),
                                         self.topo.n, 1)
                    f = sim.sample_slot_fading(range(self.topo.n),
                                               range(self.topo.n),
                                               self.topo, draws)
                    vals += [10 * math.log10(v) for v in f.values()]
        self.assertAlmostEqual(float(np.std(vals)), sim.FAST_FADE_DB,
                               delta=0.3)

    def test_the_sequential_path_keeps_its_draw_count(self):
        """Mesmo numero de sorteios do rng sequencial (o legado depende
        disso): a parte lenta nao consome o rng."""
        sim.apply_config({"FADING_MODEL": "two_scale"})
        a, b = random.Random(5), random.Random(5)
        rx = list(range(self.topo.n))
        with sim.slow_fading_epoch(self.seed, 0, self.topo.n):
            sim.sample_slot_fading(rx, rx, self.topo, a)
        sim.apply_config({"FADING_MODEL": "iid"})
        sim.sample_slot_fading(rx, rx, self.topo, b)
        self.assertEqual(a.getstate(), b.getstate())

    def test_parameters_are_validated(self):
        for bad in ({"FADING_MODEL": "magico"},
                    {"FADING_SLOW_FRACTION": 1.5},
                    {"FADING_EPOCH_CORR": -0.1}):
            with self.subTest(cfg=bad), self.assertRaises(ValueError):
                sim.apply_config(bad)


class SensitivitySetTests(unittest.TestCase):
    def test_the_bench_separates_memory_within_and_between_epochs(self):
        import sensibilidade
        cfgs = sensibilidade.conjuntos()["canal_memoria"]
        self.assertEqual(cfgs[0][1], {})
        self.assertIn("(padrao)", cfgs[0][0])
        two = [c for _, c in cfgs[1:]]
        self.assertTrue(all(c["FADING_MODEL"] == "two_scale" for c in two))
        pairs = {(c["FADING_SLOW_FRACTION"], c["FADING_EPOCH_CORR"])
                 for c in two}
        self.assertIn((1.0, 1.0), pairs)          # ambiente estatico
        self.assertTrue(any(r == 0.0 and w > 0 for w, r in pairs))
        self.assertTrue(any(r >= 0.9 for _, r in pairs))


class EngineTests(Base):
    def test_every_protocol_sees_the_same_environment(self):
        """O bootstrap de cada epoca roda com a mesma parte lenta em todos
        os protocolos; o agrupamento do Cupid, com a da epoca -1."""
        sim.apply_config(dict(TINY, FADING_MODEL="two_scale"))
        topo, seed = sim.make_topologies(n_topo=1, base_seed=42)[0]
        original = sim.run_bootstrap
        seen = {}

        def spy(topo, rng, *a, **k):
            slow = sim._SLOW_FADE
            seen.setdefault(proto, []).append(
                None if slow is None else slow["epoch"])
            return original(topo, rng, *a, **k)

        for proto in ("weaver", "cupid"):
            with patch.object(sim, "run_bootstrap", spy):
                sim.simulate_topology(proto, topo, seed, O=7, n_epochs=2)
        self.assertEqual(seen["weaver"], [0, 1])
        self.assertEqual(seen["cupid"][-2:], [0, 1])
        self.assertEqual(set(seen["cupid"][:-2]), {-1})   # agrupamento
        self.assertIsNone(sim._SLOW_FADE)                  # contexto fechado

    def test_the_manifest_records_the_channel_model(self):
        sim.apply_config({"FADING_MODEL": "two_scale",
                          "FADING_SLOW_FRACTION": 0.9,
                          "FADING_EPOCH_CORR": 0.95})
        src = sim.physical_model_sources()["fading"]
        self.assertEqual(src, {"model": "two_scale", "fast_fade_db": 4.0,
                               "slow_fraction": 0.9, "epoch_corr": 0.95})
        for name in ("FADING_MODEL", "FADING_SLOW_FRACTION",
                     "FADING_EPOCH_CORR"):
            self.assertIn(name, sim.active_hypotheses())


if __name__ == "__main__":
    unittest.main()
