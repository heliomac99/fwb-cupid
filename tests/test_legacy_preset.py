"""O preset legado reproduz o motor auditado (commit 5ef525b) bit a bit.

Os numeros abaixo foram medidos em 5ef525b, com a configuracao padrao
daquele commit (F28: canal 5, TX_PROB = 0,85, hops por BFS da potencia
media, captura com peso assincrono 0,06), ANTES de qualquer mudanca do
preset CLOVES. Se este teste falhar, alguma mudanca nova alterou o motor por
fora da chave que a desliga, ou a chave nao entrou em PRESET_LEGACY_V2.
"""

import hashlib
import json
import random
import unittest

import cupid_sim_v2 as sim


def digest(obj):
    blob = json.dumps(obj, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


EXPECTED_EPOCHS = {
    "cupid_fwb": ([((140, 35), "acb6a62bd75bcb09"),
                   ((122, 35), "0a14a6c7bd1dad62")], 0.1709016167693077),
    "cupid": ([((140, 35), "acb6a62bd75bcb09"),
               ((122, 35), "0a14a6c7bd1dad62")], 0.1709016167693077),
    "weaver": ([((221, 35), "bb528f1437a8cab6"),
                ((179, 35), "edd878eb23bfc3a2")], 0.1421324730559217),
    "crystal": ([((313, 35), "0de625b2a2bed2f2"),
                 ((313, 35), "c2eaf7da09962c1c")], 0.2805794899756755),
}


class LegacyPresetTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }
        sim.apply_config(dict(sim.PRESET_LEGACY_V2, SEED=42, N_NODES=36,
                              AREA_X=83.0, AREA_Y=33.0, MAX_SLOTS=1500))
        self.topo = sim.build_topology(random.Random(42))

    def tearDown(self):
        sim.apply_config(self.original_config)

    def test_topology_is_the_audited_one(self):
        topo = self.topo
        self.assertEqual(digest({str(k): v for k, v in topo.hop.items()}),
                         "78b6408e8acecb09")
        self.assertEqual(digest({str(k): v for k, v in topo.group.items()}),
                         "86d17d9bf4f9b801")
        self.assertAlmostEqual(topo.rrange, 29.37649652, places=7)
        self.assertAlmostEqual(
            float(topo.ps_db[topo.ps_db > -250].sum()), -123376.108835,
            places=5)

    def test_epochs_and_rng_consumption_are_the_audited_ones(self):
        valid = sim.valid_nodes(self.topo)
        for proto, (runs, next_draw) in EXPECTED_EPOCHS.items():
            with self.subTest(protocol=proto):
                rng = random.Random(2026)
                got = []
                for _ in range(2):
                    origs = rng.sample(valid, 35)
                    ds = {}
                    res = sim.run_epoch(proto, origs, self.topo, rng,
                                        delivery_slots=ds)
                    got.append((tuple(res),
                                digest({str(k): v for k, v in ds.items()})))
                self.assertEqual((got, rng.random()), (runs, next_draw))

    def test_crystal_t_phase_and_capture_benchmark_are_the_audited_ones(self):
        valid = sim.valid_nodes(self.topo)
        rng = random.Random(99)
        floods = []
        for _ in range(3):
            clock = [rng.gauss(0.0, sim.CLOCK_JITTER_NS)
                     for _ in range(self.topo.n)]
            floods.append(tuple(sim.crystal_t_flood(valid, self.topo, rng,
                                                    clock)))
        self.assertEqual((floods, rng.random()),
                         ([(7, 1), (None, None), (7, 2)],
                          0.3752663228285107))
        topos = sim.make_topologies(n_topo=1, base_seed=42)
        micro = sim.microbenchmark_capture(topos, k_vals=(2, 6), draws=300)
        self.assertEqual(micro, [(2, 0.8133333333333334, 0.14333333333333334),
                                 (6, 0.87, 0.0033333333333333335)])


if __name__ == "__main__":
    unittest.main()
