"""Fase 7: aleatoriedade por chave estavel.

Cada sorteio fisico de uma epoca sai de uma chave (raiz da epoca, slot) e e
indexado por no: fading por par (tx, rx), sorteio de decodificacao por
(rx, canal), atraso de TX, participacao e escolha de fila. Uma transmissao a
mais num slot nao desloca nenhum outro sorteio, a ordem de iteracao nao
importa, e a instrumentacao nao consome nada.
"""

import random
import unittest
from types import SimpleNamespace

import numpy as np

import cupid_sim_v2 as sim
import rng_streams as rs


class KeyTests(unittest.TestCase):
    def test_key_seed_is_stable_and_distinguishes_keys(self):
        self.assertEqual(rs.key_seed(42, 1042, "workload", 3),
                         rs.key_seed(42, 1042, "workload", 3))
        self.assertNotEqual(rs.key_seed(42, 1042, "workload", 3),
                            rs.key_seed(42, 1042, "workload", 4))
        # valor fixo: nao depende de hash() aleatorizado por processo
        self.assertEqual(rs.key_seed("a", 1), rs.key_seed("a", 1))
        self.assertIsInstance(rs.key_seed("a", 1), int)
        self.assertEqual(rs.stream("x").random(), rs.stream("x").random())

    def test_slot_draws_are_indexed_by_node_not_by_subset(self):
        streams = rs.EpochStreams(7, "e1")
        a = streams.slot(3, n=5, channels=2)
        b = streams.slot(3, n=5, channels=2)
        self.assertTrue(np.array_equal(a.fading_normal, b.fading_normal))
        self.assertFalse(np.array_equal(
            a.fading_normal, streams.slot(4, n=5, channels=2).fading_normal))
        topo = SimpleNamespace(audible=[{1, 2, 3}, {0}, {0}, {0}, set()])
        one = sim.sample_slot_fading([0], [1], topo, a)
        two = sim.sample_slot_fading([0], [2, 1, 3], topo, a)
        self.assertEqual(one[(1, 0)], two[(1, 0)])
        d1 = sim.sample_decode_draws([0], [0], a)
        d2 = sim.sample_decode_draws([4, 0, 2], [1, 0], a)
        self.assertEqual(d1[(0, 0)], d2[(0, 0)])


class SlotLayoutTests(unittest.TestCase):
    def test_draws_do_not_depend_on_the_number_of_channels(self):
        """Mudar N_CHANNELS nao pode deslocar o fading, o atraso de TX, a
        participacao ou a fila do slot (senao o FWB com 1 radio deixa de ver
        os mesmos numeros que o Cupid quando K muda)."""
        streams = rs.EpochStreams(3, "x")
        one = streams.slot(5, n=6, channels=1)
        many = streams.slot(5, n=6, channels=16)
        for field in ("fading_normal", "txdelay_u", "txprob_u", "queue_u"):
            self.assertTrue(np.array_equal(getattr(one, field),
                                           getattr(many, field)), field)
        self.assertTrue(np.array_equal(one.decode_u[:, 0],
                                       many.decode_u[:, 0]))
        with self.assertRaises(ValueError):
            streams.slot(5, n=6, channels=rs.MAX_DRAW_CHANNELS + 1)

    def test_fwb_one_radio_equals_cupid_for_any_channel_count(self):
        sim_cfg = {name: getattr(sim, name) for name in sim.PARAM_NAMES}
        try:
            sim.apply_config({"SEED": 42, "N_NODES": 12, "AREA_X": 40.0,
                              "AREA_Y": 12.0, "N_RADIOS_PER_NODE": 1,
                              "CHANNEL_PLAN": "ieee_nominal",
                              "GROUPING_ITERATIONS": 1})
            topo = sim.build_topology(random.Random(42))
            origs = sim.valid_nodes(topo)
            sim.apply_config({"N_CHANNELS": 1})
            ref = sim.run_epoch("cupid", origs, topo, None, slot_budget=80,
                                streams=rs.EpochStreams(9))
            for k in (2, 4, 12, 16):
                sim.apply_config({"N_CHANNELS": k})
                got = sim.run_epoch("cupid_fwb", origs, topo, None,
                                    slot_budget=80, streams=rs.EpochStreams(9))
                self.assertEqual(got, ref, k)
        finally:
            sim.apply_config(sim_cfg)


class KeyedEngineTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }
        sim.apply_config({"SEED": 42, "N_NODES": 16, "AREA_X": 60.0,
                          "AREA_Y": 12.0})
        self.topo = sim.build_topology(random.Random(42))

    def tearDown(self):
        sim.apply_config(self.original_config)

    def test_default_is_keyed_and_legacy_is_sequential(self):
        self.assertEqual(sim.RNG_MODEL, "keyed")
        self.assertEqual(sim.PRESET_LEGACY_V2["RNG_MODEL"], "sequential")
        with self.assertRaisesRegex(ValueError, "RNG_MODEL"):
            sim.apply_config({"RNG_MODEL": "magico"})

    def epoch(self, proto, origs, streams, **kw):
        slots = {}
        res = sim.run_epoch(proto, origs, self.topo, None, slot_budget=120,
                            delivery_slots=slots, streams=streams, **kw)
        return res, slots

    def test_repeating_the_configuration_reproduces_the_result(self):
        for proto in sim.PROTOCOLS:
            with self.subTest(protocol=proto):
                origs = sim.valid_nodes(self.topo)
                a = self.epoch(proto, origs, rs.EpochStreams(1, 2, 3))
                b = self.epoch(proto, origs, rs.EpochStreams(1, 2, 3))
                self.assertEqual(a, b)

    def test_iteration_order_does_not_change_the_draws(self):
        origs = sim.valid_nodes(self.topo)
        shuffled = list(reversed(origs))
        for proto in sim.PROTOCOLS:
            with self.subTest(protocol=proto):
                self.assertEqual(
                    self.epoch(proto, origs, rs.EpochStreams(5)),
                    self.epoch(proto, shuffled, rs.EpochStreams(5)))

    def test_instrumentation_consumes_no_simulation_draw(self):
        origs = sim.valid_nodes(self.topo)
        for proto in ("weaver", "cupid", "cupid_fwb", "crystal"):
            with self.subTest(protocol=proto):
                plain = self.epoch(proto, origs, rs.EpochStreams(8))
                info = {}
                traced = self.epoch(proto, origs, rs.EpochStreams(8),
                                  phy_stats=sim.new_phy_stats(),
                                  epoch_info=info,
                                  **({} if proto == "crystal" else
                                     {"holder_stats":
                                      sim.new_holder_stats()}))
                self.assertEqual(plain, traced)

    def test_protocols_share_the_physical_draws_of_a_slot(self):
        """Numeros aleatorios comuns: Cupid e Cupid FWB (um radio) sao
        identicos, e o bootstrap da epoca e o mesmo para todos os CTX."""
        origs = sim.valid_nodes(self.topo)
        infos = {}
        for proto in ("weaver", "cupid", "cupid_fwb"):
            infos[proto] = {}
            self.epoch(proto, origs, rs.EpochStreams(11),
                     epoch_info=infos[proto])
        self.assertEqual(infos["weaver"]["bootstrap"],
                         infos["cupid"]["bootstrap"])
        self.assertEqual(self.epoch("cupid", origs, rs.EpochStreams(11)),
                         self.epoch("cupid_fwb", origs, rs.EpochStreams(11)))

    def test_without_streams_the_epoch_rng_seeds_them(self):
        origs = sim.valid_nodes(self.topo)
        a = sim.run_epoch("cupid", origs, self.topo, random.Random(3),
                          slot_budget=80)
        b = sim.run_epoch("cupid", origs, self.topo, random.Random(3),
                          slot_budget=80)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
