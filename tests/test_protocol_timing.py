"""Fase 5: estruturas de slot, participacao e fases do Crystal.

Cupid (DCOSS-IoT 2025, Fig. 5b, lida celula a celula em
fontes/EXTRACT_cupid.md; sec. IV-C: "the order differs between
even-numbered and odd-numbered hops within the same group"). Weaver
(SenSys 2020): TX-RX-RX. Crystal (EWSN 2020, Tab. 1-2): slots de 404 us
(15 B) e 806 us (127 B) -- cada fase com o airtime do seu quadro.
"""

import math
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import cupid_sim_v2 as sim
import radio_profiles as rp


def star(n_sources):
    n = n_sources + 1
    power = np.zeros((n, n))
    for v in range(1, n):
        power[0][v] = power[v][0] = 1e-6
    audible = [set(np.nonzero(power[r] > 0)[0].tolist()) for r in range(n)]
    return SimpleNamespace(
        n=n, ps_lin=power, tprop_ns=np.zeros((n, n)), audible=audible,
        reliable=[set(a) for a in audible],
        hop={v: (0 if v == 0 else 1) for v in range(n)},
        pos={v: (float(v), 0.0) for v in range(n)},
        group={v: "emitter" for v in range(n)})


def chain(n_nodes, extra_hops=None):
    hop = {v: v for v in range(n_nodes)}
    hop.update(extra_hops or {})
    n = len(hop)
    power = np.zeros((n, n))
    for a in range(n_nodes - 1):
        power[a][a + 1] = power[a + 1][a] = 1e-6
    audible = [set(np.nonzero(power[r] > 0)[0].tolist()) for r in range(n)]
    return SimpleNamespace(
        n=n, ps_lin=power, tprop_ns=np.zeros((n, n)), audible=audible,
        reliable=[set(a) for a in audible], hop=hop,
        pos={v: (float(v), 0.0) for v in range(n)},
        group={v: "emitter" for v in range(n)})


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }

    def tearDown(self):
        sim.apply_config(self.original_config)

    def test_cloves_transmits_in_every_opportunity(self):
        self.assertEqual(sim.TX_PROB, 1.0)
        self.assertEqual(sim.PRESET_LEGACY_V2["TX_PROB"], 0.85)

    def test_tx_delay_never_removes_an_opportunity(self):
        """O atraso do TSM desloca a TX no slot; nunca a cancela."""
        sim.apply_config({"FAST_FADE_DB": 0.0, "CLOCK_JITTER_NS": 0.0,
                          "PER_MODEL": "threshold", "TX_DELAY_NS": 1000.0,
                          "BOOTSTRAP_MODEL": "bfs", "BOOTSTRAP_DELAY": False,
                          "ACK_MODEL": "none",
                          "LOCAL_ACK_MODEL": "overhearing"})
        sent = []
        original = sim.decode_physical_channels

        def spy(rx, tx_sets, *args, **kwargs):
            sent.append(frozenset(v for _, nodes in tx_sets for v in nodes))
            return original(rx, tx_sets, *args, **kwargs)

        with patch.object(sim, "decode_physical_channels", spy):
            sim.run_epoch("weaver", [1, 2, 3], star(3), random.Random(7),
                          slot_budget=4)
        # slot 1 (hop 1): os tres detentores transmitem juntos
        self.assertIn(frozenset({1, 2, 3}), sent)

    def test_weaver_timeline_is_tx_rx_rx(self):
        hop = {v: v for v in range(4)}
        for h in (1, 2, 3):
            tx = [s for s in range(9)
                  if sim.schedule_for("weaver", s, hop, [h])]
            self.assertEqual(tx, [s for s in range(9) if s % 3 == h % 3])

    def test_cupid_timeline_matches_fig_5b(self):
        """Fig. 5b (coluna c = slot c - 1): hop 1 emitter em 3 e 7,
        collector em 5 e 9; hop 2 collector em 4 e 8, emitter em 6; hop 3
        emitter em 5 e 9, collector em 7. O grupo que nao transmite fica em
        modo silencioso (escuta) no slot TX do proprio hop."""
        expected = {
            (1, "emitter"): [3, 7], (1, "collector"): [5, 9],
            (2, "collector"): [4, 8], (2, "emitter"): [6],
            (3, "emitter"): [5, 9], (3, "collector"): [7],
        }
        self.assertEqual(sim.CUPID_SCHEDULE, "fig5b")
        for (h, g), slots in expected.items():
            got = [s for s in range(h + 2, 10)
                   if sim.cupid_tx_group(s, h) == g]
            self.assertEqual(got, slots, (h, g))
        # cada grupo transmite a cada 4 slots
        for h in (1, 2, 3):
            for g in ("emitter", "collector"):
                tx = [s for s in range(40) if sim.cupid_tx_group(s, h) == g]
                self.assertEqual({b - a for a, b in zip(tx, tx[1:])}, {4})

    def test_cupid_sink_transmits_in_every_hop_0_tx_slot(self):
        """Fig. 5b (linha do sink: T nas colunas 3, 5, 7, 9) e Fig. 6c (ACK
        do sink nos slots 2, 4, 6). No legado o sink era do grupo emitter
        (1 a cada 4 slots)."""
        topo = SimpleNamespace(hop={0: 0}, group={0: "emitter"})
        self.assertEqual([s for s in range(10)
                          if sim.sink_scheduled("cupid", s, topo)],
                         [0, 2, 4, 6, 8])
        sim.apply_config({"CUPID_SCHEDULE": "legacy"})
        self.assertEqual([s for s in range(10)
                          if sim.sink_scheduled("cupid", s, topo)], [0, 4, 8])
        self.assertEqual(sim.PRESET_LEGACY_V2["CUPID_SCHEDULE"], "legacy")

    def test_crystal_phases_use_the_airtime_of_their_frames(self):
        """Linha de 3 nos (H = 2): S e A com quadros curtos, T com o quadro
        de dados. O relatorio 1 chega no 1o slot do primeiro T; o 2 (a dois
        hops) no 2o slot do segundo T -- a mesma sequencia de
        test_crystal.test_one_sync_flood_then_ta_pairs (5 e 13 slots), agora
        com S e A mais curtos."""
        sim.apply_config({"FAST_FADE_DB": 0.0, "CLOCK_JITTER_NS": 0.0,
                          "PER_MODEL": "threshold"})
        self.assertEqual(sim.CRYSTAL_PHASE_TIMING, "airtime")
        prof = rp.get_radio_profile(sim.RADIO_PROFILE)
        sa = rp.derive_phy(prof, sim.CRYSTAL_SA_FRAME_BYTES).slot_min_us
        self.assertAlmostEqual(sa, 404.0, delta=5.0)   # EWSN Tab. 2
        sa_units = sa / sim.SLOT_US
        slots = {}
        res = sim.run_epoch("crystal", [1, 2], chain(3), random.Random(1),
                            slot_budget=100, delivery_slots=slots)
        l_sa, l_t = 4, 3
        first = l_sa * sa_units + 1
        second = l_sa * sa_units + l_t + l_sa * sa_units + 2
        self.assertAlmostEqual(slots[1], first)
        self.assertAlmostEqual(slots[2], second)
        self.assertAlmostEqual(res[0], second)

    def test_crystal_negative_ack_is_a_physical_flood(self):
        """Com a fonte isolada, cada T fica mudo e o sink inunda um NACK: o
        flood A passa pela fisica mesmo sem ACK positivo."""
        sim.apply_config({"FAST_FADE_DB": 0.0, "CLOCK_JITTER_NS": 0.0,
                          "PER_MODEL": "threshold",
                          "BOOTSTRAP_MODEL": "bfs"})
        self.assertEqual(sim.CRYSTAL_A_MODEL, "physical")
        info = {}
        with patch.object(sim, "glossy_flood_reach",
                          wraps=sim.glossy_flood_reach) as spy:
            sim.run_epoch("crystal", [2], chain(2, extra_hops={2: 99}),
                          random.Random(1), slot_budget=200,
                          epoch_info=info)
        self.assertEqual(spy.call_count, sim.CRYSTAL_R)
        self.assertGreater(info["counters"]["tx_frames"], 0)

    def test_ack_and_control_frames_are_shorter_than_data_frames(self):
        n = 36
        data = sim.frame_bytes_of(7, n)
        ack = sim.frame_bytes_of(sim.ACK_PKT, n)
        gack = sim.frame_bytes_of(sim.GACK_PKT_BASE - 3, n)
        self.assertEqual(data, sim.DATA_FRAME_BYTES)
        self.assertEqual(ack, sim.SHORT_FRAME_BYTES + math.ceil(n / 8))
        self.assertEqual(gack, ack)
        self.assertEqual(sim.frame_bytes_of(("sync", 1), n),
                         sim.SHORT_FRAME_BYTES)
        # [F27] quadro com cabecalho: (pacote, ultimo ouvido, bitmap)
        self.assertEqual(sim.frame_bytes_of((7, 3, frozenset()), n), data)
        self.assertEqual(sim.frame_bytes_of((sim.ACK_PKT, None, None), n),
                         ack)
        prof = rp.get_radio_profile(sim.RADIO_PROFILE)
        self.assertLess(rp.frame_airtime_us(prof, ack),
                        rp.frame_airtime_us(prof, data))


if __name__ == "__main__":
    unittest.main()
