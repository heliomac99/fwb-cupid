"""[F28] Transmissoes dessincronizadas e captura de pacotes diferentes.

Fontes: Weaver (SenSys 2020, sec. 6.1: atraso aleatorio de ate 1 us em toda
TX); Lobba et al. (EWSN 2020, sec. 5.1: fontes do T do Crystal com ate 30 us
de diferenca; sec. 5.2: sucesso do T com todos os nos transmitindo);
Vecchia et al. (SECON 2019, sec. V-D: o receptor fica no primeiro preambulo
e so troca por um mais forte com simbolos alinhados, Fig. 12).
"""

import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import cupid_sim_v2 as sim

SYMBOL = 1017.6


def sink_hears(p1, p2, t2_ns):
    """O sink 0 ouve os nos 1 (chega em 0 ns) e 2 (chega em t2_ns)."""
    power = np.zeros((3, 3))
    power[0][1] = power[1][0] = p1
    power[0][2] = power[2][0] = p2
    return SimpleNamespace(ps_lin=power, tprop_ns=np.zeros((3, 3)),
                           audible=[{1, 2}, {0}, {0}]), [0.0, 0.0, t2_ns]


def decode(topo, clock):
    """Pacotes diferentes: o no 1 manda o 11, o no 2 manda o 22."""
    return sim.decode_at_receiver(
        0, [1, 2], {1: 11, 2: 22}, topo, clock,
        fading={(1, 0): 1.0, (2, 0): 1.0})


def star(n_sources):
    """Sink 0 no centro, fontes 1..n a um hop, sem enlace entre elas."""
    n = n_sources + 1
    power = np.zeros((n, n))
    for v in range(1, n):
        power[0][v] = power[v][0] = 1e-6
    audible = [set(np.nonzero(power[r] > 0)[0].tolist()) for r in range(n)]
    return SimpleNamespace(
        n=n, ps_lin=power, tprop_ns=np.zeros((n, n)), audible=audible,
        reliable=[set(a) for a in audible],
        hop={v: (0 if v == 0 else 1) for v in range(n)},
        group={v: "emitter" for v in range(n)})


class AsyncCaptureTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }
        # [P16] a aritmetica do peso supoe ruido desprezivel a -70 dBm, como
        # no radio legado (-105,7 dBm efetivos); no piso do CLOVES (-97,1
        # dBm) o ruido ja pesa 0,1 dB. O modelo testado aqui e o legado.
        sim.apply_config({"RADIO_PROFILE": "legacy_v2",
                          "CAPTURE_MODEL": "legacy_async_weight",  # [Fase 6] fisica legada de copias/peso; o empirico tem testes proprios
                          "PER_MODEL": "threshold", "REQ_SNR_DB": 3.0,
                          "ASYNC_INTERF_WEIGHT": 0.06,
                          "SYNC_WINDOW_NS": 128.0,
                          "CAPTURE_TIME_NS": 8141.0,
                          "PREAMBLE_ALIGN_NS": 136.0})

    def tearDown(self):
        sim.apply_config(self.original_config)

    def test_async_interferer_counts_with_reduced_weight(self):
        """Mesma potencia, 500 ns depois: fora de um simbolo de dados. Com
        peso integral o SINR e 0 dB; com 0,06, 12 dB."""
        topo, clock = sink_hears(1e-7, 1e-7, 500.0)
        res = decode(topo, clock)
        self.assertEqual(res[0], 11)
        self.assertAlmostEqual(res[4], -10.0 * np.log10(0.06), places=1)
        sim.apply_config({"ASYNC_INTERF_WEIGHT": 1.0})
        self.assertIsNone(decode(topo, clock))

    def test_interferer_inside_one_data_symbol_counts_fully(self):
        topo, clock = sink_hears(1e-7, 1e-7, 100.0)
        self.assertIsNone(decode(topo, clock))

    def test_stronger_preamble_wins_during_acquisition(self):
        """6 dB mais forte, 5 us depois: o receptor ainda acumula o PAC."""
        topo, clock = sink_hears(1e-7, 4e-7, 5000.0)
        self.assertEqual(decode(topo, clock)[0], 22)

    def test_after_acquisition_only_an_aligned_preamble_steals(self):
        """10 simbolos depois: alinhado rouba a trava; desalinhado em 500 ns
        nao, e o primeiro decodifica apesar do interferente 6 dB mais forte
        (SINR = -10 log10(4 x 0,06) = 6,2 dB)."""
        topo, clock = sink_hears(1e-7, 4e-7, 10 * SYMBOL)
        self.assertEqual(decode(topo, clock)[0], 22)
        topo, clock = sink_hears(1e-7, 4e-7, 10 * SYMBOL + 500.0)
        self.assertEqual(decode(topo, clock)[0], 11)
        sim.apply_config({"PREAMBLE_ALIGN_NS": 0.0})
        topo, clock = sink_hears(1e-7, 4e-7, 10 * SYMBOL)
        self.assertEqual(decode(topo, clock)[0], 11)

    def test_preset_etapa0_turns_f28_off(self):
        sim.apply_config(sim.PRESET_ETAPA0)
        self.assertEqual(
            (sim.TX_DELAY_NS, sim.CRYSTAL_T_OFFSET_NS, sim.CAPTURE_TIME_NS,
             sim.PREAMBLE_ALIGN_NS, sim.ASYNC_INTERF_WEIGHT),
            (0.0, 0.0, 40.0, 0.0, 1.0))
        sim.apply_config({"PER_MODEL": "threshold"})
        topo, clock = sink_hears(1e-7, 1e-7, 500.0)
        self.assertIsNone(decode(topo, clock))

    def test_weight_outside_its_range_is_rejected(self):
        for bad in (0.0, 1.5):
            with self.assertRaisesRegex(ValueError, "ASYNC_INTERF_WEIGHT"):
                sim.apply_config({"ASYNC_INTERF_WEIGHT": bad})


class TxDelayTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }
        sim.apply_config({"FAST_FADE_DB": 0.0, "CLOCK_JITTER_NS": 0.0,
                          "TX_PROB": 1.0, "BOOTSTRAP_DELAY": False, "BOOTSTRAP_MODEL": "bfs",
                          "PER_MODEL": "threshold"})

    def tearDown(self):
        sim.apply_config(self.original_config)

    def clocks_seen(self, delay):
        sim.apply_config({"TX_DELAY_NS": delay})
        seen = []
        original = sim.decode_physical_channels

        def spy(rx, tx_sets, pkt_of, topo, clock, *args, **kwargs):
            seen.append(({v for _, nodes in tx_sets for v in nodes},
                         list(clock)))
            return original(rx, tx_sets, pkt_of, topo, clock, *args,
                            **kwargs)

        with patch.object(sim, "decode_physical_channels", spy):
            sim.run_epoch("weaver", [1, 2, 3], star(3), random.Random(7),
                          slot_budget=12)
        return seen

    def test_ctx_transmitters_get_a_random_delay_below_the_limit(self):
        seen = self.clocks_seen(1000.0)
        self.assertTrue(seen)
        offsets = [clock[v] for txs, clock in seen for v in txs]
        self.assertTrue(all(0.0 <= o < 1000.0 for o in offsets))
        self.assertGreater(len(set(offsets)), 1)

    def test_no_delay_keeps_the_epoch_clock(self):
        seen = self.clocks_seen(0.0)
        self.assertTrue(seen)
        self.assertTrue(all(c == 0.0 for _, clock in seen for c in clock))


class CrystalOffsetTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }
        sim.apply_config({"FAST_FADE_DB": 0.0, "CLOCK_JITTER_NS": 0.0,
                          "PER_MODEL": "threshold", "N_TX_PER_FLOOD": 2,
                          "CRYSTAL_T_OFFSET_NS": 30500.0})

    def tearDown(self):
        sim.apply_config(self.original_config)

    def test_relays_inherit_the_time_base_of_the_packet_they_carry(self):
        """Fontes 2 e 3 (hop 2) so alcancam o no 1 (hop 1): no slot 1 ele
        retransmite o pacote adotado com o mesmo atraso da fonte dele."""
        n = 4
        power = np.zeros((n, n))
        for a, b in ((0, 1), (1, 2), (1, 3)):
            power[a][b] = power[b][a] = 1e-6
        audible = [set(np.nonzero(power[r] > 0)[0].tolist())
                   for r in range(n)]
        topo = SimpleNamespace(n=n, ps_lin=power, tprop_ns=np.zeros((n, n)),
                               audible=audible, hop={0: 0, 1: 1, 2: 2, 3: 2})
        calls = []
        original = sim.decode_at_receiver

        def spy(rx, txs, pkt_of, topo_, clock, *args, **kwargs):
            calls.append((tuple(txs), dict(pkt_of), list(clock)))
            return original(rx, txs, pkt_of, topo_, clock, *args, **kwargs)

        with patch.object(sim, "decode_at_receiver", spy):
            sim.crystal_t_flood([2, 3], topo, random.Random(3),
                                [0.0] * n)
        relay_slots = [(pkt_of, clock) for txs, pkt_of, clock in calls
                       if 1 in txs]
        self.assertTrue(relay_slots)
        for pkt_of, clock in relay_slots:
            self.assertEqual(clock[1], clock[pkt_of[1]])
        first = calls[0][2]
        self.assertNotEqual(first[2], first[3])
        self.assertTrue(all(0.0 <= first[v] < 30500.0 for v in (2, 3)))

    def test_all_but_the_sink_transmitting_matches_the_measured_t_success(self):
        """Calibracao do F28: com todos os nos transmitindo no T, o sink
        decodifica ~75% das vezes (EWSN 2020, sec. 5.2). Duas topologias do
        cenario-base, 60 floods cada. [P16] E a calibracao do modelo LEGADO
        (peso 0,06 no radio do canal 5): roda no preset legado explicito."""
        sim.apply_config(sim.PRESET_LEGACY_V2)
        topos = sim.make_topologies(n_topo=2, base_seed=sim.SEED)
        ok = total = 0
        for topo, seed in topos:
            rng = random.Random(seed)
            sources = sim.valid_nodes(topo)
            for _ in range(60):
                clock = [rng.gauss(0.0, sim.CLOCK_JITTER_NS)
                         for _ in range(topo.n)]
                pkt, _ = sim.crystal_t_flood(sources, topo, rng, clock)
                ok += pkt is not None
                total += 1
        self.assertGreater(ok / total, 0.6)
        self.assertLess(ok / total, 0.95)


if __name__ == "__main__":
    unittest.main()
