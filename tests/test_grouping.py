"""Fase 4: periodo de agrupamento do Cupid (emitter/collector por ranging).

Cupid (Park et al., DCOSS-IoT 2025): sec. IV-B e Alg. 1 (quatro casos,
media de M iteracoes, comparacao com o hop da epoca), sec. V-B (M = 10, slot
de 460 us, custo ">= (Bootstrapping + 3 x N) x slot duration x M", erro
medio de 1,31 m). O ranging passa pela mesma fisica dos dados; a distancia
verdadeira so e usada para medir o erro.
"""

import math
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import cupid_grouping as cg
import cupid_sim_v2 as sim


def make_topology(n, links, pos):
    """Enlaces reciprocos com potencia dada; atraso de propagacao pela
    posicao (a distancia fisica que o ranging mede)."""
    power = np.zeros((n, n))
    for (a, b), p in links.items():
        power[a][b] = power[b][a] = p
    tprop = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            tprop[a][b] = math.dist(pos[a], pos[b]) / sim.C_M_PER_NS
    audible = [set(np.nonzero(power[r] > 0)[0].tolist()) for r in range(n)]
    return SimpleNamespace(
        n=n, ps_lin=power, tprop_ns=tprop, audible=audible,
        reliable=[set(a) for a in audible], pos=dict(pos),
        hop={v: v for v in range(n)},
        group={v: "emitter" for v in range(n)})


def line(n, spacing=10.0):
    return make_topology(n, {(a, a + 1): 1e-6 for a in range(n - 1)},
                         {v: (spacing * v, 0.0) for v in range(n)})


class VirtualHopRuleTests(unittest.TestCase):
    def test_the_four_cases_of_algorithm_1(self):
        f = cg.virtual_hop_value
        self.assertEqual(f(2, 10.0, 20.0), (2.25, "both"))
        self.assertEqual(f(2, 20.0, 10.0), (2.75, "both"))
        self.assertEqual(f(2, 10.0, 10.0), (2.5, "both"))
        self.assertEqual(f(2, 10.0, None), (2.25, "up_only"))
        self.assertEqual(f(2, None, 10.0), (2.75, "down_only"))
        self.assertEqual(f(2, None, None), (2.5, "none"))

    def test_group_follows_the_hop_of_the_epoch(self):
        """Virtual 2,3: emitter no hop 2 (mais perto de 2 que de 3);
        collector se a epoca o poe no hop 1 (mais perto de 2 que de 1)."""
        grouping = cg.GroupingResult(
            virtual_hop={1: 2.3, 2: 2.5, 3: None}, observations=(),
            successful_upper=0, successful_lower=0, both_failed=0,
            total_slots=0, total_time_ms=0.0, iterations=1,
            measurement_errors=(), operations=())

        def state(hops):
            return cg.BootstrapState(
                hop_by_node=hops, synchronized_nodes=frozenset(hops),
                first_sync_slot={v: h - 1 for v, h in hops.items()},
                parent={}, bootstrap_slots=0, failure_reason={},
                clock_ns=(0.0,) * 4)

        g = cg.assign_epoch_groups(grouping, state({0: 0, 1: 2, 2: 2, 3: 3}))
        self.assertEqual(g[1], "emitter")
        # empate exato em h + 0,5: emitter (Fig. 8 desenha o 2,5 como
        # circulo); virtual desconhecido: emitter
        self.assertEqual(g[2], "emitter")
        self.assertEqual(g[3], "emitter")
        self.assertEqual(g[0], "emitter")
        g = cg.assign_epoch_groups(grouping, state({0: 0, 1: 1, 2: 2, 3: 3}))
        self.assertEqual(g[1], "collector")


class GroupingPeriodTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }
        sim.apply_config({"FAST_FADE_DB": 0.0, "CLOCK_JITTER_NS": 0.0,
                          "PER_MODEL": "threshold"})

    def tearDown(self):
        sim.apply_config(self.original_config)

    def config(self, **kw):
        base = dict(iterations=3, clock_ppm=0.0, lde_sigma_ns=0.0)
        base.update(kw)
        return sim.grouping_config(**base)

    def test_defaults_follow_the_paper(self):
        cfg = sim.grouping_config()
        self.assertEqual(cfg.iterations, 10)
        self.assertEqual(cfg.slot_us, 460.0)
        self.assertEqual(cfg.bootstrap_slots, 10)
        self.assertEqual(sim.GROUPING_MODEL, "ranging")
        self.assertEqual(sim.PRESET_LEGACY_V2["GROUPING_MODEL"], "static")

    def test_line_measures_both_neighbours_and_averages(self):
        """Linha 0-1-2-3 com 10 m entre nos: o no 1 mede o sink (cima) e o
        no 2 (baixo) na mesma distancia -> depende do erro; o no 3 (ultimo)
        so tem hop de cima -> +0,25 em toda iteracao."""
        res = sim.run_grouping_period(line(4), self.config(),
                                      random.Random(1))
        self.assertEqual(res.iterations, 3)
        self.assertEqual(res.virtual_hop[3], 3.25)
        obs3 = [o for o in res.observations if o.node == 3]
        self.assertEqual(len(obs3), 3)
        self.assertTrue(all(o.case == "up_only" for o in obs3))
        for v in (1, 2, 3):
            values = [o.value for o in res.observations if o.node == v]
            self.assertAlmostEqual(res.virtual_hop[v],
                                   sum(values) / len(values))

    def test_cost_is_the_sum_of_the_simulated_operations(self):
        cfg = self.config()
        res = sim.run_grouping_period(line(4), cfg, random.Random(1))
        self.assertEqual(res.total_slots,
                         sum(slots for _, _, slots in res.operations))
        self.assertEqual(res.total_slots,
                         cfg.iterations * (cfg.bootstrap_slots + 3 * 3))
        self.assertAlmostEqual(res.total_time_ms,
                               res.total_slots * cfg.slot_us / 1000.0)
        kinds = {kind for _, kind, _ in res.operations}
        self.assertEqual(kinds, {"bootstrap", "poll", "upper", "lower"})

    def test_physical_loss_gives_a_missing_observation(self):
        """O no 3 esta abaixo da sensibilidade do no 2: a resposta dele se
        perde e o no 2 so mede o de cima -- nunca uma distancia inventada."""
        weak = sim.NOISE_EFF_LIN * 10 ** ((sim.REQ_SNR_DB - 3) / 10)
        topo = make_topology(4, {(0, 1): 1e-6, (1, 2): 1e-6, (2, 3): weak},
                             {v: (10.0 * v, 0.0) for v in range(4)})
        res = sim.run_grouping_period(topo, self.config(), random.Random(1))
        obs2 = [o for o in res.observations if o.node == 2]
        self.assertTrue(obs2)
        self.assertTrue(all(o.d_down_m is None for o in obs2))
        self.assertTrue(all(o.case == "up_only" for o in obs2))
        # o no 3 nem sincroniza: fica sem observacao nenhuma
        self.assertIsNone(res.virtual_hop[3])

    def test_true_distance_is_only_used_for_the_error(self):
        """Posicoes adulteradas mudam o erro calculado, nunca o grupo."""
        topo = line(5)
        honest = sim.run_grouping_period(topo, self.config(clock_ppm=10.0),
                                         random.Random(3))
        topo.pos = {v: (0.0, 0.0) for v in range(5)}
        lied = sim.run_grouping_period(topo, self.config(clock_ppm=10.0),
                                       random.Random(3))
        self.assertEqual(honest.virtual_hop, lied.virtual_hop)
        self.assertEqual([o.value for o in honest.observations],
                         [o.value for o in lied.observations])
        self.assertNotEqual(honest.measurement_errors,
                            lied.measurement_errors)

    def test_measurement_error_comes_from_modelled_mechanisms(self):
        """Sem deriva nem ruido de borda, o erro e so o truncamento de 8 ns
        do delayed TX: a distancia medida fica em (d - 1,2 m, d]."""
        res = sim.run_grouping_period(line(3), self.config(),
                                      random.Random(2))
        self.assertTrue(res.measurement_errors)
        limit = sim.C_M_PER_NS * (4.0 / 0.4992) / 2.0
        for err in res.measurement_errors:
            self.assertLessEqual(err, 1e-9)
            self.assertGreater(err, -limit - 1e-9)

    def test_deterministic_per_seed_and_telemetry_is_passive(self):
        topo = line(4)
        a = sim.run_grouping_period(topo, self.config(clock_ppm=10.0),
                                    random.Random(9))
        b_rng = random.Random(9)
        stats = sim.new_phy_stats()
        b = sim.run_grouping_period(topo, self.config(clock_ppm=10.0),
                                    b_rng, phy_stats=stats)
        self.assertEqual(a, b)
        c_rng = random.Random(9)
        sim.run_grouping_period(topo, self.config(clock_ppm=10.0), c_rng)
        self.assertEqual(b_rng.random(), c_rng.random())
        self.assertGreater(stats["n_groups_singleton"], 0)

    def test_ranging_goes_through_the_receiver_physics(self):
        calls = []
        original = sim.decode_at_receiver

        def spy(rx, txs, pkt_of, *args, **kwargs):
            calls.append(set(pkt_of.values()))
            return original(rx, txs, pkt_of, *args, **kwargs)

        with patch.object(sim, "decode_at_receiver", spy):
            sim.run_grouping_period(line(3), self.config(iterations=1),
                                    random.Random(1))
        kinds = {frame[0] for frames in calls for frame in frames}
        self.assertEqual(kinds, {"sync", "poll", "resp"})


class RangingBenchmarkTests(unittest.TestCase):
    def test_error_statistics_and_no_fitting_to_the_published_value(self):
        import microbenchmark_ranging as mb
        st = mb.error_stats([-1.0, 0.5, -0.5, 2.0])
        self.assertEqual(st["n"], 4)
        self.assertAlmostEqual(st["mae"], 1.0)
        self.assertAlmostEqual(st["bias"], 0.25)
        self.assertAlmostEqual(st["rmse"], math.sqrt((1 + .25 + .25 + 4) / 4))
        self.assertTrue(math.isnan(mb.error_stats([])["mae"]))
        # o numero publicado e so referencia: nenhum parametro do simulador
        # e derivado dele
        self.assertEqual(mb.PUBLISHED_AVG_ERROR_M, 1.31)
        for name in sim.PARAM_NAMES:
            self.assertNotEqual(getattr(sim, name), 1.31, name)

    def test_benchmark_grids_contain_the_values_in_force(self):
        import microbenchmark_ranging as mb
        ppm, lde = mb.default_grids()
        self.assertIn(sim.RANGING_CLOCK_PPM, ppm)
        self.assertIn(sim.RANGING_LDE_SIGMA_NS, lde)
        self.assertIn(20.0, ppm)


if __name__ == "__main__":
    unittest.main()
