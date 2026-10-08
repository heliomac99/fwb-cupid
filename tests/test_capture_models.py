"""Fase 6: modelos de captura (interface, legado e empirico).

Pontos medidos com o radio do CLOVES (canal 4, PRF 64, codigo 17):
EWSN 2020 Fig. 8 (mesmo quadro, 15 B, desalinhamento 0-128 ns) e Fig. 9
(mesmo quadro, 127 B, atenuacao relativa 0-8 dB); SECON 2019 Fig. 13
(quadros diferentes, fontes somadas da mais proxima, Delta-t ate 20 us;
alturas extraidas do vetor do PDF). Fora da regiao medida o modelo devolve
uma faixa e marca a extrapolacao.
"""

import math
import unittest

import capture_models as cm
import cupid_sim_v2 as sim


def ctx(**kw):
    base = dict(frame_bytes=126, signal_mw=1e-7, noise_mw=1e-12,
                sync_interf_mw=0.0, async_interf_mw=0.0, same_copy_dt_ns=None,
                same_copy_dp_db=None, async_senders=0,
                acquisition_ns=8141.0, max_async_dt_ns=None)
    base.update(kw)
    return cm.CaptureContext(**base)


class TableTests(unittest.TestCase):
    def test_interpolation_hits_the_measured_points(self):
        t = cm.SAME_LONG_DP
        self.assertAlmostEqual(t.value(5.0)[0], 0.72)
        self.assertAlmostEqual(t.value(4.5)[0], (0.44 + 0.72) / 2)
        self.assertAlmostEqual(cm.SAME_SHORT_DT.value(62.0)[0], 0.927)
        self.assertAlmostEqual(cm.DIFF_ASYNC_K.value(4)[0], 0.9285, places=4)
        for table in (cm.SAME_LONG_DP, cm.SAME_SHORT_DT, cm.DIFF_ASYNC_K):
            self.assertTrue(table.source)
            self.assertTrue(table.figure)
            for x, _ in table.points:
                self.assertEqual(table.value(x)[1], "medido")

    def test_out_of_region_is_identified_with_a_band(self):
        v, status, lo, hi = cm.DIFF_ASYNC_K.value(12, band=True)
        self.assertEqual(status, "extrapolacao")
        self.assertLess(lo, hi)
        self.assertLessEqual(lo, v)
        self.assertLessEqual(v, hi)
        self.assertEqual(hi, cm.DIFF_ASYNC_K.value(9)[0])
        v, status = cm.SAME_LONG_DP.value(10.0)
        self.assertEqual(status, "saturado")
        self.assertEqual(v, cm.SAME_LONG_DP.value(8.0)[0])

    def test_isotonic_cleanup_is_declared_and_monotone(self):
        """PAV: os pontos brutos que contradizem a monotonia sao agrupados
        (0-2 dB na Fig. 9; 7-9 fontes na Fig. 13), e a correcao e
        registrada, nao escondida."""
        self.assertTrue(cm.SAME_LONG_DP.isotonic_adjusted)
        raw = dict(cm.SAME_LONG_DP.raw_points)
        self.assertEqual(raw[0.0], 0.05)
        ys = [y for _, y in cm.SAME_LONG_DP.points]
        self.assertEqual(ys, sorted(ys))
        ks = [y for _, y in cm.DIFF_ASYNC_K.points]
        self.assertEqual(ks, sorted(ks, reverse=True))


class ModelTests(unittest.TestCase):
    def test_long_and_short_frames_can_diverge(self):
        model = cm.EmpiricalUwbCapture()
        short, _ = model.success_probability(ctx(
            frame_bytes=15, same_copy_dt_ns=0.0, same_copy_dp_db=0.0))
        long_, _ = model.success_probability(ctx(
            frame_bytes=127, same_copy_dt_ns=0.0, same_copy_dp_db=0.0))
        self.assertAlmostEqual(short, 0.970, places=3)
        self.assertAlmostEqual(long_, cm.SAME_LONG_DP.value(0.0)[0],
                               places=3)
        self.assertGreater(short - long_, 0.5)

    def test_identical_and_different_frames_can_diverge(self):
        """Duas fontes, mesma potencia, sincronizadas: copias identicas
        curtas quase sempre passam; quadros diferentes colidem (SINR 0 dB)."""
        model = cm.EmpiricalUwbCapture()
        same, _ = model.success_probability(ctx(
            frame_bytes=15, same_copy_dt_ns=0.0, same_copy_dp_db=0.0))
        diff, _ = model.success_probability(ctx(
            frame_bytes=15, sync_interf_mw=1e-7))
        self.assertGreater(same, 0.9)
        self.assertLess(diff, 0.2)

    def test_more_power_difference_never_hurts_the_strongest(self):
        model = cm.EmpiricalUwbCapture()
        for fb in (15, 60, 127):
            ps = [model.success_probability(ctx(
                frame_bytes=fb, same_copy_dt_ns=0.0, same_copy_dp_db=dp))[0]
                for dp in [0.5 * i for i in range(17)]]
            self.assertEqual(ps, sorted(ps), fb)
        ps = [model.success_probability(ctx(
            sync_interf_mw=1e-7 * 10 ** (-dp / 10)))[0]
            for dp in range(0, 12)]
        self.assertEqual(ps, sorted(ps))

    def test_more_concurrency_never_helps(self):
        for mode in ("lower", "mid", "upper"):
            model = cm.EmpiricalUwbCapture(extrapolation=mode)
            ps = [model.success_probability(ctx(async_senders=k))[0]
                  for k in range(0, 40)]
            self.assertEqual(ps, sorted(ps, reverse=True), mode)

    def test_flags_report_every_assumption_used(self):
        model = cm.EmpiricalUwbCapture()
        _, flags = model.success_probability(ctx(async_senders=20))
        self.assertIn("async_k_extrapolacao", flags)
        _, flags = model.success_probability(ctx(
            async_senders=2, max_async_dt_ns=900.0))
        self.assertIn("async_dentro_da_aquisicao", flags)
        _, flags = model.success_probability(ctx(
            frame_bytes=60, same_copy_dt_ns=10.0, same_copy_dp_db=1.0))
        self.assertIn("comprimento_interpolado", flags)
        # abaixo do menor comprimento medido (15 B) nao e interpolacao: usa a
        # tabela de 15 B e marca a hipotese
        _, flags = model.success_probability(ctx(
            frame_bytes=11, same_copy_dt_ns=10.0, same_copy_dp_db=1.0))
        self.assertIn("comprimento_abaixo_do_medido", flags)
        self.assertNotIn("comprimento_interpolado", flags)
        _, flags = model.success_probability(ctx())
        self.assertEqual(flags, ())

    def test_intra_acquisition_regime_has_explicit_bounds(self):
        """Concorrentes fora de 1 simbolo mas TODOS dentro da aquisicao
        (atraso <= 1 us dos CTX): regime nao medido. 'async_table' (padrao)
        usa a curva do SECON com Delta-t ate 20 us -- o melhor caso medido;
        'sync_table', a curva medida com pacotes sincronizados -- o pior
        caso medido; 'no_penalty' e um limite teorico, sem medicao."""
        inside = ctx(async_senders=3, max_async_dt_ns=500.0)   # k = 4
        per = sim.decode_success_prob(10 * math.log10(1e-7 / 1e-12))
        expected = {"async_table": cm.DIFF_ASYNC_K.value(4)[0],
                    "sync_table": min(cm.DIFF_SYNC_K_BOUND.value(4)[0],
                                      cm.DIFF_ASYNC_K.value(4)[0]),
                    "no_penalty": 1.0}
        self.assertEqual(cm.EmpiricalUwbCapture().intra_acquisition,
                         "async_table")
        for mode, factor in expected.items():
            model = cm.EmpiricalUwbCapture(intra_acquisition=mode)
            p, flags = model.success_probability(inside)
            self.assertAlmostEqual(p, per * factor, places=12, msg=mode)
            self.assertIn("async_dentro_da_aquisicao", flags)
        self.assertAlmostEqual(expected["sync_table"], 0.6685, places=4)
        with self.assertRaisesRegex(ValueError, "intra"):
            cm.EmpiricalUwbCapture(intra_acquisition="magico")

    def test_bounds_only_change_the_intra_acquisition_regime(self):
        """Com algum concorrente alem da aquisicao (o regime medido do
        SECON, ate 20 us) os tres modos coincidem."""
        outside = ctx(async_senders=3, max_async_dt_ns=12000.0)
        ps = {mode: cm.EmpiricalUwbCapture(
            intra_acquisition=mode).success_probability(outside)[0]
            for mode in cm.INTRA_ACQUISITION_MODES}
        self.assertEqual(len(set(ps.values())), 1)

    def test_bounds_are_ordered_and_monotone_in_the_sources(self):
        for extrapolation in ("lower", "mid", "upper"):
            per_mode = {}
            for mode in cm.INTRA_ACQUISITION_MODES:
                model = cm.EmpiricalUwbCapture(extrapolation,
                                               intra_acquisition=mode)
                per_mode[mode] = [model.success_probability(ctx(
                    async_senders=k, max_async_dt_ns=500.0))[0]
                    for k in range(0, 40)]
                self.assertEqual(per_mode[mode],
                                 sorted(per_mode[mode], reverse=True))
            for lo, mid, hi in zip(per_mode["sync_table"],
                                   per_mode["async_table"],
                                   per_mode["no_penalty"]):
                self.assertLessEqual(lo, mid)
                self.assertLessEqual(mid, hi)

    def test_the_worst_case_table_is_the_measured_one(self):
        """Mesmos pontos da Fig. 13 (pacotes sincronizados), com o ajuste
        isotonico declarado e faixa acima de 9 fontes, como a assincrona."""
        self.assertEqual(cm.DIFF_SYNC_K_BOUND.raw_points,
                         cm.DIFF_SYNC_K.raw_points)
        ys = [y for _, y in cm.DIFF_SYNC_K_BOUND.points]
        self.assertEqual(ys, sorted(ys, reverse=True))
        v, status, lo, hi = cm.DIFF_SYNC_K_BOUND.value(20, band=True)
        self.assertEqual(status, "extrapolacao")
        self.assertLessEqual(lo, hi)
        model = cm.EmpiricalUwbCapture(intra_acquisition="sync_table")
        _, flags = model.success_probability(ctx(async_senders=20,
                                                 max_async_dt_ns=500.0))
        self.assertIn("async_k_extrapolacao", flags)

    def test_legacy_model_is_the_old_weighted_sinr(self):
        model = cm.LegacyStepCapture(async_weight=0.06)
        p, flags = model.success_probability(ctx(
            signal_mw=1e-7, noise_mw=0.0, async_interf_mw=1e-7))
        sinr = -10 * math.log10(0.06)
        self.assertAlmostEqual(p, sim.decode_success_prob(sinr))
        self.assertEqual(flags, ())


class EngineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }

    def tearDown(self):
        sim.apply_config(self.original_config)

    def test_legacy_preset_keeps_the_legacy_capture(self):
        self.assertEqual(sim.PRESET_LEGACY_V2["CAPTURE_MODEL"],
                         "legacy_async_weight")
        with self.assertRaisesRegex(ValueError, "CAPTURE_MODEL"):
            sim.apply_config({"CAPTURE_MODEL": "magico"})
        with self.assertRaisesRegex(ValueError, "CAPTURE_EXTRAPOLATION"):
            sim.apply_config({"CAPTURE_EXTRAPOLATION": "magico"})
        with self.assertRaisesRegex(ValueError, "CAPTURE_INTRA_ACQUISITION"):
            sim.apply_config({"CAPTURE_INTRA_ACQUISITION": "magico"})

    def test_intra_acquisition_mode_reaches_the_receiver(self):
        """No 1 travado; nos 2 e 3 com quadros diferentes, 10 dB mais
        fracos (decodificaveis sozinhos), 500 ns depois: assincronos dentro
        da aquisicao, k = 3. Sorteio 0,9: passa com a curva de ate 20 us
        (0,989) e sem penalidade; cai com a curva sincronizada (0,850)."""
        from types import SimpleNamespace
        import numpy as np
        power = np.zeros((4, 4))
        for v, p in ((1, 1e-7), (2, 1e-8), (3, 1e-8)):
            power[0][v] = power[v][0] = p
        topo = SimpleNamespace(ps_lin=power, tprop_ns=np.zeros((4, 4)),
                               audible=[{1, 2, 3}, {0}, {0}, {0}])
        kw = dict(fading={(v, 0): 1.0 for v in (1, 2, 3)}, draw=0.9)
        args = (0, [1, 2, 3], {1: 7, 2: 8, 3: 9}, topo,
                [0.0, 0.0, 500.0, 500.0])
        self.assertEqual(sim.CAPTURE_INTRA_ACQUISITION, "async_table")
        outcome = {}
        for mode in cm.INTRA_ACQUISITION_MODES:
            sim.apply_config({"CAPTURE_MODEL": "empirical_uwb",
                              "CAPTURE_INTRA_ACQUISITION": mode})
            self.assertEqual(sim.CAPTURE.intra_acquisition, mode)
            self.assertIs(sim.CAPTURE.per, sim.decode_success_prob)
            outcome[mode] = sim.decode_at_receiver(*args, **kw) is not None
        self.assertEqual(outcome, {"async_table": True, "sync_table": False,
                                   "no_penalty": True})
        src = sim.physical_model_sources()["capture"]
        self.assertEqual(src["intra_acquisition"], "no_penalty")
        self.assertIn("CAPTURE_INTRA_ACQUISITION", sim.active_hypotheses())

    def test_empirical_model_goes_through_decode_at_receiver(self):
        """Quadros longos identicos, mesma potencia: o empirico derruba o
        que o legado 'nondestructive' deixava passar."""
        from types import SimpleNamespace
        import numpy as np
        power = np.zeros((3, 3))
        power[0][1] = power[1][0] = power[0][2] = power[2][0] = 1e-7
        topo = SimpleNamespace(ps_lin=power, tprop_ns=np.zeros((3, 3)),
                               audible=[{1, 2}, {0}, {0}])
        kw = dict(fading={(1, 0): 1.0, (2, 0): 1.0}, draw=0.5)
        frames = {1: 7, 2: 7}
        sim.apply_config({"CAPTURE_MODEL": "legacy_async_weight",
                          "PER_MODEL": "logistic"})
        self.assertIsNotNone(sim.decode_at_receiver(
            0, [1, 2], frames, topo, [0.0] * 3, **kw))
        sim.apply_config({"CAPTURE_MODEL": "empirical_uwb"})
        self.assertIsNone(sim.decode_at_receiver(
            0, [1, 2], frames, topo, [0.0] * 3, **kw))
        stats = sim.new_phy_stats()
        sim.decode_at_receiver(0, [1, 2], frames, topo, [0.0] * 3,
                               phy_stats=stats, **kw)
        self.assertEqual(stats["capture_evaluations"], 1)


if __name__ == "__main__":
    unittest.main()
