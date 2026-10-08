"""Fase 1: automacao e documentacao coerentes com o codigo que roda.

Cada teste pega uma inconsistencia encontrada na auditoria de 26/09: o
cenarios.json anunciava ACK_MODEL 'physical' como padrao; a
bancada de captura chamava 40 ns de padrao (o padrao era 8141 ns); a grade de
calibracao nao continha o peso escolhido; e U = 35 era apresentado como a
medicao da EWSN 2020, que foi feita com U = 21.
"""

import json
import unittest
from pathlib import Path

import calibracao_captura
import cupid_sim_v2 as sim
import sensibilidade

ROOT = Path(__file__).resolve().parents[1]


class AutomationDocsTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }

    def tearDown(self):
        sim.apply_config(self.original_config)

    def test_scenario_help_matches_effective_default(self):
        """A ajuda do cenarios.json diz qual e o ACK_MODEL padrao de fato."""
        with (ROOT / "cenarios.json").open(encoding="utf-8") as f:
            help_lines = json.load(f)["_ajuda"]
        ack = [line for line in help_lines if "ACK_MODEL" in line]
        self.assertEqual(len(ack), 1)
        self.assertIn(f"'{sim.ACK_MODEL}' (padrao)", ack[0])
        for model in sim.ACK_MODELS:
            self.assertIn(f"'{model}'", ack[0])

    def test_fwb_scenarios_flag_hypothetical_hardware(self):
        """Mais de um radio por no nao existe no DW1000: o cenario o declara."""
        with (ROOT / "cenarios.json").open(encoding="utf-8") as f:
            scenarios = json.load(f)["cenarios"]
        for scenario in scenarios:
            radios = scenario.get("parametros", {}).get("N_RADIOS_PER_NODE", 1)
            with self.subTest(scenario=scenario["nome"]):
                self.assertEqual(scenario.get("hardware_hipotetico", False),
                                 radios > 1)

    def test_capture_sensitivity_contains_active_default(self):
        """O item marcado '(padrao)' e o padrao em vigor, sem sobrescrita."""
        for conjunto in ("captura", "limiar"):
            with self.subTest(conjunto=conjunto):
                itens = sensibilidade.conjuntos()[conjunto]
                padrao = [(nome, cfg) for nome, cfg in itens
                          if "(padrao)" in nome]
                self.assertEqual(len(padrao), 1)
                self.assertEqual(padrao[0][1], {})
                others = [cfg for nome, cfg in itens if "(padrao)" not in nome]
                self.assertTrue(others)
                for cfg in others:
                    self.assertTrue(cfg)
                    for key, value in cfg.items():
                        self.assertNotEqual(value, getattr(sim, key), key)
        nome = [n for n, _ in sensibilidade.conjuntos()["captura"]
                if "(padrao)" in n][0]
        self.assertIn(f"{sim.CAPTURE_TIME_NS:g}", nome)
        # temporal e limiar nao se misturam
        temporal = {k for _, cfg in sensibilidade.conjuntos()["captura"]
                    for k in cfg}
        limiar = {k for _, cfg in sensibilidade.conjuntos()["limiar"]
                  for k in cfg}
        self.assertFalse(temporal & limiar)
        self.assertLessEqual(temporal, {"CAPTURE_TIME_NS",
                                        "PREAMBLE_ALIGN_NS"})

    def test_calibration_grid_contains_selected_weight(self):
        """A grade padrao contem o peso que o preset legado usa."""
        grid = calibracao_captura.default_weights()
        self.assertIn(sim.PRESET_LEGACY_V2["ASYNC_INTERF_WEIGHT"], grid)
        self.assertEqual(len(grid), len(set(grid)))

    def test_calibration_marks_extrapolated_source_count(self):
        """U medido na fonte e calibracao; qualquer outro U e extrapolacao.

        EWSN 2020, sec. 5.2: U em {1, 2, 5, 7, 10} (Fig. 10 mostra 2-10) e
        o caso extremo com os 21 nao-sink de uma rede de 22 nos. U = 21 so e
        comparavel numa rede do mesmo tamanho: roda com N_NODES = 22."""
        measured = calibracao_captura.EWSN_MEASURED_U
        self.assertEqual(measured, {2, 5, 7, 10, 21})
        self.assertNotIn(35, measured)
        self.assertEqual(calibracao_captura.EWSN_T_SUCCESS[21][2], 22)
        for u in calibracao_captura.US:
            with self.subTest(U=u):
                label = calibracao_captura.u_status(u)
                if u in measured:
                    self.assertEqual(label, "calibracao")
                else:
                    self.assertEqual(label, "extrapolacao")
        md = calibracao_captura.render_tables(
            t_rows=[{"configuracao": "x", "U": u, "sucesso_nos": 1.0,
                     "sucesso_sink": 1.0} for u in calibracao_captura.US],
            c_rows=[{"configuracao": "y", "k": k, "prr": 1.0}
                    for k in calibracao_captura.KS],
            n_topo=1, tentativas=1, weight=0.06)
        calib, _, valid = md.partition("## Validacao")
        self.assertTrue(valid)
        self.assertIn("U=35 (extrapolacao)", calib)
        self.assertIn("U=21 (medido, rede de 22)", calib)
        self.assertIn("U=7 (medido)", calib)
        self.assertNotIn("U = 35)", calib)
        self.assertNotIn("extrapolacao", valid)


if __name__ == "__main__":
    unittest.main()
