"""Regime de captura nao medido: concorrentes todos dentro da aquisicao.

Weaver e Cupid atrasam cada TX ate 1 us, e ~9 em 10 decisoes de captura
concorrentes deles caem nesse regime, que nenhuma medicao isola. O
CAPTURE_INTRA_ACQUISITION poe o regime entre o pior e o melhor caso medidos
(e um limite teorico), e a bancada e a suite core mostram se a conclusao
muda -- com comparacao pareada por topologia.
"""

import json
import unittest
from pathlib import Path

import cupid_sim_v2 as sim
import run_cenarios as rc
import sensibilidade

ROOT = Path(__file__).resolve().parents[1]


class SensitivitySetTests(unittest.TestCase):
    def test_bracket_set_spans_the_measured_worst_and_best_cases(self):
        cfgs = dict(sensibilidade.conjuntos()["intra_aquisicao"])
        labels = list(cfgs)
        self.assertEqual(cfgs[labels[0]], {})              # padrao em vigor
        self.assertIn("(padrao)", labels[0])
        modes = {c.get("CAPTURE_INTRA_ACQUISITION", "async_table")
                 for c in cfgs.values()}
        self.assertEqual(modes, set(sim.CAPTURE_INTRA_ACQUISITIONS))
        # pior caso combinado: curva sincronizada + extrapolacao inferior
        self.assertIn({"CAPTURE_INTRA_ACQUISITION": "sync_table",
                       "CAPTURE_EXTRAPOLATION": "lower"}, list(cfgs.values()))
        theory = [lbl for lbl, c in cfgs.items()
                  if c.get("CAPTURE_INTRA_ACQUISITION") == "no_penalty"]
        self.assertTrue(theory and all("sem medicao" in t for t in theory))

    def test_paired_comparison_uses_the_topology_as_the_unit(self):
        def row(proto, rmst):
            return {"configuracao": "x", "protocolo": proto,
                    "rmst_por_topologia": rmst}
        linhas = [row("cupid", {"1": 60.0, "2": 90.0, "3": 100.0}),
                  row("weaver", {"1": 100.0, "2": 100.0, "3": 80.0}),
                  row("crystal", {"1": 200.0, "2": 200.0, "3": 200.0})]
        comps = {(c["a"], c["b"]): c
                 for c in sensibilidade.comparacoes_pareadas(linhas)}
        cw = comps[("cupid", "weaver")]
        self.assertEqual((cw["a_mais_rapido"], cw["n"]), (2, 3))
        self.assertAlmostEqual(cw["razao_media"], (0.6 + 0.9 + 1.25) / 3)
        self.assertEqual((cw["razao_min"], cw["razao_max"]), (0.6, 1.25))
        self.assertEqual(comps[("weaver", "crystal")]["a_mais_rapido"], 3)
        md = sensibilidade.tabela_pareada(list(comps.values()))
        self.assertIn("2/3", md)


class SuiteTests(unittest.TestCase):
    def test_core_suite_runs_the_bracket_with_paired_statistics(self):
        data = json.loads((ROOT / "cenarios_cloves_core.json").read_text(
            encoding="utf-8"))
        by_name = {c["nome"]: c for c in data["cenarios"]}
        worst = by_name["captura_intra_aquisicao_pior"]["parametros"]
        theory = by_name["captura_intra_aquisicao_sem_penalidade"]
        self.assertEqual(worst["CAPTURE_INTRA_ACQUISITION"], "sync_table")
        self.assertEqual(worst["CAPTURE_EXTRAPOLATION"], "lower")
        self.assertEqual(theory["parametros"]["CAPTURE_INTRA_ACQUISITION"],
                         "no_penalty")
        self.assertIn("sem medicao", theory["descricao"])
        for name in ("captura_intra_aquisicao_pior",
                     "captura_intra_aquisicao_sem_penalidade"):
            cmd = rc.construir_comando(name, by_name[name]["parametros"], "o")
            self.assertIn("--CAPTURE_INTRA_ACQUISITION", cmd)


if __name__ == "__main__":
    unittest.main()
