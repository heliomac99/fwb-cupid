"""Fases 7 e 9 de ponta a ponta: o runner em paralelo produz os mesmos bytes
que em sequencia, e uma segunda rodada compativel pula os cenarios."""

import json
import tempfile
import unittest
from pathlib import Path

import run_cenarios as rc

MINI = {"N_NODES": 10, "AREA_X": 35.0, "AREA_Y": 10.0, "EPOCHS": 2,
        "N_TOPOLOGIES": 2, "GROUPING_ITERATIONS": 1, "MAX_SLOTS": 400}
DETERMINISTIC = ("resumo_resultados.txt", "amostras_epocas.csv",
                 "config_efetiva.json")


class RunnerIntegrationTests(unittest.TestCase):
    def test_parallel_equals_sequential_and_resume_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp) / "mini.json"
            suite.write_text(json.dumps({
                "preset": "cloves", "suite": "core",
                "protocolos": ["cupid", "weaver", "crystal"],
                "cenarios": [
                    {"nome": "a", "suite": "core", "parametros": MINI},
                    {"nome": "b", "suite": "core",
                     "parametros": dict(MINI, SEED=7)}]}))
            common = [str(suite), "--experiments", "cdf", "--outputs", tmp]
            self.assertEqual(rc.main(common + ["--run-id", "seq",
                                               "--jobs", "1"]), 0)
            self.assertEqual(rc.main(common + ["--run-id", "par",
                                               "--jobs", "2"]), 0)
            for cen in ("a", "b"):
                for name in DETERMINISTIC:
                    with self.subTest(cenario=cen, arquivo=name):
                        seq = (Path(tmp) / "seq" / cen / name).read_bytes()
                        par = (Path(tmp) / "par" / cen / name).read_bytes()
                        self.assertEqual(seq, par)
                ms = json.loads((Path(tmp) / "seq" / cen /
                                 "run_manifest.json").read_text())
                mp = json.loads((Path(tmp) / "par" / cen /
                                 "run_manifest.json").read_text())
                for m in (ms, mp):
                    m.pop("timestamp")
                self.assertEqual(ms, mp)
            # segunda rodada compativel: nada roda de novo
            self.assertEqual(rc.main(common + ["--run-id", "seq",
                                               "--jobs", "2"]), 0)
            status = json.loads((Path(tmp) / "seq" /
                                 "run_manifest.json").read_text())
            self.assertEqual(status["cenarios"], {"a": "PULADO",
                                                  "b": "PULADO"})
            # outro --epochs muda a configuracao efetiva: recusa sem --force
            with self.assertRaisesRegex(SystemExit, "run-id"):
                rc.main(common + ["--run-id", "seq", "--epochs", "3"])


if __name__ == "__main__":
    unittest.main()
