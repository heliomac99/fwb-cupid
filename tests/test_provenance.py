"""Fases 9 e 10: proveniencia, retomada segura e suites.

Um cenario so e pulado se o manifesto que ja esta na pasta for compativel
com o que se pede agora (versao do modelo, commit, configuracao efetiva,
cenario, suite e experimentos). Saida velha ou incompativel e erro, salvo
--force.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cupid_sim_v2 as sim
import run_cenarios as rc

ROOT = Path(__file__).resolve().parents[1]
TINY = {"N_NODES": 8, "AREA_X": 30.0, "AREA_Y": 8.0, "EPOCHS": 2,
        "N_TOPOLOGIES": 1, "MAX_SLOTS": 200, "GROUPING_ITERATIONS": 1}


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }

    def tearDown(self):
        sim.apply_config(self.original_config)

    def test_manifest_has_the_required_provenance(self):
        m = sim.build_run_manifest(
            scenario="x", preset="cloves", requested={"EPOCHS": 3},
            experiments=["cdf"], suite="core", scenario_file_hash="abc")
        for key in ("git_commit", "git_dirty", "model_version", "preset",
                    "requested_config", "effective_config",
                    "effective_config_hash", "scenario_file_hash",
                    "python_version", "package_versions", "seeds",
                    "n_topologies", "epochs", "experiments", "suite",
                    "timestamp", "physical_model_sources",
                    "active_hypotheses", "extrapolation", "scenario",
                    "hypothetical_hardware"):
            self.assertIn(key, m)
        self.assertEqual(set(m["package_versions"]),
                         {"numpy", "matplotlib"})
        self.assertEqual(m["effective_config_hash"],
                         sim.config_hash(m["effective_config"]))
        self.assertIn("capture", m["physical_model_sources"])
        self.assertIn("radio_profile", m["physical_model_sources"])
        self.assertTrue(m["active_hypotheses"])

    def test_presets_are_named_and_the_default_is_cloves(self):
        self.assertEqual(set(sim.PRESETS), {"cloves", "legacy_v2", "etapa0",
                                            "2026_09_10"})
        self.assertEqual(sim.PRESETS["cloves"], {})
        self.assertEqual(sim.PRESETS["legacy_v2"], sim.PRESET_LEGACY_V2)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "cen"
        self.out.mkdir()
        self.cen = {"nome": "cen", "suite": "core", "parametros": {"SEED": 7}}
        self.request = rc.scenario_request(self.cen, preset="cloves",
                                           suite="core", experiments=["cdf"],
                                           overrides={})

    def tearDown(self):
        sim.apply_config(self.original_config)
        self.tmp.cleanup()

    def write(self, manifest, summary=True):
        if summary:
            (self.out / "resumo_resultados.txt").write_text("ok")
        if manifest is not None:
            (self.out / "run_manifest.json").write_text(json.dumps(manifest))

    def compatible_manifest(self):
        """Manifesto de uma execucao do mesmo pedido que TERMINOU."""
        return dict(rc.expected_manifest_fields(self.request),
                    status="completo")

    def test_old_summary_without_manifest_is_not_valid(self):
        self.write(None)
        status, reason = rc.check_resume(self.out, self.request)
        self.assertEqual(status, "incompativel")
        self.assertIn("manifesto", reason)

    def test_compatible_scenario_is_skipped(self):
        self.write(self.compatible_manifest())
        self.assertEqual(rc.check_resume(self.out, self.request)[0], "pular")

    def test_complete_manifest_without_the_marker_runs_again(self):
        self.write(self.compatible_manifest(), summary=False)
        self.assertEqual(rc.check_resume(self.out, self.request)[0], "rodar")

    def test_incomplete_manifest_of_the_same_request_runs_again(self):
        self.write(dict(self.compatible_manifest(), status="incompleto"))
        self.assertEqual(rc.check_resume(self.out, self.request)[0], "rodar")

    def test_incomplete_manifest_of_another_request_is_incompatible(self):
        """A compatibilidade vem antes do status: uma rodada interrompida de
        OUTRA configuracao nunca e sobrescrita sem --force."""
        other = rc.scenario_request(
            dict(self.cen, parametros={"SEED": 8}), preset="cloves",
            suite="core", experiments=["cdf"], overrides={})
        self.write(dict(rc.expected_manifest_fields(other),
                        status="incompleto"), summary=False)
        self.assertEqual(rc.check_resume(self.out, self.request)[0],
                         "incompativel")

    def test_empty_directory_runs(self):
        self.assertEqual(rc.check_resume(self.out, self.request)[0],
                         "rodar")

    def test_parameter_change_invalidates(self):
        self.write(self.compatible_manifest())
        other = rc.scenario_request(
            dict(self.cen, parametros={"SEED": 8}), preset="cloves",
            suite="core", experiments=["cdf"], overrides={})
        status, reason = rc.check_resume(self.out, other)
        self.assertEqual(status, "incompativel")
        self.assertIn("configuracao efetiva", reason)

    def test_preset_change_invalidates(self):
        self.write(self.compatible_manifest())
        other = rc.scenario_request(self.cen, preset="legacy_v2",
                                    suite="core", experiments=["cdf"],
                                    overrides={})
        self.assertEqual(rc.check_resume(self.out, other)[0], "incompativel")

    def test_commit_or_model_version_change_invalidates(self):
        for field, value in (("git_commit", "0" * 40),
                             ("model_version", "outro"),
                             ("git_dirty", not self.request["git_dirty"])):
            with self.subTest(field=field):
                m = self.compatible_manifest()
                m[field] = value
                self.write(m)
                self.assertEqual(rc.check_resume(self.out, self.request)[0],
                                 "incompativel")

    def test_suite_and_experiments_are_part_of_the_request(self):
        self.write(self.compatible_manifest())
        other = rc.scenario_request(self.cen, preset="cloves", suite="core",
                                    experiments=["cdf", "pdr"], overrides={})
        self.assertEqual(rc.check_resume(self.out, other)[0], "incompativel")

    def test_interrupted_run_leaves_an_incomplete_manifest_and_is_redone(self):
        """O manifesto e gravado ANTES dos experimentos (status incompleto)
        e reescrito no fim; um cenario interrompido com o mesmo pedido volta
        a rodar sem --force."""
        sim.apply_config(dict(TINY, SEED=7))
        with patch.object(sim, "experiment_latency_cdf",
                          side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                sim.main(out_dir=str(self.out), scenario_name="cen",
                         requested={"SEED": 7}, experiments=["cdf"],
                         preset="cloves", suite="core")
        m = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(m["status"], "incompleto")
        self.assertFalse((self.out / "resumo_resultados.txt").exists())
        req = rc.scenario_request(dict(self.cen, parametros=dict(
            TINY, SEED=7)), preset="cloves", suite="core",
            experiments=["cdf"], overrides={})
        self.assertEqual(rc.check_resume(self.out, req)[0], "rodar")

    def test_incompatible_output_fails_unless_forced(self):
        self.write(None)
        with self.assertRaisesRegex(SystemExit, "run-id"):
            rc.plan_scenarios([self.cen], Path(self.tmp.name), "cloves",
                              "core", ["cdf"], {}, force=False)
        todo, _ = rc.plan_scenarios([self.cen], Path(self.tmp.name),
                                    "cloves", "core", ["cdf"], {},
                                    force=True)
        self.assertEqual([c["nome"] for c, _ in todo], ["cen"])


class RunnerTests(unittest.TestCase):
    def test_failed_scenario_gives_a_nonzero_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps({"preset": "cloves", "cenarios": [
                {"nome": "a", "suite": "core", "parametros": TINY}]}))
            fail = ("a", "FALHOU", 0.0, "erro", {})
            with patch.object(rc, "rodar_cenario", return_value=fail):
                code = rc.main([str(path), "--run-id", "t", "--jobs", "1",
                                "--outputs", tmp])
        self.assertNotEqual(code, 0)

    def test_two_radios_need_a_declared_hypothetical_hardware(self):
        cen = {"nome": "f", "suite": "fwb",
               "parametros": {"N_RADIOS_PER_NODE": 2}}
        with self.assertRaisesRegex(SystemExit, "hardware_hipotetico"):
            rc.validar_hardware([cen])
        rc.validar_hardware([dict(cen, hardware_hipotetico=True)])
        with tempfile.TemporaryDirectory() as out_dir:
            base = [sys.executable, str(ROOT / "cupid_sim_v2.py"), "--out",
                    out_dir, "--experiments", "cdf", "--N_RADIOS_PER_NODE",
                    "2"] + [x for k, v in TINY.items()
                            for x in (f"--{k}", str(v))]
            refused = subprocess.run(base, capture_output=True, text=True)
            self.assertEqual(refused.returncode, 2, refused.stderr)
            self.assertIn("--hypothetical-hardware", refused.stderr)

    def test_epochs_and_topologies_overrides_reach_the_command(self):
        cmd = rc.construir_comando("x", {"EPOCHS": 1000}, "o/x",
                                   preset="cloves", experiments=["cdf"],
                                   overrides={"EPOCHS": 50,
                                              "N_TOPOLOGIES": 2})
        self.assertEqual(cmd[cmd.index("--EPOCHS") + 1], "50")
        self.assertEqual(cmd[cmd.index("--N_TOPOLOGIES") + 1], "2")
        self.assertEqual(cmd[cmd.index("--preset") + 1], "cloves")
        self.assertEqual(cmd[cmd.index("--experiments") + 1], "cdf")


class SuiteFileTests(unittest.TestCase):
    def load(self, name):
        with (ROOT / name).open(encoding="utf-8") as f:
            return json.load(f)

    def test_core_suite_compares_only_cupid_weaver_crystal(self):
        data = self.load("cenarios_cloves_core.json")
        self.assertEqual(data["preset"], "cloves")
        self.assertEqual(data["protocolos"], ["cupid", "weaver", "crystal"])
        names = [c["nome"] for c in data["cenarios"]]
        self.assertEqual(len(names), len(set(names)))
        for c in data["cenarios"]:
            self.assertEqual(c["suite"], "core")
            self.assertLessEqual(c["parametros"].get("N_RADIOS_PER_NODE", 1),
                                 1)
            rc.construir_comando(c["nome"], c["parametros"], "o")
        themes = {c["tema"] for c in data["cenarios"]}
        self.assertLessEqual({"base", "sementes", "densidade", "profundidade",
                              "canal", "carga", "tamanho", "captura",
                              "shadowing", "estresse"}, themes)

    def test_fwb_suite_declares_hardware_and_never_labels_16_as_dw1000(self):
        data = self.load("cenarios_cloves_fwb.json")
        self.assertEqual(data["protocolos"], ["cupid", "cupid_fwb"])
        for c in data["cenarios"]:
            p = c["parametros"]
            self.assertEqual(c["suite"], "fwb")
            radios = p.get("N_RADIOS_PER_NODE", 1)
            self.assertEqual(c.get("hardware_hipotetico", False), radios > 1)
            if p.get("N_CHANNELS", 1) > 4 or radios > 2:
                self.assertIn("ideal", c["nome"])
                self.assertNotEqual(p.get("CHANNEL_PLAN"), "dw1000")
        rc.validar_hardware(data["cenarios"])
        one = [c for c in data["cenarios"]
               if c["parametros"].get("N_RADIOS_PER_NODE", 1) == 1]
        self.assertTrue(one)


class FwbEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }

    def tearDown(self):
        sim.apply_config(self.original_config)

    def test_fwb_with_one_radio_equals_cupid_in_the_experiments(self):
        sim.apply_config(dict(TINY, N_NODES=12, AREA_X=40.0,
                              N_RADIOS_PER_NODE=1))
        topo, seed = sim.make_topologies(n_topo=1, base_seed=42)[0]
        a = sim.simulate_topology("cupid", topo, seed, O=11, n_epochs=3)
        b = sim.simulate_topology("cupid_fwb", topo, seed, O=11, n_epochs=3)
        self.assertEqual(a["records"], b["records"])
        self.assertEqual(a["grouping_setup_ms"], b["grouping_setup_ms"])


if __name__ == "__main__":
    unittest.main()
