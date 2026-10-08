"""[F20] Parametro pedido e truncado vira erro (auditoria Item 5).

O cenario "fwb_funil_16canais" pedia N_CHANNELS=16 no plano 'dw1000' (K=4):
enforce_channel_plan() truncava para 4 com um aviso no log e o cenario rodava
o baseline -- tres cenarios sairam identicos.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cupid_sim_v2 as sim
import run_cenarios

ROOT = Path(__file__).resolve().parents[1]


class ChannelClampTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }

    def tearDown(self):
        sim.apply_config(self.original_config)

    def test_requested_channels_beyond_the_plan_are_reported(self):
        clamps = sim.requested_clamps({"N_CHANNELS": 16})
        self.assertEqual(len(clamps), 1)
        self.assertEqual((clamps[0]["pedido"], clamps[0]["efetivo"]), (16, 4))
        self.assertEqual(sim.requested_clamps(
            {"N_CHANNELS": 16, "CHANNEL_PLAN": "ieee_nominal"}), [])
        # o padrao ajustado ao plano nao foi pedido: nao e erro
        self.assertEqual(sim.requested_clamps({"CHANNEL_PLAN": "dw3000"}), [])

    def cli(self, *extra):
        with tempfile.TemporaryDirectory() as out_dir:
            command = [
                sys.executable, str(Path(sim.__file__)),
                "--experiments", "cdf", "--protocols", "cupid",
                "--EPOCHS", "1", "--N_TOPOLOGIES", "1",
                "--out", out_dir, "--N_NODES", "2", "--AREA_X", "8",
                "--AREA_Y", "1", "--N_CHANNELS", "16", *extra,
            ]
            return subprocess.run(command, capture_output=True, text=True)

    def test_cli_refuses_a_clamped_request_unless_allowed(self):
        refused = self.cli()
        self.assertEqual(refused.returncode, 2)
        self.assertIn("--allow-clamp", refused.stderr)
        allowed = self.cli("--allow-clamp")
        self.assertEqual(allowed.returncode, 0, allowed.stderr)

    def test_runner_fails_fast_unless_the_scenario_allows_clamping(self):
        clamped = {"nome": "fwb_velho", "parametros": {"N_CHANNELS": 16}}
        with self.assertRaisesRegex(SystemExit, "fwb_velho"):
            run_cenarios.validar_truncamentos([clamped])
        run_cenarios.validar_truncamentos([{**clamped, "allow_clamp": True}])

    def test_runner_forwards_allow_clamp_only_when_declared(self):
        plain = run_cenarios.construir_comando("x", {}, "outputs/x")
        forced = run_cenarios.construir_comando("x", {}, "outputs/x",
                                                allow_clamp=True)
        self.assertNotIn("--allow-clamp", plain)
        self.assertIn("--allow-clamp", forced)

    def test_bundled_scenarios_request_nothing_that_would_be_clamped(self):
        with (ROOT / "cenarios.json").open(encoding="utf-8") as f:
            scenarios = json.load(f)["cenarios"]
        for scenario in scenarios:
            with self.subTest(scenario=scenario["nome"]):
                self.assertFalse(scenario.get("allow_clamp", False))
                self.assertEqual(sim.requested_clamps(
                    scenario.get("parametros", {})), [])

    def test_fwb_scenarios_declare_their_hardware(self):
        """Cenario FWB sem plano/radios explicitos e o erro do Item 5."""
        with (ROOT / "cenarios.json").open(encoding="utf-8") as f:
            scenarios = json.load(f)["cenarios"]
        fwb = [s for s in scenarios if s["nome"].startswith("fwb")]
        self.assertTrue(fwb)
        for scenario in fwb:
            with self.subTest(scenario=scenario["nome"]):
                self.assertIn("CHANNEL_PLAN", scenario["parametros"])
                self.assertIn("N_RADIOS_PER_NODE", scenario["parametros"])

    def test_config_hash_follows_the_effective_configuration(self):
        base = sim.effective_config()
        self.assertEqual(sim.config_hash(base), sim.config_hash(dict(base)))
        sim.apply_config({"N_CHANNELS": 2})
        self.assertNotEqual(sim.config_hash(sim.effective_config()),
                            sim.config_hash(base))


if __name__ == "__main__":
    unittest.main()
