import json
import unittest
from pathlib import Path

import run_cenarios


ROOT = Path(__file__).resolve().parents[1]


class V2EntrypointTests(unittest.TestCase):
    def test_scenario_command_invokes_v2_and_accepts_v2_parameters(self):
        """Catches invoking the removed v1 or rejecting a v2 radio parameter."""
        command = run_cenarios.construir_comando(
            "v2", {"TX_POWER_DBM": -14.3, "CHANNEL_SWITCH_US": 1000.0},
            "outputs/v2")

        self.assertEqual(command[2], "cupid_sim_v2.py")
        self.assertIn("--TX_POWER_DBM", command)
        self.assertIn("--CHANNEL_SWITCH_US", command)

    def test_bundled_scenarios_use_parameters_accepted_by_the_runner(self):
        """Catches shipping a scenario file that the active runner cannot execute."""
        with (ROOT / "cenarios.json").open(encoding="utf-8") as f:
            scenarios = json.load(f)["cenarios"]

        for scenario in scenarios:
            with self.subTest(scenario=scenario["nome"]):
                run_cenarios.construir_comando(
                    scenario["nome"], scenario.get("parametros", {}),
                    f"outputs/{scenario['nome']}",
                )

if __name__ == "__main__":
    unittest.main()
