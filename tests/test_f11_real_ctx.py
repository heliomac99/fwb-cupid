import math
import random
import unittest
from types import SimpleNamespace

import numpy as np

import cupid_sim_v2 as sim


class F11RealCtxTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }

    def tearDown(self):
        sim.apply_config(self.original_config)

    @staticmethod
    def controlled_phy_topology():
        return SimpleNamespace(
            ps_lin=np.array([
                [0.0, 1e-7, 1e-7, 1e-7],
                [1e-7, 0.0, 1e-7, 1e-7],
                [1e-7, 1e-7, 0.0, 1e-7],
                [1e-7, 1e-7, 1e-7, 0.0],
            ]),
            tprop_ns=np.array([
                [0.0, 0.0, 3.0, 10.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]),
            audible=[{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}],
        )

    def test_phy_stats_condition_comb_fraction_on_multi_groups(self):
        """Catches counting singleton signals as constructive combining."""
        sim.apply_config({
            "COMBINE_WINDOW_NS": 4.0,
            "COMBINE_EFF": 1.0,
            "REQ_SNR_DB": -100.0,
            "PER_MODEL": "threshold",
            "SAME_PACKET_MODEL": "additive",
            "CAPTURE_MODEL": "legacy_async_weight",  # [Fase 6] fisica legada de copias/peso; o empirico tem testes proprios
        })
        topo = self.controlled_phy_topology()
        self.assertTrue(
            hasattr(sim, "new_phy_stats"),
            "F11 telemetry API new_phy_stats is missing",
        )
        stats = sim.new_phy_stats()

        sim.decode_at_receiver(
            0, [1], {1: 7}, topo, [0.0] * 4,
            fading={(1, 0): 1.0}, phy_stats=stats)
        sim.decode_at_receiver(
            0, [1, 2, 3], {1: 8, 2: 8, 3: 8}, topo, [0.0] * 4,
            fading={(1, 0): 1.0, (2, 0): 1.0, (3, 0): 1.0},
            phy_stats=stats)

        summary = sim.summarize_phy_stats(stats)
        self.assertIn(
            "n_groups_singleton", summary,
            "singleton and multi groups need separate denominators",
        )
        self.assertEqual(summary["n_groups_singleton"], 1)
        self.assertEqual(summary["n_groups_multi"], 1)
        self.assertEqual(summary["comb_signals"], 2)
        self.assertEqual(summary["resid_signals"], 1)
        self.assertAlmostEqual(summary["comb_power"], 2e-7)
        self.assertAlmostEqual(summary["resid_power"], 1e-7)
        self.assertAlmostEqual(summary["comb_fraction"], 2 / 3)
        self.assertAlmostEqual(summary["resid_fraction"], 1 / 3)
        self.assertFalse(math.isnan(summary["comb_fraction"]))

    @staticmethod
    def fanout_topology():
        """Sink em 0, dois relays de hop 1 e um originador de hop 2.

        Todos se ouvem com folga e sem atraso: o unico fator que decide
        quem retem o pacote e a logica de detentores, nao a fisica.
        """
        n = 4
        power = np.full((n, n), 1e-6)
        np.fill_diagonal(power, 0.0)
        return SimpleNamespace(
            n=n,
            ps_lin=power,
            tprop_ns=np.zeros((n, n)),
            audible=[set(range(n)) - {v} for v in range(n)],
            reliable=[set(range(n)) - {v} for v in range(n)],
            hop={0: 0, 1: 1, 2: 1, 3: 2},
            group={v: "emitter" for v in range(n)},
        )

    @staticmethod
    def deterministic_config():
        """Remove toda aleatoriedade fisica: fading, jitter e sorteio de TX.

        [F21][F22] Tambem fixa a fisica idealizada em que este teste foi
        escrito (limiar duro, copias identicas que somam): aqui o que se testa
        e a logica de detentores, nao a fisica das copias concorrentes.
        """
        return {
            "FAST_FADE_DB": 0.0,
            "CLOCK_JITTER_NS": 0.0,
            "TX_PROB": 1.0,
            "RANDOM_QUEUE_HEAD": False,
            "PER_MODEL": "threshold",
            "SAME_PACKET_MODEL": "additive",
            "CAPTURE_MODEL": "legacy_async_weight",  # [Fase 6] fisica legada de copias/peso; o empirico tem testes proprios
            "BOOTSTRAP_DELAY": False, "BOOTSTRAP_MODEL": "bfs",
            # [Fase 5] slots exatos escritos para a grade legada do Cupid; a
            # da Fig. 5b tem teste proprio (test_protocol_timing)
            "CUPID_SCHEDULE": "legacy",
        }

    @staticmethod
    def loaded_scenario():
        """Cenario carregado e semeado: varios pacotes disputam os mesmos slots."""
        sim.apply_config({
            "SEED": 42,
            "N_NODES": 24,
            "AREA_X": 60.0,
            "AREA_Y": 20.0,
            "MAX_SLOTS": 205,
        })
        topo = sim.build_topology(random.Random(42))
        return topo, sim.valid_nodes(topo)

    def test_holder_telemetry_is_observation_only(self):
        """Ligar a telemetria nao pode mexer em entrega, PDR nem no rng."""
        topo, originators = self.loaded_scenario()
        self.assertTrue(
            hasattr(sim, "new_holder_stats"),
            "instrumentacao de detentores exigida por F11 esta ausente",
        )
        for protocol in ("weaver", "cupid", "cupid_fwb"):
            with self.subTest(protocol=protocol):
                plain_rng = random.Random(2026)
                measured_rng = random.Random(2026)
                plain_slots, measured_slots = {}, {}
                holder_stats = sim.new_holder_stats()
                plain = sim.run_epoch(protocol, originators, topo, plain_rng,
                                      delivery_slots=plain_slots)
                measured = sim.run_epoch(protocol, originators, topo,
                                         measured_rng,
                                         delivery_slots=measured_slots,
                                         holder_stats=holder_stats)
                self.assertEqual(measured, plain)
                self.assertEqual(measured_slots, plain_slots)
                self.assertEqual(measured_rng.random(), plain_rng.random())
                summary = sim.summarize_holder_stats(holder_stats)
                self.assertGreater(summary["n_samples"], 0)
                self.assertGreaterEqual(summary["max_holders"], 1)
                self.assertIn("overhearing_suppressions", summary)
                self.assertIn("post_delivery_suppressions", summary)
                self.assertEqual(
                    len(summary["samples"]), summary["n_samples"],
                    "a serie por slot precisa ficar disponivel para o relatorio")

    def test_all_eligible_receivers_become_holders(self):
        """Parte 1: sem handoff onisciente, todo receptor elegivel retem.

        No slot 0 o originador de hop 2 transmite e os dois relays de hop 1
        decodificam. O modelo correto tem TRES copias vivas no slot seguinte
        (originador + dois relays); o handoff deixava uma so.
        """
        cfg = self.deterministic_config()
        sim.apply_config(cfg)
        topo = self.fanout_topology()
        holder_stats = sim.new_holder_stats()
        delivery_slots = {}
        slots, delivered = sim.run_epoch(
            "cupid", [3], topo, random.Random(7), slot_budget=8,
            delivery_slots=delivery_slots, holder_stats=holder_stats)

        summary = sim.summarize_holder_stats(holder_stats)
        self.assertEqual(
            summary["max_holders"], 3,
            "o originador e os dois relays precisam reter a mesma copia")
        self.assertGreaterEqual(
            summary["overhearing_suppressions"], 1,
            "o originador deve largar a copia ao decodificar o proprio "
            "pacote vindo do hop de baixo, nao por decreto")
        self.assertEqual(delivered, 1)
        self.assertEqual(delivery_slots, {3: 2})
        self.assertEqual(slots, 2)

    def test_delivered_packet_can_still_trigger_physical_overhearing(self):
        """Parte 2: entrega e contabilidade, nao mordaca fisica.

        Depois que o sink recebe um relatorio, os detentores que ainda tem a
        copia continuam ouvindo o meio. Ouvir o proprio pacote vindo de um hop
        menor tem de continuar suprimindo: 'delivered' so pode barrar nova
        contabilizacao de PDR, nunca o ACK implicito.
        """
        topo, originators = self.loaded_scenario()
        holder_stats = sim.new_holder_stats()
        rng = random.Random(2026)
        delivered_total = 0
        for _ in range(5):
            delivery_slots = {}
            _, delivered = sim.run_epoch(
                "cupid", originators, topo, rng, slot_budget=205,
                delivery_slots=delivery_slots, holder_stats=holder_stats)
            delivered_total += delivered
            # a contabilidade de PDR nao pode contar duas vezes o mesmo relatorio
            self.assertEqual(delivered, len(delivery_slots))
            self.assertLessEqual(delivered, len(originators))
        summary = sim.summarize_holder_stats(holder_stats)
        self.assertGreater(
            summary["post_delivery_suppressions"], 0,
            "um detentor que decodifica o proprio pacote depois da entrega "
            "precisa parar: o guard de 'delivered' vinha antes do overhearing")
        self.assertGreater(delivered_total, 0)


if __name__ == "__main__":
    unittest.main()
