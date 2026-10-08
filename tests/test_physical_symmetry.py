import random
import unittest
from types import SimpleNamespace

import numpy as np

import cupid_sim_v2 as sim


class PhysicalSymmetryTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }

    def tearDown(self):
        sim.apply_config(self.original_config)

    @staticmethod
    def branched_topology():
        """Sink 0 <- nodes 1/2 <- nodes 3/4, all adjacent links reliable."""
        return SimpleNamespace(
            n=5,
            hop={0: 0, 1: 1, 2: 1, 3: 2, 4: 2},
            reliable=[{1, 2}, {0, 3, 4}, {0, 3, 4}, {1, 2}, {1, 2}],
        )

    def test_slow_switching_forces_one_channel_per_connected_component(self):
        """Catches treating receiver count as permission for slotwise hopping."""
        sim.apply_config({
            "N_CHANNELS": 4,
            "N_RADIOS_PER_NODE": 1,
            "CHANNEL_SWITCH_US": 1000.0,
            "SLOT_US": 813.0,
        })

        assignment = sim.build_channel_assignment(
            "cupid_fwb", self.branched_topology(), {1: 2, 2: 2})

        self.assertEqual(set(assignment.values()), {0})

    def test_invalid_assignment_is_rejected_when_switch_exceeds_slot(self):
        """Catches accepting adjacent relays that need a too-fast channel switch."""
        sim.apply_config({
            "N_RADIOS_PER_NODE": 1,
            "CHANNEL_SWITCH_US": 1000.0,
            "SLOT_US": 813.0,
        })
        invalid = {0: 0, 1: 0, 2: 1, 3: 0, 4: 1}

        with self.assertRaisesRegex(ValueError, "CHANNEL_SWITCH_US"):
            sim.validate_channel_assignment(invalid, self.branched_topology())

    def test_two_fixed_radios_allow_a_two_channel_assignment(self):
        """Catches collapsing the network to one channel by receiver-count fiat."""
        sim.apply_config({
            "N_CHANNELS": 4,
            "N_RADIOS_PER_NODE": 2,
            "CHANNEL_SWITCH_US": 1000.0,
            "SLOT_US": 813.0,
        })

        assignment = sim.build_channel_assignment(
            "cupid_fwb", self.branched_topology(), {1: 2, 2: 2})

        self.assertEqual(set(assignment.values()), {0, 1})
        sim.validate_channel_assignment(assignment, self.branched_topology())

    def test_same_channel_partitions_have_identical_physics(self):
        """Catches co-channel transmitters disappearing across tx_set boundaries."""
        sim.apply_config({"PER_MODEL": "threshold"})
        topo = SimpleNamespace(
            ps_lin=np.array([
                [0.0, 1e-7, 1e-7],
                [1e-7, 0.0, 1e-7],
                [1e-7, 1e-7, 0.0],
            ]),
            tprop_ns=np.zeros((3, 3)),
            audible=[{1, 2}, {0, 2}, {0, 1}],
        )
        clock = [0.0, 0.0, 0.0]
        pkt_of = {1: 1, 2: 2}
        fading = {(1, 0): 1.0, (2, 0): 1.0}

        merged = sim.decode_physical_channels(
            0, [(0, [1, 2])], pkt_of, topo, clock, fading, max_channels=1)
        partitioned = sim.decode_physical_channels(
            0, [(0, [1]), (0, [2])], pkt_of, topo, clock, fading,
            max_channels=1)

        self.assertEqual(merged, [(0, None)])
        self.assertEqual(partitioned, merged)

    def test_slot_fading_is_sampled_once_per_pair_in_canonical_order(self):
        """Catches repartitioning transmitters changing which node gets each fade."""
        topo = SimpleNamespace(audible=[{1, 2}, {0, 2}, {0, 1}])

        forward = sim.sample_slot_fading([0], [1, 2], topo,
                                         random.Random(123))
        reversed_inputs = sim.sample_slot_fading([0], [2, 1], topo,
                                                 random.Random(123))

        self.assertEqual(forward, reversed_inputs)
        self.assertEqual(set(forward), {(1, 0), (2, 0)})

    def test_cupid_and_fwb_are_exactly_equal_with_one_fixed_radio(self):
        """Catches any residual protocol-specific physics with one radio."""
        sim.apply_config({
            "SEED": 42,
            "N_NODES": 6,
            "AREA_X": 12.0,
            "AREA_Y": 4.0,
            "TX_PROB": 1.0,
            "SHADOW_DB": 0.0,
            "FAST_FADE_DB": 4.0,
            "CLOCK_JITTER_NS": 0.0,
            "N_CHANNELS": 4,
            "N_RADIOS_PER_NODE": 1,
            "CHANNEL_SWITCH_US": 1000.0,
            "MAX_SLOTS": 80,
        })
        topo = sim.build_topology(random.Random(42))
        originators = list(range(1, 6))
        cupid_rng = random.Random(2026)
        fwb_rng = random.Random(2026)
        cupid_slots = {}
        fwb_slots = {}

        cupid = sim.run_epoch(
            "cupid", originators, topo, cupid_rng, delivery_slots=cupid_slots)
        fwb = sim.run_epoch(
            "cupid_fwb", originators, topo, fwb_rng, delivery_slots=fwb_slots)

        self.assertEqual(fwb, cupid)
        self.assertEqual(fwb_slots, cupid_slots)
        self.assertEqual(fwb_rng.random(), cupid_rng.random())


if __name__ == "__main__":
    unittest.main()
