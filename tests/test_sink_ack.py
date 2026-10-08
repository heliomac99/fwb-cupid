"""[F18] ACK do sink fisico e igual para todos os protocolos CTX.

Antes, o sink nunca transmitia: o hop 1 do Cupid nao tinha como saber que um
relatorio ja tinha chegado (o ACK implicito exige ouvir o pacote vindo de um
hop menor, e o hop 0 e o proprio sink), enquanto o Weaver tinha um ACK global
oraculo. As topologias abaixo sao sinteticas e deterministicas (sem fading,
sem jitter, TX_PROB = 1) para que cada efeito tenha uma causa unica.
"""

import random
import unittest
from types import SimpleNamespace

import numpy as np

import cupid_sim_v2 as sim


def make_topology(n, links, hop):
    """Topologia sintetica: 'links' = {(a, b): potencia_mW}, reciprocos."""
    power = np.zeros((n, n))
    for (a, b), p in links.items():
        power[a][b] = power[b][a] = p
    audible = [set(np.nonzero(power[r] > 0)[0].tolist()) for r in range(n)]
    return SimpleNamespace(
        n=n, ps_lin=power, tprop_ns=np.zeros((n, n)), audible=audible,
        reliable=[set(a) for a in audible], hop=hop,
        group={v: "emitter" for v in range(n)})


class SinkAckTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }

    def tearDown(self):
        sim.apply_config(self.original_config)

    @staticmethod
    def deterministic(ack_model, local_ack="overhearing"):
        """Sem sorteio nenhum. O ACK local fica no modo em que estes cenarios
        foram escritos (F11), salvo quando o teste pede o do Weaver [F23]."""
        sim.apply_config({
            "FAST_FADE_DB": 0.0, "CLOCK_JITTER_NS": 0.0, "TX_PROB": 1.0,
            "RANDOM_QUEUE_HEAD": False, "ACK_MODEL": ack_model,
            "PER_MODEL": "threshold", "LOCAL_ACK_MODEL": local_ack,
            "BOOTSTRAP_DELAY": False, "BOOTSTRAP_MODEL": "bfs",
            # [Fase 5] cenarios da premissa P14 (sink no grupo emitter),
            # escritos para a grade legada; a da Fig. 5b tem teste proprio
            "CUPID_SCHEDULE": "legacy",
        })

    @staticmethod
    def traced_epoch(protocol, originators, topo, budget):
        holder_stats = sim.new_holder_stats()
        delivery_slots = {}
        events = []
        result = sim.run_epoch(
            protocol, originators, topo, random.Random(1),
            slot_budget=budget, delivery_slots=delivery_slots,
            sink_events=events, holder_stats=holder_stats)
        duplicates = sum(1 for e in events
                         if e["flags"] & sim.SINK_FLAG_DUPLICATE)
        return result, delivery_slots, duplicates, holder_stats

    def test_legacy_mode_reproduces_the_pre_fix_engine_bit_for_bit(self):
        """PRESET_2026_09_10 e o motor anterior intacto (numeros de 10/09).

        Valores medidos no commit 2efd6ba, antes do F18. Se falhar, algum
        item novo nao entrou no preset ou mudou o motor por fora da chave.
        """
        loaded = {
            "weaver": ((205, 4), 0.9293881983417874),
            "cupid": ((205, 15), 0.946204115789721),
            "cupid_fwb": ((205, 15), 0.946204115789721),
        }
        sim.apply_config(dict(sim.PRESET_2026_09_10, SEED=42, N_NODES=24,
                              AREA_X=60.0, AREA_Y=20.0, MAX_SLOTS=205))
        topo = sim.build_topology(random.Random(42))
        for protocol, (result, next_draw) in loaded.items():
            with self.subTest(protocol=protocol):
                rng = random.Random(2026)
                got = sim.run_epoch(protocol, sim.valid_nodes(topo), topo,
                                    rng, slot_budget=205)
                self.assertEqual((got, rng.random()), (result, next_draw))

        default = {
            "weaver": ([(1500, 21), (1500, 24)], 0.19183695821959645),
            "cupid": ([(1500, 26), (1500, 28)], 0.40653271844292105),
        }
        sim.apply_config(dict(sim.PRESET_2026_09_10, SEED=42, N_NODES=36,
                              AREA_X=83.0, AREA_Y=33.0, MAX_SLOTS=1500))
        topo = sim.build_topology(random.Random(42))
        valid = sim.valid_nodes(topo)
        for protocol, (results, next_draw) in default.items():
            with self.subTest(protocol=protocol):
                rng = random.Random(7)
                got = [sim.run_epoch(protocol, rng.sample(valid, 35), topo,
                                     rng) for _ in range(2)]
                self.assertEqual((got, rng.random()), (results, next_draw))

    def test_cupid_hop1_stops_only_after_decoding_the_sink_ack(self):
        """P1: o hop 1 do Cupid so para de retransmitir ao OUVIR o sink.

        No 1 entrega o proprio relatorio no slot 1. No 2 nao tem rota e
        mantem a epoca aberta. Sem ACK do sink ("legacy"), o no 1 repete o
        relatorio entregue em todo slot seu; com o ACK fisico ele o apaga no
        slot 4, quando decodifica o quadro do sink.
        """
        topo = make_topology(3, {(0, 1): 1e-6}, {0: 0, 1: 1, 2: 99})

        self.deterministic("physical")
        result, slots, dups, stats = self.traced_epoch(
            "cupid", [1, 2], topo, 40)
        self.assertEqual(result, (40, 1))
        self.assertEqual(slots, {1: 2})
        self.assertEqual(dups, 0)
        self.assertEqual(stats["sink_ack_decodes"], 1)
        self.assertEqual(stats["global_ack_suppressions"], 1)

        self.deterministic("legacy")
        result, slots, dups, stats = self.traced_epoch(
            "cupid", [1, 2], topo, 40)
        self.assertEqual(result, (40, 1))
        self.assertEqual(dups, 9)
        self.assertEqual(stats["sink_ack_tx"], 0)
        self.assertEqual(stats["global_ack_suppressions"], 0)

    def test_sink_ack_is_decoded_like_any_frame_and_can_be_lost(self):
        """O ACK concorre com os dados do slot na mesma decode_at_receiver().

        No Cupid o sink transmite junto com os emitters do hop 2. Se o no 2
        chega 6 dB mais forte que o sink no no 1, o no 1 captura o dado e
        perde o ACK -- para sempre, neste cenario deterministico. Invertendo
        as potencias, o mesmo ACK e decodificado e o relatorio 2 avanca.
        """
        hop = {0: 0, 1: 1, 2: 2, 3: 99}
        self.deterministic("physical")

        lost = make_topology(4, {(0, 1): 1e-6, (1, 2): 4e-6}, hop)
        result, _, dups, stats = self.traced_epoch(
            "cupid", [1, 2, 3], lost, 40)
        self.assertEqual(result, (40, 1))
        self.assertGreater(stats["sink_ack_tx"], 0)
        self.assertEqual(stats["sink_ack_decodes"], 0)
        self.assertEqual(stats["global_ack_suppressions"], 0)
        self.assertGreater(dups, 0)

        heard = make_topology(4, {(0, 1): 4e-6, (1, 2): 1e-6}, hop)
        result, slots, _, stats = self.traced_epoch(
            "cupid", [1, 2, 3], heard, 40)
        self.assertEqual(result, (40, 2))
        self.assertEqual(slots, {1: 2, 2: 6})
        self.assertGreaterEqual(stats["sink_ack_decodes"], 1)
        self.assertGreaterEqual(stats["global_ack_suppressions"], 1)

        # [F23] com o ACK local do Weaver o mesmo cenario "perdido" se
        # resolve: no slot 1 o no 2 le no cabecalho do no 1 que o relatorio 2
        # ja subiu e se cala; sem a colisao, o ACK do sink passa no slot 4.
        self.deterministic("physical", local_ack="weaver")
        result, slots, _, stats = self.traced_epoch(
            "cupid", [1, 2, 3], lost, 40)
        self.assertEqual(result, (40, 2))
        self.assertEqual(slots, {1: 2, 2: 6})
        self.assertGreaterEqual(stats["local_ack_suppressions"], 1)
        # slot 4 (confirma o 1) e slot 8 (confirma o 2, que o no 1 segurava)
        self.assertEqual(stats["sink_ack_decodes"], 2)

    def test_weaver_ack_is_no_longer_an_oracle(self):
        """Item 4: o Weaver perde o ACK quando ele colide, como o Cupid.

        No slot do sink (hop 0) tambem transmite o hop 3. Um no do hop 3
        6 dB mais forte no no 1 encobre o ACK: o no 1 repete o relatorio
        entregue. O oraculo antigo apagava a copia por decreto. (O no 2, sem
        enlace nenhum, so ocupa o slot do hop 2: o oraculo antigo nao rodava
        em slot sem transmissao.)
        """
        topo = make_topology(4, {(0, 1): 1e-6, (1, 3): 4e-6},
                             {0: 0, 1: 1, 2: 2, 3: 3})

        self.deterministic("physical")
        _, _, dups, stats = self.traced_epoch("weaver", [1, 2, 3], topo, 30)
        self.assertEqual(dups, 9)
        self.assertGreater(stats["sink_ack_tx"], 0)
        self.assertEqual(stats["sink_ack_decodes"], 0)

        self.deterministic("legacy")
        _, _, dups, _ = self.traced_epoch("weaver", [1, 2, 3], topo, 30)
        self.assertEqual(dups, 0)

    def test_delivery_knowledge_travels_only_inside_decoded_headers(self):
        """A confirmacao chega ao hop 2 "de carona", nunca por decreto.

        O relatorio 5 sobe por 5 -> 4 -> 2 -> sink; o no 3 tambem ficou com
        uma copia, mas nao ouve nem o no 2 nem o no 4. No slot 6 os nos 1 e
        2 decodificam o ACK do sink; no slot 7 o no 3 decodifica o quadro do
        no 1 (que carrega o relatorio 3) e aprende pelo cabecalho que o 5 ja
        chegou. Sem a carona, o no 3 ficaria com a copia ate o fim.
        """
        topo = make_topology(
            6,
            {(0, 1): 1e-6, (0, 2): 4e-6, (1, 3): 1e-6, (2, 4): 1e-6,
             (3, 5): 1e-6, (4, 5): 1e-6},
            {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3})
        self.deterministic("physical")
        result, slots, _, stats = self.traced_epoch("weaver", [5, 3], topo, 30)
        self.assertEqual(result, (8, 2))
        self.assertEqual(slots, {5: 5, 3: 8})
        self.assertEqual(stats["sink_ack_tx"], 1)
        self.assertEqual(stats["sink_ack_decodes"], 2)
        # no 2 pelo ACK do sink + no 3 pela carona no quadro do no 1
        self.assertEqual(stats["global_ack_suppressions"], 2)

    def test_unknown_ack_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ACK_MODEL"):
            sim.apply_config({"ACK_MODEL": "magico"})


if __name__ == "__main__":
    unittest.main()
