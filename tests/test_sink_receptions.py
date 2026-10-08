"""[A1] Recepcoes no sink sao observabilidade pura.

O teste central e test_sink_events_never_change_the_epoch: se a emissao de
eventos alterar entrega, PDR ou consumo de aleatoriedade, todos os resultados
de fisica dos Itens 1-8 ficam contaminados.
"""

import random
import unittest

import cupid_sim_v2 as sim


class SinkReceptionTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }

    def tearDown(self):
        sim.apply_config(self.original_config)

    def loaded_scenario(self):
        """Topologia com pressao de deadline suficiente para gerar duplicatas.

        [F18] O prazo era 205 slots. Com o ACK fisico do sink o Cupid passou a
        fechar esta epoca em 172 slots, e sem epoca aberta nao ha cauda para
        observar; 100 slots devolvem a pressao a todos os protocolos.
        """
        sim.apply_config({
            "SEED": 42,
            "N_NODES": 24,
            "AREA_X": 60.0,
            "AREA_Y": 20.0,
            "MAX_SLOTS": 100,
        })
        topo = sim.build_topology(random.Random(42))
        return topo, sim.valid_nodes(topo)

    def open_epoch_budget(self, protocol, topo, originators):
        """Prazo que corta a epoca no meio: metade chega antes, metade depois.

        Escolhido a partir de uma rodada sem prazo, para que os testes de
        cauda nao dependam de quao rapido cada protocolo esta no modelo atual
        (a cada item de realismo esse numero muda).
        """
        slots = {}
        sim.run_epoch(protocol, originators, topo, random.Random(2026),
                      slot_budget=1500, delivery_slots=slots)
        arrivals = sorted(slots.values())
        self.assertGreaterEqual(len(arrivals), 2,
                                f"{protocol} quase nao entrega")
        return arrivals[len(arrivals) // 2]

    def run_pair(self, protocol, topo, originators, slot_budget=None,
                 **kwargs):
        """Roda a mesma epoca com e sem instrumentacao, mesma semente."""
        plain_rng = random.Random(2026)
        traced_rng = random.Random(2026)
        plain_slots = {}
        traced_slots = {}
        events = []
        plain = sim.run_epoch(protocol, originators, topo, plain_rng,
                              slot_budget=slot_budget,
                              delivery_slots=plain_slots)
        traced = sim.run_epoch(protocol, originators, topo, traced_rng,
                               slot_budget=slot_budget,
                               delivery_slots=traced_slots,
                               sink_events=events, **kwargs)
        return (plain, plain_slots, plain_rng), (traced, traced_slots,
                                                 traced_rng), events

    def test_sink_events_never_change_the_epoch(self):
        """Criterio 1: entrega, PDR e stream de rng identicos com e sem trace."""
        topo, originators = self.loaded_scenario()
        # 10 fontes: com todos os nos transmitindo, a fase T do Crystal
        # "tapairs" quase nunca sucede no modelo de captura atual (P10) e a
        # epoca acaba sem recepcao nenhuma -- o teste ficaria vazio
        originators = originators[:10]
        for protocol in sim.PROTOCOLS:
            with self.subTest(protocol=protocol):
                plain, traced, events = self.run_pair(
                    protocol, topo, originators, slot_budget=1500)
                self.assertEqual(traced[0], plain[0])
                self.assertEqual(traced[1], plain[1])
                # a assercao forte: o rng consumiu exatamente o mesmo tanto
                self.assertEqual(traced[2].random(), plain[2].random())
                self.assertTrue(events, f"{protocol} nao emitiu recepcoes")

    def test_tail_does_not_touch_the_main_rng_or_the_result(self):
        """A cauda pos-deadline roda num stream separado."""
        topo, originators = self.loaded_scenario()
        for protocol in ("weaver", "cupid", "cupid_fwb"):
            with self.subTest(protocol=protocol):
                budget = self.open_epoch_budget(protocol, topo, originators)
                plain, traced, events = self.run_pair(
                    protocol, topo, originators, slot_budget=budget,
                    tail_slots=200, tail_rng=random.Random("cauda"))
                self.assertEqual(traced[0], plain[0])
                self.assertEqual(traced[1], plain[1])
                self.assertEqual(traced[2].random(), plain[2].random())
                self.assertTrue(
                    any(e["flags"] & sim.SINK_FLAG_LATE for e in events),
                    f"{protocol} nao produziu chegadas atrasadas")

    def test_tail_without_its_own_rng_is_rejected(self):
        topo, originators = self.loaded_scenario()
        with self.assertRaisesRegex(ValueError, "tail_rng"):
            sim.run_epoch("cupid", originators, topo, random.Random(1),
                          sink_events=[], tail_slots=10)

    def test_late_events_never_add_a_delivery(self):
        """A cauda observa sem entregar: delivery_slots ignora o pos-deadline.

        [F11] Depois do fan-out, 'late' deixou de implicar 'nunca entregue':
        um relatorio entregue dentro do prazo continua a ser retransmitido
        pelos detentores que sobreviveram a entrega, e o sink o ouve de novo
        na cauda. O que a cauda nao pode fazer e criar entrega -- por isso a
        comparacao e contra a mesma epoca rodada sem cauda nenhuma.
        """
        topo, originators = self.loaded_scenario()
        budget = self.open_epoch_budget("cupid", topo, originators)
        slots = {}
        events = []
        sim.run_epoch("cupid", originators, topo, random.Random(2026),
                      slot_budget=budget, delivery_slots=slots,
                      sink_events=events, tail_slots=200,
                      tail_rng=random.Random("cauda"))
        baseline = {}
        sim.run_epoch("cupid", originators, topo, random.Random(2026),
                      slot_budget=budget, delivery_slots=baseline)
        self.assertEqual(slots, baseline)
        late = [e for e in events if e["flags"] & sim.SINK_FLAG_LATE]
        self.assertTrue(late)
        for event in late:
            self.assertGreaterEqual(event["slot"], budget)
        for packet, delivery_slot in slots.items():
            self.assertLessEqual(delivery_slot, budget)

    def test_every_duplicate_follows_a_first_reception(self):
        """Criterio 2: a chave (originator, epoch, seq) se repete de fato."""
        topo, originators = self.loaded_scenario()
        for protocol in sim.PROTOCOLS:
            with self.subTest(protocol=protocol):
                events = []
                sim.run_epoch(protocol, originators, topo,
                              random.Random(2026), sink_events=events)
                for event in events:
                    if not event["flags"] & sim.SINK_FLAG_DUPLICATE:
                        continue
                    earlier = [e for e in events
                               if e["originator"] == event["originator"]
                               and e["slot"] <= event["slot"]
                               and not e["flags"] & sim.SINK_FLAG_DUPLICATE]
                    self.assertTrue(earlier,
                                    "duplicata sem recepcao original antes")

    def test_crystal_floods_produce_duplicates(self):
        """O filtro 'not sink_arrival_slot' escondia todas as repeticoes.

        Vale para o Crystal sequencial, em que o sink volta a ouvir o mesmo
        flood; no modelo "tapairs" [F26] o sink adota o primeiro pacote e
        passa a retransmiti-lo, e a duplicata so aparece quando uma fonte
        perde o proprio ACK (test_crystal.py).
        """
        topo, originators = self.loaded_scenario()
        sim.apply_config({"CRYSTAL_MODEL": "sequential"})
        events = []
        sim.run_epoch("crystal", originators, topo, random.Random(2026),
                      sink_events=events)
        self.assertTrue(any(e["flags"] & sim.SINK_FLAG_DUPLICATE
                            for e in events))

    def test_cupid_keeps_several_holders_per_packet_after_f11(self):
        """[F11] Com fan-out real, o pacote vive em varios detentores.

        A prova era indireta: duplicatas no sink. Com os ACKs do Weaver
        [F23][F24] as duplicatas ficaram raras -- e para isso que esses ACKs
        existem --, entao o teste mede o invariante direto: o maximo de
        detentores por pacote. No motor da etapa 0, sem esses ACKs, as
        duplicatas continuam aparecendo.
        """
        topo, originators = self.loaded_scenario()
        for protocol in ("cupid", "cupid_fwb"):
            with self.subTest(protocol=protocol):
                stats = sim.new_holder_stats()
                sim.run_epoch(protocol, originators, topo,
                              random.Random(2026), holder_stats=stats)
                self.assertGreaterEqual(
                    sim.summarize_holder_stats(stats)["max_holders"], 2,
                    f"{protocol} continua com um detentor por pacote")
        # [P16] o preset leva o radio legado: a topologia e refeita com ele,
        # senao o motor da etapa 0 rodaria sobre enlaces do canal 4
        sim.apply_config(sim.PRESET_ETAPA0)
        topo, originators = self.loaded_scenario()
        events = []
        sim.run_epoch("cupid", originators, topo, random.Random(2026),
                      sink_events=events, tail_slots=60,
                      tail_rng=random.Random("cauda"))
        self.assertTrue(
            [e for e in events if e["flags"] & sim.SINK_FLAG_DUPLICATE])

    def test_weaver_has_no_duplicates_in_legacy_mode_because_of_the_oracle(self):
        """Item 4 da auditoria, preservado so no modo "legacy".

        O ACK global antigo purga o pacote de TODAS as filas por decreto, sem
        passar por decode_at_receiver(), e isso zera as duplicatas do Weaver.
        Com o ACK fisico [F18] o zero deixa de ser garantido: ver
        test_sink_ack.test_weaver_ack_is_no_longer_an_oracle.
        """
        topo, originators = self.loaded_scenario()
        sim.apply_config(sim.PRESET_2026_09_10)
        events = []
        sim.run_epoch("weaver", originators, topo, random.Random(2026),
                      sink_events=events, tail_slots=60,
                      tail_rng=random.Random("cauda"))
        self.assertTrue(events)
        self.assertEqual(
            [e for e in events if e["flags"] & sim.SINK_FLAG_DUPLICATE], [])

    def test_first_reception_agrees_with_the_delivery_slot(self):
        """O evento de primeira recepcao e o mesmo slot que gera a entrega."""
        topo, originators = self.loaded_scenario()
        slots = {}
        events = []
        sim.run_epoch("cupid", originators, topo, random.Random(2026),
                      delivery_slots=slots, sink_events=events)
        for packet, delivery_slot in slots.items():
            advancing = [e for e in events
                         if e["originator"] == packet
                         and not e["flags"] & sim.SINK_FLAG_NO_ADVANCE]
            self.assertTrue(advancing)
            self.assertEqual(min(e["slot"] for e in advancing) + 1,
                             delivery_slot)

    def test_receptions_carry_physical_measurements(self):
        """rssi/sinr vem de decode_at_receiver, nao de um recalculo."""
        topo, originators = self.loaded_scenario()
        events = []
        sim.run_epoch("cupid", originators, topo, random.Random(2026),
                      sink_events=events)
        # [F21] com a curva suave um quadro pode passar abaixo de REQ_SNR_DB,
        # mas nunca muito abaixo (a 3 larguras, a chance e < 1e-5)
        floor = (sim.REQ_SNR_DB if sim.PER_MODEL == "threshold"
                 else sim.REQ_SNR_DB - 3 * sim.PER_WIDTH_DB)
        for event in events:
            self.assertGreater(event["sinr_db"], floor - 1e-9)
            self.assertLess(event["rssi_dbm"], 0.0)
            self.assertGreater(event["t_arrival_ns"], 0.0)


if __name__ == "__main__":
    unittest.main()
