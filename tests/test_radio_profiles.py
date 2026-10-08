"""Fase 2: perfis de radio, parametros derivados e airtime.

O padrao e o DW1000 do testbed CLOVES (canal 4, Cupid DCOSS-IoT 2025 sec.
V-A; Weaver SenSys 2020 sec. 6.1). O perfil 'legacy_v2' guarda os valores do
motor auditado (canal 5, 6,5 GHz, 500 MHz). Estes testes verificam
CONSISTENCIA de software e de derivacao; a unica comparacao com medicao
(Tab. 1 da EWSN 2020) e uma checagem de consistencia do airtime, nao
validacao da fisica de recepcao.
"""

import math
import unittest
from dataclasses import replace

import cupid_sim_v2 as sim
import radio_profiles as rp

CLOVES = "cloves_dw1000_ch4"


class RadioProfileTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }

    def tearDown(self):
        sim.apply_config(self.original_config)

    def test_default_profile_is_cloves(self):
        self.assertEqual(sim.RADIO_PROFILE, CLOVES)
        profile = rp.get_radio_profile(CLOVES)
        self.assertEqual(profile.name, CLOVES)
        with self.assertRaisesRegex(ValueError, "RADIO_PROFILE"):
            rp.get_radio_profile("magico")
        with self.assertRaisesRegex(ValueError, "RADIO_PROFILE"):
            sim.apply_config({"RADIO_PROFILE": "magico"})

    def test_channel_4_documented_values(self):
        p = rp.get_radio_profile(CLOVES)
        self.assertEqual(p.channel, 4)
        self.assertEqual(p.center_freq_mhz, 3993.6)
        self.assertEqual(p.nominal_bandwidth_mhz, 1331.2)
        self.assertEqual(p.rx_bandwidth_mhz, 900.0)
        self.assertEqual(p.noise_bandwidth_mhz, 900.0)
        self.assertEqual(p.mask_bandwidth_mhz, 1331.2)
        self.assertEqual(p.data_rate_mbps, 6.81)
        self.assertEqual(p.prf_mhz, 64)
        self.assertEqual(p.preamble_code, 17)
        self.assertEqual(p.pac_symbols, 8)
        self.assertEqual(p.preamble_symbols, 64)
        self.assertEqual(p.sfd_symbols, 8)
        self.assertEqual(p.phr_bits, 19)
        d = rp.derive_phy(p, 126)
        self.assertAlmostEqual(d.preamble_symbol_ns, 508 / 0.4992, places=6)
        self.assertAlmostEqual(d.data_symbol_ns, 64 / 0.4992, places=6)
        self.assertAlmostEqual(d.phr_symbol_ns, 8 * 64 / 0.4992, places=6)
        # 9 bits baixos do delayed TX ignorados: 512 ticks de 1/63,8976 GHz
        self.assertAlmostEqual(d.delayed_tx_granularity_ns, 4 / 0.4992,
                               places=6)
        self.assertAlmostEqual(d.timestamp_resolution_ps, 1e3 / 63.8976,
                               places=6)
        self.assertAlmostEqual(d.acquisition_ns, 8 * 508 / 0.4992, places=6)

    def test_legacy_profile_keeps_the_audited_values(self):
        p = rp.get_radio_profile("legacy_v2")
        self.assertEqual((p.center_freq_mhz, p.nominal_bandwidth_mhz),
                         (6500.0, 500.0))
        derived = rp.derived_params(p)
        self.assertEqual(derived, {
            "TX_POWER_DBM": -14.3, "PL0_DB": 48.7, "NOISE_DBM": -87.0,
            "PROC_GAIN_DB": 18.7, "PREAMBLE_SYMBOL_NS": 1017.6,
            "CAPTURE_TIME_NS": 8141.0, "SYNC_WINDOW_NS": 128.0})
        sim.apply_config({"RADIO_PROFILE": "legacy_v2"})
        for name, value in derived.items():
            self.assertEqual(getattr(sim, name), value, name)
        self.assertEqual(sim.RADIO_OVERRIDES, {})

    def test_noise_changes_with_the_effective_bandwidth(self):
        """O ruido integrado segue a banda de ruido; o ruido efetivo apos o
        despreading (N0 x Rb) nao depende dela -- por isso a banda nominal de
        1331,2 MHz nao pode entrar em todas as contas."""
        p = rp.get_radio_profile(CLOVES)
        wide = rp.derived_params(p)
        narrow = rp.derived_params(replace(p, noise_bandwidth_mhz=500.0))
        self.assertAlmostEqual(wide["NOISE_DBM"] - narrow["NOISE_DBM"],
                               10 * math.log10(900 / 500), places=9)
        eff = lambda d: d["NOISE_DBM"] - d["PROC_GAIN_DB"]
        self.assertAlmostEqual(eff(wide), eff(narrow), places=9)
        # a potencia captura a parte do espectro dentro da banda do receptor
        tx = rp.derived_params(replace(p, rx_bandwidth_mhz=500.0))
        self.assertAlmostEqual(wide["TX_POWER_DBM"] - tx["TX_POWER_DBM"],
                               10 * math.log10(900 / 500), places=9)

    def test_path_loss_at_1_m_changes_with_frequency(self):
        p = rp.get_radio_profile(CLOVES)
        pl4 = rp.derived_params(p)["PL0_DB"]
        # Friis a 1 m na frequencia central
        self.assertAlmostEqual(
            pl4, 20 * math.log10(4 * math.pi * 3993.6e6 / 299792458.0),
            places=9)
        pl5 = rp.derived_params(replace(p, center_freq_mhz=6489.6))["PL0_DB"]
        self.assertAlmostEqual(pl5 - pl4, 20 * math.log10(6489.6 / 3993.6),
                               places=9)

    def test_sensitivity_anchor_reproduces_the_datasheet(self):
        """Com a curva de erro do modelo, o enlace no limite de sensibilidade
        do datasheet (PER 1%) decodifica com 99%: a NF equivalente foi
        inferida para isso, e nao para bater latencia de protocolo."""
        p = rp.get_radio_profile(CLOVES)
        d = rp.derived_params(p)
        noise_eff = d["NOISE_DBM"] - d["PROC_GAIN_DB"]
        sinr = p.sensitivity_dbm - noise_eff
        sim.apply_config({"PER_MODEL": "logistic",
                          "REQ_SNR_DB": p.sensitivity_req_snr_db,
                          "PER_WIDTH_DB": p.sensitivity_per_width_db})
        self.assertAlmostEqual(sim.decode_success_prob(sinr),
                               1 - p.sensitivity_per, places=9)
        self.assertAlmostEqual(p.sensitivity_dbm, -92.0)

    def test_airtime_grows_with_frame_bytes(self):
        p = rp.get_radio_profile(CLOVES)
        times = [rp.frame_airtime_us(p, b) for b in range(0, 128)]
        self.assertTrue(all(b > a for a, b in zip(times, times[1:])))
        with self.assertRaises(ValueError):
            rp.frame_airtime_us(p, -1)
        with self.assertRaises(ValueError):
            rp.frame_airtime_us(p, 128)

    def test_airtime_is_consistent_with_the_ewsn_measurements(self):
        """EWSN 2020, Tab. 1 (EVB1000, mesma configuracao de radio):
        SHR 73 us; PHR + payload 45 us (15 B) e 178 us (127 B); slots de
        404 us e 806 us. Checagem de consistencia, com a folga da tabela
        ("approximate durations")."""
        p = rp.get_radio_profile(CLOVES)
        d15, d127 = rp.derive_phy(p, 15), rp.derive_phy(p, 127)
        self.assertAlmostEqual(d15.shr_us, 73.0, delta=0.5)
        self.assertAlmostEqual(d15.phr_us + d15.data_us, 45.0, delta=5.0)
        self.assertAlmostEqual(d127.phr_us + d127.data_us, 178.0, delta=5.0)
        self.assertAlmostEqual(d15.slot_min_us, 404.0, delta=5.0)
        self.assertAlmostEqual(d127.slot_min_us, 806.0, delta=5.0)

    def test_declared_slots_accommodate_their_frames(self):
        p = rp.get_radio_profile(CLOVES)
        self.assertLessEqual(rp.derive_phy(p, sim.DATA_FRAME_BYTES)
                             .slot_min_us, sim.SLOT_US)
        self.assertLessEqual(rp.derive_phy(p, sim.SHORT_FRAME_BYTES)
                             .slot_min_us, sim.RANGING_SLOT_US)
        self.assertEqual(sim.DATA_FRAME_BYTES, 126)
        self.assertEqual(sim.RANGING_SLOT_US, 460.0)
        self.assertEqual(sim.SHORT_FRAME_BYTES, p.short_frame_bytes)

    def test_effective_config_has_profile_and_derived_values(self):
        cfg = sim.effective_config()
        self.assertEqual(cfg["RADIO_PROFILE"], CLOVES)
        derived = rp.derived_params(rp.get_radio_profile(CLOVES))
        for name, value in derived.items():
            self.assertEqual(cfg[name], value)
        phy = cfg["DERIVED_PHY"]
        self.assertAlmostEqual(phy["data_frame"]["airtime_us"],
                               rp.frame_airtime_us(
                                   rp.get_radio_profile(CLOVES), 126))
        self.assertIn("short_frame", phy)
        self.assertEqual(cfg["RADIO_OVERRIDES"], {})
        for name in ("RADIO_PROFILE", "DATA_FRAME_BYTES",
                     "SHORT_FRAME_BYTES", "RANGING_SLOT_US"):
            self.assertIn(name, sim.PARAM_NAMES)

    def test_derived_values_are_never_overridden_silently(self):
        derived = rp.derived_params(rp.get_radio_profile(CLOVES))
        sim.apply_config({"PL0_DB": 50.0})
        self.assertEqual(sim.PL0_DB, 50.0)
        self.assertEqual(sim.RADIO_OVERRIDES, {
            "PL0_DB": {"derivado": derived["PL0_DB"], "efetivo": 50.0}})
        cfg = sim.effective_config()
        self.assertEqual(cfg["RADIO_OVERRIDES"], sim.RADIO_OVERRIDES)
        # um apply_config seguinte, sem tocar no PL0_DB, mantem o override
        sim.apply_config({"FAST_FADE_DB": 3.0})
        self.assertEqual(sim.PL0_DB, 50.0)
        # voltar ao valor derivado desfaz o override
        sim.apply_config({"PL0_DB": derived["PL0_DB"]})
        self.assertEqual(sim.RADIO_OVERRIDES, {})
        # trocar de perfil rederiva tudo e descarta overrides antigos
        sim.apply_config({"PL0_DB": 50.0})
        sim.apply_config({"RADIO_PROFILE": "legacy_v2"})
        self.assertEqual((sim.PL0_DB, sim.RADIO_OVERRIDES), (48.7, {}))


if __name__ == "__main__":
    unittest.main()
