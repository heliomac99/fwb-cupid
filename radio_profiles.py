# -*- coding: utf-8 -*-
"""
Perfis de radio UWB e grandezas derivadas (airtime, ruido, perda a 1 m).

Cada campo de RadioProfile tem uma CLASSE de origem, documentada ao lado:
  [medido]    medido diretamente (numa publicacao, com a configuracao dita);
  [spec]      derivado de especificacao (datasheet/User Manual/IEEE);
  [inferido]  inferido de resultado experimental publicado;
  [hipotese]  hipotese explicita, sem fonte que fixe o valor;
  [sens]      parametro de sensibilidade (varrido, nao fixado).

Fontes:
  DS  = DW1000 Datasheet v2.23;  UM = DW1000 User Manual v2.15;
  CUP = Park et al., Cupid, DCOSS-IoT 2025;  WEA = Trobinger et al., Weaver,
  SenSys 2020;  EWSN = Lobba et al., EWSN 2020.

Bandas -- quatro papeis distintos, que nao podem ser trocados entre si:
  nominal_bandwidth_mhz : banda do canal na norma (1331,2 MHz no canal 4);
  mask_bandwidth_mhz    : banda sobre a qual a mascara de -41,3 dBm/MHz
                          limita a potencia RADIADA (so para relatorio);
  rx_bandwidth_mhz      : banda efetiva do receptor (DS: ~900 MHz nos canais
                          4 e 7) -- a parte do espectro que ele CAPTURA;
  noise_bandwidth_mhz   : banda do ruido termico integrado.
O ruido efetivo apos o despreading, N0 x Rb, NAO depende da banda de ruido:
ela entra no ruido integrado e sai no ganho de processamento. O que a banda
do receptor muda e a potencia de sinal capturada com PSD fixa.
"""

import math
from dataclasses import dataclass, field, asdict

C_M_PER_S = 299792458.0
CHIP_MHZ = 499.2                    # [spec] UM "Clock Periods": 499,2 MHz
KT_DBM_HZ = -174.0                  # [spec] kT a ~290 K


@dataclass(frozen=True)
class RadioProfile:
    """Configuracao de radio imutavel. Valores fixos em 'fixed' substituem a
    derivacao (so no perfil legado, para reproduzir o motor auditado)."""
    name: str
    description: str
    channel: int
    center_freq_mhz: float
    nominal_bandwidth_mhz: float
    mask_bandwidth_mhz: float
    rx_bandwidth_mhz: float
    noise_bandwidth_mhz: float
    tx_psd_dbm_mhz: float
    data_rate_mbps: float
    prf_mhz: int
    preamble_code: int
    pac_symbols: int
    preamble_symbols: int
    sfd_symbols: int
    phr_bits: int
    antenna_gain_db: float
    # ancora de sensibilidade: PER, valor e condicoes do datasheet
    sensitivity_ref_dbm: float
    sensitivity_per: float
    sensitivity_channel_penalty_db: float
    sensitivity_preamble_penalty_db: float
    sensitivity_req_snr_db: float
    sensitivity_per_width_db: float
    # plataforma (EVB1000): SPI e processamento por slot (EWSN, Tab. 1)
    spi_us_at_15b: float
    spi_us_at_127b: float
    slot_other_us: float
    short_frame_bytes: int
    fixed: dict = field(default_factory=dict)

    @property
    def sensitivity_dbm(self):
        """Potencia total na banda do receptor no limite de sensibilidade
        (PER = sensitivity_per) para este canal e preambulo."""
        return (self.sensitivity_ref_dbm + self.sensitivity_channel_penalty_db
                + self.sensitivity_preamble_penalty_db)


@dataclass(frozen=True)
class DerivedPhy:
    """Grandezas derivadas de um perfil para um quadro de 'frame_bytes'."""
    profile: str
    frame_bytes: int
    preamble_symbol_ns: float
    data_symbol_ns: float
    phr_symbol_ns: float
    shr_us: float
    phr_us: float
    data_us: float
    airtime_us: float
    payload_bits: int
    slot_min_us: float
    acquisition_ns: float
    delayed_tx_granularity_ns: float
    timestamp_resolution_ps: float

    def as_dict(self):
        return asdict(self)


# [spec] UM Tab. 60: simbolo de preambulo = 496 (PRF 16) ou 508 (PRF 64)
# chips a 499,2 MHz. Simbolo de dados a 6,8 Mbps = 64 chips (128,2 ns); o
# PHR vai a "just less than 980 kbps" (UM sec. 10.2) = 512 chips (1025,6 ns).
_CHIPS_PER_PREAMBLE_SYMBOL = {16: 496, 64: 508}
_CHIPS_PER_DATA_SYMBOL = {6.81: 64, 0.85: 512, 0.11: 4096}
_CHIPS_PER_PHR_SYMBOL = {6.81: 512, 0.85: 512, 0.11: 4096}
MAX_FRAME_BYTES = 127               # [spec] quadro padrao IEEE 802.15.4


PROFILES = {
    "cloves_dw1000_ch4": RadioProfile(
        name="cloves_dw1000_ch4",
        description=("DW1000 (EVB1000) do testbed CLOVES: canal 4, PRF 64, "
                     "codigo 17, PAC 8, 6,8 Mbps, preambulo de 64 simbolos"),
        channel=4,                          # [medido] CUP sec. V-A
        center_freq_mhz=3993.6,             # [spec] CUP V-A; DS 5.1.1
        nominal_bandwidth_mhz=1331.2,       # [spec] CUP V-A; DS 5.1.1
        mask_bandwidth_mhz=1331.2,          # [spec] = banda nominal
        rx_bandwidth_mhz=900.0,             # [spec] DS Tab. 5 e 5.1.1
        noise_bandwidth_mhz=900.0,          # [hipotese] = banda do receptor
        tx_psd_dbm_mhz=-41.3,               # [spec]+[hipotese] limite da
        # mascara; WEA 6.1 e EWSN 4.2 usam a potencia "maxima recomendada"
        # para o canal/PRF; PSD plana na banda (hipotese)
        data_rate_mbps=6.81,                # [medido] CUP V-A
        prf_mhz=64,                         # [spec] CUP V-A imprime "6.4 MHz"
        # (erro de digitacao: o DW1000 so tem 16/64 MHz, DS 5.1.2; o codigo
        # 17 so existe em PRF 64, UM Tab. 61)
        preamble_code=17,                   # [medido] CUP V-A
        pac_symbols=8,                      # [medido] CUP V-A
        preamble_symbols=64,                # [inferido] WEA 6.1 "shortest
        # ~64 us preamble" e EWSN 4.2 "64 us preamble" (mesmo grupo e placa);
        # o CUP nao informa
        sfd_symbols=8,                      # [spec] UM 10.3 (SFD padrao)
        phr_bits=19,                        # [spec] UM 10.1/10.4
        antenna_gain_db=0.0,                # [hipotese] nao publicado
        sensitivity_ref_dbm=-93.0,          # [spec] DS Tab. 6: 6,8 Mbps,
        # PER 1%, preambulo 256, 20 B, canal 5, PRF 16, CFO +-10 ppm
        sensitivity_per=0.01,
        sensitivity_channel_penalty_db=1.0,  # [spec] DS Tab. 6: canal 2
        # (mesma frequencia central do 4) "approximately 1 dB less
        # sensitive"; o canal 4 nao e tabelado -> hipotese: igual ao 2
        sensitivity_preamble_penalty_db=0.0,  # [hipotese]+[sens] o datasheet
        # mediu com preambulo 256; o CLOVES usa 64
        sensitivity_req_snr_db=3.0,         # [hipotese] REQ_SNR_DB do modelo
        sensitivity_per_width_db=2.0,       # [hipotese] PER_WIDTH_DB do modelo
        spi_us_at_15b=36.0,                 # [medido] EWSN Tab. 1
        spi_us_at_127b=304.0,               # [medido] EWSN Tab. 1
        slot_other_us=250.0,                # [medido] EWSN Tab. 1 ("Other")
        short_frame_bytes=11,               # [hipotese] CUP IV-B: "header-
        # only, with payload length of 0"; cabecalho MAC 802.15.4 com
        # enderecos curtos e PAN comprimido (9 B) + FCS (2 B). O tamanho do
        # cabecalho nao e publicado.
    ),
    "legacy_v2": RadioProfile(
        name="legacy_v2",
        description=("motor auditado 5ef525b: numeros do canal 5 (6,5 GHz, "
                     "500 MHz) com ruido termico sem figura de ruido"),
        channel=5,
        center_freq_mhz=6500.0,
        nominal_bandwidth_mhz=500.0,
        mask_bandwidth_mhz=500.0,
        rx_bandwidth_mhz=500.0,
        noise_bandwidth_mhz=500.0,
        tx_psd_dbm_mhz=-41.3,
        data_rate_mbps=6.81,
        prf_mhz=64,
        preamble_code=17,
        pac_symbols=8,
        preamble_symbols=64,                # nao modelado no legado
        sfd_symbols=8,
        phr_bits=19,
        antenna_gain_db=0.0,
        sensitivity_ref_dbm=-102.7,         # o que o legado implica (NF 0)
        sensitivity_per=0.5,
        sensitivity_channel_penalty_db=0.0,
        sensitivity_preamble_penalty_db=0.0,
        sensitivity_req_snr_db=3.0,
        sensitivity_per_width_db=2.0,
        spi_us_at_15b=36.0,
        spi_us_at_127b=304.0,
        slot_other_us=250.0,
        short_frame_bytes=11,
        # valores do motor auditado, arredondados como estavam no codigo
        fixed={"TX_POWER_DBM": -14.3, "PL0_DB": 48.7, "NOISE_DBM": -87.0,
               "PROC_GAIN_DB": 18.7, "PREAMBLE_SYMBOL_NS": 1017.6,
               "CAPTURE_TIME_NS": 8141.0, "SYNC_WINDOW_NS": 128.0},
    ),
}


def get_radio_profile(name):
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"RADIO_PROFILE desconhecido '{name}'. Validos: "
                         f"{', '.join(PROFILES)}") from None


def _rate_key(profile):
    for key in _CHIPS_PER_DATA_SYMBOL:
        if abs(profile.data_rate_mbps - key) < 1e-9:
            return key
    raise ValueError(f"taxa de dados sem estrutura de simbolo conhecida: "
                     f"{profile.data_rate_mbps} Mbps")


def payload_bits(frame_bytes):
    """[spec] Bits no ar para um PSDU de 'frame_bytes' (FCS incluido): o
    Reed-Solomon RS(63,55) sobre simbolos de 6 bits acrescenta 48 bits de
    paridade a cada bloco de ate 330 bits (UM 10.2: taxas ja descontam o
    fator ~0,87 do RS)."""
    bits = 8 * int(frame_bytes)
    return bits + 48 * math.ceil(bits / 330)


def _check_frame(frame_bytes):
    if int(frame_bytes) != frame_bytes or not 0 <= frame_bytes <= \
            MAX_FRAME_BYTES:
        raise ValueError(f"frame_bytes deve ser inteiro em 0..{MAX_FRAME_BYTES}"
                         f" (recebido {frame_bytes})")


def derive_phy(profile, frame_bytes):
    """Airtime e temporizacao do quadro. SHR = (preambulo + SFD) simbolos de
    preambulo; PHR = phr_bits simbolos a ~975 kbps; dados = payload_bits a
    6,8 Mbps. slot_min soma, como a Tab. 1 da EWSN 2020, o ar, o SPI (linear
    em bytes entre os dois pontos medidos) e o processamento fixo."""
    _check_frame(frame_bytes)
    rate = _rate_key(profile)
    t_psym = _CHIPS_PER_PREAMBLE_SYMBOL[profile.prf_mhz] / CHIP_MHZ * 1e3
    t_dsym = _CHIPS_PER_DATA_SYMBOL[rate] / CHIP_MHZ * 1e3
    t_phr = _CHIPS_PER_PHR_SYMBOL[rate] / CHIP_MHZ * 1e3
    bits = payload_bits(frame_bytes)
    shr_us = (profile.preamble_symbols + profile.sfd_symbols) * t_psym / 1e3
    phr_us = profile.phr_bits * t_phr / 1e3
    data_us = bits * t_dsym / 1e3
    slope = ((profile.spi_us_at_127b - profile.spi_us_at_15b)
             / (127 - 15))
    spi_us = profile.spi_us_at_15b + (frame_bytes - 15) * slope
    airtime = shr_us + phr_us + data_us
    return DerivedPhy(
        profile=profile.name, frame_bytes=int(frame_bytes),
        preamble_symbol_ns=t_psym, data_symbol_ns=t_dsym,
        phr_symbol_ns=t_phr, shr_us=shr_us, phr_us=phr_us, data_us=data_us,
        airtime_us=airtime, payload_bits=bits,
        slot_min_us=airtime + spi_us + profile.slot_other_us,
        acquisition_ns=profile.pac_symbols * t_psym,
        # [spec] UM 3.3: os 9 bits baixos do delayed TX sao ignorados ->
        # 4 / 499,2 MHz; UM "Clock Periods": tick de 1/63,8976 GHz
        delayed_tx_granularity_ns=4.0 / CHIP_MHZ * 1e3,
        timestamp_resolution_ps=1e6 / (128 * CHIP_MHZ),
    )


def frame_airtime_us(profile, frame_bytes):
    return derive_phy(profile, frame_bytes).airtime_us


def _per_margin_db(per, width_db):
    """SINR acima do ponto de 50% em que a curva logistica do modelo
    (decode_success_prob) atinge sucesso 1 - per."""
    k = 2.0 * math.log(9.0) / width_db
    return math.log((1.0 - per) / per) / k


def equivalent_noise_figure_db(profile):
    """[inferido] Figura de ruido + perda de implementacao que faz o enlace
    no limite de sensibilidade do datasheet decodificar com 1 - PER na curva
    de erro do modelo (REQ_SNR_DB e PER_WIDTH_DB de referencia do perfil):
        S = -174 + 10 log10(Rb) + NF + REQ + margem(PER)."""
    req = profile.sensitivity_req_snr_db + _per_margin_db(
        profile.sensitivity_per, profile.sensitivity_per_width_db)
    n0_rb = KT_DBM_HZ + 10.0 * math.log10(profile.data_rate_mbps * 1e6)
    return profile.sensitivity_dbm - n0_rb - req


def derived_params(profile):
    """Valores que o simulador toma do perfil (nomes de PARAM_NAMES)."""
    if profile.fixed:
        return dict(profile.fixed)
    phy = derive_phy(profile, 0)
    nf = equivalent_noise_figure_db(profile)
    return {
        # potencia capturada na banda do receptor com PSD fixa
        "TX_POWER_DBM": (profile.tx_psd_dbm_mhz
                         + 10.0 * math.log10(profile.rx_bandwidth_mhz)
                         + profile.antenna_gain_db),
        # Friis a 1 m na frequencia central
        "PL0_DB": 20.0 * math.log10(4.0 * math.pi * profile.center_freq_mhz
                                    * 1e6 / C_M_PER_S),
        "NOISE_DBM": (KT_DBM_HZ
                      + 10.0 * math.log10(profile.noise_bandwidth_mhz * 1e6)
                      + nf),
        "PROC_GAIN_DB": 10.0 * math.log10(profile.noise_bandwidth_mhz
                                          / profile.data_rate_mbps),
        "PREAMBLE_SYMBOL_NS": phy.preamble_symbol_ns,
        "CAPTURE_TIME_NS": phy.acquisition_ns,
        "SYNC_WINDOW_NS": phy.data_symbol_ns,
    }


def radiated_power_dbm(profile):
    """Potencia total radiada sob a mascara, na banda da mascara (relatorio)."""
    return profile.tx_psd_dbm_mhz + 10.0 * math.log10(
        profile.mask_bandwidth_mhz)


def profile_summary(profile, data_frame_bytes, short_frame_bytes):
    """Resumo para a configuracao efetiva e o manifesto."""
    return {
        "profile": profile.name,
        "description": profile.description,
        "fields": {k: v for k, v in asdict(profile).items() if k != "fixed"},
        "fixed_legacy_values": dict(profile.fixed),
        "equivalent_noise_figure_db": (None if profile.fixed
                                       else equivalent_noise_figure_db(
                                           profile)),
        "radiated_power_dbm": radiated_power_dbm(profile),
        "data_frame": derive_phy(profile, data_frame_bytes).as_dict(),
        "short_frame": derive_phy(profile, short_frame_bytes).as_dict(),
    }
