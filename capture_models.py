# -*- coding: utf-8 -*-
"""
Modelos de captura: probabilidade de decodificar o quadro travado diante dos
concorrentes.

  LegacyStepCapture   ('legacy_async_weight'): o F28 -- SINR contra o ruido,
                      a interferencia dentro de um simbolo de dados com peso
                      integral e a de fora com um peso unico (0,06,
                      calibrado uma vez no T do Crystal). Preservado para
                      reproducao; nao e o modelo empirico.
  EmpiricalUwbCapture ('empirical_uwb'): tabelas MEDIDAS com o radio do
                      CLOVES (canal 4, PRF 64, codigo 17), cada ponto com
                      fonte e figura, interpolacao linear, e fora da regiao
                      medida uma faixa [inferior, superior] marcada como
                      extrapolacao.

EmpiricalUwbCapture, termo a termo (hipoteses marcadas):
  P = PER(SINR_sync) x F_mesmo x F_assinc
  PER(SINR_sync)  curva de erro do motor contra ruido + vazamento + quadros
                  DIFERENTES que chegam dentro de 1 simbolo de dados do
                  travado, com peso integral: os pulsos caem nas mesmas
                  posicoes de salto do sinal util [derivado, F17].
  F_mesmo         se o quadro travado tem uma 2a copia identica dentro da
                  janela de 128 ns: quadro curto -> Fig. 8 da EWSN 2020 pelo
                  desalinhamento; quadro longo -> Fig. 9 pela diferenca de
                  potencia; comprimento intermediario -> interpolacao linear
                  em bytes [hipotese, marcada].
  F_assinc        k = 1 + numero de quadros diferentes (ou copias fora da
                  janela) que chegam FORA de 1 simbolo e seriam decodificaveis
                  sozinhos: Fig. 13 do SECON 2019 (Delta-t ate 20 us) em
                  funcao de k. Concorrentes abaixo da sensibilidade nao contam
                  [hipotese]. A potencia dos assincronos nao entra: no SECON
                  (Fig. 11) o quadro travado passa com um concorrente mais
                  forte ou mais fraco [hipotese sustentada por essa medicao].
                  k > 9: faixa; 'extrapolation' escolhe o limite.
                  REGIME NAO MEDIDO: com TODOS os assincronos dentro da
                  janela de aquisicao (o atraso <= 1 us dos CTX), nenhuma
                  medicao isola o efeito. 'intra_acquisition' escolhe o
                  limite: 'async_table' (padrao) = a curva de ate 20 us, o
                  MELHOR caso medido; 'sync_table' = a curva medida com
                  pacotes sincronizados, o PIOR caso medido (em cada k, o
                  menor dos dois valores medidos); 'no_penalty' =
                  limite teorico (esses concorrentes nao atrapalham), sem
                  medicao nenhuma [sensibilidade].
  Fatores independentes [hipotese].
"""

import math
import sys
from dataclasses import dataclass


def _pav(points, increasing):
    """Regressao isotonica (pool adjacent violators), pesos iguais."""
    blocks = [[y, 1, [x]] for x, y in points]
    sign = 1 if increasing else -1
    i = 0
    while i < len(blocks) - 1:
        if sign * (blocks[i][0] - blocks[i + 1][0]) > 1e-12:
            a, b = blocks[i], blocks[i + 1]
            n = a[1] + b[1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / n, n, a[2] + b[2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return [(x, blk[0]) for blk in blocks for x in blk[2]]


class MeasuredTable:
    """Pontos medidos (x, y) de uma figura, com proveniencia."""

    def __init__(self, raw_points, source, figure, uncertainty, x_unit,
                 monotone, right="saturate"):
        self.raw_points = tuple((float(x), float(y)) for x, y in raw_points)
        self.source = source
        self.figure = figure
        self.uncertainty = uncertainty
        self.x_unit = x_unit
        self.right = right
        pts = list(self.raw_points)
        if monotone:
            pts = _pav(pts, monotone == "increasing")
        self.points = tuple(pts)
        self.isotonic_adjusted = self.points != self.raw_points

    def _interp(self, x):
        pts = self.points
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                if x1 == x0:
                    return y0
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        raise ValueError(x)

    def value(self, x, band=False, mode="mid"):
        """(y, status) ou, com band, (y, status, inferior, superior).
        status: 'medido' | 'interpolado' | 'saturado' | 'extrapolacao'."""
        xs = [p[0] for p in self.points]
        if x < xs[0]:
            y, status, lo, hi = self.points[0][1], "saturado", None, None
        elif x <= xs[-1]:
            y = self._interp(x)
            status = "medido" if x in xs else "interpolado"
            lo = hi = None
        elif self.right == "band":
            hi = self.points[-1][1]
            (xa, ya), (xb, yb) = self.points[-4], self.points[-1]
            slope = (yb - ya) / (xb - xa)
            lo = max(0.0, min(hi, yb + slope * (x - xb)))
            y = {"lower": lo, "upper": hi, "mid": 0.5 * (lo + hi)}[mode]
            status = "extrapolacao"
        else:
            y, status, lo, hi = self.points[-1][1], "saturado", None, None
        if band:
            return (y, status, y if lo is None else lo,
                    y if hi is None else hi)
        return y, status


# EWSN 2020 (Lobba et al.), Fig. 8: 2 TX sincronizados a 1 m do RX, mesma
# potencia, quadro curto de 15 B, desalinhamento da TX (digitalizado).
SAME_SHORT_DT = MeasuredTable(
    [(0, 0.970), (2, 0.990), (6, 1.000), (16, 0.997), (32, 0.990),
     (40, 1.000), (50, 0.990), (60, 0.993), (62, 0.927), (64, 0.993),
     (80, 0.990), (96, 0.993), (110, 0.980), (120, 0.980), (128, 0.993)],
    source="Lobba et al., EWSN 2020, sec. 4.3",
    figure="Fig. 8 (short 15 B frame)", uncertainty="+-0,005 (digitalizado)",
    x_unit="ns", monotone=None)
# EWSN 2020, Fig. 9: 2 TX de quadros longos (127 B), 4 RX a mesma distancia,
# atenuacao relativa de TX 0-8 dB; PRR medio dos 4 RX (digitalizado).
SAME_LONG_DP = MeasuredTable(
    [(0, 0.05), (1, 0.04), (2, 0.03), (3, 0.13), (4, 0.44), (5, 0.72),
     (6, 0.84), (7, 0.99), (8, 0.99)],
    source="Lobba et al., EWSN 2020, sec. 4.3",
    figure="Fig. 9 (Average, long packets)",
    uncertainty="+-0,02 (digitalizado)", x_unit="dB",
    monotone="increasing", right="saturate")
# SECON 2019 (Vecchia et al.), Fig. 13: receptor unico, fontes somadas da
# mais proxima para a mais distante num corredor, pacotes diferentes com
# atraso aleatorio ate 20 us ("Different Packet + Delta-t"); alturas das
# barras lidas do vetor do PDF (erro < 0,1 ponto percentual).
DIFF_ASYNC_K = MeasuredTable(
    [(2, 0.9987), (3, 0.9893), (4, 0.9285), (5, 0.9062), (6, 0.8667),
     (7, 0.8246), (8, 0.8259), (9, 0.8265)],
    source="Vecchia et al., SECON 2019, sec. V-E",
    figure="Fig. 13 (Different Packet + Delta-t)",
    uncertainty="< 0,001 (vetor do PDF)", x_unit="fontes",
    monotone="decreasing", right="band")
# Mesma figura, para VALIDACAO (nao entram no modelo): pacotes diferentes
# sincronizados e o mesmo pacote.
DIFF_SYNC_K = MeasuredTable(
    [(2, 1.0), (3, 0.8504), (4, 0.6685), (5, 0.6582), (6, 0.5747),
     (7, 0.5730), (8, 0.4603), (9, 0.4663)],
    source="Vecchia et al., SECON 2019, sec. V-E",
    figure="Fig. 13 (Different Packet)", uncertainty="< 0,001 (vetor)",
    x_unit="fontes", monotone=None)
# A curva sincronizada como LIMITE PESSIMISTA do regime dentro da aquisicao
# (CAPTURE_INTRA_ACQUISITION = "sync_table"): os mesmos pontos, monotonos
# (PAV junta 8-9 fontes, declarado) e com faixa acima de 9 fontes, como a
# assincrona. A tabela acima fica intacta para a validacao.
DIFF_SYNC_K_BOUND = MeasuredTable(
    DIFF_SYNC_K.raw_points, source=DIFF_SYNC_K.source,
    figure=DIFF_SYNC_K.figure + ", limite pessimista",
    uncertainty=DIFF_SYNC_K.uncertainty, x_unit="fontes",
    monotone="decreasing", right="band")
INTRA_ACQUISITION_MODES = ("async_table", "sync_table", "no_penalty")
SAME_K = MeasuredTable(
    [(2, 1.0), (3, 1.0), (4, 0.9998), (5, 1.0), (6, 1.0), (7, 0.9998),
     (8, 0.9996), (9, 0.9920)],
    source="Vecchia et al., SECON 2019, sec. V-E",
    figure="Fig. 13 (Same Packet)", uncertainty="< 0,001 (vetor)",
    x_unit="fontes", monotone=None)

SHORT_FRAME_REF_BYTES = 15      # comprimento da Fig. 8
LONG_FRAME_REF_BYTES = 127      # comprimento da Fig. 9
SAME_WINDOW_NS = 128.0          # extensao medida na Fig. 8
LENGTH_TOLERANCE_BYTES = 2      # 126 B (Cupid) conta como o 127 B medido


@dataclass(frozen=True)
class CaptureContext:
    """O que o receptor 've' do quadro travado e dos concorrentes."""
    frame_bytes: int
    signal_mw: float
    noise_mw: float            # ruido efetivo + vazamento de outros canais
    sync_interf_mw: float      # quadros diferentes dentro de 1 simbolo
    async_interf_mw: float     # potencia dos que chegam fora de 1 simbolo
    same_copy_dt_ns: object    # 2a copia identica na janela (None: nao ha)
    same_copy_dp_db: object
    async_senders: int         # assincronos decodificaveis sozinhos
    acquisition_ns: float      # janela de aquisicao (PAC)
    max_async_dt_ns: object    # maior atraso entre os assincronos


def _engine_per(sinr_db):
    engine = sys.modules.get("cupid_sim_v2")
    if engine is None:
        import cupid_sim_v2 as engine
    return engine.decode_success_prob(sinr_db)


def _db(num, den):
    if num <= 0.0:
        return -math.inf
    if den <= 0.0:
        return math.inf
    return 10.0 * math.log10(num / den)


class CaptureModel:
    name = "abstract"

    def success_probability(self, ctx):
        """(probabilidade de decodificar, flags de hipotese usadas)."""
        raise NotImplementedError


class LegacyStepCapture(CaptureModel):
    name = "legacy_async_weight"

    def __init__(self, async_weight, per=_engine_per):
        self.async_weight = async_weight
        self.per = per

    def success_probability(self, ctx):
        interf = ctx.sync_interf_mw + self.async_weight * ctx.async_interf_mw
        return self.per(_db(ctx.signal_mw, interf + ctx.noise_mw)), ()


class EmpiricalUwbCapture(CaptureModel):
    name = "empirical_uwb"

    def __init__(self, extrapolation="mid", per=_engine_per,
                 intra_acquisition="async_table"):
        if extrapolation not in ("lower", "mid", "upper"):
            raise ValueError("CAPTURE_EXTRAPOLATION deve ser lower, mid ou "
                             "upper")
        if intra_acquisition not in INTRA_ACQUISITION_MODES:
            raise ValueError("CAPTURE_INTRA_ACQUISITION (intra_acquisition) "
                             "deve ser " + ", ".join(INTRA_ACQUISITION_MODES))
        self.extrapolation = extrapolation
        self.per = per
        self.intra_acquisition = intra_acquisition

    def _same(self, ctx, flags):
        span = LONG_FRAME_REF_BYTES - SHORT_FRAME_REF_BYTES
        w = min(1.0, max(0.0, (ctx.frame_bytes - SHORT_FRAME_REF_BYTES)
                         / span))
        tol = LENGTH_TOLERANCE_BYTES
        if ctx.frame_bytes < SHORT_FRAME_REF_BYTES - tol:
            # mais curto que o medido: vale a tabela de 15 B [hipotese]
            flags.append("comprimento_abaixo_do_medido")
        elif (SHORT_FRAME_REF_BYTES + tol < ctx.frame_bytes
              < LONG_FRAME_REF_BYTES - tol):
            flags.append("comprimento_interpolado")
        f_short, _ = SAME_SHORT_DT.value(min(abs(ctx.same_copy_dt_ns),
                                             SAME_WINDOW_NS))
        f_long, status = SAME_LONG_DP.value(ctx.same_copy_dp_db)
        if w > 0.0 and status == "saturado":
            flags.append("mesmo_quadro_dp_saturado")
        return (1.0 - w) * f_short + w * f_long

    def success_probability(self, ctx):
        flags = []
        p = self.per(_db(ctx.signal_mw, ctx.noise_mw + ctx.sync_interf_mw))
        if ctx.same_copy_dt_ns is not None:
            p *= self._same(ctx, flags)
        if ctx.async_senders > 0:
            # o SECON sorteou atrasos ate 20 us; com tudo dentro da janela de
            # aquisicao o regime nao foi medido em separado e o limite usado
            # e escolhido por intra_acquisition
            inside = (ctx.max_async_dt_ns is not None
                      and ctx.max_async_dt_ns < ctx.acquisition_ns)
            mode = self.intra_acquisition if inside else "async_table"
            k = ctx.async_senders + 1
            if mode == "no_penalty":
                y, status = 1.0, "medido"   # limite teorico: nada a extrapolar
            else:
                y, status, _, _ = DIFF_ASYNC_K.value(
                    k, band=True, mode=self.extrapolation)
                if mode == "sync_table":
                    # o pior dos dois medidos em cada k: com 2 fontes a curva
                    # sincronizada (1,0) passa a assincrona (0,9987) por
                    # ruido de medicao, e o limite nunca pode ser otimista
                    ys, ss, _, _ = DIFF_SYNC_K_BOUND.value(
                        k, band=True, mode=self.extrapolation)
                    if ys < y:
                        y, status = ys, ss
            if status == "extrapolacao":
                flags.append("async_k_extrapolacao")
            if inside:
                flags.append("async_dentro_da_aquisicao")
            p *= y
        return p, tuple(flags)
