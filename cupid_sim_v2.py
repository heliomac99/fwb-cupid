# -*- coding: utf-8 -*-
"""
============================================================================
 SIMULADOR CTX/UWB v2:  Crystal  vs  Weaver  vs  Cupid  vs  Cupid-FWB
 Coleta de dados (convergecast) por Transmissao Concorrente sobre UWB.
============================================================================

O QUE MUDOU EM RELACAO A v1 (resumo; detalhes em cada secao)
-------------------------------------------------------------
 F1. Decodificacao por SINR com interferencia AGREGADA (nao mais top1-vs-top2).
 F2. Piso de ruido ABSOLUTO + ganho de processamento -> sensibilidade real.
 F3. Captura TEMPORAL do UWB (first-path lock), nao so razao de potencia.
 F4. Interferencia CONSTRUTIVA (sinais do mesmo pacote se somam) -> Glossy.
 F5. Shadowing log-normal ESTATICO por enlace + fast fading por slot.
 F6. Vizinhanca deixou de ser binaria: alcance EMERGE da sensibilidade.
 F7. Atraso de propagacao + jitter de clock (sincronizacao imperfeita).
 F8. Avanco de EXATAMENTE um hop por recepcao (nao mais "teleporte").
 F9. Crystal roda no MESMO motor de slots (floods Glossy reais).
 F10. Estruturas de slot reais: Weaver TX-RX-RX (3), Cupid TX-RX (2), com os
      grupos emitter/collector do Cupid em slots DISTINTOS (nao simultaneos).
 F11. Multiplos holders por pacote + supressao por overhearing + ACK global.
 F12. FWB: crosstalk entre canais (ACLR) + limite de canais por radio.
 F13. PDR medido com deadline REAL (orcamento de slots), nao por rateio.
 F14. Multiplas topologias por ponto, com intervalo de confianca de 95%.

CORRECOES POS-AUDITORIA (set/2026)
----------------------------------
 F18. ACK do sink FISICO e igual para todos os CTX: o sink transmite o bitmap
      no seu slot e a confirmacao so chega a quem decodifica (auditoria Item 4
      + achado P1: sem isso, o hop 1 do Cupid nunca era confirmado).
 F19. Latencia com epocas incompletas CENSURADAS, nunca descartadas; o resumo
      mostra latencia e confiabilidade juntas (auditoria Item 3).
 F20. Parametro pedido que seria truncado (ex.: N_CHANNELS=16 no plano DW1000)
      vira ERRO, salvo --allow-clamp; cada rodada grava a configuracao
      efetiva e o seu hash (auditoria Item 5).

REALISMO -- PRIORIDADE 1 (set/2026), a partir dos artigos do Cupid, Weaver,
Crystal e de Lobba et al. (EWSN 2020). Cada item tem chave propria; os
presets PRESET_2026_09_10 e PRESET_ETAPA0 reproduzem os motores anteriores.
 F21. Curva de erro suave (PER_MODEL) em vez do limiar duro de SINR.
 F22. Copias do mesmo pacote (SAME_PACKET_MODEL): interferem, como medido no
      DW1000 com quadros longos, em vez de somar numa janela de 4 ns.
 F23. ACK local do Weaver (LOCAL_ACK_MODEL): ID do ultimo pacote ouvido no
      cabecalho e suspensao temporaria com prazo L, em vez de descarte.
 F24. ACK global do Weaver (ACK_MODEL="weaver"): o sink anuncia so quando ha
      novidade e nos ociosos repropagam o bitmap, agrupado a cada Y=4.
 F25. Retransmissao no mesmo hop (SAME_HOP_RELAY): no Weaver, nos ociosos
      ajudam vizinhos do hop; no Cupid, o grupo em silencio recebe do outro.
 F26. Crystal real (CRYSTAL_MODEL="tapairs"): um S por epoca, pares TA com
      todas as fontes disputando o T, ACK que pode se perder e encerramento
      apos R pares silenciosos; CTX so transmitem depois do bootstrap.
 F27. Copias identicas nao se destroem (calibrado no Glossy em UWB); no CTX
      o quadro inclui o cabecalho, entao copias de detentores diferentes
      disputam por captura (fecha o P15).
 F28. Transmissoes dessincronizadas como nas implementacoes reais (atraso
      aleatorio de ate 1 us no Weaver/Cupid, fontes do T do Crystal com ate
      30 us de diferenca), aquisicao do preambulo por potencia e interferencia
      assincrona com peso reduzido (fecha o P10).

PRINCIPIO DE COMPARACAO JUSTA
-----------------------------
Toda a camada fisica (propagacao, ruido, captura, combinacao construtiva) e
IDENTICA e vive numa unica funcao: decode_at_receiver(). Os protocolos
diferem APENAS em (a) quais nos transmitem em qual slot, (b) em qual canal, e
(c) quais mecanismos de confirmacao/supressao possuem. Nenhum protocolo tem
atalho de fisica.
============================================================================
"""

import csv
import hashlib
import math
import os
import random
import json
import subprocess
import sys
from contextlib import contextmanager
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

import capture_models as cm
import cupid_grouping as cg
import experiment_stats as es
import radio_profiles as rp
import rng_streams as rs

C_M_PER_NS = 0.299792458          # velocidade da luz em m/ns
LN10_OVER_10 = math.log(10.0) / 10.0   # [PERF] converte dB -> fator linear

# ===========================================================================
# 1) PARAMETROS
# ===========================================================================

SEED          = 42
N_NODES       = 36
AREA_X        = 83.0
AREA_Y        = 33.0

# --- Radio UWB (IEEE 802.15.4 HRP) ------------------------------------------
# [F2] Agora o alcance NAO e um parametro arbitrario: ele EMERGE do balanco
# de enlace (potencia - perda >= sensibilidade). COMM_RANGE virou saida.
# [P16] Os numeros do radio saem de um PERFIL (radio_profiles.py). O padrao e
# o DW1000 do testbed CLOVES -- canal 4, 3993,6 MHz, PRF 64, codigo 17, PAC 8,
# 6,8 Mbps (Cupid, sec. V-A) --; 'legacy_v2' guarda os numeros do canal 5 do
# motor auditado (6,5 GHz, 500 MHz, ruido sem figura de ruido). Os parametros
# de RADIO_DERIVED_PARAMS sao DERIVADOS do perfil em _apply_radio_profile():
# pedir outro valor e um override EXPLICITO, registrado em RADIO_OVERRIDES e
# na configuracao efetiva -- nunca uma sobrescrita silenciosa.
RADIO_PROFILE = "cloves_dw1000_ch4"
RADIO_DERIVED_PARAMS = ("TX_POWER_DBM", "PL0_DB", "NOISE_DBM", "PROC_GAIN_DB",
                        "PREAMBLE_SYMBOL_NS", "CAPTURE_TIME_NS",
                        "SYNC_WINDOW_NS")
RADIO_OVERRIDES = {}      # nome -> {"derivado": x, "efetivo": y}
_RADIO = rp.derived_params(rp.get_radio_profile(RADIO_PROFILE))
DATA_FRAME_BYTES = 126    # [medido] quadro de dados do Cupid/Weaver, com o
                          # cabecalho TSM (Cupid, sec. V-A)
SHORT_FRAME_BYTES = rp.get_radio_profile(RADIO_PROFILE).short_frame_bytes
                          # [hipotese] quadro do agrupamento, "payload length
                          # = 0" (Cupid, sec. IV-B): cabecalho MAC + FCS
RANGING_SLOT_US = 460.0   # [medido] slot do agrupamento (Cupid, sec. V-B)
TX_POWER_DBM  = _RADIO["TX_POWER_DBM"]   # potencia capturada na banda do
                          # receptor: PSD de -41,3 dBm/MHz x 900 MHz no canal 4
PL0_DB        = _RADIO["PL0_DB"]         # Friis a 1 m na frequencia central
PATHLOSS_N    = 2.5       # [hipotese] expoente de perda (ambiente interno)
NOISE_DBM     = _RADIO["NOISE_DBM"]      # kTB na banda de ruido + figura de
                          # ruido equivalente inferida da sensibilidade do
                          # datasheet (Tab. 6); no legado, kTB em 500 MHz sem NF
PROC_GAIN_DB  = _RADIO["PROC_GAIN_DB"]   # 10*log10(banda de ruido / taxa)
REQ_SNR_DB    = 3.0       # SINR efetivo exigido para decodificar (no modelo
                          # "logistic" abaixo, e o SINR com 50% de sucesso)
# [F21] Curva de erro. Um receptor real nao tem limiar duro (3,0 dB decodifica,
# 2,9 dB perde): a taxa de sucesso sobe numa transicao de poucos dB. Largura
# de 2 dB (10% -> 90% de sucesso): e a do quadro de 1008 bits em BPSK, e casa
# com a medicao de quadros longos concorrentes no DW1000 (Lobba et al., EWSN
# 2020, Fig. 9: com 5 dB de diferenca ao menos 1 de 4 receptores decodifica;
# com 7 dB, todos). "threshold" reproduz o limiar duro anterior.
PER_MODEL     = "logistic"
PER_MODELS    = ("logistic", "threshold")
PER_WIDTH_DB  = 2.0
# [F17] PONTO CRITICO. O ganho de processamento do UWB vale contra o RUIDO
# TERMICO, mas NAO contra outro transmissor UWB concorrente: em transmissao
# concorrente todos os nos usam o MESMO codigo de preambulo, entao o
# interferente e despreadeado junto com o sinal util. Logo:
#
#     SINR_efetivo = P_sinal / ( I_cocanal  +  N_termico / G )
#
# A v1 nao tinha ruido nenhum e tratava tudo como razao de potencias; uma
# versao intermediaria deste simulador aplicava o ganho de processamento a
# TUDO, o que tornava a contencao quase inofensiva (P_cap ~87% mesmo com 16
# concorrentes) e fazia o Weaver ganhar do Cupid. Com o tratamento correto, a
# contencao volta a doer -- que e o fenomeno que o Cupid ataca.
NOISE_EFF_DBM = NOISE_DBM - PROC_GAIN_DB        # ruido apos despreading

# --- Captura ---------------------------------------------------------------
CAPTURE_SIR_DB   = 3.0    # [F1] razao de captura: quanto um sinal posterior
                          # precisa exceder o ja travado para re-capturar
CAPTURE_TIME_NS  = _RADIO["CAPTURE_TIME_NS"]  # [F28] PAC x simbolo de
                          # preambulo (8141 ns): janela de AQUISICAO; dentro dela
                          # a decisao e por POTENCIA (um sinal posterior rouba
                          # a trava se exceder o travado por CAPTURE_SIR_DB).
                          # O DW1000 decide a deteccao depois de acumular um
                          # PAC inteiro (8 simbolos de 1017,6 ns no testbed do
                          # Cupid), com todos os preambulos presentes; e no
                          # ranging concorrente o pacote decodificado as vezes
                          # e o do respondedor MAIS DISTANTE, embora o primeiro
                          # caminho venha do mais proximo (Corbalan e Picco,
                          # EWSN 2018, Fig. 7). O valor antigo, 40 ns ([F3],
                          # "first-path lock"), era uma suposicao sem medicao e
                          # continua no PRESET_ETAPA0.
# [F28] TRANSMISSOES DESSINCRONIZADAS DE PROPOSITO. As implementacoes reais nao
# alinham as transmissoes concorrentes em ns: o Weaver atrasa TODA transmissao
# por um valor aleatorio de ate 1 us, "roughly the duration of one preamble
# symbol", porque isso "significantly reduces the chance of collision in UWB"
# (SenSys 2020, sec. 5 e 6.1; o Cupid usa o mesmo gerenciador de slots, TSM).
# O Crystal em UWB agenda as fases com o timer de 32 kHz: no T, as fontes
# "may overlap within 30 us" (Lobba et al., EWSN 2020, sec. 5.1), enquanto os
# retransmissores de cada flood seguem o relogio do radio. A causa esta em
# Vecchia et al. (SECON 2019, sec. V-D): pacotes DIFERENTES sincronizados
# colidem (PRR < 50% com 9 fontes), mas com atraso aleatorio de ate 20 us o
# receptor trava no primeiro preambulo e fica nele (PRR > 85%); depois de
# travado, um preambulo mais forte so rouba a recepcao quando seus simbolos
# coincidem com os do travado (~136 ns a cada simbolo de ~1 us, Fig. 12).
TX_DELAY_NS = 1000.0          # atraso aleatorio U[0, x) de cada TX nos CTX
CRYSTAL_T_OFFSET_NS = 30500.0 # inicio de cada fonte no T do Crystal (1 tick
                              # do timer de 32 kHz); S e A nao tem atraso
PREAMBLE_SYMBOL_NS = _RADIO["PREAMBLE_SYMBOL_NS"]  # 508 chips a 499,2 MHz
                              # (PRF 64, UM Tab. 60): 1017,6 ns
PREAMBLE_ALIGN_NS = 136.0     # janela em que um preambulo posterior rouba a
                              # trava (SECON 2019, Fig. 12); 0 = nunca
SYNC_WINDOW_NS = _RADIO["SYNC_WINDOW_NS"]  # 1 simbolo de dados a 6,8 Mbps
                              # (64 chips, 128,2 ns): dentro dele o
                              # interferente cai nas mesmas posicoes de salto
                              # temporal do sinal util e pesa integralmente
# [Fase 6] Modelo de captura (capture_models.py). "legacy_async_weight": o
# F28 abaixo (peso unico ASYNC_INTERF_WEIGHT para quem chega fora de 1
# simbolo, calibrado uma vez). "empirical_uwb": tabelas MEDIDAS com o radio
# do CLOVES (EWSN 2020 Figs. 8-9; SECON 2019 Fig. 13), com faixa e marcacao
# fora da regiao medida; CAPTURE_EXTRAPOLATION escolhe o limite da faixa.
# A aquisicao/trava do preambulo e a mesma nos dois.
CAPTURE_MODEL = "empirical_uwb"
CAPTURE_MODELS = ("legacy_async_weight", "empirical_uwb")
CAPTURE_EXTRAPOLATION = "mid"
CAPTURE_EXTRAPOLATIONS = ("lower", "mid", "upper")
# Regime NAO MEDIDO: concorrentes fora de 1 simbolo, mas todos dentro da
# janela de aquisicao (o atraso <= 1 us do TSM). A tabela do SECON e de
# Delta-t ate 20 us. "async_table" (padrao) = essa tabela, o MELHOR caso
# medido; "sync_table" = a curva medida com pacotes sincronizados, o PIOR
# caso medido; "no_penalty" = limite teorico sem medicao. Serve para
# mostrar se uma conclusao depende do que nao foi medido.
CAPTURE_INTRA_ACQUISITION = "async_table"
CAPTURE_INTRA_ACQUISITIONS = cm.INTRA_ACQUISITION_MODES
ASYNC_INTERF_WEIGHT = 0.06    # peso da interferencia que chega fora de
                              # SYNC_WINDOW_NS do sinal travado; 1.0 =
                              # interferencia integral, como no F17.
# Calibrado no primeiro T do Crystal (calibracao_captura.py): com 35 fontes o
# sink decodifica 78% das vezes, contra ~75% medidos com todos os nos
# transmitindo (EWSN 2020, sec. 5.2, 127 B, N = 2); ate 20 fontes, 100%
# (medido: >= 95%). Bate com a conta de primeira ordem: a 6,8 Mbps com PRF de
# 64 MHz o burst tem 8 chips e ha 2 posicoes de salto por meio simbolo; um
# interferente defasado de mais de um simbolo cai na posicao que o receptor
# examina em metade das vezes (1/2) e, quando cai, traz outro trecho da
# sequencia de embaralhamento nos 8 chips (1/8): 1/16 = 0,0625. Dentro de um
# simbolo o embaralhamento e o salto coincidem e o peso e integral.
# [F22] Copias do MESMO pacote vindas de transmissores diferentes. Lobba et al.
# (EWSN 2020, sec. 4.3) mediram no DW1000: com quadros curtos as copias nao se
# destroem mesmo desalinhadas em ate 128 ns (PRR >= 98%) -- "mais interferencia
# nao destrutiva do que construtiva"; com quadros longos (127 B), duas copias
# de mesma potencia dao PRR de 0-10% (desvio de frequencia dos cristais) e so
# uma diferenca de 5-7 dB garante a recepcao. Os quadros do Cupid e do Weaver
# tem 126 B e, no CTX, o cabecalho (ACK local, bitmap) difere entre
# detentores: as copias nem sao identicas. A janela antiga de 4 ns nao tinha
# base nenhuma.
#   "nondestructive": (padrao) dentro de COMBINE_WINDOW_NS as copias nao somam
#                     nem atrapalham (vale a mais forte); fora da janela,
#                     interferem.
#   "interference"  : copias do mesmo pacote interferem como pacotes
#                     diferentes; a captura decide (o pior caso medido).
#   "additive"      : F4 antigo -- dentro da janela as potencias somam
#                     (x COMBINE_EFF, aplicado ate a copia unica).
# [F27] Calibracao: com "interference" ate o flood de UMA fonte so chegava ao
# sink em 78% das vezes, mas o Glossy em UWB teve confiabilidade perfeita no
# testbed (EWSN 2020, sec. 4.2); "nondestructive" da 100% com 1 fonte e 96%
# com 5 fontes simultaneas no T do Crystal, contra >= 0,96 medido (Fig. 12).
# O caso de quadros longos com potencia igual e o pior caso controlado. E
# no CTX o quadro inclui o cabecalho (ultimo pacote ouvido, bitmap): copias
# do mesmo relatorio vindas de detentores diferentes, com cabecalhos
# diferentes, NAO sao o mesmo quadro e disputam por captura.
SAME_PACKET_MODEL = "nondestructive"
SAME_PACKET_MODELS = ("interference", "nondestructive", "additive")
FRAME_HEADERS = True      # [F27] o cabecalho (ultimo ouvido, bitmap) faz parte
                          # do quadro nos CTX; False = copias do mesmo relatorio
                          # sao sempre o mesmo quadro (como na etapa 0)
COMBINE_WINDOW_NS = 128.0 # [F4][F22] desalinhamento medido sem perda (EWSN
                          # 2020, Fig. 8); o valor antigo era 4 ns
COMBINE_EFF      = 0.85   # eficiencia da combinacao (so no modo "additive")

# --- Canal -----------------------------------------------------------------
SHADOW_DB     = 4.0       # [F5] shadowing log-normal ESTATICO por enlace
FAST_FADE_DB  = 4.0       # [F5] fast fading por pacote (NLOS UWB, cf. [10]).
                          # ATENCAO: com valores baixos (<3 dB) e shadowing
                          # estatico, o modelo entra em DEADLOCK DE CAPTURA --
                          # dois nos equidistantes nunca diferem pelos 3 dB
                          # exigidos e a epoca nunca termina. Isso e fisica
                          # correta (ver cenario 'canal bom' do artigo), mas
                          # exige o mecanismo de sorteio abaixo para nao
                          # travar a simulacao.
# [Canal com memoria] Estrutura no TEMPO do desvanecimento (a variancia num
# instante continua FAST_FADE_DB). "iid" (padrao): cada slot redesenha o
# desvanecimento de cada enlace, sem memoria -- o bootstrap de cada epoca ve
# um canal novo, e o hop dos nos muda muito entre epocas. "two_scale": uma
# fracao FADING_SLOW_FRACTION da variancia e LENTA -- constante na epoca e
# reciproca, com correlacao FADING_EPOCH_CORR entre epocas seguidas (AR(1);
# o periodo de agrupamento do Cupid e a epoca -1) --, e o resto segue por
# slot. Aproximacao de um processo continuo em duas escalas [hipotese]; os
# tres valores sao de sensibilidade: nenhuma fonte fixa a coerencia do canal
# no testbed.
FADING_MODEL = "iid"
FADING_MODELS = ("iid", "two_scale")
FADING_SLOW_FRACTION = 0.5
FADING_EPOCH_CORR = 0.9
_SLOW_CACHE = {}          # (SEED, topologia, n, rho) -> {epoca: matriz}
_SLOW_FADE = None         # parte lenta da epoca em curso (slow_fading_epoch)
CLOCK_JITTER_NS = 2.0     # [F7] erro de sincronizacao entre nos (CTX)

# [F16] Participacao aleatoria na transmissao. Implementacoes reais de CTX nao
# fazem todos os detentores transmitirem em toda oportunidade: ha diversidade
# de retransmissao. Sem isso, com shadowing estatico, um conjunto de nos que
# colide uma vez colide PARA SEMPRE (a v1 escondia esse efeito re-sorteando
# todo o fading a cada slot). Aplicado IDENTICAMENTE a todos os protocolos.
TX_PROB       = 1.0       # prob. de um detentor transmitir na sua oportunidade.
                          # [Fase 5] 1,0 no CLOVES: o Weaver real transmite em
                          # toda oportunidade (o atraso do TSM desloca a TX, nao
                          # a cancela) e o impasse de captura se desfaz pelo
                          # atraso sorteado a cada slot [F28]. 0,85 fica no
                          # preset legado e na sensibilidade.
RANDOM_QUEUE_HEAD = True  # [F16] evita head-of-line blocking deterministico:
                          # um no com varios pacotes sorteia qual enviar

# --- Confirmacao global (ACK do sink) ----------------------------------------
# [F18] So o sink sabe o que ja chegou. Os nos do hop 1 nunca recebem o ACK
# implicito (o hop 0 e o sink, que nao repassa pacote), entao sem um ACK do
# sink eles retransmitem relatorios entregues ate o fim da epoca. Modelos:
#   "physical": o sink transmite um quadro de ACK (bitmap dos entregues) no slot
#               do hop 0, pela MESMA regra de escalonamento e de participacao
#               (TX_PROB) de qualquer no. A recepcao passa por
#               decode_at_receiver() e concorre com os dados do slot. Todo
#               quadro leva no cabecalho o bitmap que o transmissor conhece
#               ("carona"), e o bitmap so passa adiante em quadro DECODIFICADO.
#               Igual para Weaver, Cupid e Cupid FWB.
#   "legacy"  : comportamento anterior (auditoria Item 4): so o Weaver tem ACK,
#               e ele e um ORACULO -- purga todas as filas a cada 3 slots, sem
#               custo, alcance ou falha. Serve so para reproduzir numeros
#               antigos e para comparacao A/B.
#   "oracle"  : oraculo para todos os CTX, no slot do sink (limite superior).
#   "none"    : nenhum protocolo tem ACK global (so o ACK implicito).
#   "weaver"  : (padrao) [F24] G-ACK como no artigo do Weaver (sec. 4): o sink
#               transmite o bitmap quando recebe pacote novo -- ou repetido,
#               sinal de que alguem perdeu o ACK --; nos SEM dado repropagam o
#               bitmap num quadro sem carga quando tem bits novos, uma vez por
#               periodo de agrupamento (GACK_BATCH_Y oportunidades de TX); quem
#               recebe um pacote ja confirmado volta a anunciar o bitmap.
# CONFIRMADO NO ARTIGO DO CUPID (sec. V-A): ele usa os mesmos dois ACKs do
# Weaver, "configured identically".
ACK_MODEL = "weaver"
ACK_MODELS = ("weaver", "physical", "legacy", "oracle", "none")
ACK_PKT = -1              # [F18] id do quadro de ACK (originador e sempre >= 1)
GACK_PKT_BASE = -1000     # [F24] quadro sem carga do no v: id GACK_PKT_BASE - v

# --- Confirmacao local (L-ACK) -----------------------------------------------
# [F23] Weaver (Trobinger et al., SenSys 2020, sec. 4) -- e o Cupid, que usa
# "two types of ACKs ... configured identically" (Park et al., DCOSS-IoT 2025,
# sec. V-A): "Upon packet TX, nodes embed the initiator ID of their last heard
# packet in the Weaver header". Quem recebe esse ID de um no mais perto do
# sink SUSPENDE aquele pacote por L = 2(h-2) + h + D slots (eq. 4) e volta a
# transmiti-lo se o ACK global nao chegar nesse prazo.
#   "weaver"     : (padrao) cabecalho com o ultimo pacote ouvido + suspensao
#                  temporaria; ouvir o proprio pacote vindo de cima tambem
#                  suspende.
#   "overhearing": F11 -- so ouvir o proprio pacote vindo de cima, e a copia e
#                  descartada de vez.
LOCAL_ACK_MODEL = "weaver"
LOCAL_ACK_MODELS = ("weaver", "overhearing")

# [F25] Retransmissao dentro do mesmo hop. Weaver: "a node relay[s] only
# packets from nodes at the same hop distance, or higher" e "nodes without new
# data listen during TX slots to hear same-hop neighbors and help them deliver
# their packets" (SenSys 2020, sec. 3.2 e 4). No Cupid e o que o modo
# silencioso habilita: "allowing nodes within the same hop to receive data"
# (DCOSS-IoT 2025, sec. IV-C). Antes, um pacote so era adotado vindo de
# exatamente um hop abaixo (F8); o avanco maximo continua sendo de um hop.
SAME_HOP_RELAY = True
GACK_BATCH_Y = 4          # [F23] agrupamento do ACK global: Y=4 no Weaver e,
                          # "configured identically", no Cupid

# --- Roteamento ------------------------------------------------------------
ROUTING_MARGIN_DB = 3.0   # margem exigida para um enlace ser "confiavel"
                          # (define a arvore de hops via busca em largura)
MAX_HOP       = 8         # nos alem disto sao considerados sem rota

# --- Tempo -----------------------------------------------------------------
SLOT_US       = 813.0     # duracao do slot (quadro de 126 B a 6.8 Mbps)
# [F9] O Crystal e composto por rodadas de TRES floods Glossy: S (sincronizacao,
# desce do sink), T (dados, sobe) e A (confirmacao, desce). Um flood Glossy tem
# duracao FIXA -- ele nao termina cedo quando o sink recebe, porque todos os nos
# precisam retransmitir N vezes para cobrir a rede. A duracao segue a regra
# classica do Glossy:  H + N - 1  slots, com H = profundidade e N = numero de
# retransmissoes por no. Assim o custo do Crystal e DERIVADO da topologia, e nao
# um numero escolhido a mao como na v1 (que usava "6 slots por rodada" fixos e
# uma probabilidade de falha de 0.05*pendentes inventada).
CRYSTAL_FLOODS_PER_ROUND = 3   # fases S, T, A (so no modelo "sequential")
N_TX_PER_FLOOD = 2             # retransmissoes por no dentro de um flood
                               # (fase T do Crystal: N_T = 2)
# [F26] Estrutura real do Crystal (Istomin et al., SenSys 2016, sec. 5 e
# Tab. 4; confirmada no artigo do Cupid, sec. III): UM flood S no inicio da
# epoca e depois pares TA. Na fase T "all nodes with data to send become
# Glossy initiators" ao mesmo tempo e a captura decide qual chega ao sink; na
# fase A o sink inunda o ACK (positivo com o ID recebido, ou negativo); quem
# nao ouviu o proprio ACK tenta de novo no proximo T. O sink encerra a epoca
# depois de R pares silenciosos seguidos. Configuracao "high power" da Tab. 4:
# N_S = N_A = 3, N_T = 2, R = 2.
#   "tapairs"   : (padrao) o protocolo descrito acima; S e cobrado (L slots)
#                 mas nao simulado; T e A (positivo) passam pela fisica.
#   "sequential": v2 anterior -- uma fonte por rodada, rodada = 3 floods, A
#                 nunca falha e o originador e abandonado apos 3 falhas.
CRYSTAL_MODEL = "tapairs"
CRYSTAL_MODELS = ("tapairs", "sequential")
N_TX_SYNC_ACK = 3              # [F26] N_S = N_A
CRYSTAL_R = 2                  # [F26] pares silenciosos que encerram a epoca
# [F26] Bootstrap dos CTX: "the nodes h hops away from the sink must wait h
# slots before they can TX data, to first receive the epoch bootstrap packet"
# (Weaver, sec. 3.3). Antes todos transmitiam desde o slot 0.
BOOTSTRAP_DELAY = True
# [P-boot] Bootstrap FISICO por epoca (Cupid, sec. IV-A: "Nodes determine their
# logical hop from the relay count of received synchronization packets"; o hop
# "can vary between epochs"). O sink inicia a onda de sync e cada no
# sincronizado a retransmite BOOTSTRAP_TX vezes, a primeira no slot seguinte e
# as demais a cada BOOTSTRAP_RELAY_PERIOD slots (Weaver, sec. 3.3 e Tab. 1:
# B = 2 copias em slots TX consecutivos do TX-RX-RX; o Cupid usa o mesmo
# processo, "configured identically", sec. V-A). Tudo passa por
# decode_physical_channels(). O hop e o do primeiro sync aceito; quem nao
# decodifica nenhum fica INATIVO na epoca (premissa da especificacao; o Weaver
# deixaria o no reusar o hop anterior). O no transmite a partir do slot
# seguinte ao do sync, o que, sem perdas, e a regra "o hop h espera h slots".
# No Crystal o bootstrap e o proprio flood S (N_TX_SYNC_ACK retransmissoes em
# slots consecutivos). "bfs" = motor legado: hop fixo da BFS da potencia media
# e a regra BOOTSTRAP_DELAY. A BFS continua na topologia so como diagnostico.
BOOTSTRAP_MODEL = "physical"
BOOTSTRAP_MODELS = ("physical", "bfs")
BOOTSTRAP_TX = 2               # [medido] B do Weaver (Tab. 1)
BOOTSTRAP_RELAY_PERIOD = 3     # [spec] slots TX consecutivos no TX-RX-RX
# [P-group] Agrupamento emitter/collector do Cupid (sec. IV-B, Alg. 1): um
# PERIODO de agrupamento antes da coleta, com M iteracoes de bootstrap +
# ranging concorrente (cupid_grouping.run_grouping_period), e em cada epoca de
# dados o grupo sai da comparacao do hop virtual com o hop DESSA epoca.
#   "ranging": (padrao) o que o artigo descreve, com a fisica do motor;
#   "static" : legado -- grupo fixo da topologia, pela distancia VERDADEIRA
#              ao hop de cima x ao de baixo da BFS (um oraculo).
GROUPING_MODEL = "ranging"
GROUPING_MODELS = ("ranging", "static")
GROUPING_ITERATIONS = 10       # [medido] M (Cupid, sec. V-B)
GROUPING_BOOTSTRAP_SLOTS = 10  # [hipotese] "bootstrapping iteration is set to
                               # 10" (sec. V-B), lido como 10 slots
GROUPING_WAIT_SLOTS = 1        # [hipotese] dT_wait = 1 slot de ranging (o
                               # artigo nao informa; 3 slots por no)
RANGING_CLOCK_PPM = 3.0        # [spec][sens] desvio maximo do cristal de cada
                               # no: "the EVB1000 platform we use, factory-
                               # trimmed at 3 ppm" (EWSN 2020, sec. 4.3) -- a
                               # placa do CLOVES. Varrido em 0-20 ppm (a
                               # tolerancia do DW1000 sem trim e +-20 ppm)
RANGING_LDE_SIGMA_NS = 0.33    # [hipotese] ruido da deteccao do primeiro
                               # caminho (~10 cm, "precision of 10 cm" do DS)
GROUPING_RENEW = False         # refaz o periodo antes de cada epoca de dados
                               # (opcao explicita; o artigo agrupa uma vez)

# [Fase 5] Grade de slots do Cupid. "fig5b": a da Fig. 5b do artigo (lida
# celula a celula; sec. IV-C: "the order differs between even-numbered and
# odd-numbered hops within the same group") -- no hop impar o emitter usa o
# primeiro slot TX, no par o collector -- e o sink transmite em TODO slot TX
# do hop 0 (Fig. 5b; Fig. 6c: ACK do sink nos slots 2, 4, 6). "legacy": a
# grade antiga, com a mesma ordem em todos os hops e o sink no grupo emitter
# (1 a cada 4 slots; era a premissa P14).
CUPID_SCHEDULE = "fig5b"
CUPID_SCHEDULES = ("fig5b", "legacy")
# [Fase 5] Fases do Crystal com o airtime do proprio quadro: S e A levam
# quadros curtos (CRYSTAL_SA_FRAME_BYTES), T o quadro de dados; o slot de cada
# fase sai do modelo de slot da EWSN 2020 (Tab. 1-2: 404 us com 15 B, 806 us
# com 127 B) -- no T, o slot de dados SLOT_US. "uniform" = legado (todo slot
# com SLOT_US).
CRYSTAL_PHASE_TIMING = "airtime"
CRYSTAL_PHASE_TIMINGS = ("airtime", "uniform")
CRYSTAL_SA_FRAME_BYTES = 15    # [hipotese] quadro de sync/ACK curto; o
                               # tamanho medido com slot na EWSN 2020 e 15 B
# [Fase 5] Flood A: "physical" simula tambem o A NEGATIVO (T mudo), que o
# sink inunda do mesmo jeito; "positive_only" = legado (o A negativo so e
# cobrado em tempo).
CRYSTAL_A_MODEL = "physical"
CRYSTAL_A_MODELS = ("physical", "positive_only")

# --- Aleatoriedade (Fase 7) ---------------------------------------------------
# "keyed": (padrao) cada slot tem um gerador proprio, chaveado pela raiz da
#          epoca e pelo slot, e os sorteios sao indexados por no (fading por
#          par tx-rx, decodificacao por rx-canal, atraso de TX, participacao,
#          fila) -- rng_streams.py. Uma transmissao a mais nao desloca nenhum
#          outro sorteio, e protocolos diferentes veem os mesmos numeros no
#          mesmo slot. Bootstrap, S do Crystal e agrupamento tem streams
#          proprios pela chave.
# "sequential": legado -- um unico rng consumido em sequencia pela epoca.
RNG_MODEL = "keyed"
RNG_MODELS = ("keyed", "sequential")

# --- Canais UWB / FWB ------------------------------------------------------
# [F15] REALIDADE DOS CANAIS UWB. Dizer "16 canais" e repetir a contagem
# nominal do IEEE 802.15.4, e ela NAO corresponde a 16 canais utilizaveis
# simultaneamente. A contagem cai em quatro estagios:
#
#   (a) O padrao define 16 canais (0..15), mas os canais 4, 7, 11 e 15 sao
#       canais LARGOS (1082-1355 MHz) que SE SOBREPOEM a canais de 500 MHz
#       vizinhos. Nao sao ortogonais aos vizinhos -> restam 12.
#   (b) Regulacao. Na Europa (ETSI EN 302 065) so ha operacao sem mitigacao
#       em 6.0-8.5 GHz -> sobram os canais 5, 6, 8, 9 = 4 canais.
#       Nos EUA a FCC libera 3.1-10.6 GHz, mas 3.1-4.8 GHz exige DAA/LDC.
#   (c) Hardware. O DW1000 (o radio do testbed CLOVES, usado pelo Cupid)
#       suporta apenas os canais 1,2,3,4,5,7 -> 4 nao sobrepostos (1,2,3,5).
#       O DW3000/DW3110, o radio UWB atual (celulares, tags), suporta
#       SOMENTE os canais 5 e 9 -> 2 canais.
#   (d) Simultaneidade. Um DW1000/DW3000 tem UM front-end: ele recebe UM canal
#       por vez, e trocar de canal custa da ordem de ms -- MAIOR que o slot de
#       813 us. Receber k canais ao mesmo tempo exige k RADIOS por no.
#
# Conclusao: K=16 com recepcao simultanea nao existe em nenhum hardware UWB
# disponivel. O valor defensavel para um artigo e K=2 (DW3000) ou K=4
# (DW1000 / faixa europeia), com 1 radio por no salvo hardware customizado.
CHANNEL_PLANS = {
    "ieee_nominal":   {"K": 16, "desc": "IEEE 802.15.4 nominal (inclui os "
                                        "canais largos sobrepostos) - IRREAL"},
    "ieee_orthogonal": {"K": 12, "desc": "canais de 500 MHz nao sobrepostos "
                                         "(exclui 4, 7, 11, 15)"},
    "fcc_practical":  {"K": 8,  "desc": "EUA, evitando a faixa com DAA/LDC"},
    "etsi_eu":        {"K": 4,  "desc": "Europa 6.0-8.5 GHz (ch 5, 6, 8, 9)"},
    "dw1000":         {"K": 4,  "desc": "DW1000 (radio do testbed CLOVES): "
                                        "ch 1, 2, 3, 5 nao sobrepostos"},
    "dw3000":         {"K": 2,  "desc": "DW3000/DW3110 (radio UWB atual): "
                                        "ch 5 e 9 -- o caso realista hoje"},
    # [Fase 10] com o canal 4 (largo, 3328-4659 MHz) em uso, os canais 1, 2 e
    # 3 se sobrepoem a ele (DS 5.1.1); so o 5 ou o 7 ficam livres
    "dw1000_ch4":     {"K": 2,  "desc": "DW1000 com o canal 4 do CLOVES: so "
                                        "o 5 (ou o 7) nao se sobrepoe"},
}
CHANNEL_PLAN  = "dw1000"  # plano vigente; limita N_CHANNELS ao K viavel
N_CHANNELS    = 4         # total de canais a distribuir em funil (sera
                          # truncado pelo CHANNEL_PLAN em enforce_channel_plan)
N_RADIOS_PER_NODE = 1     # [F15] radios por no. 1 = hardware real (DW1000/
                          # DW3000). >1 exige placa customizada e custa energia.
CHANNEL_SWITCH_US = 1000.0 # [F15] limite conservador de sistema para retunar
                          # um DW1000. O User Manual v2.18, sec. 7.2.32, exige
                          # reprogramar CHAN_CTRL, RF_RXCTRLH, RF_TXCTRL,
                          # TC_PGDELAY e FS_PLLCFG ao mudar de canal; o driver
                          # oficial faz a reconfiguracao completa da cadeia RF.
                          # O manual nao especifica latencia de hot-switch; 1 ms
                          # e uma premissa conservadora, explicita e ajustavel.
ACLR_DB       = 30.0      # [F12] rejeicao de canal adjacente. Para UWB de
                          # 500 MHz com filtros praticos, 25-35 dB e realista;
                          # entre canais LARGOS sobrepostos e ~0 dB (por isso
                          # eles nao contam como ortogonais).
MAX_RX_CHANNELS = 1       # derivado de N_RADIOS_PER_NODE em _derived()

# --- Execucao --------------------------------------------------------------
EPOCHS        = 200
MAX_SLOTS     = 1500      # teto de slots por epoca. Precisa acomodar o pior
                          # protocolo (Crystal, ~600 slots): se cortar, as
                          # epocas incompletas somem da mediana e o protocolo
                          # lento parece artificialmente bom.
N_TOPOLOGIES  = 5         # [F14] topologias (sementes) agregadas por ponto

PROTOCOLS     = ["cupid_fwb", "cupid", "weaver", "crystal"]

PARAM_NAMES = [
    "SEED", "N_NODES", "AREA_X", "AREA_Y",
    "RADIO_PROFILE", "DATA_FRAME_BYTES", "SHORT_FRAME_BYTES",
    "RANGING_SLOT_US",
    "TX_POWER_DBM", "PL0_DB", "PATHLOSS_N", "NOISE_DBM", "PROC_GAIN_DB",
    "REQ_SNR_DB", "PER_MODEL", "PER_WIDTH_DB",
    "CAPTURE_SIR_DB", "CAPTURE_TIME_NS", "SAME_PACKET_MODEL",
    "FRAME_HEADERS", "TX_DELAY_NS", "CRYSTAL_T_OFFSET_NS",
    "PREAMBLE_SYMBOL_NS", "PREAMBLE_ALIGN_NS", "SYNC_WINDOW_NS",
    "ASYNC_INTERF_WEIGHT", "CAPTURE_MODEL", "CAPTURE_EXTRAPOLATION",
    "CAPTURE_INTRA_ACQUISITION", "COMBINE_WINDOW_NS", "COMBINE_EFF",
    "SHADOW_DB", "FAST_FADE_DB",
    "FADING_MODEL", "FADING_SLOW_FRACTION", "FADING_EPOCH_CORR",
    "CLOCK_JITTER_NS",
    "TX_PROB", "RANDOM_QUEUE_HEAD", "ACK_MODEL", "LOCAL_ACK_MODEL",
    "GACK_BATCH_Y", "SAME_HOP_RELAY",
    "ROUTING_MARGIN_DB", "MAX_HOP", "SLOT_US",
    "CRYSTAL_FLOODS_PER_ROUND", "N_TX_PER_FLOOD", "CRYSTAL_MODEL",
    "N_TX_SYNC_ACK", "CRYSTAL_R", "BOOTSTRAP_DELAY",
    "BOOTSTRAP_MODEL", "BOOTSTRAP_TX", "BOOTSTRAP_RELAY_PERIOD",
    "GROUPING_MODEL", "GROUPING_ITERATIONS", "GROUPING_BOOTSTRAP_SLOTS",
    "GROUPING_WAIT_SLOTS", "RANGING_CLOCK_PPM", "RANGING_LDE_SIGMA_NS",
    "GROUPING_RENEW", "RNG_MODEL", "CUPID_SCHEDULE",
    "CRYSTAL_PHASE_TIMING", "CRYSTAL_SA_FRAME_BYTES", "CRYSTAL_A_MODEL",
    "CHANNEL_PLAN", "N_CHANNELS", "N_RADIOS_PER_NODE", "CHANNEL_SWITCH_US",
    "ACLR_DB",
    "EPOCHS", "MAX_SLOTS", "N_TOPOLOGIES",
]


# Configuracoes de referencia para reproduzir numeros antigos (comparacao A/B).
# Cada item novo do realismo acrescenta aqui o valor que o desliga.
#
# PRESET_LEGACY_V2 e o motor auditado de 5ef525b (F28), com TODOS os valores
# fisicos e de protocolo explicitos -- nao "o padrao de hoje", que muda. O
# desenho do experimento (SEED, N_NODES, area, EPOCHS, MAX_SLOTS,
# N_TOPOLOGIES) fica de fora: e escolha de cada cenario. Reproducao conferida
# bit a bit em tests/test_legacy_preset.py.
PRESET_LEGACY_V2 = {
    "RADIO_PROFILE": "legacy_v2",
    "TX_POWER_DBM": -14.3, "PL0_DB": 48.7, "PATHLOSS_N": 2.5,
    "NOISE_DBM": -87.0, "PROC_GAIN_DB": 18.7, "REQ_SNR_DB": 3.0,
    "PER_MODEL": "logistic", "PER_WIDTH_DB": 2.0, "CAPTURE_SIR_DB": 3.0,
    "CAPTURE_TIME_NS": 8141.0, "SAME_PACKET_MODEL": "nondestructive",
    "FRAME_HEADERS": True, "TX_DELAY_NS": 1000.0,
    "CRYSTAL_T_OFFSET_NS": 30500.0, "PREAMBLE_SYMBOL_NS": 1017.6,
    "PREAMBLE_ALIGN_NS": 136.0, "SYNC_WINDOW_NS": 128.0,
    "ASYNC_INTERF_WEIGHT": 0.06, "CAPTURE_MODEL": "legacy_async_weight",
    "CAPTURE_EXTRAPOLATION": "mid", "COMBINE_WINDOW_NS": 128.0,
    "COMBINE_EFF": 0.85, "SHADOW_DB": 4.0, "FAST_FADE_DB": 4.0,
    "CLOCK_JITTER_NS": 2.0, "TX_PROB": 0.85, "RANDOM_QUEUE_HEAD": True,
    "ACK_MODEL": "weaver", "LOCAL_ACK_MODEL": "weaver", "GACK_BATCH_Y": 4,
    "SAME_HOP_RELAY": True, "ROUTING_MARGIN_DB": 3.0, "MAX_HOP": 8,
    "SLOT_US": 813.0, "CRYSTAL_FLOODS_PER_ROUND": 3, "N_TX_PER_FLOOD": 2,
    "CRYSTAL_MODEL": "tapairs", "N_TX_SYNC_ACK": 3, "CRYSTAL_R": 2,
    "BOOTSTRAP_DELAY": True, "BOOTSTRAP_MODEL": "bfs",
    "GROUPING_MODEL": "static", "RNG_MODEL": "sequential",
    "CUPID_SCHEDULE": "legacy", "CRYSTAL_PHASE_TIMING": "uniform",
    "CRYSTAL_A_MODEL": "positive_only",
    "CHANNEL_PLAN": "dw1000", "N_CHANNELS": 4,
    "N_RADIOS_PER_NODE": 1, "CHANNEL_SWITCH_US": 1000.0, "ACLR_DB": 30.0,
}
PRESET_ETAPA0 = dict(PRESET_LEGACY_V2, **{  # motor da etapa 0 (fix/etapa0-...)
    "ACK_MODEL": "physical",
    "PER_MODEL": "threshold",
    "SAME_PACKET_MODEL": "additive", "COMBINE_WINDOW_NS": 4.0,
    "FRAME_HEADERS": False,
    "LOCAL_ACK_MODEL": "overhearing",
    "SAME_HOP_RELAY": False,
    "CRYSTAL_MODEL": "sequential", "BOOTSTRAP_DELAY": False,
    "TX_DELAY_NS": 0.0, "CRYSTAL_T_OFFSET_NS": 0.0,
    "CAPTURE_TIME_NS": 40.0, "PREAMBLE_ALIGN_NS": 0.0,
    "ASYNC_INTERF_WEIGHT": 1.0,
})
PRESET_2026_09_10 = dict(    # motor auditado de 10/09/2026, antes da etapa 0
    PRESET_ETAPA0, ACK_MODEL="legacy")


def enforce_channel_plan(verbose=True):
    """[F15] Trunca N_CHANNELS ao numero de canais que o plano vigente
    (regulacao + hardware) realmente oferece, e avisa quando o pedido excede
    a realidade. Impede que o simulador produza numeros com K=16 sem que isso
    fique explicito no log e no artigo."""
    global N_CHANNELS
    plan = CHANNEL_PLANS.get(CHANNEL_PLAN)
    if plan is None:
        raise KeyError(f"CHANNEL_PLAN desconhecido '{CHANNEL_PLAN}'. "
                       f"Validos: {', '.join(CHANNEL_PLANS)}")
    kmax = plan["K"]
    if N_CHANNELS > kmax:
        if verbose:
            print(f"  [AVISO] N_CHANNELS={N_CHANNELS} excede o plano "
                  f"'{CHANNEL_PLAN}' ({plan['desc']}), que oferece {kmax}. "
                  f"Truncando para {kmax}.")
        N_CHANNELS = kmax
    return N_CHANNELS


def requested_clamps(cfg):
    """[F20] Parametros PEDIDOS em 'cfg' que o plano de canais descartaria.

    enforce_channel_plan() trunca e so avisa no log: foi assim que o cenario
    "16 canais" rodou o baseline de 4 canais sem deixar rastro (auditoria
    Item 5, pcaps byte-identicos). Um valor so conta como pedido se estiver
    em 'cfg'; o padrao ajustado ao plano nao e erro."""
    plan_name = cfg.get("CHANNEL_PLAN", CHANNEL_PLAN)
    plan = CHANNEL_PLANS.get(plan_name)
    if plan is None:
        raise KeyError(f"CHANNEL_PLAN desconhecido '{plan_name}'. "
                       f"Validos: {', '.join(CHANNEL_PLANS)}")
    clamps = []
    if "N_CHANNELS" in cfg and int(cfg["N_CHANNELS"]) > plan["K"]:
        clamps.append({
            "param": "N_CHANNELS", "pedido": cfg["N_CHANNELS"],
            "efetivo": plan["K"],
            "motivo": f"CHANNEL_PLAN='{plan_name}' oferece K={plan['K']}",
        })
    return clamps


def effective_config():
    """[F20] Valores efetivos de todos os parametros (pos-truncamento), mais
    os overrides de parametros derivados do radio e as grandezas derivadas do
    perfil (airtime, slot minimo, figura de ruido equivalente)."""
    cfg = {name: globals()[name] for name in PARAM_NAMES}
    cfg["RADIO_OVERRIDES"] = {k: dict(v) for k, v in RADIO_OVERRIDES.items()}
    cfg["DERIVED_PHY"] = rp.profile_summary(
        rp.get_radio_profile(RADIO_PROFILE), DATA_FRAME_BYTES,
        SHORT_FRAME_BYTES)
    return cfg


def config_hash(config):
    """[F20] Impressao digital curta de uma configuracao."""
    blob = json.dumps(config, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _derived():
    """Recalcula constantes derivadas apos qualquer mudanca de parametro."""
    global NOISE_EFF_DBM, NOISE_EFF_LIN, ACLR_LIN, MAX_RX_CHANNELS, CAPTURE
    # [F17] ganho de processamento so contra o ruido termico
    NOISE_EFF_DBM = NOISE_DBM - PROC_GAIN_DB
    NOISE_EFF_LIN = 10.0 ** (NOISE_EFF_DBM / 10.0)
    ACLR_LIN = 10.0 ** (-ACLR_DB / 10.0)
    # [F15] um radio recebe um canal por vez; k canais simultaneos = k radios
    MAX_RX_CHANNELS = max(1, int(N_RADIOS_PER_NODE))
    _SLOW_CACHE.clear()   # a parte lenta depende de SEED, N_NODES e rho
    # [Fase 6] instancia do modelo empirico com o limite de extrapolacao e a
    # curva de erro DESTE modulo (sob 'python cupid_sim_v2.py' ele e
    # __main__: importar "cupid_sim_v2" daria outra copia, com os padroes)
    per = globals().get("decode_success_prob")
    CAPTURE = (cm.EmpiricalUwbCapture(
        CAPTURE_EXTRAPOLATION, per=per,
        intra_acquisition=CAPTURE_INTRA_ACQUISITION)
        if per is not None else None)


def _apply_radio_profile(explicit, previous_profile):
    """[P16] Deriva do perfil os parametros de RADIO_DERIVED_PARAMS. Um valor
    pedido em 'explicit' que difere do derivado vira override registrado;
    pedir o proprio valor derivado desfaz o override; trocar de perfil
    descarta os overrides antigos."""
    derived = rp.derived_params(rp.get_radio_profile(RADIO_PROFILE))
    if RADIO_PROFILE != previous_profile:
        RADIO_OVERRIDES.clear()
    for name in RADIO_DERIVED_PARAMS:
        if name in explicit:
            if explicit[name] == derived[name]:
                RADIO_OVERRIDES.pop(name, None)
            else:
                RADIO_OVERRIDES[name] = {"derivado": derived[name],
                                         "efetivo": explicit[name]}
        globals()[name] = (RADIO_OVERRIDES[name]["efetivo"]
                           if name in RADIO_OVERRIDES else derived[name])


_CHOICES = (("ACK_MODEL", "ACK_MODELS"), ("PER_MODEL", "PER_MODELS"),
            ("SAME_PACKET_MODEL", "SAME_PACKET_MODELS"),
            ("LOCAL_ACK_MODEL", "LOCAL_ACK_MODELS"),
            ("CRYSTAL_MODEL", "CRYSTAL_MODELS"),
            ("BOOTSTRAP_MODEL", "BOOTSTRAP_MODELS"),
            ("GROUPING_MODEL", "GROUPING_MODELS"),
            ("RNG_MODEL", "RNG_MODELS"),
            ("CUPID_SCHEDULE", "CUPID_SCHEDULES"),
            ("CRYSTAL_PHASE_TIMING", "CRYSTAL_PHASE_TIMINGS"),
            ("CRYSTAL_A_MODEL", "CRYSTAL_A_MODELS"),
            ("CAPTURE_MODEL", "CAPTURE_MODELS"),
            ("CAPTURE_EXTRAPOLATION", "CAPTURE_EXTRAPOLATIONS"),
            ("CAPTURE_INTRA_ACQUISITION", "CAPTURE_INTRA_ACQUISITIONS"),
            ("FADING_MODEL", "FADING_MODELS"))


def fresh_config(cfg, defaults):
    """Configuracao COMPLETA para uma tarefa de bancada: os padroes do
    processo, os parametros derivados do perfil pedido (sem herdar overrides
    de tarefas anteriores) e, por cima, 'cfg'. Workers de ProcessPoolExecutor
    sao reutilizados, e apply_config so troca as chaves que recebe."""
    base = {k: v for k, v in defaults.items() if k not in RADIO_DERIVED_PARAMS}
    profile = cfg.get("RADIO_PROFILE", base["RADIO_PROFILE"])
    base.update(rp.derived_params(rp.get_radio_profile(profile)))
    base.update(cfg)
    return base


def _validate(view):
    """Valida a configuracao resultante ANTES de gravar qualquer global: um
    valor invalido nunca fica preso no modulo depois do erro."""
    if "RADIO_PROFILE" in view:
        rp.get_radio_profile(view["RADIO_PROFILE"])   # ValueError
    for name, choices in _CHOICES:
        valid = globals()[choices]
        if view[name] not in valid:
            raise ValueError(f"{name} desconhecido '{view[name]}'. "
                             f"Validos: {', '.join(valid)}")
    if not 0.0 < view["ASYNC_INTERF_WEIGHT"] <= 1.0:
        raise ValueError("ASYNC_INTERF_WEIGHT deve estar em (0, 1]")
    if (view["CAPTURE_MODEL"] == "empirical_uwb"
            and view["SAME_PACKET_MODEL"] != "nondestructive"):
        raise ValueError("CAPTURE_MODEL 'empirical_uwb' define ele mesmo as "
                         "copias identicas: exige SAME_PACKET_MODEL "
                         "'nondestructive' (agrupamento sem soma)")
    for name in ("TX_DELAY_NS", "CRYSTAL_T_OFFSET_NS", "PREAMBLE_ALIGN_NS"):
        if view[name] < 0.0:
            raise ValueError(f"{name} nao pode ser negativo")
    for name in ("FADING_SLOW_FRACTION", "FADING_EPOCH_CORR"):
        if not 0.0 <= view[name] <= 1.0:
            raise ValueError(f"{name} deve estar em [0, 1]")


def apply_config(cfg):
    for key in cfg:
        if key not in PARAM_NAMES:
            raise KeyError(f"Parametro desconhecido '{key}'. "
                           f"Validos: {', '.join(PARAM_NAMES)}")
    view = {name: cfg.get(name, globals()[name]) for name in PARAM_NAMES}
    _validate(view)
    previous_profile = RADIO_PROFILE
    for key, value in cfg.items():
        globals()[key] = value
    _apply_radio_profile(cfg, previous_profile)
    _derived()
    enforce_channel_plan()
    random.seed(SEED)
    np.random.seed(SEED)

_derived()
random.seed(SEED)
np.random.seed(SEED)


def reliable_range_m():
    """[F6] Alcance confiavel DERIVADO do balanco de enlace (nao arbitrado).
    Resolve: TX - PL0 - 10n log10(d) - NOISE >= SINR_THRESH + margem."""
    budget = (TX_POWER_DBM - PL0_DB - NOISE_EFF_DBM
              - REQ_SNR_DB - ROUTING_MARGIN_DB)
    return 10.0 ** (budget / (10.0 * PATHLOSS_N))


# ===========================================================================
# 2) TOPOLOGIA E CANAL
# ===========================================================================

class Topo:
    """Guarda tudo que e fixo por topologia."""
    __slots__ = ("pos", "ps_db", "ps_lin", "tprop_ns", "hop", "group", "n",
                 "rrange", "reliable", "audible")


def build_topology(rng):
    """[F5][F6][F7] Gera posicoes, perda de percurso, shadowing ESTATICO por
    enlace, atraso de propagacao e a arvore de hops derivada da sensibilidade.

    Diferenca central para a v1: nao existe mais matriz binaria 'hears'. Quem
    ouve quem passa a ser decidido slot a slot pelo SINR. O alcance de ~30 m do
    testbed EMERGE do balanco de enlace em vez de ser imposto."""
    t = Topo()
    t.n = N_NODES
    t.rrange = reliable_range_m()
    pos = {0: (2.0, AREA_Y / 2.0)}                 # sink na extremidade
    for i in range(1, N_NODES):
        pos[i] = (rng.uniform(8.0, AREA_X), rng.uniform(0.0, AREA_Y))
    t.pos = pos

    n = N_NODES
    ps_db = np.full((n, n), -300.0)     # perda + shadowing (dBm recebidos)
    mean_db = np.full((n, n), -300.0)   # sem shadowing (para o roteamento)
    tprop = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = max(math.dist(pos[i], pos[j]), 0.5)
            base = TX_POWER_DBM - PL0_DB - 10.0 * PATHLOSS_N * math.log10(d)
            # [F5] shadowing sorteado UMA VEZ por enlace e reciproco: um no em
            # sombra CONTINUA em sombra ao longo da epoca (correlacao temporal
            # que a v1 destruia ao re-sortear o fading a cada slot).
            sh = rng.gauss(0.0, SHADOW_DB)
            ps_db[i][j] = ps_db[j][i] = base + sh
            mean_db[i][j] = mean_db[j][i] = base
            tp = d / C_M_PER_NS                    # [F7] atraso de propagacao
            tprop[i][j] = tprop[j][i] = tp
    t.ps_db = ps_db
    # [PERF] potencia linear pre-computada: evita 10**(x/10) em cada slot
    t.ps_lin = np.where(ps_db > -250.0, 10.0 ** (ps_db / 10.0), 0.0)
    # [PERF] transmissores que o receptor pode plausivelmente ouvir
    floor_lin = 10.0 ** ((NOISE_DBM - PROC_GAIN_DB) / 10.0) * 1e-2
    t.audible = [set(np.nonzero(t.ps_lin[r] > floor_lin)[0].tolist())
                 for r in range(n)]
    t.tprop_ns = tprop

    # --- Enlaces confiaveis (para a arvore de roteamento) ---
    thr = NOISE_EFF_DBM + REQ_SNR_DB + ROUTING_MARGIN_DB
    reliable = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and mean_db[i][j] >= thr:
                reliable[i].add(j)
    t.reliable = reliable

    # --- Hop logico por busca em largura sobre os enlaces confiaveis ---
    hop = {0: 0}
    frontier = [0]
    while frontier:
        nxt = []
        for u in frontier:
            for v in reliable[u]:
                if v not in hop:
                    hop[v] = hop[u] + 1
                    nxt.append(v)
        frontier = nxt
    for i in range(n):
        hop.setdefault(i, 99)
    t.hop = hop
    t.group = assign_cupid_groups(pos, hop, n)
    return t


def assign_cupid_groups(pos, hop, n):
    """Agrupamento intra-hop do Cupid (emitter / collector) por distancia
    fisica ao hop de cima vs ao hop de baixo. Inalterado em relacao a v1."""
    group = {}
    for i in range(n):
        h = hop[i]
        if h == 0 or h >= 99:
            group[i] = "emitter"
            continue
        up = [math.dist(pos[i], pos[j]) for j in range(n) if hop[j] == h - 1]
        dn = [math.dist(pos[i], pos[j]) for j in range(n) if hop[j] == h + 1]
        d_up = min(up) if up else 1e9
        d_dn = min(dn) if dn else 1e9
        group[i] = "emitter" if d_up <= d_dn else "collector"
    return group


def allocate_funnel_channels(hops, n_channels):
    """FWB: reparte os canais em funil (mais canais nos hops perto do sink).
    Piso de 1 canal por hop; sobra do arredondamento vai aos hops menores."""
    hops = sorted(hops)
    if not hops:
        return {}
    max_h = max(hops)
    w = {h: (max_h - h + 1) for h in hops}
    alloc = {h: 1 for h in hops}
    rest = int(n_channels) - len(hops)
    if rest > 0:
        tw = sum(w.values())
        exact = {h: rest * w[h] / tw for h in hops}
        for h in hops:
            alloc[h] += int(exact[h])
        sobra = rest - sum(int(exact[h]) for h in hops)
        ordem = sorted(hops, key=lambda h: (-(exact[h] - int(exact[h])), h))
        for h in ordem[:sobra]:
            alloc[h] += 1
    return alloc


# ===========================================================================
# 3) FISICA DE RECEPCAO  -- UNICA para TODOS os protocolos
# ===========================================================================

def new_holder_stats():
    """[F11] Acumulador OBSERVACIONAL da distribuicao de detentores.

    Sem fan-out real esta serie e degenerada (um detentor por pacote em todo
    slot). Com fan-out ela e o unico jeito de saber se o overhearing de fato
    suprime: se a media por pacote so cresce ate o deadline, o ACK implicito
    nao esta funcionando e isso e um achado, nao algo a limitar.
    """
    return {
        "samples": [],                  # um registro por slot percorrido
        "overhearing_suppressions": 0,  # remocoes por ACK implicito decodificado
        "post_delivery_suppressions": 0,  # idem, com o sink ja tendo o pacote
        # [F18] ACK do sink e desperdicio de transmissoes
        "sink_ack_tx": 0,               # quadros de ACK que o sink transmitiu
        "sink_ack_decodes": 0,          # quadros de ACK decodificados
        "global_ack_suppressions": 0,   # copias apagadas pelo bitmap conhecido
        "data_tx": 0,                   # transmissoes de dados
        "data_tx_delivered": 0,         # ... de relatorio ja entregue ao sink
        "data_tx_hop1": 0,              # transmissoes de dados do hop 1
        "data_tx_hop1_delivered": 0,    # ... de relatorio ja entregue ao sink
        # [F23] ACK local do Weaver
        "local_ack_suppressions": 0,    # suspensoes pelo cabecalho (L-ACK)
        "local_ack_resumes": 0,         # prazo L venceu sem ACK global
        # [F24] ACK global do Weaver repropagado por nos ociosos
        "node_gack_tx": 0,              # quadros sem carga (so bitmap)
        "node_gack_decodes": 0,
    }


def summarize_holder_stats(stats):
    """Media ponderada por pacote, maximo global e contagem de supressoes."""
    samples = stats.get("samples", [])
    total_copies = sum(s["mean_holders"] * s["packet_count"] for s in samples)
    total_pkts = sum(s["packet_count"] for s in samples)

    def frac(num, den):
        total = stats.get(den, 0)
        return stats.get(num, 0) / total if total else float("nan")

    return {
        "n_samples": len(samples),
        "mean_holders": (total_copies / total_pkts
                         if total_pkts else float("nan")),
        "max_holders": max((s["max_holders"] for s in samples), default=0),
        "overhearing_suppressions": stats.get("overhearing_suppressions", 0),
        "post_delivery_suppressions": stats.get(
            "post_delivery_suppressions", 0),
        "sink_ack_tx": stats.get("sink_ack_tx", 0),
        "sink_ack_decodes": stats.get("sink_ack_decodes", 0),
        "global_ack_suppressions": stats.get("global_ack_suppressions", 0),
        "local_ack_suppressions": stats.get("local_ack_suppressions", 0),
        "local_ack_resumes": stats.get("local_ack_resumes", 0),
        "node_gack_tx": stats.get("node_gack_tx", 0),
        "node_gack_decodes": stats.get("node_gack_decodes", 0),
        "wasted_tx_fraction": frac("data_tx_delivered", "data_tx"),
        "wasted_tx_fraction_hop1": frac("data_tx_hop1_delivered",
                                        "data_tx_hop1"),
        "samples": samples,
    }


def _sample_holders(holder_stats, queues, n, slot):
    """Conta copias por pacote no inicio do slot. Nao sorteia nada."""
    counts = defaultdict(int)
    for v in range(n):
        for p in queues[v]:
            counts[p] += 1
    k = len(counts)
    holder_stats["samples"].append({
        "slot": slot,
        "packet_count": k,
        "mean_holders": (sum(counts.values()) / k) if k else 0.0,
        "max_holders": max(counts.values(), default=0),
    })


def new_phy_stats():
    """Cria telemetria opcional para combinacao real de multiplos sinais."""
    return {
        "n_groups_singleton": 0,
        "n_groups_multi": 0,
        "comb_signals": 0,
        "resid_signals": 0,
        "comb_power": 0.0,
        "resid_power": 0.0,
        # [Fase 6] modelo empirico: avaliacoes e hipoteses usadas
        "capture_evaluations": 0,
        "capture_flags": {},
    }


def summarize_phy_stats(stats):
    """Resume apenas grupos com >=2 transmissores do mesmo pacote."""
    comb = stats.get("comb_signals", 0)
    resid = stats.get("resid_signals", 0)
    comb_power = stats.get("comb_power", 0.0)
    resid_power = stats.get("resid_power", 0.0)
    total_power = comb_power + resid_power
    return {
        "n_groups_singleton": stats.get("n_groups_singleton", 0),
        "n_groups_multi": stats.get("n_groups_multi", 0),
        "comb_signals": comb,
        "resid_signals": resid,
        "total_signals": comb + resid,
        "comb_power": comb_power,
        "resid_power": resid_power,
        "total_power": total_power,
        "comb_fraction": (comb_power / total_power
                          if total_power else float("nan")),
        "resid_fraction": (resid_power / total_power
                           if total_power else float("nan")),
    }

# --- [A1] Observabilidade de recepcoes no sink ---
# Bits do campo 'flags' de cada evento de recepcao (sink_events).
SINK_FLAG_DUPLICATE = 1 << 0    # o sink ja havia decodificado este relatorio
SINK_FLAG_LATE = 1 << 1         # chegou depois do slot_budget da epoca
SINK_FLAG_SUPPRESSED = 1 << 2   # originador ja abandonado (existe so no
                                # Crystal, via fails >= 3)
SINK_FLAG_NO_ADVANCE = 1 << 3   # decodificado no sink, mas nao avanca a rota:
                                # 'audible' e muito mais largo que 'reliable',
                                # entao o sink ouve nos que nao estao a um hop.
                                # Fisicamente real; nunca conta como entrega.

PKT_TYPE_DATA = 0
PKT_TYPE_SYNC = 1
PKT_TYPE_ACK = 2


def sink_reception_event(originator, first_path_node, slot, channel,
                         t_arrival_ns, rssi_dbm, sinr_db, flags,
                         n_hops, n_retx, pkt_type=PKT_TYPE_DATA):
    """Monta um evento de recepcao no sink.

    Funcao pura: nao le nem escreve estado de simulacao e nao sorteia nada.
    """
    return {
        "slot": slot,
        "originator": originator,
        "first_path_node": first_path_node,
        "channel": channel,
        "t_arrival_ns": t_arrival_ns,
        "rssi_dbm": rssi_dbm,
        "sinr_db": sinr_db,
        "flags": flags,
        "n_hops": n_hops,
        "n_retx": n_retx,
        "pkt_type": pkt_type,
    }


def slow_fading_matrix(topo_seed, epoch, n):
    """[Canal com memoria] Parte lenta (normal padrao, reciproca) de cada
    enlace na epoca 'epoch' >= -1 da topologia: AR(1) entre epocas com
    FADING_EPOCH_CORR, a partir da epoca -1 (o periodo de agrupamento). Cada
    inovacao sai da chave (SEED, topologia, epoca): igual em qualquer ordem
    de chamada e para todos os protocolos."""
    key = (SEED, topo_seed, n, FADING_EPOCH_CORR)
    chain = _SLOW_CACHE.get(key)
    if chain is None:
        _SLOW_CACHE.clear()               # so a topologia em curso fica
        chain = _SLOW_CACHE[key] = {}
    rho = FADING_EPOCH_CORR
    innov = math.sqrt(max(0.0, 1.0 - rho * rho))
    for e in range(max(chain, default=-2) + 1, epoch + 1):
        g = np.random.Generator(np.random.Philox(
            key=rs.key_seed(SEED, topo_seed, "slowfade", e)))
        upper = np.triu(g.standard_normal((n, n)), 1)
        eta = upper + upper.T              # reciproca, diagonal nula
        chain[e] = eta if e == -1 else rho * chain[e - 1] + innov * eta
    return chain[epoch]


@contextmanager
def slow_fading_epoch(topo_seed, epoch, n):
    """Poe em vigor a parte lenta da epoca (so com FADING_MODEL
    "two_scale"); fora de um contexto o desvanecimento e o iid."""
    global _SLOW_FADE
    previous = _SLOW_FADE
    _SLOW_FADE = ({"epoch": epoch,
                   "matrix": slow_fading_matrix(topo_seed, epoch, n)}
                  if FADING_MODEL == "two_scale" else None)
    try:
        yield
    finally:
        _SLOW_FADE = previous


def sample_slot_fading(receivers, transmitters, topo, rng):
    """Sorteia uma vez o fading de cada par (tx, rx) audivel neste slot.

    A ordem canonica impede que a particao dos transmissores em tx_sets mude
    qual amostra aleatoria e atribuida a cada enlace. [Canal com memoria]
    Com uma parte lenta em vigor, a variancia FAST_FADE_DB^2 se divide entre
    ela (fracao w) e o sorteio do slot (1 - w), sem mudar quantos numeros o
    rng consome.
    """
    fading = {}
    txs = sorted(set(transmitters))
    slow = (_SLOW_FADE["matrix"] if _SLOW_FADE is not None
            and FADING_SLOW_FRACTION > 0.0 else None)
    if slow is not None:
        fast_db = FAST_FADE_DB * math.sqrt(1.0 - FADING_SLOW_FRACTION)
        slow_scale = (FAST_FADE_DB * math.sqrt(FADING_SLOW_FRACTION)
                      * LN10_OVER_10)
    if isinstance(rng, rs.SlotDraws):
        # [Fase 7] sorteio do par (tx, rx) por chave: nao depende de quem
        # mais esta no slot
        z = rng.fading_normal
        if slow is None:
            scale = FAST_FADE_DB * LN10_OVER_10
            for rx in sorted(set(receivers)):
                audible = topo.audible[rx]
                for tx in txs:
                    if tx in audible:
                        fading[(tx, rx)] = math.exp(float(z[tx][rx]) * scale)
            return fading
        scale = fast_db * LN10_OVER_10
        for rx in sorted(set(receivers)):
            audible = topo.audible[rx]
            for tx in txs:
                if tx in audible:
                    fading[(tx, rx)] = math.exp(
                        float(z[tx][rx]) * scale
                        + float(slow[tx][rx]) * slow_scale)
        return fading
    for rx in sorted(set(receivers)):
        audible = topo.audible[rx]
        for tx in txs:
            if tx in audible:
                if slow is None:
                    fading[(tx, rx)] = math.exp(
                        rng.gauss(0.0, FAST_FADE_DB) * LN10_OVER_10)
                else:
                    fading[(tx, rx)] = math.exp(
                        rng.gauss(0.0, fast_db) * LN10_OVER_10
                        + float(slow[tx][rx]) * slow_scale)
    return fading


def decode_success_prob(sinr_db):
    """[F21] Probabilidade de decodificar um quadro que chega com esse SINR."""
    if PER_MODEL == "threshold":
        return 1.0 if sinr_db >= REQ_SNR_DB else 0.0
    x = 2.0 * math.log(9.0) / PER_WIDTH_DB * (sinr_db - REQ_SNR_DB)
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


CAPTURE = cm.EmpiricalUwbCapture(CAPTURE_EXTRAPOLATION,
                                 per=decode_success_prob,
                                 intra_acquisition=CAPTURE_INTRA_ACQUISITION)


def sample_decode_draws(receivers, channels, rng):
    """[F21] Um sorteio uniforme por (receptor, canal) do slot, em ordem
    canonica -- como o fading, independe de como os transmissores foram
    particionados. No limiar duro nao sorteia nada (reproduz a v2 anterior)."""
    if PER_MODEL == "threshold":
        return None
    if isinstance(rng, rs.SlotDraws):
        u = rng.decode_u
        return {(rx, ch): float(u[rx][ch])
                for rx in sorted(set(receivers)) for ch in sorted(set(channels))}
    return {(rx, ch): rng.random()
            for rx in sorted(set(receivers)) for ch in sorted(set(channels))}


def _interf_weight(dt_ns):
    """[F28] Peso de um interferente que chega dt_ns depois (ou antes) do
    sinal travado: integral dentro de um simbolo de dados, reduzido fora."""
    return 1.0 if abs(dt_ns) <= SYNC_WINDOW_NS else ASYNC_INTERF_WEIGHT


def decode_at_receiver(rx, txs, pkt_of, topo, clock, rng=None, leak_txs=(),
                       fading=None, phy_stats=None, draw=None):
    """[F1][F2][F3][F4][F12][F28] Decide o que o receptor 'rx' decodifica
    num slot.

    Modelo (nesta ordem):
      1. Potencia recebida = perda + shadowing estatico + fast fading do slot.
      2. Tempo de chegada  = inicio da TX (offset de clock do transmissor,
         mais o atraso aleatorio de TX, se houver [F28]) + propagacao.
      3. Copias do MESMO quadro seguem SAME_PACKET_MODEL [F22][F27]: por
         padrao, dentro de COMBINE_WINDOW_NS vale a mais forte e as outras
         nao atrapalham; fora da janela, interferem.
      4. AQUISICAO: entre os quadros que chegam ate CAPTURE_TIME_NS depois do
         primeiro, um posterior rouba a trava se exceder o travado por
         CAPTURE_SIR_DB. Depois disso, so rouba um preambulo mais forte com
         simbolos alinhados aos do travado (PREAMBLE_ALIGN_NS) [F28].
      5. Decodifica com probabilidade decode_success_prob(SINR), SINR contra
         ruido + interferencia agregada (inclusive vazamento de outros
         canais) [F21]. A interferencia que chega fora de SYNC_WINDOW_NS do
         sinal travado pesa ASYNC_INTERF_WEIGHT [F28]; 'draw' e o sorteio
         uniforme deste receptor no slot.

    Retorna (pacote, no_do_primeiro_caminho, t_chegada_ns, rssi_dbm, sinr_db)
    ou None. Os tres ultimos campos JA existiam internamente e eram
    descartados; exporta-los e observabilidade pura -- nenhum deles e
    recalculado, para nao consumir aleatoriedade adicional.
    """
    plin = topo.ps_lin[rx]
    tp = topo.tprop_ns[rx]
    aud = topo.audible[rx]
    floor = NOISE_EFF_LIN * 1e-2
    if fading is None:
        if rng is None:
            raise ValueError("rng ou fading pre-sorteado e obrigatorio")
        fading = sample_slot_fading([rx], list(txs) + list(leak_txs),
                                    topo, rng)
    sigs = []
    for t in txs:
        if t not in aud:
            continue
        p = plin[t] * fading[(t, rx)]
        if p < floor:
            continue
        sigs.append((t, pkt_of[t], p, clock[t] + tp[t]))
    if not sigs:
        return None

    # [F12] vazamento de canais adjacentes entra como interferencia atenuada
    leak = 0.0
    for t in leak_txs:
        if t in aud:
            leak += plin[t] * fading[(t, rx)]
    leak *= ACLR_LIN

    # --- agrupa por pacote; [F22] SAME_PACKET_MODEL decide as copias ---
    by_pkt = defaultdict(list)
    for s in sigs:
        by_pkt[s[1]].append(s)
    # (pkt, first_node, t_first, p_sinal, p_residual, t_sinal, residuais)
    groups = []
    total_power = leak
    for pkt, lst in by_pkt.items():
        lst.sort(key=lambda s: s[3])
        t_first = lst[0][3]
        first_node = lst[0][0]
        is_multi = len(lst) >= 2
        if phy_stats is not None:
            key = "n_groups_multi" if is_multi else "n_groups_singleton"
            phy_stats[key] += 1
        if SAME_PACKET_MODEL == "interference":
            # cada copia disputa sozinha, como um pacote diferente
            for i, s in enumerate(lst):
                groups.append((pkt, s[0], s[3], s[2], 0.0, s[3], (), None,
                               s[2]))
                total_power += s[2]
                if phy_stats is not None and is_multi:
                    kind = "comb" if i == 0 else "resid"
                    phy_stats[f"{kind}_signals"] += 1
                    phy_stats[f"{kind}_power"] += s[2]
            continue
        comb = 0.0
        resid = 0.0
        strongest = 0.0
        t_strongest = t_first
        second = None                      # [Fase 6] (potencia, chegada)
        late = []
        for s in lst:
            if s[3] - t_first <= COMBINE_WINDOW_NS:
                comb += s[2]
                if s[2] > strongest:
                    if strongest > 0.0:
                        second = (strongest, t_strongest)
                    strongest = s[2]
                    t_strongest = s[3]
                elif second is None or s[2] > second[0]:
                    second = (s[2], s[3])
                if phy_stats is not None and is_multi:
                    phy_stats["comb_signals"] += 1
                    phy_stats["comb_power"] += s[2]
            else:
                resid += s[2]              # chegou tarde: atrapalha
                late.append((s[2], s[3]))
                if phy_stats is not None and is_multi:
                    phy_stats["resid_signals"] += 1
                    phy_stats["resid_power"] += s[2]
        window_sum = comb                  # [Fase 6] energia das copias
        if SAME_PACKET_MODEL == "nondestructive":
            comb = strongest               # as outras nao somam nem atrapalham
        else:
            comb *= COMBINE_EFF            # "additive": F4 antigo, intacto
            t_strongest = t_first
        groups.append((pkt, first_node, t_first, comb, resid, t_strongest,
                       late, second, window_sum))
        total_power += comb + resid

    # --- [F28] AQUISICAO DO PREAMBULO ---
    # Enquanto o receptor ainda acumula o primeiro PAC, todos os preambulos
    # que chegaram estao presentes e a decisao e por potencia: um sinal
    # posterior rouba a trava se exceder o travado por CAPTURE_SIR_DB. Depois
    # da aquisicao, o receptor segue a temporizacao do preambulo travado e um
    # mais forte so o arrasta se os simbolos dos dois coincidirem (SECON 2019,
    # Fig. 12). No PRESET_ETAPA0 (40 ns, sem alinhamento) este e o
    # first-path lock do F3.
    groups.sort(key=lambda g: g[2])
    locked = groups[0]
    t_lock = locked[2]
    steal = 10.0 ** (CAPTURE_SIR_DB / 10.0)
    for g in groups[1:]:
        if g[2] - t_lock > CAPTURE_TIME_NS:
            if PREAMBLE_ALIGN_NS <= 0.0:
                break                               # tarde demais para roubar
            phase = (g[2] - locked[2]) % PREAMBLE_SYMBOL_NS
            if min(phase, PREAMBLE_SYMBOL_NS - phase) > PREAMBLE_ALIGN_NS / 2:
                continue                            # simbolos desalinhados
        if g[3] > locked[3] * steal:
            locked = g                              # re-captura
    signal = locked[3]
    if signal <= 0.0:
        return None
    if CAPTURE_MODEL == "empirical_uwb":
        return _decode_empirical(locked, groups, leak, topo, draw, rng,
                                 phy_stats)

    # [F17] SINR efetivo: o ganho de processamento vale contra o ruido termico,
    # mas NAO contra os concorrentes co-codigo.
    if ASYNC_INTERF_WEIGHT >= 1.0:
        interf = total_power - signal
    else:
        # [F28] o interferente que chega fora de um simbolo de dados do sinal
        # travado cai fora das posicoes de salto temporal dele e pesa menos
        t0 = locked[5]
        interf = leak
        for g in groups:
            if g is not locked:
                interf += g[3] * _interf_weight(g[5] - t0)
            for p_late, t_late in g[6]:
                interf += p_late * _interf_weight(t_late - t0)
    sinr_db = 10.0 * math.log10(signal / (interf + NOISE_EFF_LIN))
    if PER_MODEL == "threshold":
        if sinr_db < REQ_SNR_DB:
            return None
    else:
        # [F21] curva de erro suave: o quadro trava, mas pode falhar
        if draw is None:
            if rng is None:
                raise ValueError("PER_MODEL='logistic' exige 'draw' ou rng")
            draw = rng.random()
        if draw >= decode_success_prob(sinr_db):
            return None
    # locked = (pkt, first_node, t_first, p_sinal, p_residual, ...)
    return locked[0], locked[1], locked[2], 10.0 * math.log10(signal), sinr_db


def _decode_empirical(locked, groups, leak, topo, draw, rng, phy_stats):
    """[Fase 6] Decisao do quadro travado pelo EmpiricalUwbCapture. Quadros
    diferentes dentro de 1 simbolo de dados entram no SINR com peso
    integral; fora dele contam como fontes assincronas (as decodificaveis
    sozinhas) na tabela do SECON; a 2a copia identica na janela leva as
    tabelas da EWSN. Com PER_MODEL = "threshold" (sem sorteio) decide p >= 0,5."""
    signal, t0 = locked[3], locked[5]
    alone = NOISE_EFF_LIN * 10.0 ** (REQ_SNR_DB / 10.0)
    sync_i = async_i = 0.0
    k = 0
    max_dt = None

    def classify(power, arrival):
        nonlocal sync_i, async_i, k, max_dt
        dt = abs(arrival - t0)
        if dt <= SYNC_WINDOW_NS:
            sync_i += power
            return
        async_i += power
        if power >= alone:
            k += 1
            max_dt = dt if max_dt is None else max(max_dt, dt)

    for g in groups:
        if g is not locked:
            # [Fase 6] um quadro CONCORRENTE interfere com a energia de todas
            # as suas copias na janela (g[8]); so o quadro travado tem as
            # copias tratadas como nao destrutivas. Com o maximo (g[3]) o T do
            # Crystal, cujas copias sao identicas, ficava com menos
            # interferencia que os CTX, cujos cabecalhos diferem.
            classify(g[8], g[5])
        for p_late, t_late in g[6]:
            classify(p_late, t_late)
    second = locked[7]
    ctx = cm.CaptureContext(
        frame_bytes=frame_bytes_of(locked[0], len(topo.audible)),
        signal_mw=signal, noise_mw=NOISE_EFF_LIN + leak,
        sync_interf_mw=sync_i, async_interf_mw=async_i,
        same_copy_dt_ns=None if second is None else abs(second[1] - t0),
        same_copy_dp_db=(None if second is None
                         else 10.0 * math.log10(signal / second[0])),
        async_senders=k, acquisition_ns=CAPTURE_TIME_NS,
        max_async_dt_ns=max_dt)
    p, flags = CAPTURE.success_probability(ctx)
    if phy_stats is not None:
        phy_stats["capture_evaluations"] += 1
        for f in flags:
            phy_stats["capture_flags"][f] = (
                phy_stats["capture_flags"].get(f, 0) + 1)
    sinr_db = 10.0 * math.log10(signal / (sync_i + leak + NOISE_EFF_LIN))
    if PER_MODEL == "threshold" and draw is None:
        ok = p >= 0.5
    else:
        if draw is None:
            if rng is None:
                raise ValueError("PER_MODEL='logistic' exige 'draw' ou rng")
            draw = rng.random()
        ok = draw < p
    if not ok:
        return None
    return locked[0], locked[1], locked[2], 10.0 * math.log10(signal), sinr_db


# ===========================================================================
# 4) MOTOR DE SLOTS  (Weaver / Cupid / Cupid-FWB)
# ===========================================================================
# [F10] Estruturas de slot REAIS. Na v1 os dois grupos do Cupid eram resolvidos
# DENTRO DO MESMO SLOT, o que dava ao Cupid 2x oportunidades de decodificacao
# de graca. Agora:
#   Weaver : quadro TX-RX-RX -> cada hop transmite 1 vez a cada 3 slots, com
#            todos os nos do hop juntos, e o quadro inclui a fase de ACK global.
#   Cupid  : quadro TX-RX    -> cada hop transmite 1 vez a cada 2 slots, mas
#            emitter e collector se ALTERNAM: cada grupo a cada 4 slots.
# A vantagem estrutural do Cupid fica em ~1.5x oportunidades por hop, que e a
# razao real, e nao os 2x artificiais da v1.

def schedule_for(protocol, slot, hop_of_node, nodes):
    """Retorna a lista de nos autorizados a transmitir neste slot (Weaver)
    ou (candidatos do hop, grupo da vez na grade legada) no Cupid."""
    if protocol == "weaver":
        return [v for v in nodes if hop_of_node[v] % 3 == slot % 3]
    # cupid e cupid_fwb
    want = "emitter" if (slot // 2) % 2 == 0 else "collector"
    return [v for v in nodes if hop_of_node[v] % 2 == slot % 2], want


def cupid_tx_group(slot, h):
    """[Fase 5] Grupo do hop h que transmite no slot (None: slot RX do hop).
    "fig5b": nos slots TX do hop (slot e h com a mesma paridade), o primeiro
    grupo -- emitter no hop impar, collector no par -- e o outro alternam;
    cada grupo transmite a cada 4 slots e o outro fica em modo silencioso."""
    if slot % 2 != h % 2:
        return None
    if CUPID_SCHEDULE == "legacy":
        return "emitter" if (slot // 2) % 2 == 0 else "collector"
    first, second = (("emitter", "collector") if h % 2 == 1
                     else ("collector", "emitter"))
    return first if ((slot - h) // 2) % 2 == 1 else second


def cupid_active(slot, nodes, hop, group):
    """Nos (de 'nodes') cujo grupo transmite neste slot do Cupid."""
    return [v for v in nodes if hop[v] < 99
            and cupid_tx_group(slot, hop[v]) == group[v]]


def tx_period(protocol):
    """Slots entre duas oportunidades de TX do mesmo no (F10): 3 no Weaver
    (TX-RX-RX); 4 no Cupid (TX-RX com os grupos alternando)."""
    return 3 if protocol == "weaver" else 4


def suppression_slots(protocol, h):
    """[F23] Prazo L de suspensao apos um L-ACK (Weaver, eq. 4): 2(h-2) slots
    para o pacote chegar ao sink, h para o ACK global voltar, e D do
    agrupamento do ACK global -- aqui o pior caso, Y oportunidades de TX.
    O Cupid usa a mesma regra ("configured identically")."""
    return 2 * max(h - 2, 0) + h + GACK_BATCH_Y * tx_period(protocol)


def sink_scheduled(protocol, slot, topo):
    """[F18] O sink transmite o ACK no slot do hop 0, pela MESMA regra de
    escalonamento de qualquer no. No Cupid ele e do grupo emitter (como todo
    hop 0 em assign_cupid_groups), entao a cadencia e 1 a cada 4 slots; no
    Weaver, 1 a cada 3. Nenhum atalho: e a consequencia das estruturas F10."""
    if protocol == "weaver":
        return bool(schedule_for("weaver", slot, topo.hop, [0]))
    if CUPID_SCHEDULE == "fig5b":
        return slot % 2 == 0          # sink em todo slot TX do hop 0
    cand, want = schedule_for(protocol, slot, topo.hop, [0])
    return bool(cand) and topo.group[0] == want


def frame_bytes_of(pkt, n):
    """[Fase 5] Bytes no ar de cada tipo de quadro. Dados: DATA_FRAME_BYTES
    (126 B com o cabecalho TSM, Cupid V-A). ACK do sink e quadro so de
    bitmap: cabecalho curto + 1 bit por no (Weaver sec. 4) [hipotese no
    tamanho do cabecalho]. Sync, poll e resposta: o quadro curto."""
    if isinstance(pkt, tuple):
        if pkt[0] in ("crystal_s", "crystal_a"):   # S e A do Crystal
            return CRYSTAL_SA_FRAME_BYTES
        if isinstance(pkt[0], str):             # sync, poll, resposta
            return SHORT_FRAME_BYTES
        pkt = pkt[0]                            # [F27] quadro com cabecalho
    if pkt == ACK_PKT or pkt <= GACK_PKT_BASE:
        return SHORT_FRAME_BYTES + math.ceil(n / 8)
    return DATA_FRAME_BYTES


def validate_channel_assignment(assignment, topo):
    """Rejeita um plano que exija retuning mais rapido que o hardware.

    Quando CHANNEL_SWITCH_US excede um slot, os radios de cada no permanecem
    fixos. Logo, o canal de TX do proprio no e os canais dos vizinhos de hops
    adjacentes precisam caber simultaneamente em N_RADIOS_PER_NODE front-ends.
    """
    missing = set(range(topo.n)) - set(assignment)
    if missing:
        raise ValueError(f"atribuicao sem canal para os nos {sorted(missing)}")
    if CHANNEL_SWITCH_US <= SLOT_US:
        return
    for rx in range(topo.n):
        required = {assignment[rx]}
        for tx in topo.reliable[rx]:
            if abs(topo.hop[tx] - topo.hop[rx]) == 1:
                required.add(assignment[tx])
        if len(required) > N_RADIOS_PER_NODE:
            raise ValueError(
                f"no {rx} precisaria de {len(required)} canais fixos com "
                f"N_RADIOS_PER_NODE={N_RADIOS_PER_NODE}; alternar em "
                f"SLOT_US={SLOT_US} us viola CHANNEL_SWITCH_US="
                f"{CHANNEL_SWITCH_US} us")


def _candidate_channel_assignment(topo, funnel, channel_limit):
    assignment = {v: 0 for v in range(topo.n)}
    by_hop = defaultdict(list)
    for v in range(topo.n):
        if 0 < topo.hop[v] < 99:
            by_hop[topo.hop[v]].append(v)
    for h, nodes in sorted(by_hop.items()):
        k = min(max(1, int(funnel.get(h, 1))), channel_limit, N_CHANNELS)
        for index, node in enumerate(sorted(nodes)):
            assignment[node] = index % k
    return assignment


def build_channel_assignment(protocol, topo, funnel=None):
    """Monta uma atribuicao fisica estatica e validada para uma topologia."""
    if protocol != "cupid_fwb":
        assignment = {v: 0 for v in range(topo.n)}
        validate_channel_assignment(assignment, topo)
        return assignment
    if funnel is None:
        funnel = allocate_funnel_channels(
            {h for h in topo.hop.values() if 0 < h < 99}, N_CHANNELS)
    max_requested = min(N_CHANNELS, max(funnel.values(), default=1))
    last_error = None
    for channel_limit in range(max_requested, 0, -1):
        assignment = _candidate_channel_assignment(
            topo, funnel, channel_limit)
        try:
            validate_channel_assignment(assignment, topo)
            return assignment
        except ValueError as exc:
            last_error = exc
    raise last_error


def build_tx_sets(protocol, active, topo, funnel,
                  channel_assignment=None):
    """Agrupa transmissores ativos por canal fisico, nunca por particao."""
    if channel_assignment is None:
        channel_assignment = build_channel_assignment(protocol, topo, funnel)
    by_channel = defaultdict(list)
    for v in sorted(active):
        by_channel[channel_assignment[v]].append(v)
    return [(channel, nodes) for channel, nodes in sorted(by_channel.items())]


def decode_physical_channels(rx, tx_sets, pkt_of, topo, clock, fading,
                             max_channels=None, phy_stats=None, draws=None):
    """Decodifica uma vez por canal; co-canal interfere integralmente.
    [F21] 'draws' vem de sample_decode_draws(), chave (receptor, canal)."""
    by_channel = defaultdict(set)
    for channel, nodes in tx_sets:
        by_channel[channel].update(nodes)
    channels = [(channel, sorted(nodes))
                for channel, nodes in sorted(by_channel.items())]
    limit = MAX_RX_CHANNELS if max_channels is None else max_channels
    results = []
    for channel, txs in channels[:limit]:
        leak = sorted(
            tx for other_channel, other_txs in channels
            if other_channel != channel
            for tx in other_txs)
        result = decode_at_receiver(
            rx, txs, pkt_of, topo, clock, leak_txs=leak, fading=fading,
            phy_stats=phy_stats,
            draw=None if draws is None else draws[(rx, channel)])
        results.append((channel, result))
    return results


def run_bootstrap(topo, rng, n_tx=None, relay_period=None, max_slots=None,
                  tx_delay_ns=None, phy_stats=None, frame_kind="sync"):
    """[P-boot] Onda de sincronizacao fisica de uma epoca (cupid_grouping).
    Sem argumentos, e o bootstrap dos CTX (Weaver/Cupid/FWB); o Crystal pede
    o flood S com n_tx = N_TX_SYNC_ACK e relay_period = 1."""
    n_tx = BOOTSTRAP_TX if n_tx is None else n_tx
    relay_period = BOOTSTRAP_RELAY_PERIOD if relay_period is None         else relay_period
    if max_slots is None:
        max_slots = (topo.n + 1) * relay_period * n_tx
    return cg.run_bootstrap(
        topo, rng, phy=sys.modules[__name__], n_tx=n_tx,
        relay_period=relay_period, max_slots=max_slots,
        clock_jitter_ns=CLOCK_JITTER_NS,
        tx_delay_ns=TX_DELAY_NS if tx_delay_ns is None else tx_delay_ns,
        phy_stats=phy_stats, frame_kind=frame_kind)


def grouping_config(**overrides):
    """[P-group] GroupingConfig com os parametros em vigor (e overrides)."""
    phy = rp.derive_phy(rp.get_radio_profile(RADIO_PROFILE),
                        SHORT_FRAME_BYTES)
    base = dict(iterations=GROUPING_ITERATIONS, slot_us=RANGING_SLOT_US,
                bootstrap_slots=GROUPING_BOOTSTRAP_SLOTS,
                wait_slots=GROUPING_WAIT_SLOTS, clock_ppm=RANGING_CLOCK_PPM,
                lde_sigma_ns=RANGING_LDE_SIGMA_NS,
                delayed_tx_granularity_ns=phy.delayed_tx_granularity_ns)
    base.update(overrides)
    return cg.GroupingConfig(**base)


def run_grouping_period(topo, config=None, rng=None, phy_stats=None):
    """[P-group] Periodo de agrupamento do Cupid sobre a fisica do motor."""
    return cg.run_grouping_period(
        topo, config or grouping_config(), rng, phy=sys.modules[__name__],
        dist=math.dist, c_m_per_ns=C_M_PER_NS, phy_stats=phy_stats)


def _bootstrap_substream(rng):
    """Stream do bootstrap quando o chamador nao fornece um BootstrapState:
    um unico sorteio do rng da epoca semeia um stream proprio. Com o mesmo
    rng, todos os protocolos recebem o mesmo bootstrap."""
    return random.Random(rng.getrandbits(64))


def _start_slots(topo):
    """Primeiro slot em que cada no participa. Bootstrap fisico: slot
    seguinte ao do sync aceito (inativos: nunca). Legado: hop h espera h
    slots com BOOTSTRAP_DELAY, senao todos desde o slot 0."""
    state = getattr(topo, "bootstrap", None)
    if state is not None:
        return {v: state.start_slot(v) for v in range(topo.n)}
    if BOOTSTRAP_DELAY:
        return {v: topo.hop[v] for v in range(topo.n)}
    return {v: 0 for v in range(topo.n)}


def _sendable(queue, sup, slot, holder_stats=None):
    """[F23] Pacotes que o no pode transmitir agora (fora de suspensao).
    Suspensao vencida sem ACK global e apagada: o pacote volta a sair."""
    if not sup:
        return queue
    ok = []
    for p in queue:
        until = sup.get(p)
        if until is None:
            ok.append(p)
        elif until <= slot:
            del sup[p]
            ok.append(p)
            if holder_stats is not None:
                holder_stats["local_ack_resumes"] += 1
    return ok


def run_epoch_ctx(protocol, originators, topo, rng, slot_budget=None,
                  funnel=None, delivery_slots=None, phy_stats=None,
                  sink_events=None, tail_slots=0, tail_rng=None,
                  holder_stats=None, streams=None, counters=None):
    """Simula uma epoca de coleta para Weaver / Cupid / Cupid-FWB.

    [F11] Cada pacote pode ter MULTIPLOS holders (como em CTX real). Um holder
    para de transmitir quando ouve o proprio pacote sendo repassado por um no
    de hop menor (ACK implicito por overhearing) ou quando o ACK global do
    Weaver confirma a entrega. Na v1 havia um unico holder e nenhuma
    supressao, o que apagava justamente os mecanismos que diferenciam os
    protocolos na pratica.

    Ate este commit a promessa nao era cumprida: 'advances' guardava UM
    receptor por pacote e um laco pos-avanco removia a copia de todo
    transmissor de hop maior no mesmo slot. Isso e confirmacao ONISCIENTE --
    o transmissor estava transmitindo, nao passou por decode_at_receiver() e
    portanto nao podia saber de avanco nenhum. Era a semantica de dono unico
    da v1 sobrevivendo dentro do motor novo. As duas correcoes entram juntas:
    fan-out completo (todo receptor elegivel retem) e separacao entre entrega
    CONTABIL e entrega FISICA (o sink ja ter o pacote nao pode calar o ACK
    implicito). Corrigir so uma delas troca um artefato por outro.

    [F18] O ACK implicito nunca alcanca o hop 1: quem repassaria o pacote
    "de cima" e o sink, que nao transmite dados. Sem outro aviso, o hop 1 do
    Cupid gastava ~78% das suas transmissoes com relatorios ja entregues,
    enquanto o Weaver tinha um ACK global oraculo. Com ACK_MODEL="physical",
    o sink transmite o bitmap de entregas no slot do hop 0 e cada no so
    aprende o que decodificou -- do proprio sink ou "de carona" no cabecalho
    de qualquer quadro. Um no apaga uma copia apenas pelo que ELE sabe.

    [A1] Com sink_events, cada decodificacao bem-sucedida no sink vira um
    evento de trace -- inclusive as duplicatas que a logica de entrega
    descarta. E observabilidade pura: 'delivered', 'delivery_slots', o valor
    de retorno e o consumo do rng principal sao identicos com e sem o
    parametro. A cauda opcional (tail_slots) observa chegadas apos o deadline
    usando tail_rng, um stream separado, depois de congelar o resultado.
    """
    hop = topo.hop
    n = topo.n
    budget = slot_budget if slot_budget is not None else MAX_SLOTS
    if funnel is None and protocol == "cupid_fwb":
        funnel = allocate_funnel_channels(
            {h for h in hop.values() if 0 < h < 99}, N_CHANNELS)
    channel_assignment = build_channel_assignment(protocol, topo, funnel)

    # [F7] offsets de clock re-sorteados por epoca (deriva entre epocas);
    # [P-boot] com bootstrap fisico eles vem do proprio bootstrap
    state = getattr(topo, "bootstrap", None)
    if state is not None:
        clock = list(state.clock_ns)
    else:
        crng = streams.sub("clock") if streams is not None else rng
        clock = [crng.gauss(0.0, CLOCK_JITTER_NS) for _ in range(n)]
    n_ch = max(1, int(N_CHANNELS))
    start = _start_slots(topo)
    never = float("inf")
    start = {v: never if s is None else s for v, s in start.items()}

    queues = {v: [] for v in range(n)}
    for s in originators:
        queues[s].append(s)
    delivered = set()
    # [F18] bitmap de entregas que cada no conhece (so no modelo fisico; o
    # do sink e o proprio 'delivered')
    ack_model = ACK_MODEL
    gack_weaver = ack_model == "weaver"
    known = ([set() for _ in range(n)]
             if ack_model in ("physical", "weaver") else None)
    # [F24] bitmap com novidade a repropagar, ultimo periodo de agrupamento em
    # que o no mandou um quadro so de bitmap, e novidade no sink
    gack_pending = [False] * n
    last_gack_batch = [-1] * n
    sink_news = False
    # [F23] ultimo pacote que cada no ouviu (vai no cabecalho como L-ACK) e
    # pacotes suspensos: no -> {pacote: slot em que ele pode voltar a sair}
    local_weaver = LOCAL_ACK_MODEL == "weaver"
    same_hop = SAME_HOP_RELAY               # [F25]
    # [F27] cabecalho so existe se leva algo: ultimo ouvido ou bitmap
    headers_on = FRAME_HEADERS and (local_weaver or known is not None)
    last_heard = [None] * n
    suppressed = [dict() for _ in range(n)] if local_weaver else None

    tracing = sink_events is not None
    if tracing and tail_slots and tail_rng is None:
        raise ValueError("tail_slots exige tail_rng: a cauda nunca pode "
                         "consumir o rng principal da epoca")
    # transmissoes acumuladas por pacote, so para preencher n_retx no trace
    tx_count = defaultdict(int) if tracing else None
    horizon = budget + (tail_slots if tracing else 0)
    final = None
    in_budget = True

    for slot in range(horizon):
        if slot == budget:
            # [A1] Fronteira do deadline. O resultado da epoca e congelado
            # AQUI, antes de qualquer slot extra. A partir deste ponto:
            # 'delivered' e uma copia local, delivery_slots deixa de ser
            # escrito e o rng passa a ser o stream separado da cauda. Nada
            # abaixo pode alterar 'final'.
            final = (budget, len(delivered))
            delivered = set(delivered)
            delivery_slots = None
            rng = tail_rng
            if streams is not None:      # a cauda tem streams proprios
                streams = rs.EpochStreams.from_rng(tail_rng).child("tail")
            in_budget = False
        if len(delivered) == len(originators):
            if final is None:
                final = (slot, len(delivered))
            break

        if holder_stats is not None:
            _sample_holders(holder_stats, queues, n, slot)
        # [Fase 7] sorteios do slot: por chave (raiz, slot) ou o rng sequencial
        srng = streams.slot(slot, n, n_ch) if streams is not None else rng
        keyed = streams is not None
        sink_turn = (ack_model in ("physical", "oracle", "weaver")
                     and sink_scheduled(protocol, slot, topo))
        if ack_model == "oracle" and sink_turn:
            # [F18] limite superior para comparacao A/B: todos sabem tudo
            for v in range(1, n):
                queues[v] = [p for p in queues[v] if p not in delivered]
        # [F26] no do hop h so esta sincronizado a partir do slot h;
        # [P-boot] no bootstrap fisico, a partir do slot seguinte ao do sync
        if local_weaver:
            # [F23] so disputa o slot quem tem pacote nao suspenso
            sendable = {}
            for v in range(1, n):
                if queues[v] and start[v] <= slot:
                    ok = _sendable(queues[v], suppressed[v], slot,
                                   holder_stats)
                    if ok:
                        sendable[v] = ok
            holders = sorted(sendable)
        else:
            sendable = queues
            holders = [v for v in range(n) if v != 0 and queues[v]
                       and start[v] <= slot]
        if protocol == "weaver":
            active = schedule_for("weaver", slot, hop, holders)
        else:
            active = cupid_active(slot, holders, hop, topo.group)
        holders_set = set(holders)
        # [F18] o sink entra como mais um transmissor escalonado quando ha
        # entrega a confirmar -- e passa pelo mesmo sorteio de participacao
        if ack_model == "physical" and sink_turn and delivered:
            active = [0] + active
        if gack_weaver:
            # [F24] nos sem dado repropagam o bitmap num quadro sem carga, se
            # tem novidade e ainda nao o fizeram neste periodo de agrupamento
            batch = slot // (GACK_BATCH_Y * tx_period(protocol))
            idle = [v for v in range(1, n)
                    if gack_pending[v] and v not in holders_set
                    and last_gack_batch[v] < batch]
            if idle:
                if protocol == "weaver":
                    idle = schedule_for("weaver", slot, hop, idle)
                else:
                    idle = cupid_active(slot, idle, hop, topo.group)
                active = sorted(active + idle)
            # ... e o sink anuncia quando recebeu pacote novo ou repetido
            if sink_turn and sink_news:
                active = [0] + active
        # [F16] participacao aleatoria, IDENTICA para todos os protocolos
        if TX_PROB < 1.0:
            if keyed:
                active = [v for v in active if srng.txprob_u[v] < TX_PROB]
            else:
                active = [v for v in active if rng.random() < TX_PROB]
        if not active:
            continue

        # [F16] escolha do pacote a enviar (evita head-of-line deterministico)
        data_nodes = [v for v in active if v != 0 and v in holders_set]
        if RANDOM_QUEUE_HEAD and keyed:
            pkt_of = {v: sendable[v][int(srng.queue_u[v] * len(sendable[v]))]
                      for v in data_nodes}
        elif RANDOM_QUEUE_HEAD:
            pkt_of = {v: rng.choice(sendable[v]) for v in data_nodes}
        else:
            pkt_of = {v: sendable[v][0] for v in data_nodes}
        for v in active:
            if v == 0:
                pkt_of[0] = ACK_PKT              # [F18] quadro de ACK do sink
            elif v not in pkt_of:
                pkt_of[v] = GACK_PKT_BASE - v    # [F24] so bitmap
        if gack_weaver:
            # o bitmap sai em qualquer quadro do no (carona ou sem carga)
            for v in active:
                if v == 0:
                    sink_news = False
                else:
                    gack_pending[v] = False
                    if pkt_of[v] <= GACK_PKT_BASE:
                        last_gack_batch[v] = batch
        if tx_count is not None:
            for v in active:
                if pkt_of[v] > 0:
                    tx_count[pkt_of[v]] += 1
        if counters is not None and in_budget:
            # observacao pura: quadros no ar e quadros de dados
            counters["tx_frames"] += len(active)
            counters["tx_data"] += len(data_nodes)
        if holder_stats is not None:
            # [F18] so observacao: quanto do meio vai para relatorio entregue
            for v in active:
                if v == 0:
                    holder_stats["sink_ack_tx"] += 1
                    continue
                if pkt_of[v] <= GACK_PKT_BASE:
                    holder_stats["node_gack_tx"] += 1
                    continue
                wasted = pkt_of[v] in delivered
                holder_stats["data_tx"] += 1
                holder_stats["data_tx_delivered"] += wasted
                if hop[v] == 1:
                    holder_stats["data_tx_hop1"] += 1
                    holder_stats["data_tx_hop1_delivered"] += wasted
        tx_sets = build_tx_sets(protocol, active, topo, funnel,
                                channel_assignment=channel_assignment)
        active_set = set(active)
        if headers_on:
            # [F27] o que vai ao ar e o quadro inteiro: pacote + cabecalho
            # (ultimo pacote ouvido e bitmap). Copias do mesmo relatorio com
            # cabecalhos diferentes nao sao o mesmo quadro (P15).
            frame_of = {v: (pkt_of[v],
                            last_heard[v] if local_weaver else None,
                            frozenset(known[v])
                            if known is not None and v != 0 else None)
                        for v in active}
        else:
            frame_of = pkt_of

        # [F11] pkt -> TODOS os receptores elegiveis do slot. Guardar um
        # unico receptor era o que reduzia o CTX a dono unico: o pacote
        # avancava por um caminho so, mesmo quando varios nos o ouviam.
        advances = defaultdict(set)
        overheard = defaultdict(set)   # no -> pacotes que ele ouviu de cima
        # so telemetria: pares (no, pkt) cujo ACK implicito chegou com o
        # sink JA de posse do relatorio -- medido no instante da
        # decodificacao, antes de qualquer entrega deste slot.
        late_ack = set() if holder_stats is not None else None
        # [F18] no -> entregas que ele aprendeu NESTE slot (aplicadas no fim,
        # como advances/overheard)
        learned = defaultdict(set) if known is not None else None
        # [F23] pacote ouvido por cada receptor neste slot e L-ACKs recebidos
        # no cabecalho de quadros vindos de cima
        heard_now = {}
        header_lack = defaultdict(set)

        # [PERF] So importam dois tipos de receptor: (a) os que estao um hop
        # ACIMA de algum transmissor (podem fazer o pacote avancar) e (b) os
        # detentores um hop ABAIXO (podem se suprimir por overhearing).
        # Avaliar os 36 nos a cada slot era desperdicio puro.
        # [F24] com o G-ACK do Weaver, qualquer no um hop abaixo escuta: o
        # bitmap desce por nos que nao tem pacote nenhum
        active_hops = {hop[v] for v in active}
        # [F25] ... e, com a regra do Weaver, quem esta calado no proprio hop
        # [P-boot] no ainda nao sincronizado (ou inativo) nao escuta
        relevant = [v for v in range(n)
                    if v not in active_set
                    and (state is None or start[v] <= slot)
                    and ((hop[v] + 1) in active_hops
                         or (same_hop and v != 0 and hop[v] in active_hops)
                         or ((queues[v] or gack_weaver)
                             and (hop[v] - 1) in active_hops))]
        fading = sample_slot_fading(relevant, active, topo, srng)
        draws = sample_decode_draws(
            relevant, [channel for channel, _ in tx_sets], srng)
        slot_clock = clock
        if TX_DELAY_NS > 0.0:
            # [F28] o TSM atrasa cada TX por um valor aleatorio (Weaver,
            # sec. 6.1), sorteado em ordem canonica
            slot_clock = list(clock)
            for v in sorted(active_set):
                u = float(srng.txdelay_u[v]) if keyed else rng.random()
                slot_clock[v] += u * TX_DELAY_NS

        for rx in relevant:
            for channel, res in decode_physical_channels(
                    rx, tx_sets, frame_of, topo, slot_clock, fading,
                    phy_stats=phy_stats, draws=draws):
                if res is None:
                    continue
                frame, src, t_arrival_ns, rssi_dbm, sinr_db = res
                pkt = frame[0] if headers_on else frame
                if counters is not None and in_budget:
                    counters["rx_frames"] += 1
                    counters["rx_data"] += pkt > 0
                if pkt == ACK_PKT or pkt <= GACK_PKT_BASE:
                    # [F18][F24] quadro so de bitmap (do sink ou de no
                    # ocioso): o receptor aprende o que o transmissor sabia
                    # no inicio deste slot
                    if rx != 0 and learned is not None:
                        learned[rx] |= delivered if src == 0 else known[src]
                    if local_weaver and rx != 0:
                        lack = last_heard[src]
                        if (lack is not None and hop[src] < hop[rx]
                                and lack in queues[rx]):
                            header_lack[rx].add(lack)
                    if holder_stats is not None:
                        holder_stats["sink_ack_decodes" if src == 0
                                     else "node_gack_decodes"] += 1
                    continue
                if gack_weaver:
                    # [F24] pacote ja confirmado chegando: quem o mandou
                    # perdeu o G-ACK -- volta a anunciar o bitmap
                    if rx == 0:
                        if pkt in delivered:
                            sink_news = True
                    elif pkt in known[rx]:
                        gack_pending[rx] = True
                if learned is not None and rx != 0 and known[src]:
                    # [F18] carona: o cabecalho de todo quadro leva o bitmap
                    # que o transmissor conhece -- mas so para quem o
                    # decodificou
                    learned[rx] |= known[src]
                if local_weaver and rx != 0:
                    heard_now[rx] = pkt
                    # [F23] L-ACK: o cabecalho de um no mais perto do sink traz
                    # o ultimo pacote que ELE ouviu -- esse pacote ja avancou
                    lack = last_heard[src]
                    if (lack is not None and hop[src] < hop[rx]
                            and lack in queues[rx]):
                        header_lack[rx].add(lack)
                if rx == 0 and tracing:
                    # [A1] Ponto unico de emissao. Roda ANTES do filtro
                    # 'pkt in delivered' logo abaixo, que permanece intacto,
                    # e nao muta nada: so faz append numa lista do chamador.
                    flags = 0
                    if pkt in delivered:
                        flags |= SINK_FLAG_DUPLICATE
                    if not in_budget:
                        flags |= SINK_FLAG_LATE
                    if hop[rx] != hop[src] - 1:
                        flags |= SINK_FLAG_NO_ADVANCE
                    sink_events.append(sink_reception_event(
                        pkt, src, slot, channel, t_arrival_ns, rssi_dbm,
                        sinr_db, flags,
                        n_hops=hop[pkt] - hop[src] + 1,
                        n_retx=max(0, tx_count[pkt] - 1)))
                # [F11] ACK implicito por overhearing: ouvi meu proprio
                # pacote vindo de um no mais perto do sink -> posso parar de
                # retransmitir. Isto e FISICA e roda ANTES da contabilidade:
                # o sink ja ter o relatorio nao emudece o meio nem cala este
                # detentor. Com o guard de 'delivered' na frente, todo holder
                # que sobrevivia a primeira entrega ficava surdo ate o fim da
                # epoca -- e a unica saida era reintroduzir o handoff.
                if hop[src] < hop[rx] and pkt in queues[rx]:
                    overheard[rx].add(pkt)
                    if late_ack is not None and pkt in delivered:
                        late_ack.add((rx, pkt))
                # Daqui para baixo e CONTABILIDADE: um relatorio ja entregue
                # nao avanca de novo nem conta uma segunda vez no PDR.
                if pkt in delivered:
                    continue
                # [F8] avanco de EXATAMENTE um hop (a v1 permitia teleporte);
                # [F25] ou adocao de um vizinho do mesmo hop
                if hop[rx] == hop[src] - 1 or (
                        same_hop and rx != 0 and hop[rx] == hop[src]):
                    advances[pkt].add(rx)

        if learned:
            for v, ids in learned.items():
                new = ids - known[v]
                if new:
                    known[v] |= new
                    if gack_weaver:
                        gack_pending[v] = True   # [F24] bits novos a repassar

        for pkt, receivers in advances.items():
            for rx in sorted(receivers):
                if rx == 0:
                    delivered.add(pkt)
                    sink_news = sink_news or gack_weaver   # [F24]
                    if delivery_slots is not None:
                        delivery_slots.setdefault(pkt, slot + 1)
                elif pkt not in queues[rx]:
                    queues[rx].append(pkt)

        if local_weaver:
            # [F23] o cabecalho vale a partir do proximo quadro do no
            for v, p in heard_now.items():
                last_heard[v] = p
        for v in set(overheard) | set(header_lack):
            for p in overheard.get(v, set()) | header_lack.get(v, set()):
                if p not in queues[v]:
                    continue
                if local_weaver:
                    # [F23] suspende, nao descarta: se o ACK global nao
                    # chegar em L slots, o pacote volta a disputar o meio
                    suppressed[v][p] = slot + 1 + suppression_slots(
                        protocol, hop[v])
                else:
                    queues[v].remove(p)          # F11: descarte definitivo
                if holder_stats is None:
                    continue
                if p in overheard.get(v, ()):
                    holder_stats["overhearing_suppressions"] += 1
                    if (v, p) in late_ack:
                        holder_stats["post_delivery_suppressions"] += 1
                else:
                    holder_stats["local_ack_suppressions"] += 1

        # [F11] NAO existe supressao por avanco observado. Um transmissor
        # nao pode saber que o pacote avancou: ele estava transmitindo e
        # nao passou por decode_at_receiver(). A unica confirmacao que ele
        # recebe e o ACK implicito acima (overhearing decodificado) ou o
        # ACK global abaixo.

        # [F18] ACK global fisico: cada no apaga so o que ELE sabe que chegou
        # (inclusive uma copia recem-recebida de relatorio ja confirmado)
        if known is not None:
            for v in range(1, n):
                if queues[v] and known[v]:
                    kept = [p for p in queues[v] if p not in known[v]]
                    if len(kept) != len(queues[v]):
                        if holder_stats is not None:
                            holder_stats["global_ack_suppressions"] += (
                                len(queues[v]) - len(kept))
                        if local_weaver:
                            for p in known[v]:
                                suppressed[v].pop(p, None)
                        queues[v] = kept

        # [F11] ACK global ORACULO do Weaver, preservado so no modo "legacy"
        # para reproduzir os numeros anteriores (auditoria Item 4)
        if (ack_model == "legacy" and protocol == "weaver"
                and slot % 3 == 2):
            for v in range(n):
                queues[v] = [p for p in queues[v] if p not in delivered]

    if final is None:
        final = (budget, len(delivered))
    return final


# ===========================================================================
# 5) CRYSTAL -- agora no MESMO motor de fisica  [F9]
# ===========================================================================

def flood_len(topo, n_tx=None):
    """[F9] Duracao FIXA de um flood Glossy: H + N - 1 slots (regra classica).
    Derivada da profundidade da topologia, nao arbitrada. Sem n_tx, usa o N
    da fase T (N_TX_PER_FLOOD)."""
    hs = [h for h in topo.hop.values() if h < 99]
    return max(hs) + (N_TX_PER_FLOOD if n_tx is None else n_tx) - 1


def crystal_t_flood(sources, topo, rng, clock, phy_stats=None,
                    sink_events=None, slot_offset=0, slot_budget=None,
                    delivered=(), reached=None, participants=None,
                    streams=None, counters=None):
    """[F26] Fase T do Crystal: TODAS as fontes pendentes iniciam o proprio
    flood Glossy no mesmo slot ("all nodes with data to send become Glossy
    initiators"); quem ainda nao pegou pacote adota o primeiro que decodificar
    e o retransmite N_T vezes -- o sink tambem. A captura decide o que chega:
    "only one data packet can be successfully delivered to the sink per phase"
    (artigo do Cupid, sec. III). Devolve (pacote, slot_local + 1) da primeira
    decodificacao no sink, ou (None, None). Com 'reached' (um set), acrescenta
    nele os nos que decodificaram algum pacote (calibracao, F28). [P-boot] Com
    'participants', so esses nos (os sincronizados pelo S) escutam."""
    n = topo.n
    pkt_of = {s: s for s in sources}
    # [F28] cada fonte comeca o seu flood num instante proprio (timer de
    # 32 kHz); quem retransmite um pacote herda a base de tempo dele
    t0_of = {s: 0.0 for s in sources}
    if CRYSTAL_T_OFFSET_NS > 0.0 and streams is not None:
        # [Fase 7] instante de cada fonte por chave (fase T que comeca em
        # slot_offset), indexado pela fonte
        u = streams.slot(slot_offset, n, kind="crystal_t0").txdelay_u
        t0_of = {s: float(u[s]) * CRYSTAL_T_OFFSET_NS for s in sources}
    elif CRYSTAL_T_OFFSET_NS > 0.0:
        t0_of = {s: rng.random() * CRYSTAL_T_OFFSET_NS for s in sources}
    tx_count = defaultdict(int)
    first = (None, None)
    for slot in range(flood_len(topo)):
        txs = sorted(v for v in pkt_of if tx_count[v] < N_TX_PER_FLOOD)
        if not txs:
            break
        for v in txs:
            tx_count[v] += 1
        if counters is not None:
            counters["tx_frames"] += len(txs)
            counters["tx_data"] += len(txs)
        receivers = [rx for rx in range(n) if rx not in pkt_of
                     and (participants is None or rx in participants)]
        if not receivers:
            continue
        srng = (streams.slot(slot_offset + slot, n) if streams is not None
                else rng)
        fading = sample_slot_fading(receivers, txs, topo, srng)
        draws = sample_decode_draws(receivers, [0], srng)
        slot_clock = clock
        if CRYSTAL_T_OFFSET_NS > 0.0:
            slot_clock = list(clock)
            for v in txs:
                slot_clock[v] += t0_of[pkt_of[v]]
        adopted = {}
        for rx in receivers:
            res = decode_at_receiver(
                rx, txs, pkt_of, topo, slot_clock, fading=fading,
                phy_stats=phy_stats,
                draw=None if draws is None else draws[(rx, 0)])
            if res is None:
                continue
            if counters is not None:
                counters["rx_frames"] += 1
                counters["rx_data"] += 1
            adopted[rx] = res[0]
            if rx != 0:
                continue
            first = (res[0], slot + 1)
            if sink_events is not None:
                abs_slot = slot_offset + slot
                flags = SINK_FLAG_DUPLICATE if res[0] in delivered else 0
                if slot_budget is not None and abs_slot >= slot_budget:
                    flags |= SINK_FLAG_LATE
                sink_events.append(sink_reception_event(
                    res[0], res[1], abs_slot, 0, res[2], res[3], res[4],
                    flags, n_hops=topo.hop[res[0]] - topo.hop[res[1]] + 1,
                    n_retx=0))
        pkt_of.update(adopted)
        if reached is not None:
            reached.update(adopted)
    return first


def glossy_flood_reach(src, topo, rng, clock, n_tx, phy_stats=None,
                       participants=None, streams=None, slot_offset=0,
                       counters=None):
    """[F26] Flood Glossy de uma so fonte (S ou A do Crystal) com n_tx
    retransmissoes por no e duracao fixa H + n_tx - 1. Todas as copias sao o
    mesmo quadro, e SAME_PACKET_MODEL decide se elas atrapalham. Devolve o
    conjunto de nos alcancados (inclui a fonte)."""
    n = topo.n
    holders = {src}
    tx_count = defaultdict(int)
    for local in range(flood_len(topo, n_tx)):
        txs = sorted(v for v in holders if tx_count[v] < n_tx)
        receivers = [rx for rx in range(n) if rx not in holders
                     and (participants is None or rx in participants)]
        if not txs or not receivers:
            break
        for v in txs:
            tx_count[v] += 1
        if counters is not None:
            counters["tx_frames"] += len(txs)
        pkt_of = {v: ("crystal_a",) for v in txs}   # [Fase 5] quadro de A
        srng = (streams.slot(slot_offset + local, n) if streams is not None
                else rng)
        fading = sample_slot_fading(receivers, txs, topo, srng)
        draws = sample_decode_draws(receivers, [0], srng)
        new = set()
        for rx in receivers:
            if decode_at_receiver(
                    rx, txs, pkt_of, topo, clock, fading=fading,
                    phy_stats=phy_stats,
                    draw=None if draws is None else draws[(rx, 0)]):
                new.add(rx)
        if counters is not None:
            counters["rx_frames"] += len(new)
        holders |= new
    return holders


def glossy_flood(src, topo, rng, clock, sink_arrival_slot=None,
                 phy_stats=None, sink_events=None, slot_offset=0,
                 slot_budget=None, extra_flags=0):
    """Um flood Glossy de duracao FIXA: o originador transmite e todos que
    recebem retransmitem o MESMO pacote. Como o pacote e o mesmo, os sinais
    COMBINAM construtivamente (F4) em vez de colidir -- e por isso o Crystal e
    robusto mesmo sem captura entre pacotes distintos. O flood NAO termina
    cedo: em Glossy todos os nos precisam retransmitir N vezes.
    Retorna (chegou_ao_sink, slots_gastos).

    [A1] Com sink_events, TODA decodificacao no sink e emitida. Num flood o
    sink decodifica o mesmo pacote em varios slots, o que faz do Crystal a
    fonte mais rica de duplicatas do simulador -- e todas elas eram
    descartadas pelo filtro 'not sink_arrival_slot', que so registrava a
    primeira. A emissao nao altera holders, o retorno nem o consumo do rng.
    """
    n = topo.n
    holders = {src}
    tx_count = defaultdict(int)
    pkt_of = {}
    L = flood_len(topo)
    pkt_tx_total = 0
    sink_seen = False
    for slot in range(L):
        txs = [v for v in holders if tx_count[v] < N_TX_PER_FLOOD]
        if txs:
            for v in txs:
                tx_count[v] += 1
                pkt_of[v] = src
            pkt_tx_total += len(txs)
            new = set()
            receivers = [rx for rx in range(n) if rx not in txs]
            fading = sample_slot_fading(receivers, txs, topo, rng)
            draws = sample_decode_draws(receivers, [0], rng)
            for rx in receivers:
                res = decode_at_receiver(
                    rx, txs, pkt_of, topo, clock, fading=fading,
                    phy_stats=phy_stats,
                    draw=None if draws is None else draws[(rx, 0)])
                if res is None:
                    continue
                new.add(rx)
                if rx != 0:
                    continue
                if sink_arrival_slot is not None and not sink_arrival_slot:
                    sink_arrival_slot.append(slot + 1)
                if sink_events is not None:
                    abs_slot = slot_offset + slot
                    flags = extra_flags
                    if sink_seen:
                        flags |= SINK_FLAG_DUPLICATE
                    if slot_budget is not None and abs_slot >= slot_budget:
                        flags |= SINK_FLAG_LATE
                    sink_events.append(sink_reception_event(
                        src, res[1], abs_slot, 0, res[2], res[3], res[4],
                        flags,
                        n_hops=topo.hop[src] - topo.hop[res[1]] + 1,
                        n_retx=max(0, pkt_tx_total - 1)))
                    sink_seen = True
            holders |= new
    return (0 in holders), L


def run_epoch_crystal(originators, topo, rng, slot_budget=None,
                      delivery_slots=None, phy_stats=None,
                      sink_events=None, epoch_info=None, streams=None,
                      sync_rng=None):
    """Crystal segundo CRYSTAL_MODEL [F26]."""
    if CRYSTAL_MODEL == "sequential":
        # no modo chaveado nao ha rng sequencial: um stream proprio da epoca
        if rng is None:
            rng = streams.sub("crystal_sequential")
        return _run_epoch_crystal_sequential(
            originators, topo, rng, slot_budget, delivery_slots, phy_stats,
            sink_events)
    counters = None if epoch_info is None else epoch_info.get("counters")
    return _run_epoch_crystal_tapairs(
        originators, topo, rng, slot_budget, delivery_slots, phy_stats,
        sink_events, epoch_info, streams, counters, sync_rng)


def _run_epoch_crystal_tapairs(originators, topo, rng, slot_budget=None,
                               delivery_slots=None, phy_stats=None,
                               sink_events=None, epoch_info=None,
                               streams=None, counters=None, sync_rng=None):
    """[F26] Crystal como no artigo: um S (cobrado, nao simulado), depois
    pares TA ate todas as fontes ouvirem o proprio ACK, o prazo acabar ou o
    sink ver CRYSTAL_R fases T seguidas sem dado nenhum.

    Cada fase T e uma disputa de todas as fontes pendentes; o sink fica com o
    primeiro pacote que decodificar (se ja o tinha, e duplicata, mas o ACK sai
    de novo). Um ACK positivo so cala a fonte se o flood A chegar ate ela --
    senao ela repete o pacote no proximo T. A latencia e a chegada do ultimo
    relatorio ao sink, como no Cupid/Weaver e no artigo do Cupid."""
    budget = slot_budget if slot_budget is not None else MAX_SLOTS
    l_sa = flood_len(topo, N_TX_SYNC_ACK)
    l_t = flood_len(topo)
    participants = None
    if BOOTSTRAP_MODEL == "physical":
        # [P-boot] o S e simulado: sem atraso de TX (o S e iniciado pelo sink)
        # e com a duracao fixa do flood; so quem o decodificou participa da
        # epoca, e uma fonte sem sync nunca transmite
        state = run_bootstrap(topo, (streams.sub("crystal_S")
                                     if streams is not None
                                     else sync_rng or
                                     _bootstrap_substream(rng)),
                              n_tx=N_TX_SYNC_ACK, relay_period=1,
                              max_slots=l_sa, tx_delay_ns=0.0,
                              phy_stats=phy_stats, frame_kind="crystal_s")
        if epoch_info is not None:
            epoch_info["crystal_sync"] = state
        clock = list(state.clock_ns)
        participants = state.synchronized_nodes
    else:
        crng = streams.sub("clock") if streams is not None else rng
        clock = [crng.gauss(0.0, CLOCK_JITTER_NS) for _ in range(topo.n)]
    # [Fase 5] 'now' e o tempo em unidades de SLOT_US; 't' segue sendo o
    # indice de slot das chaves de sorteio
    sa_u, t_u = crystal_phase_units()
    t = l_sa                       # fase S
    now = l_sa * sa_u
    pending = set(originators)     # fontes que ainda nao ouviram o proprio ACK
    if participants is not None:
        pending &= participants
    arrivals = {}
    silent = 0
    while now < budget and pending and len(arrivals) < len(originators):
        pkt, local = crystal_t_flood(
            sorted(pending), topo, rng, clock, phy_stats, sink_events,
            slot_offset=t, slot_budget=budget, delivered=arrivals,
            participants=participants, streams=streams,
            counters=counters)
        if pkt is None:
            silent += 1
        else:
            silent = 0
            arrival = now + local * t_u
            if pkt not in arrivals and arrival <= budget:
                arrivals[pkt] = arrival
                if delivery_slots is not None:
                    delivery_slots.setdefault(pkt, arrival)
        t += l_t
        now += l_t * t_u
        # ACK negativo nao cala ninguem; com CRYSTAL_A_MODEL = "physical" ele
        # tambem vai ao ar pela fisica, como o sink faz
        if pkt is not None or CRYSTAL_A_MODEL == "physical":
            reach = glossy_flood_reach(0, topo, rng, clock, N_TX_SYNC_ACK,
                                       phy_stats, participants, streams,
                                       slot_offset=t, counters=counters)
            if pkt is not None and pkt in reach:
                pending.discard(pkt)
        t += l_sa
        now += l_sa * sa_u
        if silent >= CRYSTAL_R:    # o sink dorme; o que faltou se perdeu
            break
    if len(arrivals) == len(originators):
        return max(arrivals.values()), len(arrivals)
    return budget, len(arrivals)


def crystal_phase_units():
    """(slot de S/A, slot de T) em unidades de SLOT_US. "uniform": (1, 1)."""
    if CRYSTAL_PHASE_TIMING == "uniform":
        return 1, 1               # inteiros: o legado segue bit a bit
    prof = rp.get_radio_profile(RADIO_PROFILE)
    sa = rp.derive_phy(prof, CRYSTAL_SA_FRAME_BYTES).slot_min_us
    return sa / SLOT_US, 1.0


def _run_epoch_crystal_sequential(originators, topo, rng, slot_budget=None,
                                  delivery_slots=None, phy_stats=None,
                                  sink_events=None):
    """Crystal da v2 anterior: rodadas de tres floods Glossy (S, T, A) com uma
    fonte por rodada, A que nunca falha e originador abandonado apos 3
    falhas. A v1 sorteava uma probabilidade de rodada perdida (0.05*pendentes)
    escolhida a mao; aqui custo e perda ja vinham do mesmo motor de fisica."""
    budget = slot_budget if slot_budget is not None else MAX_SLOTS
    clock = [rng.gauss(0.0, CLOCK_JITTER_NS) for _ in range(topo.n)]
    pending = list(originators)
    delivered = 0
    slots = 0
    fails = 0
    L = flood_len(topo)
    while pending and slots < budget:
        src = pending[0]
        sink_arrival_slot = [] if delivery_slots is not None else None
        round_start = slots
        # [A1] O flood T e logicamente precedido por um flood S de L slots;
        # e o mesmo deslocamento usado no calculo de arrival_slot abaixo.
        ok, used = glossy_flood(
            src, topo, rng, clock, sink_arrival_slot, phy_stats=phy_stats,
            sink_events=sink_events, slot_offset=round_start + L,
            slot_budget=budget)
        # custo da rodada: flood de dados + os outros floods (S e A)
        slots += used + (CRYSTAL_FLOODS_PER_ROUND - 1) * L
        if ok:
            # A ordem logica e S-T-A. Apenas T e simulado fisicamente; por isso
            # o primeiro slot local de T recebe o deslocamento de um flood S.
            if delivery_slots is not None and sink_arrival_slot:
                arrival_slot = round_start + L + sink_arrival_slot[0]
                if arrival_slot <= budget:
                    delivery_slots.setdefault(src, arrival_slot)
            delivered += 1
            pending.pop(0)
            fails = 0
        else:
            fails += 1
            if fails >= 3:              # desiste deste originador
                pending.pop(0)
                fails = 0
    return min(slots, budget), delivered


def run_epoch(protocol, originators, topo, rng, slot_budget=None, funnel=None,
              delivery_slots=None, phy_stats=None, sink_events=None,
              tail_slots=0, tail_rng=None, holder_stats=None,
              bootstrap=None, groups=None, epoch_info=None, streams=None,
              grouping=None, bootstrap_rng=None):
    """Uma epoca de coleta. [P-boot] Com BOOTSTRAP_MODEL = "physical", os CTX
    rodam sobre a EpochTopology do bootstrap desta epoca -- o fornecido ou,
    sem ele, um sorteado de um stream proprio semeado pelo rng --, e 'groups'
    (emitter/collector da epoca, Fase 4) substitui o grupo da topologia.
    'epoch_info' (observacao pura) recebe o BootstrapState usado.
    [Fase 7] Com RNG_MODEL = "keyed", 'streams' (EpochStreams) fornece todos
    os sorteios; sem ele, um unico sorteio do rng semeia a raiz."""
    if RNG_MODEL == "keyed" and streams is None:
        streams = rs.EpochStreams.from_rng(rng)
    elif RNG_MODEL != "keyed":
        streams = None
    counters = None
    if epoch_info is not None:
        counters = epoch_info.setdefault("counters", {
            "tx_frames": 0, "tx_data": 0, "rx_frames": 0, "rx_data": 0})
    if protocol == "crystal":
        # O Crystal ja produz chegadas atrasadas sem cauda: uma rodada pode
        # comecar dentro do orcamento e terminar fora dele.
        return run_epoch_crystal(originators, topo, rng, slot_budget,
                                 delivery_slots, phy_stats, sink_events,
                                 epoch_info, streams, bootstrap_rng)
    if BOOTSTRAP_MODEL == "physical":
        if bootstrap is None:
            bootstrap = run_bootstrap(topo, (streams.sub("bootstrap")
                                             if streams is not None
                                             else bootstrap_rng or
                                             _bootstrap_substream(rng)))
        # [P-group] grupo da epoca: do periodo de agrupamento (ranging) ou,
        # sem ele, o da topologia -- e o fallback fica registrado
        source = "n/a"
        if protocol in ("cupid", "cupid_fwb"):
            if groups is not None:
                source = "given"
            elif GROUPING_MODEL == "ranging" and grouping is not None:
                groups = cg.assign_epoch_groups(grouping, bootstrap)
                source = "ranging"
            else:
                source = ("static" if GROUPING_MODEL == "static"
                          else "static_fallback")
        topo = cg.epoch_view(topo, bootstrap, groups)
        if epoch_info is not None:
            epoch_info["bootstrap"] = bootstrap
            epoch_info["group_source"] = source
    elif epoch_info is not None and protocol in ("cupid", "cupid_fwb"):
        epoch_info["group_source"] = "static"
    return run_epoch_ctx(protocol, originators, topo, rng, slot_budget, funnel,
                         delivery_slots, phy_stats, sink_events, tail_slots,
                         tail_rng, holder_stats, streams, counters)


# ===========================================================================
# 6) EXPERIMENTOS
# ===========================================================================

def sweep_epochs():
    """Numero de epocas usado nas varreduras (menor que o das CDFs, porque
    cada varredura tem dezenas de pontos x protocolos x topologias)."""
    return max(15, EPOCHS // 4)


def slots_to_ms(slots):
    return slots * SLOT_US / 1000.0


def valid_nodes(topo):
    """Originadores possiveis. [P-boot] Com bootstrap fisico, todo no sensor
    (o hop so existe dentro da epoca; quem nao sincroniza perde o relatorio);
    no legado, os que a BFS estatica alcanca ate MAX_HOP."""
    if BOOTSTRAP_MODEL == "physical":
        return list(range(1, topo.n))
    return [i for i in range(1, topo.n) if topo.hop[i] <= MAX_HOP]


def mean_ci(vals, conf=0.95):
    """[F14][Fase 8] media e semi-largura do IC entre topologias, com o t de
    Student de n - 1 graus de liberdade (o 1,96 fixo subestimava o IC com 5
    topologias em 42%). Com uma topologia so, a semi-largura e nan."""
    m, half, _ = es.mean_ci_t(vals, conf)
    return m, half


def epoch_cap_ms():
    """[F19] Teto de observacao de uma epoca (MAX_SLOTS) em ms."""
    return slots_to_ms(MAX_SLOTS)


def censored_median(lats):
    """[F19] Mediana com amostras incompletas CENSURADAS (+inf).

    Toda epoca incompleta duraria mais que o orcamento, entao a ordenacao das
    amostras e preservada: a mediana e exata sempre que mais da metade
    completou; senao ela vale +inf ("mais da metade nao completou"). Descartar
    as incompletas, como antes, dava a CDF mais bonita justamente ao
    protocolo menos confiavel (auditoria Item 3)."""
    if not lats:
        return float("nan")
    return float(np.median(np.asarray(lats, dtype=float)))


def mean_ci_censored(vals, cap):
    """[F14][F19] Media e IC entre topologias quando alguma mediana por
    topologia e censurada (+inf): usa o teto 'cap' no lugar e sinaliza que o
    resultado e um LIMITE INFERIOR. Devolve (media, ic, censurado)."""
    clean = [v for v in vals if not math.isnan(v)]
    censored = any(math.isinf(v) for v in clean)
    m, ci = mean_ci([min(v, cap) for v in clean])
    return m, ci, censored


def fmt_latency(ms, censored=False):
    """[F19] Texto de uma latencia: '>= X ms' quando e limite inferior."""
    if ms is None or math.isnan(ms):
        return "n/d"
    if math.isinf(ms) or censored:
        return f">= {min(ms, epoch_cap_ms()):.1f} ms"
    return f"{ms:.1f} ms"


def make_topologies(n_topo=None, base_seed=None):
    """[F14] Gera N topologias independentes. Cada ponto das figuras passa a
    ser uma media sobre topologias, com IC -- em vez de uma unica topologia,
    onde efeito de parametro e ruido topologico ficavam confundidos."""
    n_topo = n_topo or N_TOPOLOGIES
    base_seed = SEED if base_seed is None else base_seed
    topos = []
    for k in range(n_topo):
        rng = random.Random(base_seed + 1000 * k)
        topos.append((build_topology(rng), base_seed + 1000 * k))
    return topos


def _bootstrap_slots_of(proto, topo, info):
    """Duracao do bootstrap da epoca, em slots. CTX com bootstrap fisico: ate
    o ultimo no sincronizar (a onda se sobrepoe aos slots de dados). Legado:
    o hop mais fundo (regra "hop h espera h slots"). Crystal: o flood S."""
    if proto == "crystal":
        return flood_len(topo, N_TX_SYNC_ACK) * crystal_phase_units()[0]
    state = info.get("bootstrap")
    if state is not None:
        return state.last_sync_slot() + 1
    hops = [h for h in topo.hop.values() if h <= MAX_HOP]
    return max(hops) if (BOOTSTRAP_DELAY and hops) else 0


def _epoch_record(proto, topo, epoch, origs, result, ds, info):
    slots, deliv = result
    complete = deliv == len(origs)
    boot = _bootstrap_slots_of(proto, topo, info)
    c = info.get("counters", {})
    return {
        "epoch": epoch, "originators": tuple(origs), "slots": slots,
        "delivered": deliv, "complete": complete, "delivery_slots": dict(ds),
        # [Fase 5] latencia da coleta: do inicio da epoca ao ultimo relatorio
        # de uma epoca COMPLETA (incompleta = +inf, censurada)
        "latency_data_epoch_ms": slots_to_ms(slots) if complete else math.inf,
        # [P17] metrica do artigo do Cupid: ate o ultimo pacote recebido,
        # mesmo em epoca incompleta (None se nada chegou)
        "last_report_ms": (slots_to_ms(max(ds.values())) if ds else None),
        "bootstrap_slots": boot, "bootstrap_ms": slots_to_ms(boot),
        "group_source": info.get("group_source", "n/a"),
        "tx_frames": c.get("tx_frames", 0), "tx_data": c.get("tx_data", 0),
        "rx_frames": c.get("rx_frames", 0), "rx_data": c.get("rx_data", 0),
    }


def simulate_topology(proto, topo, seed, O, n_epochs, slot_budget=None,
                      funnel=None, tag="cdf", legacy_seed=None,
                      phy_stats=None):
    """[Fases 5/7/8] Epocas de um protocolo numa topologia.

    RNG_MODEL = "keyed": o WorkloadPlan (experiment_stats) e os streams de
    cada epoca sao chaveados por (SEED, topologia, tag, epoca) -- identicos
    para todos os protocolos do experimento --, o bootstrap sai do stream da
    epoca e o Cupid usa o periodo de agrupamento da topologia, cujo custo
    fica em 'grouping_setup_ms', separado da latencia da coleta.
    RNG_MODEL = "sequential" (legado): exatamente as chamadas antigas, com o
    rng por protocolo semeado por legacy_seed (padrao: SEED*7919 + offset)."""
    budget = slot_budget if slot_budget is not None else MAX_SLOTS
    valid = valid_nodes(topo)
    records, grouping, grouping_ms, plan_id = [], None, 0.0, None
    # o periodo de agrupamento tem stream proprio (nao consome o rng do
    # protocolo) e vale nos dois modos de sorteio
    use_grouping = (proto in ("cupid", "cupid_fwb")
                    and GROUPING_MODEL == "ranging"
                    and BOOTSTRAP_MODEL == "physical")
    # [Canal com memoria] o periodo de agrupamento e a epoca -1 do canal;
    # refeito antes de cada epoca, ve o canal dela
    if use_grouping and not GROUPING_RENEW:
        with slow_fading_epoch(seed, -1, topo.n):
            grouping = run_grouping_period(
                topo, rng=rs.stream(SEED, seed, "grouping"))
        grouping_ms = grouping.total_time_ms
    if RNG_MODEL == "keyed":
        plan = es.make_workload_plan(SEED, seed, valid, O, n_epochs,
                                     DATA_FRAME_BYTES, budget, tag)
        plan_id = plan.plan_id
        for ep in plan.epochs:
            with slow_fading_epoch(seed, ep.epoch, topo.n):
                if use_grouping and GROUPING_RENEW:
                    grouping = run_grouping_period(topo, rng=rs.stream(
                        SEED, seed, "grouping", tag, ep.epoch))
                    grouping_ms += grouping.total_time_ms
                ds, info = {}, {}
                origs = list(ep.originators)
                res = run_epoch(proto, origs, topo, None, slot_budget=budget,
                                funnel=funnel, delivery_slots=ds,
                                streams=rs.EpochStreams(SEED, seed, tag,
                                                        ep.epoch),
                                grouping=grouping, epoch_info=info,
                                phy_stats=phy_stats)
            records.append(_epoch_record(proto, topo, ep.epoch, origs, res,
                                         ds, info))
    else:
        rng = random.Random(seed * 7919 + PROTO_SEED_OFFSET[proto]
                            if legacy_seed is None else legacy_seed)
        if funnel is None and proto == "cupid_fwb":
            funnel = allocate_funnel_channels(
                {h for h in topo.hop.values() if 0 < h < 99}, N_CHANNELS)
        for e in range(n_epochs):
            with slow_fading_epoch(seed, e, topo.n):
                if use_grouping and GROUPING_RENEW:
                    grouping = run_grouping_period(topo, rng=rs.stream(
                        SEED, seed, "grouping", tag, e))
                    grouping_ms += grouping.total_time_ms
                origs = rng.sample(valid, min(O, len(valid)))
                ds, info = {}, {}
                # bootstrap por chave da epoca, a mesma do modo chaveado:
                # igual para todos os CTX; o S do Crystal tem a sua
                # ("crystal_S"). No legado (BOOTSTRAP_MODEL = "bfs") nenhum
                # dos dois e usado.
                boot_key = "crystal_S" if proto == "crystal" else "bootstrap"
                res = run_epoch(proto, origs, topo, rng,
                                slot_budget=slot_budget, funnel=funnel,
                                delivery_slots=ds, epoch_info=info,
                                phy_stats=phy_stats, grouping=grouping,
                                bootstrap_rng=rs.stream(
                                    SEED, seed, tag, e, boot_key))
            records.append(_epoch_record(proto, topo, e, origs, res, ds,
                                         info))
    return {"records": records, "grouping": grouping,
            "grouping_setup_ms": grouping_ms, "plan_id": plan_id,
            "budget_slots": budget}


def _latency_stats(proto, topo, seed, O, epochs=None, phy_stats=None):
    """Latencia e confiabilidade de um protocolo numa topologia.

    [F19] Antes so entravam as epocas em que TODOS os relatorios chegaram
    (auditoria Item 3). Agora toda epoca entra: as incompletas como +inf.
    Tambem guarda a latencia de cada relatorio entregue (slot de chegada ao
    sink), que junto com 'sent' da o PDR sem prazo."""
    n_epochs = epochs or EPOCHS
    run = simulate_topology(proto, topo, seed, O, n_epochs,
                            phy_stats=phy_stats)
    recs = run["records"]
    lats = [r["latency_data_epoch_ms"] for r in recs]
    report_lats = [slots_to_ms(s) for r in recs
                   for s in r["delivery_slots"].values()]
    return {"median": censored_median(lats), "lats": lats,
            "report_lats": report_lats, "epochs": n_epochs,
            "complete": sum(r["complete"] for r in recs),
            "sent": sum(len(r["originators"]) for r in recs),
            "records": recs, "grouping_setup_ms": run["grouping_setup_ms"],
            "plan_id": run["plan_id"]}


def experiment_latency_cdf(topos, O=35, protos=None, capture_flags=None):
    """CDF agregada sobre todas as topologias.

    [F19] Devolve (cdf, stats): cdf[p] tem TODAS as epocas (incompletas =
    +inf); stats[p] soma epocas, completas, relatorios enviados e a latencia
    de cada relatorio entregue. [Fase 8] stats[p]["by_topology"][seed] guarda
    as amostras de cada topologia -- a unidade independente das comparacoes.
    [Fase 6] Com 'capture_flags' (dict), acumula as hipoteses de captura
    usadas (observacao pura)."""
    protos = protos or PROTOCOLS
    out, stats = {}, {}
    for proto in protos:
        agg = {"lats": [], "report_lats": [], "epochs": 0, "complete": 0,
               "sent": 0, "by_topology": {}}
        for topo, seed in topos:
            phy = new_phy_stats() if capture_flags is not None else None
            st = _latency_stats(proto, topo, seed, O, phy_stats=phy)
            if phy is not None:
                for k, v in phy["capture_flags"].items():
                    capture_flags[k] = capture_flags.get(k, 0) + v
            agg["by_topology"][seed] = {
                "lats": st["lats"], "records": st["records"],
                "grouping_setup_ms": st["grouping_setup_ms"],
                "plan_id": st["plan_id"]}
            for key in ("lats", "report_lats"):
                agg[key].extend(st[key])
            for key in ("epochs", "complete", "sent"):
                agg[key] += st[key]
        out[proto] = sorted(agg["lats"])
        agg["report_lats"].sort()
        stats[proto] = agg
    return out, stats


def experiment_latency_vs_originators(topos, protos=None):
    """[F14] Latencia media vs carga, com IC 95% ENTRE topologias.
    [F19] Mediana censurada por topologia; 'censored' marca os pontos que sao
    limite inferior (alguma topologia nao completou metade das epocas)."""
    protos = protos or PROTOCOLS
    Ovals = [5, 15, 25, 35]
    cap = epoch_cap_ms()
    out = {p: {"mean": [], "ci": [], "censored": []} for p in protos}
    for O in Ovals:
        for proto in protos:
            per_topo = [_latency_stats(proto, topo, seed, O,
                                       epochs=sweep_epochs())["median"]
                        for topo, seed in topos]
            m, ci, cens = mean_ci_censored(per_topo, cap)
            out[proto]["mean"].append(m)
            out[proto]["ci"].append(ci)
            out[proto]["censored"].append(cens)
    return Ovals, out


def experiment_pdr_vs_frequency(topos, O=35, protos=None):
    """[F13] PDR com DEADLINE REAL: a epoca recebe um orcamento de slots igual
    ao periodo (1000/f ms) e conta-se quantos pacotes de fato chegaram dentro
    dele. A v1 rodava sem prazo e depois rateava a entrega pela fracao do
    periodo -- um artificio aritmetico, nao uma simulacao."""
    protos = protos or PROTOCOLS
    freqs = [1, 2, 3, 4, 5, 6, 7]  # indice 5 = 6 epocas/s (usado no resumo)
    out = {p: {"mean": [], "ci": []} for p in protos}
    for f in freqs:
        period_ms = 1000.0 / f
        budget = max(1, int(period_ms * 1000.0 / SLOT_US))
        for proto in protos:
            per_topo = []
            for topo, seed in topos:
                run = simulate_topology(
                    proto, topo, seed, O, sweep_epochs(), slot_budget=budget,
                    tag=f"pdr{f}",
                    legacy_seed=seed * 104729 + PROTO_SEED_OFFSET[proto] + f)
                sent = sum(len(r["originators"]) for r in run["records"])
                recv = sum(r["delivered"] for r in run["records"])
                per_topo.append(recv / sent if sent else float("nan"))
            m, ci = mean_ci(per_topo)
            out[proto]["mean"].append(m)
            out[proto]["ci"].append(ci)
    return freqs, out


def experiment_funnel(topos, O=35, channel_vals=(2, 4, 8, 12, 16)):
    """Latencia do FWB conforme o total de canais, com IC entre topologias.

    [F15] Agora sao DUAS curvas, e a diferenca entre elas e o ponto do artigo:
      - 'real'    : 1 radio por no (DW1000/DW3000). O no so escuta um canal.
      - 'ideal'   : K radios por no -- a premissa implicita da v1, que nenhum
                    hardware UWB atual satisfaz.
    A regiao K > 12 fica hachurada porque nem o padrao oferece tantos canais
    de 500 MHz nao sobrepostos.
    [F19] curves[modo] = (medias, ics, censurados)."""
    saved_k = globals()["N_CHANNELS"]
    saved_r = globals()["N_RADIOS_PER_NODE"]
    cap = epoch_cap_ms()
    curves = {}
    for mode in ("real", "ideal"):
        ms, cis, cens = [], [], []
        for nch in channel_vals:
            globals()["N_CHANNELS"] = nch
            globals()["N_RADIOS_PER_NODE"] = 1 if mode == "real" else nch
            _derived()
            per_topo = [_latency_stats("cupid_fwb", t, s, O,
                                       epochs=sweep_epochs())["median"]
                        for t, s in topos]
            m, ci, c = mean_ci_censored(per_topo, cap)
            ms.append(m); cis.append(ci); cens.append(c)
        curves[mode] = (ms, cis, cens)
    globals()["N_CHANNELS"] = saved_k
    globals()["N_RADIOS_PER_NODE"] = saved_r
    _derived()
    ref = [_latency_stats("cupid", t, s, O, epochs=sweep_epochs())["median"]
           for t, s in topos]
    ref_m, ref_ci, _ = mean_ci_censored(ref, cap)
    return list(channel_vals), curves, ref_m, ref_ci


def experiment_crosstalk(topos, O=35, aclr_vals=(40, 30, 25, 20, 15)):
    """[F12] NOVO: sensibilidade do FWB ao vazamento entre canais. Este
    experimento nao existia na v1 -- e ele que diz se o ganho do FWB sobrevive
    a um radio real, em que os canais NAO sao perfeitamente ortogonais."""
    saved = globals()["ACLR_DB"]
    saved_r = globals()["N_RADIOS_PER_NODE"]
    # o vazamento so tem efeito se o no realmente escutar varios canais;
    # com 1 radio nao ha canais concorrentes para vazar.
    globals()["N_RADIOS_PER_NODE"] = max(2, N_CHANNELS)
    cap = epoch_cap_ms()
    m_list, ci_list, cens_list = [], [], []
    for a in aclr_vals:
        globals()["ACLR_DB"] = a
        _derived()
        per_topo = [_latency_stats("cupid_fwb", t, s, O,
                                   epochs=sweep_epochs())["median"]
                    for t, s in topos]
        m, ci, cens = mean_ci_censored(per_topo, cap)
        m_list.append(m)
        ci_list.append(ci)
        cens_list.append(cens)
    globals()["ACLR_DB"] = saved
    globals()["N_RADIOS_PER_NODE"] = saved_r
    _derived()
    return list(aclr_vals), m_list, ci_list, cens_list


def experiment_radio_limit(topos, O=35, rx_ch_vals=(1, 2, 4, 8)):
    """[F12][F15] NOVO: quanto do ganho do FWB depende de ter VARIOS RADIOS por
    no. Um DW1000/DW3000 tem um front-end so: com 1 radio (o caso real), o FWB
    deve colapsar para perto do Cupid. Se o ganho do artigo so existe com 4+
    radios por no, isso e um custo de hardware que precisa estar declarado."""
    saved = globals()["N_RADIOS_PER_NODE"]
    cap = epoch_cap_ms()
    m_list, ci_list, cens_list = [], [], []
    for k in rx_ch_vals:
        globals()["N_RADIOS_PER_NODE"] = k
        _derived()
        per_topo = [_latency_stats("cupid_fwb", t, s, O,
                                   epochs=sweep_epochs())["median"]
                    for t, s in topos]
        m, ci, cens = mean_ci_censored(per_topo, cap)
        m_list.append(m)
        ci_list.append(ci)
        cens_list.append(cens)
    globals()["N_RADIOS_PER_NODE"] = saved
    _derived()
    return list(rx_ch_vals), m_list, ci_list, cens_list


def experiment_channel_plans(topos, O=35):
    """[F15] NOVO E CENTRAL: avalia o FWB-Cupid sob cada plano de canais
    REALISTA (regulacao + hardware), em vez de assumir 16 canais ortogonais.

    Para cada plano usa-se K = numero de canais realmente disponiveis e
    N_RADIOS_PER_NODE = min(K, radios), reportando o ganho sobre o Cupid de
    canal unico. E esta tabela, e nao a de K=16, que deve ir para o artigo."""
    saved_plan = globals()["CHANNEL_PLAN"]
    saved_k = globals()["N_CHANNELS"]
    saved_r = globals()["N_RADIOS_PER_NODE"]
    cap = epoch_cap_ms()
    ref = [_latency_stats("cupid", t, s, O, epochs=sweep_epochs())["median"]
           for t, s in topos]
    ref_m, _, ref_cens = mean_ci_censored(ref, cap)
    rows = []
    for plan, info in CHANNEL_PLANS.items():
        for radios in (1, 2):
            globals()["CHANNEL_PLAN"] = plan
            globals()["N_CHANNELS"] = info["K"]
            globals()["N_RADIOS_PER_NODE"] = min(radios, info["K"])
            _derived()
            enforce_channel_plan(verbose=False)
            per_topo = [_latency_stats("cupid_fwb", t, s, O,
                                       epochs=sweep_epochs())["median"]
                        for t, s in topos]
            m, ci, cens = mean_ci_censored(per_topo, cap)
            # [F19] ganho entre limites inferiores nao e ganho: fica n/d
            ganho = (100 * (1 - m / ref_m)
                     if (ref_m and not math.isnan(m)
                         and not cens and not ref_cens)
                     else float("nan"))
            rows.append({"plano": plan, "desc": info["desc"], "K": info["K"],
                         "radios": min(radios, info["K"]), "lat": m,
                         "ci": ci, "ganho": ganho, "censored": cens})
    globals()["CHANNEL_PLAN"] = saved_plan
    globals()["N_CHANNELS"] = saved_k
    globals()["N_RADIOS_PER_NODE"] = saved_r
    _derived()
    return rows, ref_m


def microbenchmark_capture(topos, k_vals=(2, 4, 6, 8, 12, 16), draws=20000):
    """Micro-benchmark do modelo de captura, agora com a fisica nova: mede
    P_cap (algum pacote decodificado) e P_far (o decodificado ser o do
    transmissor MAIS DISTANTE) em funcao do numero k de concorrentes."""
    topo, seed = topos[0]
    rng = random.Random(seed + 555)
    clock = [rng.gauss(0.0, CLOCK_JITTER_NS) for _ in range(topo.n)]
    valid = valid_nodes(topo)
    rows = []
    for k in k_vals:
        ok = far = 0
        for _ in range(draws):
            rx = rng.choice(valid)
            cands = [v for v in range(topo.n) if v != rx]
            txs = rng.sample(cands, min(k, len(cands)))
            pkt_of = {t: t for t in txs}
            tx_clock = clock
            if TX_DELAY_NS > 0.0:          # [F28] como num slot CTX
                tx_clock = list(clock)
                for t in sorted(txs):
                    tx_clock[t] += rng.random() * TX_DELAY_NS
            res = decode_at_receiver(rx, txs, pkt_of, topo, tx_clock, rng)
            if res is not None:
                ok += 1
                dmax = max(txs, key=lambda t: math.dist(topo.pos[rx],
                                                        topo.pos[t]))
                if res[1] == dmax:
                    far += 1
        rows.append((k, ok / draws, far / draws))
    return rows


# ===========================================================================
# 7) GRAFICOS
# ===========================================================================

PROTO_SEED_OFFSET = {"weaver": 11, "cupid": 22, "cupid_fwb": 33,
                     "crystal": 44}   # offsets ESTAVEIS (hash() de str e
                                      # aleatorizado por processo em Python 3)


COLORS = {"cupid": "#f4b400", "weaver": "#1f3a93", "crystal": "#2e8b57",
          "cupid_fwb": "#8e44ad"}
LABELS = {"cupid": "Cupid", "weaver": "Weaver", "crystal": "Crystal",
          "cupid_fwb": "Cupid FWB"}


def plot_topology(topo, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    hc = {0: "#d62728", 1: "#2e8b57", 2: "#1f3a93", 3: "#9467bd",
          4: "#ff7f0e", 5: "#8c564b"}
    for i in range(topo.n):
        h = topo.hop[i]
        if h > MAX_HOP:
            continue
        marker = "o" if (h == 0 or topo.group[i] == "emitter") else "s"
        ax.scatter(*topo.pos[i], c=hc.get(h, "gray"), marker=marker,
                   s=220 if h == 0 else 90, edgecolors="black", zorder=3)
        ax.annotate(str(i), topo.pos[i], fontsize=6, ha="center", va="center",
                    color="white", zorder=4)
    ax.set_title(f"Topologia ({topo.n} nos) - alcance confiavel DERIVADO = "
                 f"{topo.rrange:.1f} m\ncor=hop, circulo=emitter, "
                 f"quadrado=collector", fontsize=10)
    ax.set_xlabel("metros"); ax.set_ylabel("metros")
    ax.set_xlim(-2, AREA_X + 2); ax.set_ylim(-2, AREA_Y + 2)
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _plot_censored_cdf(ax, values, total, proto):
    """[F19] CDF normalizada por TODAS as amostras: as censuradas (+inf) nao
    aparecem, entao a curva para na fracao que completou."""
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite or not total:
        return
    y = np.arange(1, len(finite) + 1) / total
    ax.plot(finite, y, label=f"{LABELS[proto]} (ate {y[-1]:.0%})",
            color=COLORS[proto], lw=2)


def plot_cdf(results, path):
    """[F19] Latencia da epoca; o patamar de cada curva e a fracao de epocas
    completas dentro de MAX_SLOTS."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for proto in PROTOCOLS:
        lats = results.get(proto) or []
        _plot_censored_cdf(ax, lats, len(lats), proto)
    ax.axvline(epoch_cap_ms(), color="gray", ls=":", lw=1)
    ax.set_xlabel("Latencia da epoca (ms)"); ax.set_ylabel("Fracao das epocas")
    ax.set_title("CDF da latencia da epoca (O=35) - incompletas censuradas\n"
                 "(o patamar e a fracao de epocas completas)", fontsize=10)
    ax.set_ylim(0, 1.02); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_report_cdf(stats, path):
    """[F19] Latencia POR RELATORIO normalizada por todos os relatorios
    gerados: o patamar de cada curva e o PDR sem prazo."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for proto in PROTOCOLS:
        st = stats.get(proto)
        if st:
            _plot_censored_cdf(ax, st["report_lats"], st["sent"], proto)
    ax.set_xlabel("Chegada do relatorio ao sink (ms desde o inicio da epoca)")
    ax.set_ylabel("Fracao dos relatorios gerados")
    ax.set_title("Latencia por relatorio (O=35)\n"
                 "(o patamar e o PDR sem prazo)", fontsize=10)
    ax.set_ylim(0, 1.02); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_latency_vs_o(Ovals, out, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for proto in PROTOCOLS:
        if proto not in out:
            continue
        m = np.array(out[proto]["mean"]); ci = np.array(out[proto]["ci"])
        ax.errorbar(Ovals, m, yerr=ci, fmt="-o", capsize=3,
                    label=LABELS[proto], color=COLORS[proto], lw=2)
        # [F19] ponto censurado = limite inferior: marcador vazado
        cens = [i for i, c in enumerate(out[proto].get("censored", [])) if c]
        if cens:
            ax.scatter([Ovals[i] for i in cens], m[cens], s=110,
                       facecolors="white", edgecolors=COLORS[proto],
                       linewidths=2, zorder=4)
    ax.axhline(epoch_cap_ms(), color="gray", ls=":", lw=1)
    ax.set_xlabel("Numero de originadores (O)")
    ax.set_ylabel("Latencia mediana da epoca (ms)")
    ax.set_title("Latencia vs. carga (barras: IC 95% entre topologias;\n"
                 "marcador vazado = limite inferior, epocas censuradas)",
                 fontsize=10)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_pdr(freqs, out, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for proto in PROTOCOLS:
        if proto not in out:
            continue
        m = np.array(out[proto]["mean"]); ci = np.array(out[proto]["ci"])
        ax.errorbar(freqs, m, yerr=ci, fmt="-o", capsize=3,
                    label=LABELS[proto], color=COLORS[proto], lw=2)
    ax.set_xlabel("Frequencia de epocas (epocas/s)")
    ax.set_ylabel("PDR (dentro do prazo)")
    ax.set_title("Confiabilidade com deadline real (IC 95%)")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_funnel(ch, curves, ref, ref_ci, path):
    """[F15] Duas curvas (1 radio vs K radios) + faixas de viabilidade."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.errorbar(ch, curves["ideal"][0], yerr=curves["ideal"][1], fmt="--o",
                capsize=3, color="#c39bd3", lw=2,
                label="FWB idealizado (K radios/no - premissa da v1)")
    ax.errorbar(ch, curves["real"][0], yerr=curves["real"][1], fmt="-o",
                capsize=3, color=COLORS["cupid_fwb"], lw=2.5,
                label="FWB real (1 radio/no - DW1000/DW3000)")
    if not math.isnan(ref):
        ax.axhline(ref, color=COLORS["cupid"], ls=":", lw=2,
                   label="Cupid (canal unico)")
    ymax = ax.get_ylim()[1]
    # faixas de viabilidade de K
    ax.axvspan(min(ch), 2, color="#2ecc71", alpha=0.10)
    ax.axvspan(2, 4, color="#f1c40f", alpha=0.10)
    ax.axvspan(4, 12, color="#e67e22", alpha=0.10)
    ax.axvspan(12, max(ch), color="#e74c3c", alpha=0.12)
    for xpos, txt in ((2, "DW3000\n(K=2)"), (4, "DW1000 / ETSI\n(K=4)"),
                      (12, "max. nao\nsobrepostos (12)")):
        if min(ch) <= xpos <= max(ch):
            ax.axvline(xpos, color="gray", ls="-", lw=0.8, alpha=0.7)
            ax.text(xpos, ymax * 0.97, txt, fontsize=7, rotation=90,
                    va="top", ha="right", color="#444")
    ax.set_xlabel("Total de canais UWB (K)")
    ax.set_ylabel("Latencia mediana (ms)")
    ax.set_title("FWB: canais em funil vs. canais REALMENTE disponiveis")
    ax.set_xticks(list(ch)); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_channel_plans(rows, ref, path):
    """[F15] Barras: latencia do FWB sob cada plano de canais realista."""
    r1 = [r for r in rows if r["radios"] == 1]
    r2 = [r for r in rows if r["radios"] == 2]
    labels = [f"{r['plano']}\nK={r['K']}" for r in r1]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(x - 0.2, [r["lat"] for r in r1], 0.38, yerr=[r["ci"] for r in r1],
           capsize=3, color=COLORS["cupid_fwb"], label="1 radio/no (real)")
    ax.bar(x + 0.2, [r["lat"] for r in r2], 0.38, yerr=[r["ci"] for r in r2],
           capsize=3, color="#c39bd3", label="2 radios/no")
    if not math.isnan(ref):
        ax.axhline(ref, color=COLORS["cupid"], ls="--", lw=2,
                   label="Cupid (canal unico)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Latencia mediana (ms)")
    ax.set_title("FWB sob planos de canais REAIS (regulacao + hardware)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_sensitivity(x, y, ci, xlabel, title, path, ref=None, invert=False):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(x, y, yerr=ci, fmt="-o", capsize=3,
                color=COLORS["cupid_fwb"], lw=2, label="Cupid FWB")
    if ref is not None and not math.isnan(ref):
        ax.axhline(ref, color=COLORS["cupid"], ls="--", lw=2,
                   label="Cupid (canal unico)")
    ax.set_xlabel(xlabel); ax.set_ylabel("Latencia mediana (ms)")
    ax.set_title(title)
    ax.set_xticks(list(x))
    if invert:
        ax.invert_xaxis()
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ===========================================================================
# 8) MAIN, PROVENIENCIA E CLI
# ===========================================================================

# [Fase 9] Versao do modelo: muda sempre que a fisica ou os protocolos mudam
# (a retomada de cenarios compara esta versao e o commit).
MODEL_VERSION = "cloves-2026.09.27.2"

PRESETS = {
    "cloves": {},                       # padrao: DW1000 canal 4 do CLOVES
    "legacy_v2": PRESET_LEGACY_V2,      # motor auditado 5ef525b (F28)
    "etapa0": PRESET_ETAPA0,
    "2026_09_10": PRESET_2026_09_10,
}

ALL_EXPERIMENTS = ("cdf", "carga", "pdr", "funil", "crosstalk", "radios",
                   "planos", "micro")
SUITES = {
    # core compara so os tres protocolos do artigo; FWB roda a parte
    "core": {"experiments": ("cdf", "carga", "pdr", "micro"),
             "protocols": ("cupid", "weaver", "crystal")},
    "fwb": {"experiments": ("cdf", "funil", "crosstalk", "radios", "planos"),
            "protocols": ("cupid", "cupid_fwb")},
    "all": {"experiments": ALL_EXPERIMENTS, "protocols": tuple(PROTOCOLS)},
    # reproducao legada (cenarios.json): tudo, como antes
    "legacy": {"experiments": ALL_EXPERIMENTS,
               "protocols": tuple(PROTOCOLS)},
}


def suite_selection(suite=None, experiments=None, protocols=None):
    """(experimentos, protocolos) de uma rodada, na ordem canonica: os
    pedidos explicitamente ou, sem eles, os da suite. A MESMA regra vale para
    cli(), main() e o pedido do runner (a identidade da rodada)."""
    suite_def = SUITES.get(suite or "all", SUITES["all"])
    exps = experiments or suite_def["experiments"]
    protos = protocols or suite_def["protocols"]
    for kind, got, valid in (("experimento", exps, ALL_EXPERIMENTS),
                             ("protocolo", protos, PROTOCOLS)):
        bad = [x for x in got if x not in valid]
        if bad:
            raise ValueError(f"{kind}(s) desconhecido(s): {bad}")
    return (tuple(e for e in ALL_EXPERIMENTS if e in exps),
            [p for p in PROTOCOLS if p in protos])

# [Fase 11] Parametros sem fonte que fixe o valor: vao para o manifesto com o
# valor em vigor.
HYPOTHESES = {
    "PATHLOSS_N": "hipotese", "SHADOW_DB": "hipotese",
    "FAST_FADE_DB": "hipotese", "CLOCK_JITTER_NS": "hipotese",
    "REQ_SNR_DB": "hipotese", "PER_WIDTH_DB": "hipotese",
    "CAPTURE_SIR_DB": "hipotese", "SHORT_FRAME_BYTES": "hipotese",
    "CRYSTAL_SA_FRAME_BYTES": "hipotese",
    "GROUPING_BOOTSTRAP_SLOTS": "hipotese", "GROUPING_WAIT_SLOTS": "hipotese",
    "RANGING_LDE_SIGMA_NS": "hipotese", "BOOTSTRAP_MODEL": "hipotese",
    "CAPTURE_EXTRAPOLATION": "sensibilidade",
    "CAPTURE_INTRA_ACQUISITION": "sensibilidade",
    "FADING_MODEL": "sensibilidade", "FADING_SLOW_FRACTION": "sensibilidade",
    "FADING_EPOCH_CORR": "sensibilidade",
    "RANGING_CLOCK_PPM": "sensibilidade", "TX_DELAY_NS": "hipotese",
    "ROUTING_MARGIN_DB": "diagnostico",
}


CODE_SUFFIXES = (".py", ".json")   # nao rastreado com esse sufixo = sujeira


def _untracked_code(porcelain):
    return [line[3:].strip() for line in porcelain.splitlines()
            if line.startswith("?? ")
            and line[3:].strip().endswith(CODE_SUFFIXES)]


def _dirty_from_porcelain(porcelain):
    """Mudanca rastreada, ou codigo/configuracao nao rastreado (um modulo
    novo que o simulador importa), suja a worktree; outros arquivos nao
    rastreados (PDFs, saidas) nao."""
    for line in porcelain.splitlines():
        if not line.strip():
            continue
        if not line.startswith("?? "):
            return True
    return bool(_untracked_code(porcelain))


def _git_info(root=None):
    """(commit, worktree suja, hash do diff) do repositorio do simulador. O
    hash cobre o diff rastreado e o conteudo do codigo nao rastreado."""
    root = root or os.path.dirname(os.path.abspath(__file__))

    def git(*a):
        # o Git escreve caminhos e diffs em UTF-8; o locale do Windows
        # (cp1252) nao decodifica bytes como 0x81 e derrubaria a rodada
        return subprocess.run(["git", *a], cwd=root, capture_output=True,
                              encoding="utf-8",
                              errors="surrogateescape").stdout

    def raw(text):
        return text.encode("utf-8", "surrogateescape")
    try:
        commit = git("rev-parse", "HEAD").strip() or None
        # um registro por ARQUIVO nao rastreado e sem aspas: sem
        # --untracked-files=all um diretorio novo vira "?? pacote/", e sem -z
        # nomes com espaco ou acento vem entre aspas (e nao terminam em .py)
        porcelain = git("status", "--porcelain", "-z",
                        "--untracked-files=all").replace("\0", "\n")
        dirty = _dirty_from_porcelain(porcelain)
        diff_hash = None
        if dirty:
            h = hashlib.sha256(raw(git("diff", "HEAD")))
            for rel in sorted(_untracked_code(porcelain)):
                try:
                    with open(os.path.join(root, rel), "rb") as f:
                        data = f.read()
                except OSError:
                    data = b""
                # nome e tamanho delimitam cada arquivo no hash
                h.update(b"\0".join((raw(rel), str(len(data)).encode(),
                                     data)))
            diff_hash = h.hexdigest()[:16]
    except OSError:
        return None, None, None
    return commit, dirty, diff_hash


def _package_versions():
    import matplotlib as mpl
    return {"numpy": np.__version__, "matplotlib": mpl.__version__}


def _table_meta(t):
    return {"source": t.source, "figure": t.figure,
            "uncertainty": t.uncertainty, "x_unit": t.x_unit,
            "isotonic_adjusted": t.isotonic_adjusted,
            "raw_points": list(t.raw_points), "points": list(t.points)}


def physical_model_sources():
    """Fontes e versoes de cada peca do modelo fisico em vigor."""
    prof = rp.get_radio_profile(RADIO_PROFILE)
    return {
        "model_version": MODEL_VERSION,
        "radio_profile": {"name": prof.name, "description": prof.description,
                          "sources": "DW1000 Datasheet v2.23; DW1000 User "
                                     "Manual v2.15; Cupid (DCOSS-IoT 2025) "
                                     "sec. V-A; Weaver (SenSys 2020) sec. "
                                     "6.1; EWSN 2020 Tab. 1"},
        "capture": {
            "model": CAPTURE_MODEL,
            "tables": {"same_short_dt": _table_meta(cm.SAME_SHORT_DT),
                       "same_long_dp": _table_meta(cm.SAME_LONG_DP),
                       "diff_async_k": _table_meta(cm.DIFF_ASYNC_K)},
            "legacy_async_weight": ASYNC_INTERF_WEIGHT,
            "intra_acquisition": CAPTURE_INTRA_ACQUISITION,
        },
        "fading": {"model": FADING_MODEL, "fast_fade_db": FAST_FADE_DB,
                   "slow_fraction": FADING_SLOW_FRACTION,
                   "epoch_corr": FADING_EPOCH_CORR},
        "bootstrap": {"model": BOOTSTRAP_MODEL,
                      "sources": "Cupid sec. IV-A; Weaver sec. 3.3, Tab. 1"},
        "grouping": {"model": GROUPING_MODEL,
                     "sources": "Cupid sec. IV-B, Alg. 1, sec. V-B"},
        "cupid_schedule": {"model": CUPID_SCHEDULE,
                           "sources": "Cupid Fig. 5b e 6c, sec. IV-C"},
        "crystal": {"model": CRYSTAL_MODEL,
                    "phase_timing": CRYSTAL_PHASE_TIMING,
                    "a_model": CRYSTAL_A_MODEL,
                    "sources": "Crystal (SenSys 2016); EWSN 2020 sec. 5"},
        "rng": RNG_MODEL,
    }


def active_hypotheses():
    out = {name: {"valor": globals()[name], "classe": cls}
           for name, cls in HYPOTHESES.items()}
    prof = rp.get_radio_profile(RADIO_PROFILE)
    if not prof.fixed:
        out["radio.noise_bandwidth_mhz"] = {
            "valor": prof.noise_bandwidth_mhz, "classe": "hipotese"}
        out["radio.antenna_gain_db"] = {"valor": prof.antenna_gain_db,
                                        "classe": "hipotese"}
        out["radio.sensitivity_preamble_penalty_db"] = {
            "valor": prof.sensitivity_preamble_penalty_db,
            "classe": "hipotese"}
        out["radio.tx_psd_flat_at_mask"] = {"valor": prof.tx_psd_dbm_mhz,
                                            "classe": "hipotese"}
    return out


def build_run_manifest(scenario=None, preset=None, requested=None,
                       experiments=None, suite=None, scenario_file_hash=None,
                       hypothetical_hardware=False, clamps=(), protocols=None,
                       capture_flags=None, outputs=None, status="completo",
                       git_info=None):
    """[Fase 9] Proveniencia completa de uma execucao. 'status' e
    'incompleto' no manifesto gravado antes dos experimentos."""
    from datetime import datetime, timezone
    eff = effective_config()
    commit, dirty, diff_hash = git_info if git_info else _git_info()
    return {
        "schema": "cupid-run-manifest/1",
        "status": status,
        "scenario": scenario, "suite": suite, "preset": preset or "cloves",
        "model_version": MODEL_VERSION,
        "git_commit": commit, "git_dirty": dirty, "git_diff_hash": diff_hash,
        "requested_config": dict(requested or {}),
        "effective_config": eff,
        "effective_config_hash": config_hash(eff),
        "clamps": list(clamps),
        "scenario_file_hash": scenario_file_hash,
        "python_version": sys.version,
        "package_versions": _package_versions(),
        "seeds": {"SEED": SEED, "topology_seeds": [
            SEED + 1000 * k for k in range(N_TOPOLOGIES)]},
        "n_topologies": N_TOPOLOGIES, "epochs": EPOCHS,
        "experiments": sorted(experiments or ALL_EXPERIMENTS),
        "protocols": list(protocols or PROTOCOLS),
        "hypothetical_hardware": bool(hypothetical_hardware),
        "timestamp": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "physical_model_sources": physical_model_sources(),
        "active_hypotheses": active_hypotheses(),
        "extrapolation": {
            "capture_model": CAPTURE_MODEL,
            "capture_extrapolation": CAPTURE_EXTRAPOLATION,
            "capture_flags_observed": dict(capture_flags or {}),
        },
        "outputs": list(outputs or []),
    }


SAMPLE_FIELDS = ["protocolo", "topologia_seed", "epoca", "n_originadores",
                 "entregues", "completa", "lat_epoca_ms",
                 "lat_ultimo_relatorio_ms", "bootstrap_ms",
                 "grouping_setup_ms", "tx_data", "rx_data", "tx_frames",
                 "rx_frames", "plan_id", "group_source"]


def _sample_rows(cdf_stats):
    rows = []
    for proto, st in cdf_stats.items():
        for seed, bt in st["by_topology"].items():
            for r in bt["records"]:
                rows.append({
                    "protocolo": proto, "topologia_seed": seed,
                    "epoca": r["epoch"],
                    "n_originadores": len(r["originators"]),
                    "entregues": r["delivered"],
                    "completa": int(r["complete"]),
                    "lat_epoca_ms": r["latency_data_epoch_ms"],
                    "lat_ultimo_relatorio_ms": r["last_report_ms"],
                    "bootstrap_ms": r["bootstrap_ms"],
                    "grouping_setup_ms": bt["grouping_setup_ms"],
                    "tx_data": r["tx_data"], "rx_data": r["rx_data"],
                    "tx_frames": r["tx_frames"], "rx_frames": r["rx_frames"],
                    "plan_id": bt["plan_id"],
                    "group_source": r["group_source"]})
    return rows


def protocol_metrics(st, tau):
    """[Fase 8] Metricas de um protocolo, com a topologia como unidade."""
    per = {"rmst": {}, "pdr": {}, "complete": {}, "median": {}}
    last, boots, tx, rx, sent, grouping = [], [], 0, 0, 0, []
    for seed, bt in st["by_topology"].items():
        recs = bt["records"]
        s = es.summarize_censored_latency(bt["lats"], tau)
        per["rmst"][seed] = s["rmst"]
        per["complete"][seed] = s["frac_complete"]
        per["median"][seed] = s["median"]
        n_sent = sum(len(r["originators"]) for r in recs)
        per["pdr"][seed] = (sum(r["delivered"] for r in recs) / n_sent
                            if n_sent else math.nan)
        last += [r["last_report_ms"] for r in recs
                 if r["last_report_ms"] is not None]
        boots += [r["bootstrap_ms"] for r in recs]
        tx += sum(r["tx_data"] for r in recs)
        rx += sum(r["rx_data"] for r in recs)
        sent += n_sent
        grouping.append(bt["grouping_setup_ms"])
    pooled = es.summarize_censored_latency(st["lats"], tau)
    report = es.summarize_censored_latency(
        st["report_lats"] + [math.inf] * (st["sent"] - len(st["report_lats"])),
        tau)
    return {
        "per_topology": per, "pooled": pooled, "report": report,
        "rmst_ci": mean_ci(list(per["rmst"].values())),
        "pdr_ci": mean_ci(list(per["pdr"].values())),
        "complete_ci": mean_ci(list(per["complete"].values())),
        "last_report_median": (float(np.median(last)) if last else None),
        "bootstrap_mean_ms": (float(np.mean(boots)) if boots else math.nan),
        "grouping_setup_ms": (float(np.mean(grouping)) if grouping
                              else math.nan),
        "tx_per_report": tx / sent if sent else math.nan,
        "rx_per_report": rx / sent if sent else math.nan,
        "retx_per_report": (tx - sent) / sent if sent else math.nan,
    }


def _fmt_ci(mc, fmt="{:.1f}"):
    m, ci = mc
    if math.isnan(m):
        return "n/d"
    return (fmt.format(m) + (" +- " + fmt.format(ci)
                             if not math.isnan(ci) else " (1 topologia)"))


def _fmt_median(summary):
    if summary["median"] is not None:
        return f"{summary['median']:.1f} ms"
    if summary["median_lower_bound"] is not None:
        return f">= {summary['median_lower_bound']:.1f} ms (nao ident.)"
    return "n/d"


def main(out_dir="outputs", scenario_name=None, requested=None, clamps=(),
         experiments=None, protocols=None, preset=None, suite=None,
         hypothetical_hardware=False, scenario_file_hash=None,
         amortize_grouping=False):
    os.makedirs(out_dir, exist_ok=True)
    # [Fase 9] commit e worktree lidos UMA vez, no inicio: o resultado e do
    # codigo que rodou, nao do HEAD de quando a rodada termina
    git_info = _git_info()
    exps, protos = suite_selection(suite, experiments, protocols)
    print("=" * 66)
    print(f"  {'CENARIO: ' + scenario_name if scenario_name else 'CUPID SIM v2'}")
    print("=" * 66)
    print(f"  Preset: {preset or 'cloves'} | modelo {MODEL_VERSION} | "
          f"experimentos {','.join(exps)} | protocolos {','.join(protos)}")
    print(f"  Perfil de radio: {RADIO_PROFILE} "
          f"({rp.get_radio_profile(RADIO_PROFILE).description})")
    if RADIO_OVERRIDES:
        print(f"  OVERRIDES de parametros derivados: {RADIO_OVERRIDES}")
    print("  Alcance confiavel da BFS diagnostica: "
          f"{reliable_range_m():.1f} m (hops da epoca: {BOOTSTRAP_MODEL})")
    print(f"  Ruido efetivo pos-despreading: {NOISE_EFF_DBM:.1f} dBm | "
          f"SINR exigido: {REQ_SNR_DB:.1f} dB | captura: {CAPTURE_MODEL}")
    print(f"  Topologias por ponto: {N_TOPOLOGIES} | Epocas: {EPOCHS}")
    enforce_channel_plan()
    print(f"  Plano de canais: {CHANNEL_PLAN} -> K={N_CHANNELS}, "
          f"{N_RADIOS_PER_NODE} radio(s)/no ({MAX_RX_CHANNELS} canal(is) "
          f"simultaneo(s))")
    print(f"  Modelo de ACK do sink: {ACK_MODEL}")
    print("-" * 66)

    topos = make_topologies()
    topo0 = topos[0][0]
    nh = defaultdict(int)
    for i in range(topo0.n):
        if topo0.hop[i] <= MAX_HOP:
            nh[topo0.hop[i]] += 1
    print("  Nos por hop na BFS diagnostica (topologia 0):",
          dict(sorted(nh.items())))

    effective = effective_config()
    cfg_hash = config_hash(effective)
    fwb_channels = [len(set(build_channel_assignment("cupid_fwb", t).values()))
                    for t, _ in topos]
    with open(f"{out_dir}/config_efetiva.json", "w", encoding="utf-8") as f:
        json.dump({"cenario": scenario_name, "preset": preset or "cloves",
                   "pedido": requested or {}, "efetivo": effective,
                   "truncamentos": list(clamps),
                   "hash_config_efetiva": cfg_hash,
                   "fwb_canais_efetivos_por_topologia": fwb_channels},
                  f, indent=2, ensure_ascii=False)
        f.write("\n")

    outputs = ["config_efetiva.json"]
    capture_flags = {}

    def write_manifest(status):
        m = build_run_manifest(
            scenario=scenario_name, preset=preset, requested=requested,
            experiments=exps, suite=suite,
            scenario_file_hash=scenario_file_hash,
            hypothetical_hardware=hypothetical_hardware, clamps=clamps,
            protocols=protos, capture_flags=capture_flags, outputs=outputs,
            status=status, git_info=git_info)
        with open(f"{out_dir}/run_manifest.json", "w",
                  encoding="utf-8") as f:
            json.dump(m, f, indent=2, ensure_ascii=False, default=str)
            f.write("\n")

    # [Fase 9] um cenario interrompido deixa um manifesto 'incompleto': a
    # retomada sabe o que foi pedido e o refaz, sem precisar de --force
    write_manifest("incompleto")
    cdf = cdf_stats = lat_o = pdr = None
    if "cdf" in exps:
        print("Exp cdf: latencia por epoca e por relatorio (O=35)...")
        cdf, cdf_stats = experiment_latency_cdf(topos, O=35, protos=protos,
                                                capture_flags=capture_flags)
    if "carga" in exps:
        print("Exp carga: latencia vs originadores...")
        Ovals, lat_o = experiment_latency_vs_originators(topos, protos=protos)
    if "pdr" in exps:
        print("Exp pdr: PDR com deadline real...")
        freqs, pdr = experiment_pdr_vs_frequency(topos, O=35, protos=protos)
    if "funil" in exps:
        print("Exp funil: FWB vs numero de canais (1 radio vs K radios)...")
        ch, curves, ref, ref_ci = experiment_funnel(topos, O=35)
    if "crosstalk" in exps:
        print("Exp crosstalk: FWB x ACLR (HIPOTETICO: >= 2 radios/no)...")
        acl, acl_m, acl_ci, acl_cens = experiment_crosstalk(topos, O=35)
    if "radios" in exps:
        print("Exp radios: dependencia de radios multiplos por no...")
        rxk, rxk_m, rxk_ci, rxk_cens = experiment_radio_limit(topos, O=35)
    if "planos" in exps:
        print("Exp planos: planos de canais UWB reais...")
        plan_rows, plan_ref = experiment_channel_plans(topos, O=35)
    if "micro" in exps:
        print("Exp micro: micro-benchmark de captura...")
        micro = microbenchmark_capture(topos)

    print("Gerando graficos...")
    plot_topology(topo0, f"{out_dir}/01_topologia.png")
    outputs.append("01_topologia.png")
    if cdf is not None:
        plot_cdf(cdf, f"{out_dir}/02_cdf_latencia.png")
        plot_report_cdf(cdf_stats, f"{out_dir}/02b_cdf_latencia_relatorio.png")
        outputs += ["02_cdf_latencia.png", "02b_cdf_latencia_relatorio.png"]
        es.export_samples(f"{out_dir}/amostras_epocas.csv",
                          _sample_rows(cdf_stats), SAMPLE_FIELDS)
        outputs.append("amostras_epocas.csv")
    if lat_o is not None:
        plot_latency_vs_o(Ovals, lat_o,
                          f"{out_dir}/03_latencia_vs_originadores.png")
        outputs.append("03_latencia_vs_originadores.png")
    if pdr is not None:
        plot_pdr(freqs, pdr, f"{out_dir}/04_pdr_vs_frequencia.png")
        outputs.append("04_pdr_vs_frequencia.png")
    if "funil" in exps:
        plot_funnel(ch, curves, ref, ref_ci, f"{out_dir}/05_funnel_canais.png")
        outputs.append("05_funnel_canais.png")
    if "crosstalk" in exps:
        plot_sensitivity(acl, acl_m, acl_ci, "ACLR - rejeicao de canal (dB)",
                         "HIPOTETICO: FWB com >= 2 radios x crosstalk",
                         f"{out_dir}/06_crosstalk.png", invert=True)
        outputs.append("06_crosstalk.png")
    if "radios" in exps:
        plot_sensitivity(rxk, rxk_m, rxk_ci, "Radios por no",
                         "Quanto do FWB depende de multiplos radios",
                         f"{out_dir}/07_radios_por_no.png")
        outputs.append("07_radios_por_no.png")
    if "planos" in exps:
        plot_channel_plans(plan_rows, plan_ref,
                           f"{out_dir}/08_planos_de_canais.png")
        outputs.append("08_planos_de_canais.png")

    tau = epoch_cap_ms()
    L = ["=" * 60, "  RESUMO (modelo " + MODEL_VERSION + ")", "=" * 60]
    L.append(f"  Preset {preset or 'cloves'} | config efetiva {cfg_hash} "
             f"(config_efetiva.json, run_manifest.json)")
    for c in clamps:
        L.append(f"  TRUNCADO (--allow-clamp): {c['param']} pedido "
                 f"{c['pedido']} -> efetivo {c['efetivo']} ({c['motivo']})")
    L.append(f"  Canais: plano {CHANNEL_PLAN}, K={N_CHANNELS}, "
             f"{N_RADIOS_PER_NODE} radio(s)/no"
             + (" [HARDWARE HIPOTETICO]" if N_RADIOS_PER_NODE > 1 else "")
             + f"; o FWB usou de fato {fwb_channels} canal(is) por topologia")
    L.append(f"  ACK do sink: {ACK_MODEL} | captura: {CAPTURE_MODEL} "
             f"(extrapolacao: {CAPTURE_EXTRAPOLATION})")
    if cdf_stats is not None:
        L.append("  [Fase 8] Unidade independente: TOPOLOGIA "
                 f"({N_TOPOLOGIES}); IC = t de Student, n-1 g.l.")
        L.append(f"  Epoca incompleta = censurada em {tau:.0f} ms (MAX_SLOTS)."
                 " RMST = area sob a sobrevivencia ate o prazo, nao media.")
        mets = {p: protocol_metrics(cdf_stats[p], tau) for p in protos}
        L.append(f"  {'protocolo':10s} {'mediana epoca':>28s} "
                 f"{'RMST epoca (ms)':>18s} {'completas':>15s} "
                 f"{'PDR s/ prazo':>15s}")
        for p in protos:
            m = mets[p]
            L.append(f"  {LABELS[p]:10s} {_fmt_median(m['pooled']):>28s} "
                     f"{_fmt_ci(m['rmst_ci']):>18s} "
                     f"{_fmt_ci(m['complete_ci'], '{:.3f}'):>15s} "
                     f"{_fmt_ci(m['pdr_ci'], '{:.3f}'):>15s}")
        L.append("")
        L.append(f"  {'protocolo':10s} {'lat. relatorio':>28s} "
                 f"{'ult. relatorio*':>16s} {'bootstrap':>10s} "
                 f"{'grouping**':>11s} {'TX/rel':>7s} {'RX/rel':>7s} "
                 f"{'reTX/rel':>8s}")
        for p in protos:
            m = mets[p]
            lr = ("n/d" if m["last_report_median"] is None
                  else f"{m['last_report_median']:.1f} ms")
            L.append(f"  {LABELS[p]:10s} {_fmt_median(m['report']):>28s} "
                     f"{lr:>16s} {m['bootstrap_mean_ms']:8.1f}ms "
                     f"{m['grouping_setup_ms']:9.1f}ms "
                     f"{m['tx_per_report']:7.1f} {m['rx_per_report']:7.1f} "
                     f"{m['retx_per_report']:8.1f}")
        L.append("  * latencia ate o ultimo pacote recebido, como no artigo do"
                 " Cupid (inclui epocas incompletas)")
        L.append("  ** periodo de agrupamento por topologia, FORA da latencia"
                 " da coleta" + (
                     "; amortizado por epoca: " + ", ".join(
                         f"{LABELS[p]} {mets[p]['grouping_setup_ms'] / EPOCHS:.2f} ms"
                         for p in protos)
                     if amortize_grouping else ""))
        L.append("  Energia: nao reportada (sem modelo com fonte).")
        L.append("")
        L.append("  Comparacoes pareadas por topologia (RMST da epoca, a - b;"
                 " negativo = a mais rapido):")
        for a_, b_ in (("cupid", "weaver"), ("cupid", "crystal"),
                       ("weaver", "crystal"), ("cupid_fwb", "cupid")):
            if a_ in mets and b_ in mets:
                eff = es.paired_effect(mets[a_]["per_topology"]["rmst"],
                                       mets[b_]["per_topology"]["rmst"])
                if eff["n"] >= 5:
                    lo, hi = es.paired_bootstrap_ci(
                        mets[a_]["per_topology"]["rmst"],
                        mets[b_]["per_topology"]["rmst"], n_boot=2000,
                        seed=SEED)
                    boot = f"bootstrap [{lo:.1f}; {hi:.1f}]"
                else:   # reamostrar menos de 5 topologias nao informa nada
                    boot = "bootstrap n/d (n < 5)"
                ci = ("IC n/d (1 topologia)" if eff["n"] < 2 else
                      f"IC t [{eff['ci_low']:.1f}; {eff['ci_high']:.1f}], "
                      f"{boot}")
                L.append(f"    {LABELS[a_]:9s} - {LABELS[b_]:9s}: "
                         f"{eff['mean_diff']:+.1f} ms ({ci}; n={eff['n']})")
        if capture_flags:
            L.append("  Hipoteses de captura usadas (contagens): " + ", ".join(
                f"{k}={v}" for k, v in sorted(capture_flags.items())))
        L.append("")
    if pdr is not None:
        L.append("  PDR a 6 epocas/s (deadline real): " + "  ".join(
            f"{LABELS[p]}: {pdr[p]['mean'][5]*100:.1f}%" for p in protos
            if p in pdr))
        L.append("")
    if "micro" in exps:
        L.append("  Micro-benchmark de captura  k | P_cap | P_far")
        for k, pc, pf in micro:
            L.append(f"      {k:3d} | {pc*100:5.1f}% | {pf*100:5.2f}%")
        L.append("")
    if "crosstalk" in exps:
        L.append("  HIPOTETICO -- FWB x crosstalk (>= 2 radios/no; ACLR dB -> "
                 "lat. epoca):")
        L.append("      " + "  ".join(f"{a}dB:{fmt_latency(m, c)}"
                                       for a, m, c in zip(acl, acl_m,
                                                          acl_cens)))
    if "radios" in exps:
        L.append("  Dependencia de radios por no (radios -> lat. epoca; >1 = "
                 "HIPOTETICO):")
        L.append("      " + "  ".join(f"{k}:{fmt_latency(m, c)}"
                                       for k, m, c in zip(rxk, rxk_m,
                                                          rxk_cens)))
    if "planos" in exps:
        L.append("")
        L.append("  -- PLANOS DE CANAIS UWB (regulacao + hardware) --")
        L.append(f"  Referencia Cupid (canal unico): {fmt_latency(plan_ref)}")
        L.append(f"  {'plano':16s} {'K':>3s} {'radios':>6s} {'latencia':>12s} "
                 f"{'ganho':>8s}   descricao")
        for r in plan_rows:
            ganho = "n/d" if math.isnan(r["ganho"]) else f"{r['ganho']:.1f}%"
            L.append(f"  {r['plano']:16s} {r['K']:3d} {r['radios']:6d} "
                     f"{fmt_latency(r['lat'], r['censored']):>12s} {ganho:>8s}"
                     f"   {r['desc']}")
        L.append("  NOTA: K > 4 ou > 1 radio/no nao e hardware DW1000 real.")
    L.append("=" * 60)
    summary = "\n".join(L)
    print("\n" + summary)
    with open(f"{out_dir}/resumo_resultados.txt", "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    outputs.append("resumo_resultados.txt")
    write_manifest("completo")
    return summary


def _parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Simulador CTX/UWB v2.")
    p.add_argument("--config")
    p.add_argument("--out", default="outputs")
    p.add_argument("--name")
    p.add_argument("--preset", choices=sorted(PRESETS),
                   help="configuracao-base (padrao: cloves).")
    p.add_argument("--suite", choices=sorted(SUITES),
                   help="experimentos e protocolos da suite.")
    p.add_argument("--experiments",
                   help="lista separada por virgulas: "
                        + ",".join(ALL_EXPERIMENTS))
    p.add_argument("--protocols", help="lista separada por virgulas.")
    p.add_argument("--hypothetical-hardware", action="store_true",
                   help="declara hardware HIPOTETICO (> 1 radio por no).")
    p.add_argument("--scenario-file-hash")
    p.add_argument("--amortize-grouping", action="store_true",
                   help="reporta tambem o agrupamento amortizado por epoca.")
    p.add_argument("--allow-clamp", action="store_true",
                   help="[F20] Aceita explicitamente que um parametro pedido "
                        "seja truncado (ex.: N_CHANNELS acima do plano).")
    for name in PARAM_NAMES:
        p.add_argument(f"--{name}")
    args = p.parse_args(argv)
    for opt, valid in (("experiments", ALL_EXPERIMENTS),
                       ("protocols", PROTOCOLS)):
        raw = getattr(args, opt)
        if raw:
            items = [x for x in raw.split(",") if x]
            bad = [x for x in items if x not in valid]
            if bad:
                p.error(f"--{opt}: desconhecido(s) {bad}; validos: "
                        f"{','.join(valid)}")
            setattr(args, opt, items)
    return args


_TRUE_WORDS = ("1", "true", "yes", "sim")
_FALSE_WORDS = ("0", "false", "no", "nao")


def _coerce(name, raw):
    """Texto da CLI (ou str(valor) do runner) -> tipo do parametro. Nada e
    truncado nem adivinhado: '2.0' vale 2, mas '2.5' ou 'ture' sao erro."""
    base = globals()[name]
    if isinstance(base, bool):
        word = raw.strip().lower()
        if word in _TRUE_WORDS or word in _FALSE_WORDS:
            return word in _TRUE_WORDS
        raise ValueError(f"{name} deve ser booleano (recebido {raw})")
    if isinstance(base, int):
        try:
            return int(raw)                 # exato, mesmo acima de 2**53
        except ValueError:
            value = float(raw)
            if not value.is_integer():      # tambem nan e inf
                raise ValueError(f"{name} deve ser inteiro (recebido {raw})")
            return int(value)
    if isinstance(base, float):
        return float(raw)
    return raw


def cli(argv=None):
    args = _parse_args(argv)
    requested = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            requested.update(json.load(f))
    for name in PARAM_NAMES:
        raw = getattr(args, name)
        if raw is not None:
            requested[name] = _coerce(name, raw)
    cfg = dict(PRESETS[args.preset or "cloves"], **requested)
    # [F20] pedido truncado e erro, nao aviso -- antes de rodar qualquer coisa
    clamps = requested_clamps(requested)
    if clamps and not args.allow_clamp:
        for c in clamps:
            print(f"ERRO: {c['param']}={c['pedido']} seria truncado para "
                  f"{c['efetivo']} ({c['motivo']}). Ajuste o cenario ou "
                  f"passe --allow-clamp.", file=sys.stderr)
        return 2
    if cfg:
        apply_config(cfg)
    if N_RADIOS_PER_NODE > 1 and not args.hypothetical_hardware:
        # [Fase 10] o DW1000 tem um front-end so: mais radios por no e
        # hardware customizado e tem de ser declarado
        print(f"ERRO: N_RADIOS_PER_NODE={N_RADIOS_PER_NODE} nao existe no "
              f"DW1000; declare hardware hipotetico com "
              f"--hypothetical-hardware.", file=sys.stderr)
        return 2
    # sem --experiments/--protocols, main() usa os da suite
    main(out_dir=args.out, scenario_name=args.name, requested=requested,
         clamps=clamps, experiments=args.experiments,
         protocols=args.protocols,
         preset=args.preset, suite=args.suite,
         hypothetical_hardware=args.hypothetical_hardware,
         scenario_file_hash=args.scenario_file_hash,
         amortize_grouping=args.amortize_grouping)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
