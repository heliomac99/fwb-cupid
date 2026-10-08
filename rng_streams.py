# -*- coding: utf-8 -*-
"""
Streams aleatorios por chave estavel.

Uma semente e derivada de uma CHAVE (tupla de inteiros e textos) por
blake2b -- nunca por hash(), que o Python aleatoriza por processo. Assim a
mesma chave da o mesmo stream em qualquer processo, ordem ou paralelismo.

Na simulacao de uma epoca, cada slot tem um gerador Philox proprio, chaveado
por (raiz da epoca, slot), que sorteia SEMPRE as mesmas matrizes, indexadas
por no: fading por par (tx, rx), sorteio de decodificacao por (rx, canal),
atraso de TX, participacao e escolha de fila. Um sorteio de (slot, tx, rx)
nao depende de quem mais transmitiu, da ordem de iteracao nem do protocolo
-- protocolos diferentes veem os mesmos numeros no mesmo slot (numeros
aleatorios comuns), o que e o que se quer numa comparacao pareada.
"""

import hashlib
import random

import numpy as np


def key_seed(*key):
    """Semente de 64 bits para uma chave (estavel entre processos)."""
    blob = repr(tuple(key)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(blob, digest_size=8).digest(),
                          "little")


def stream(*key):
    """random.Random proprio da chave."""
    return random.Random(key_seed(*key))


MAX_DRAW_CHANNELS = 16     # maior K de qualquer plano de canais


class SlotDraws:
    """Sorteios fisicos de UM slot, indexados por no (e canal)."""
    __slots__ = ("fading_normal", "decode_u", "txdelay_u", "txprob_u",
                 "queue_u")

    def __init__(self, seed, n, channels):
        if channels > MAX_DRAW_CHANNELS:
            raise ValueError(f"mais de {MAX_DRAW_CHANNELS} canais por slot")
        g = np.random.Generator(np.random.Philox(key=seed))
        # ordem e FORMA fixas: nunca dependem do que o slot usa nem de
        # N_CHANNELS (a matriz de decodificacao tem sempre o maximo de
        # colunas, para os sorteios seguintes nao se deslocarem)
        self.fading_normal = g.standard_normal((n, n))   # [tx][rx]
        self.decode_u = g.random((n, MAX_DRAW_CHANNELS))   # [rx][canal]
        self.txdelay_u = g.random(n)
        self.txprob_u = g.random(n)
        self.queue_u = g.random(n)


class EpochStreams:
    """Todos os streams de uma epoca, a partir de uma raiz de chave."""

    def __init__(self, *root):
        self.root = tuple(root)

    @classmethod
    def from_rng(cls, rng):
        """Raiz sorteada de um rng (um unico sorteio)."""
        return cls("rng", rng.getrandbits(64))

    def sub(self, *kind):
        """Stream sequencial proprio (bootstrap, S do Crystal, ...)."""
        return stream(*self.root, *kind)

    def slot(self, slot, n, channels=1, kind="slot"):
        return SlotDraws(key_seed(*self.root, kind, slot), n, channels)

    def child(self, *kind):
        """Streams de uma sub-fase com raiz propria (ex.: a cauda)."""
        return EpochStreams(*self.root, *kind)
