"""Achados da revisao independente de 27/09 (cada teste pega um defeito).

1 commit lido no fim da rodada; .py nao rastreado invisivel. 2 retomada
ignorava o status do manifesto. 3 modelo empirico consultava outra copia do
modulo sob __main__. 4 protocolos fora da comparacao. 5 suite legada nunca
retomavel. 6 hash do runner diferente do subprocesso (coercao de tipo). 7
configuracao vazando entre tarefas das bancadas (e U = 35 calculado numa rede
de 22 nos). 8 modo chaveado + Crystal sequencial. 9 limite inferior da
mediana com exatamente metade completa. 10 copias de um quadro concorrente
contavam uma vez so como interferencia. 11 tamanho dos quadros S/A do Crystal.

Fechamento da revisao: main() sem protocolos usava todos (a suite nao
contava); .py nao rastreado dentro de diretorio novo, ou com espaco/acento no
nome, escapava do 'git status' condensado; o runner lia o Git a cada cenario
e no fim; _coerce arredondava inteiros grandes e aceitava qualquer palavra
como falso; a calibracao nao registrava a rede em que cada ponto rodou; no
modo sequencial o S do Crystal reusava o stream do bootstrap dos CTX.
"""

import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import calibracao_captura as cc
import capture_models as cm
import cupid_sim_v2 as sim
import experiment_stats as es
import rng_streams as rs
import run_cenarios as rc
import sensibilidade as sens

TINY = {"N_NODES": 8, "AREA_X": 30.0, "AREA_Y": 8.0, "EPOCHS": 2,
        "N_TOPOLOGIES": 1, "MAX_SLOTS": 200, "GROUPING_ITERATIONS": 1}


class Base(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            name: getattr(sim, name) for name in sim.PARAM_NAMES
        }

    def tearDown(self):
        sim.apply_config(self.original_config)


class ProvenanceFixes(Base):
    def test_git_info_is_read_once_at_the_start_of_the_run(self):
        calls = iter([("a" * 40, False, None), ("b" * 40, False, None)])
        sim.apply_config(TINY)
        with tempfile.TemporaryDirectory() as out, \
                patch.object(sim, "_git_info", side_effect=lambda: next(calls)):
            sim.main(out_dir=out, experiments=["cdf"],
                     protocols=["weaver"], preset="cloves", suite="core")
            m = json.loads((Path(out) / "run_manifest.json").read_text())
        self.assertEqual(m["git_commit"], "a" * 40)

    def test_untracked_code_counts_as_dirty(self):
        self.assertTrue(sim._dirty_from_porcelain(" M cupid_sim_v2.py\n"))
        self.assertTrue(sim._dirty_from_porcelain("?? novo_modulo.py\n"))
        self.assertTrue(sim._dirty_from_porcelain("?? cenarios_x.json\n"))
        self.assertFalse(sim._dirty_from_porcelain("?? sbesc2026_cupid.pdf\n"))
        self.assertFalse(sim._dirty_from_porcelain(""))

    def test_untracked_code_is_found_inside_a_new_directory(self):
        """Sem --untracked-files=all o Git condensa um diretorio novo em
        '?? novo_pacote/', e o .py dentro dele ficava fora da proveniencia."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self.git_repo(repo)
            self.assertEqual(sim._git_info(repo)[1:], (False, None))
            (repo / "novo_pacote").mkdir()
            modelo = repo / "novo_pacote" / "modelo.py"
            modelo.write_text("A = 1\n")
            _, dirty, h1 = sim._git_info(repo)
            self.assertTrue(dirty)
            self.assertIsNotNone(h1)
            modelo.write_text("A = 2\n")          # so o conteudo muda
            self.assertNotEqual(sim._git_info(repo)[2], h1)

    def test_untracked_names_with_spaces_or_accents_are_hashed(self):
        """Sem -z o Git poe esses nomes entre aspas ('?? "a b.py"')."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self.git_repo(repo)
            for name in ("novo modulo.py", "ação.py"):
                with self.subTest(name=name):
                    path = repo / name
                    path.write_text("A = 1\n", encoding="utf-8")
                    _, dirty, h1 = sim._git_info(repo)
                    self.assertTrue(dirty)
                    path.write_text("A = 2\n", encoding="utf-8")
                    self.assertNotEqual(sim._git_info(repo)[2], h1)
                    path.unlink()

    def test_only_code_or_configuration_marks_the_tree_dirty(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self.git_repo(repo)
            (repo / "artigo.pdf").write_bytes(b"%PDF-1.4")
            (repo / "saidas").mkdir()
            (repo / "saidas" / "grafico.png").write_bytes(b"\x89PNG")
            self.assertEqual(sim._git_info(repo)[1:], (False, None))
            (repo / "configs").mkdir()
            (repo / "configs" / "cenario.json").write_text("{}")
            self.assertTrue(sim._git_info(repo)[1])

    @staticmethod
    def git_repo(path):
        """Repositorio minimo com um commit (identidade local ao comando)."""
        import shutil
        import subprocess
        if shutil.which("git") is None:
            raise unittest.SkipTest("git indisponivel")

        def git(*a):
            subprocess.run(["git", "-c", "user.name=t", "-c",
                            "user.email=t@t", *a], cwd=path, check=True,
                           capture_output=True)
        git("init", "-q")
        (path / "sim.py").write_text("X = 1\n")
        git("add", "sim.py")
        git("commit", "-qm", "init")

    def request(self, **kw):
        cen = {"nome": "cen", "suite": "core", "parametros": {"SEED": 7}}
        base = dict(preset="cloves", suite="core", experiments=["cdf"],
                    overrides={})
        base.update(kw)
        return rc.scenario_request(cen, **base)

    def test_incomplete_manifest_is_rerun_even_with_an_old_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            req = self.request()
            m = dict(rc.expected_manifest_fields(req), status="incompleto")
            (out / "run_manifest.json").write_text(json.dumps(m))
            (out / "resumo_resultados.txt").write_text("velho")
            self.assertEqual(rc.check_resume(out, req)[0], "rodar")

    def test_protocols_are_part_of_the_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            req = self.request(protocols=["cupid", "weaver", "crystal"])
            m = dict(rc.expected_manifest_fields(req), status="completo")
            (out / "run_manifest.json").write_text(json.dumps(m))
            (out / "resumo_resultados.txt").write_text("ok")
            self.assertEqual(rc.check_resume(out, req)[0], "pular")
            other = self.request(protocols=["cupid", "weaver"])
            self.assertEqual(rc.check_resume(out, other)[0], "incompativel")

    def test_legacy_suite_is_forwarded_and_matches_the_manifest(self):
        self.assertIn("legacy", sim.SUITES)
        cmd = rc.construir_comando("x", {}, "o", suite="legacy")
        self.assertEqual(cmd[cmd.index("--suite") + 1], "legacy")
        req = self.request(suite="legacy")
        self.assertEqual(req["protocols"], list(sim.PROTOCOLS))

    def test_runner_hash_follows_the_subprocess_coercion(self):
        cen = {"nome": "c", "suite": "core"}
        as_int = rc.scenario_request(dict(cen, parametros={"SLOT_US": 460}),
                                     "cloves", "core", ["cdf"], {})
        as_float = rc.scenario_request(dict(cen, parametros={
            "SLOT_US": 460.0}), "cloves", "core", ["cdf"], {})
        self.assertEqual(as_int["effective_config_hash"],
                         as_float["effective_config_hash"])
        with self.assertRaisesRegex(ValueError, "inteiro"):
            sim._coerce("N_TOPOLOGIES", "2.5")
        self.assertEqual(sim._coerce("N_TOPOLOGIES", "2.0"), 2)

    def test_integer_coercion_is_exact(self):
        self.assertEqual(sim._coerce("N_TOPOLOGIES", "2"), 2)
        self.assertEqual(sim._coerce("N_CHANNELS", "4.0"), 4)
        big = "12345678901234567891"          # float() arredondaria
        self.assertEqual(sim._coerce("SEED", big), int(big))
        for bad in ("2.5", "4.000001", "nan", "inf", "dois"):
            with self.subTest(raw=bad), self.assertRaises(ValueError):
                sim._coerce("N_CHANNELS", bad)

    def test_boolean_coercion_rejects_unknown_words(self):
        for raw, value in (("True", True), ("1", True), ("sim", True),
                           ("False", False), ("0", False), ("nao", False)):
            self.assertIs(sim._coerce("GROUPING_RENEW", raw), value)
        with self.assertRaisesRegex(ValueError, "GROUPING_RENEW"):
            sim._coerce("GROUPING_RENEW", "ture")

    def test_the_runner_reads_git_once_per_run(self):
        """O manifesto da rodada leva o commit do INICIO, e o pedido de
        cada cenario compara contra esse mesmo estado."""
        calls = []

        def fake_git():
            calls.append(1)
            return (("a" if len(calls) == 1 else "b") * 40, False, None)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps({"preset": "cloves", "cenarios": [
                {"nome": n, "suite": "core", "parametros": TINY}
                for n in ("a", "b")]}))
            ok = ("a", "OK", 0.0, "ok", {})
            with patch.object(sim, "_git_info", side_effect=fake_git), \
                    patch.object(rc, "rodar_cenario", return_value=ok):
                rc.main([str(path), "--run-id", "t", "--jobs", "1",
                         "--outputs", tmp])
            m = json.loads((Path(tmp) / "t" / "run_manifest.json")
                           .read_text(encoding="utf-8"))
        self.assertEqual(m["git_commit"], "a" * 40)
        self.assertEqual(len(calls), 1)


class SuiteDefaults(Base):
    """main() resolve protocolos e experimentos pela suite, como cli(): o
    manifesto de uma chamada direta e o pedido do runner tem de coincidir."""

    def manifest_of(self, **kw):
        """Manifesto 'incompleto' gravado antes do primeiro experimento."""
        sim.apply_config(dict(TINY, SEED=7))
        with tempfile.TemporaryDirectory() as out, \
                patch.object(sim, "experiment_latency_cdf",
                             side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                sim.main(out_dir=out, preset="cloves", **kw)
            return json.loads((Path(out) / "run_manifest.json").read_text())

    def test_core_suite_without_protocols_records_the_core_protocols(self):
        m = self.manifest_of(suite="core", protocols=None)
        self.assertEqual(m["protocols"], ["cupid", "weaver", "crystal"])
        self.assertEqual(m["experiments"],
                         sorted(sim.SUITES["core"]["experiments"]))

    def test_explicit_protocols_win_over_the_suite(self):
        m = self.manifest_of(suite="core", protocols=["crystal", "weaver"])
        self.assertEqual(m["protocols"], ["weaver", "crystal"])  # canonica

    def test_legacy_suite_runs_everything(self):
        m = self.manifest_of(suite="legacy")
        self.assertEqual(m["protocols"], list(sim.PROTOCOLS))
        self.assertEqual(m["experiments"], sorted(sim.ALL_EXPERIMENTS))

    def test_interrupted_core_run_matches_the_runner_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            sim.apply_config(dict(TINY, SEED=7))
            with patch.object(sim, "experiment_latency_cdf",
                              side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    sim.main(out_dir=str(out), scenario_name="cen",
                             experiments=["cdf"], preset="cloves",
                             suite="core")
            req = rc.scenario_request(
                {"nome": "cen", "suite": "core",
                 "parametros": dict(TINY, SEED=7)},
                preset="cloves", suite="core", experiments=["cdf"],
                overrides={})
            self.assertEqual(rc.check_resume(out, req),
                             ("rodar", "execucao anterior incompleta, "
                                       "mesmo pedido"))


class BenchIsolation(Base):
    def test_sensitivity_tasks_start_from_the_defaults(self):
        defaults = dict(sens.DEFAULTS)
        sim.apply_config({"TX_PROB": 0.5, "N_NODES": 22})
        sim.apply_config(sens.task_config({}))
        self.assertEqual(sim.TX_PROB, defaults["TX_PROB"])
        self.assertEqual(sim.N_NODES, defaults["N_NODES"])

    def test_calibration_gives_each_u_its_own_network(self):
        cfg = {"ASYNC_INTERF_WEIGHT": 0.06}
        self.assertEqual(cc.config_for_u(cfg, 21)["N_NODES"], 22)
        self.assertEqual(cc.config_for_u(cfg, 35)["N_NODES"],
                         cc.DEFAULTS["N_NODES"])
        self.assertEqual(cc.config_for_u(cfg, 35)["ASYNC_INTERF_WEIGHT"],
                         0.06)

    def test_fresh_config_drops_the_previous_radio_overrides(self):
        """Tarefa anterior com outro perfil e um override de parametro
        derivado: a proxima comeca do perfil pedido, sem override."""
        sim.apply_config({"RADIO_PROFILE": "legacy_v2", "NOISE_DBM": -80.0})
        self.assertIn("NOISE_DBM", sim.RADIO_OVERRIDES)
        sim.apply_config(sens.task_config({}))
        self.assertEqual(sim.RADIO_PROFILE, sens.DEFAULTS["RADIO_PROFILE"])
        self.assertEqual(sim.NOISE_DBM, sens.DEFAULTS["NOISE_DBM"])
        self.assertEqual(sim.RADIO_OVERRIDES, {})
        sim.apply_config({"NOISE_DBM": -80.0})       # override no perfil padrao
        sim.apply_config(cc.task_config({"RADIO_PROFILE": "legacy_v2"}))
        derived = sim.rp.derived_params(sim.rp.get_radio_profile("legacy_v2"))
        self.assertEqual(sim.NOISE_DBM, derived["NOISE_DBM"])
        self.assertEqual(sim.RADIO_OVERRIDES, {})

    def test_calibration_rows_record_the_network_that_ran(self):
        """U = 21 na rede de 22 nos da EWSN; U = 35 (todos os nao-sink do
        cenario de 36) na rede padrao, marcado como extrapolacao, com a
        configuracao efetiva de cada ponto identificada."""
        rows = {r["U"]: r for r in cc.t_success(
            ("x", dict(sim.PRESET_LEGACY_V2), 1, 1))}
        self.assertEqual(rows[21]["n_nodes"], 22)
        self.assertEqual(rows[35]["n_nodes"], cc.DEFAULTS["N_NODES"])
        self.assertEqual(rows[21]["status"], "calibracao")
        self.assertEqual(rows[35]["status"], "extrapolacao")
        self.assertEqual(rows[35]["fontes_por_flood"], 35)
        self.assertNotEqual(rows[21]["config_hash"], rows[35]["config_hash"])
        eff = rows[35]["config_efetiva"]
        self.assertEqual(eff["N_NODES"], 36)
        self.assertEqual(sim.config_hash(eff), rows[35]["config_hash"])
        args = SimpleNamespace(topologias=1, tentativas=1, pesos=[0.06],
                               sem_modelos=True)
        art = json.loads(json.dumps(cc.artifact_json(
            args, list(rows.values()), [], ("c" * 40, True, "d" * 16))))
        self.assertEqual(art["git_commit"], "c" * 40)
        for r in art["linhas_t"]:
            self.assertNotIn("config_efetiva", r)
            cfg = art["configuracoes_efetivas"][r["config_hash"]]
            self.assertEqual(cfg["N_NODES"], r["n_nodes"])
        md = cc.render_tables(list(rows.values()), [], 1, 1, 0.06)
        self.assertIn("U=21 (medido, rede de 22)", md)
        self.assertIn("U=35 (extrapolacao, rede de 36)", md)


class EngineFixes(Base):
    def test_empirical_model_uses_this_module_per_curve(self):
        """Sob 'python cupid_sim_v2.py' o modulo e __main__; o modelo tem de
        usar a curva de erro DESTE modulo, com os parametros em vigor."""
        self.assertIs(sim.CAPTURE.per, sim.decode_success_prob)

    def test_a_second_copy_of_the_module_uses_its_own_curve(self):
        """Outra copia do codigo (como __main__) com REQ_SNR_DB = 10 dB: a
        curva do modelo empirico e a dessa copia, e nenhuma importacao de
        'cupid_sim_v2' (com os padroes) acontece por baixo."""
        import subprocess
        import sys
        code = (
            "import runpy, sys\n"
            "g = runpy.run_path('cupid_sim_v2.py', run_name='copia')\n"
            "G = g['apply_config'].__globals__\n"
            "G['apply_config']({'PER_MODEL': 'threshold', "
            "'REQ_SNR_DB': 10.0})\n"
            "c = G['CAPTURE']\n"
            "print(c.per is G['decode_success_prob'], c.per(9.9), "
            "c.per(10.0), 'cupid_sim_v2' in sys.modules)\n")
        root = Path(__file__).resolve().parents[1]
        out = subprocess.run([sys.executable, "-c", code], cwd=root,
                             capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.split(), ["True", "0.0", "1.0", "False"])

    def test_apply_config_rebuilds_the_capture_model(self):
        sim.apply_config({"PER_MODEL": "threshold", "REQ_SNR_DB": 7.0})
        self.assertEqual(sim.CAPTURE.per(6.9), 0.0)
        self.assertEqual(sim.CAPTURE.per(7.0), 1.0)
        sim.apply_config({"PER_MODEL": "logistic", "PER_WIDTH_DB": 2.0})
        self.assertAlmostEqual(sim.CAPTURE.per(7.0), 0.5)
        self.assertAlmostEqual(sim.CAPTURE.per(8.0), 0.9)   # +-W/2: 10-90 %
        before = sim.CAPTURE
        sim.apply_config({"CAPTURE_EXTRAPOLATION": "upper"})
        self.assertIsNot(sim.CAPTURE, before)
        self.assertEqual(sim.CAPTURE.extrapolation, "upper")
        self.assertIs(sim.CAPTURE.per, sim.decode_success_prob)

    def test_copies_of_a_competing_frame_add_their_power(self):
        """Quadro travado 7 (no 1) contra o quadro 8 repetido pelos nos 2 e
        3 (copias identicas, alinhadas): a interferencia e a soma das duas."""
        power = np.zeros((4, 4))
        for v, p in ((1, 1e-7), (2, 1e-8), (3, 1e-8)):
            power[0][v] = power[v][0] = p
        topo = SimpleNamespace(ps_lin=power, tprop_ns=np.zeros((4, 4)),
                               audible=[{1, 2, 3}, {0}, {0}, {0}])
        sim.apply_config({"CAPTURE_MODEL": "empirical_uwb"})
        res = sim.decode_at_receiver(
            0, [1, 2, 3], {1: 7, 2: 8, 3: 8}, topo, [0.0, 0.0, 0.5, 0.5],
            fading={(1, 0): 1.0, (2, 0): 1.0, (3, 0): 1.0}, draw=0.0)
        expected = 10 * math.log10(1e-7 / (2e-8 + sim.NOISE_EFF_LIN))
        self.assertAlmostEqual(res[4], expected, places=6)

    def test_crystal_control_frames_use_their_declared_size(self):
        n = 36
        self.assertEqual(sim.frame_bytes_of(("crystal_a",), n),
                         sim.CRYSTAL_SA_FRAME_BYTES)
        self.assertEqual(sim.frame_bytes_of(("crystal_s", 2), n),
                         sim.CRYSTAL_SA_FRAME_BYTES)
        seen = set()
        original = sim.decode_at_receiver

        def spy(rx, txs, pkt_of, *a, **k):
            seen.update(v[0] for v in pkt_of.values()
                        if isinstance(v, tuple) and isinstance(v[0], str))
            return original(rx, txs, pkt_of, *a, **k)

        sim.apply_config({"FAST_FADE_DB": 0.0, "PER_MODEL": "threshold"})
        power = np.zeros((3, 3))
        for a, b in ((0, 1), (1, 2)):
            power[a][b] = power[b][a] = 1e-6
        topo = SimpleNamespace(
            n=3, ps_lin=power, tprop_ns=np.zeros((3, 3)),
            audible=[{1}, {0, 2}, {1}], reliable=[{1}, {0, 2}, {1}],
            hop={0: 0, 1: 1, 2: 2}, pos={v: (v, 0) for v in range(3)},
            group={v: "emitter" for v in range(3)})
        with patch.object(sim, "decode_at_receiver", spy):
            sim.run_epoch("crystal", [2], topo, random.Random(1),
                          slot_budget=60)
        self.assertLessEqual({"crystal_s", "crystal_a"}, seen)
        self.assertNotIn("sync", seen)

    def test_keyed_mode_runs_the_sequential_crystal(self):
        sim.apply_config(dict(TINY, CRYSTAL_MODEL="sequential"))
        topo, seed = sim.make_topologies(n_topo=1, base_seed=42)[0]
        run = sim.simulate_topology("crystal", topo, seed, O=7, n_epochs=2)
        self.assertEqual(len(run["records"]), 2)

    def test_sequential_mode_shares_the_bootstrap_between_protocols(self):
        sim.apply_config(dict(TINY, RNG_MODEL="sequential"))
        topo, seed = sim.make_topologies(n_topo=1, base_seed=42)[0]
        boots = {p: [r["bootstrap_slots"] for r in sim.simulate_topology(
            p, topo, seed, O=7, n_epochs=4)["records"]]
            for p in ("weaver", "cupid")}
        self.assertEqual(boots["weaver"], boots["cupid"])

    def test_bootstrap_and_crystal_s_streams_match_across_rng_models(self):
        """O bootstrap dos CTX e o S do Crystal saem da MESMA chave da epoca
        nos dois modos de sorteio (o S nao reusa o stream do bootstrap)."""
        topo, seed = sim.make_topologies(n_topo=1, base_seed=42)[0]
        original = sim.run_bootstrap

        def streams_of(proto, mode):
            seen = []

            def spy(topo, rng, *a, **k):
                seen.append((k.get("frame_kind", "sync"), rng.getstate()))
                return original(topo, rng, *a, **k)

            sim.apply_config(dict(TINY, RNG_MODEL=mode))
            with patch.object(sim, "run_bootstrap", spy):
                sim.simulate_topology(proto, topo, seed, O=7, n_epochs=2)
            return seen

        for proto in ("weaver", "crystal"):
            with self.subTest(proto=proto):
                keyed = streams_of(proto, "keyed")
                self.assertEqual(len(keyed), 2)
                self.assertEqual(streams_of(proto, "sequential"), keyed)
        self.assertNotEqual(streams_of("weaver", "keyed")[0][1],
                            streams_of("crystal", "keyed")[0][1])


class StatsFix(unittest.TestCase):
    def test_exactly_half_complete_gives_the_right_lower_bound(self):
        s = es.summarize_censored_latency([1.0, 2.0, math.inf, math.inf],
                                          10.0)
        self.assertIsNone(s["median"])
        self.assertEqual(s["median_lower_bound"], 6.0)
        s = es.summarize_censored_latency([1.0, math.inf, math.inf], 10.0)
        self.assertEqual(s["median_lower_bound"], 10.0)


if __name__ == "__main__":
    unittest.main()
