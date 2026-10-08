# -*- coding: utf-8 -*-
"""
Roda VARIAS simulacoes, lendo a lista de cenarios de um arquivo JSON, com
proveniencia e retomada segura.

EM PARALELO: cada cenario roda em um PROCESSO separado (interpretador
proprio), ate 'jobs' ao mesmo tempo. Os sorteios sao chaveados por
(SEED, topologia, experimento, epoca), entao o resultado em paralelo e
identico ao sequencial.

Saida: <outputs>/<run-id>/<cenario>/ com resumo, graficos, amostras brutas
(amostras_epocas.csv), config_efetiva.json e run_manifest.json; e
<outputs>/<run-id>/run_manifest.json + _RESUMO_GERAL.txt da rodada.

RETOMADA SEGURA. Um cenario so e pulado se a pasta ja tem um run_manifest
COMPATIVEL com o pedido de agora: mesma versao do modelo, mesmo commit (e o
mesmo diff, se a worktree estava suja), mesma configuracao efetiva (preset +
parametros + --epochs/--topologies), mesmo cenario, suite e experimentos.
Resumo antigo sem manifesto, ou manifesto diferente, e ERRO: a rodada para
antes de comecar e sugere um novo --run-id; --force sobrescreve. Falha de
qualquer cenario da codigo de saida diferente de zero.

Uso:
    py run_cenarios.py cenarios_cloves_core.json --suite core --run-id x
    py run_cenarios.py cenarios_cloves_fwb.json --suite fwb --epochs 50 \\
        --topologies 2 --jobs 2 --run-id piloto-fwb
    py run_cenarios.py cenarios.json --run-id legado     # preset legacy_v2
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cupid_sim_v2 as sim
# ---- Parametros validos: os do proprio simulador (antes era uma copia a mao,
# que envelhecia a cada parametro novo) ----
from cupid_sim_v2 import PARAM_NAMES


def fmt_dur(seg):
    m, s = divmod(int(seg), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("arquivo", nargs="?", default="cenarios.json")
    p.add_argument("--jobs", type=int,
                   default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--force", action="store_true",
                   help="sobrescreve saidas incompativeis e refaz as "
                        "compativeis")
    p.add_argument("--run-id", default="default",
                   help="pasta da rodada dentro de --outputs")
    p.add_argument("--suite", default="all",
                   choices=("core", "fwb", "legacy", "all"),
                   help="so os cenarios desta suite")
    p.add_argument("--experiments",
                   help="lista separada por virgulas (padrao: a da suite)")
    p.add_argument("--epochs", type=int,
                   help="sobrepoe EPOCHS em todos os cenarios (pilotos)")
    p.add_argument("--topologies", type=int,
                   help="sobrepoe N_TOPOLOGIES em todos os cenarios")
    p.add_argument("--outputs", default="outputs")
    a = p.parse_args(argv)
    if a.experiments:
        a.experiments = [x for x in a.experiments.split(",") if x]
        bad = [x for x in a.experiments if x not in sim.ALL_EXPERIMENTS]
        if bad:
            p.error(f"--experiments desconhecido(s): {bad}")
    return a


def carregar(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        dados = json.load(f)
    cenarios = dados.get("cenarios", [])
    if not cenarios:
        raise SystemExit(f"Nenhum cenario encontrado em {caminho}.")
    return dados, cenarios


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def validar_truncamentos(cenarios):
    """[F20] Falha ANTES de rodar qualquer coisa se um cenario pede um valor
    que o simulador truncaria em silencio (ex.: N_CHANNELS=16 com o plano
    'dw1000', que oferece 4), a menos que o cenario declare
    "allow_clamp": true. O proprio simulador tambem recusa, mas so depois
    que a rodada paralela ja comecou."""
    from cupid_sim_v2 import requested_clamps
    problemas = []
    for i, cen in enumerate(cenarios, 1):
        if cen.get("allow_clamp"):
            continue
        nome = cen.get("nome", f"cenario_{i}")
        for c in requested_clamps(cen.get("parametros", {})):
            problemas.append(f"  - {nome}: {c['param']}={c['pedido']} viraria "
                             f"{c['efetivo']} ({c['motivo']})")
    if problemas:
        raise SystemExit(
            "Cenarios com parametro que seria truncado em silencio:\n"
            + "\n".join(problemas)
            + '\nCorrija os parametros ou declare "allow_clamp": true.')


def validar_hardware(cenarios):
    """[Fase 10] Mais de um radio por no nao existe no DW1000: so roda se o
    cenario declarar "hardware_hipotetico": true (e o manifesto registra)."""
    problemas = [c.get("nome", "?") for c in cenarios
                 if c.get("parametros", {}).get("N_RADIOS_PER_NODE", 1) > 1
                 and not c.get("hardware_hipotetico")]
    if problemas:
        raise SystemExit(
            "Cenarios com mais de 1 radio por no sem declarar hardware "
            f"hipotetico: {', '.join(problemas)}. Declare "
            '"hardware_hipotetico": true no cenario.')


MARCADOR_EXPERIMENTO = "resumo_resultados.txt"
MANIFESTO = "run_manifest.json"


def construir_comando(nome, params, out_dir, allow_clamp=False,
                      preset=None, experiments=None, overrides=None,
                      suite=None, protocols=None, hypothetical=False,
                      scenario_hash=None):
    """Monta a linha de comando para rodar UM cenario via cupid_sim_v2.py.
    'overrides' (--epochs/--topologies da rodada) vencem os do cenario."""
    params = dict(params)
    params.update(overrides or {})
    cmd = [sys.executable, "-u", "cupid_sim_v2.py",
           "--name", nome, "--out", out_dir]
    if preset:
        cmd += ["--preset", preset]
    if suite in sim.SUITES:
        cmd += ["--suite", suite]
    if experiments:
        cmd += ["--experiments", ",".join(experiments)]
    if protocols:
        cmd += ["--protocols", ",".join(protocols)]
    if hypothetical:
        cmd.append("--hypothetical-hardware")
    if scenario_hash:
        cmd += ["--scenario-file-hash", scenario_hash]
    for chave, valor in params.items():
        if chave not in PARAM_NAMES:
            raise KeyError(f"Parametro invalido em '{nome}': {chave}")
        cmd += [f"--{chave}", str(valor)]
    if allow_clamp:
        cmd.append("--allow-clamp")
    return cmd


# ---- Retomada segura ----

_FIELD_LABELS = {
    "model_version": "versao do modelo", "git_commit": "commit",
    "git_dirty": "worktree suja", "git_diff_hash": "diff da worktree",
    "effective_config_hash": "configuracao efetiva", "scenario": "cenario",
    "suite": "suite", "experiments": "experimentos", "preset": "preset",
    "protocols": "protocolos",
}


def effective_hash(preset, params):
    """Hash da configuracao efetiva que o subprocesso vai calcular."""
    snapshot = {n: getattr(sim, n) for n in PARAM_NAMES}
    try:
        sim.apply_config(dict(sim.PRESETS[preset], **params))
        return sim.config_hash(sim.effective_config())
    finally:
        sim.apply_config(snapshot)


def scenario_request(cen, preset, suite, experiments, overrides,
                     protocols=None, git_info=None):
    """O que se pede agora para um cenario (comparado com o manifesto).
    'git_info' e o estado do repositorio lido no inicio da rodada."""
    params = dict(cen.get("parametros", {}))
    params.update(overrides or {})
    # a mesma coercao que o subprocesso aplica ao receber str(valor)
    coerced = {k: sim._coerce(k, str(v)) for k, v in params.items()}
    # a mesma regra de main(): sem lista explicita, os protocolos da suite
    _, protos = sim.suite_selection(suite, experiments, protocols)
    commit, dirty, diff_hash = git_info or sim._git_info()
    return {"scenario": cen["nome"], "preset": preset, "suite": suite,
            "experiments": sorted(experiments), "parametros": params,
            "protocols": protos,
            "effective_config_hash": effective_hash(preset, coerced),
            "model_version": sim.MODEL_VERSION, "git_commit": commit,
            "git_dirty": dirty, "git_diff_hash": diff_hash}


def expected_manifest_fields(request):
    return {k: request[k] for k in _FIELD_LABELS}


def check_resume(out_dir, request):
    """('rodar' | 'pular' | 'incompativel', motivo)."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / MANIFESTO
    marker = out_dir / MARCADOR_EXPERIMENTO
    if not manifest_path.exists():
        if marker.exists():
            return "incompativel", ("resumo sem manifesto (saida de versao "
                                    "antiga): nao ha como saber o que rodou")
        leftovers = ([p.name for p in out_dir.iterdir() if p.name != "run.log"]
                     if out_dir.exists() else [])
        if leftovers:
            return "incompativel", (f"arquivos sem manifesto: "
                                    f"{sorted(leftovers)[:5]}")
        return "rodar", "pasta vazia"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    diffs = [_FIELD_LABELS[k] for k, v in expected_manifest_fields(
        request).items() if manifest.get(k) != v]
    if diffs:
        return "incompativel", "manifesto difere em: " + ", ".join(diffs)
    if manifest.get("status") != "completo" or not marker.exists():
        # o resumo que sobrou (se houver) e de OUTRA execucao
        return "rodar", "execucao anterior incompleta, mesmo pedido"
    return "pular", "manifesto compativel"


def _suite_of(cen, dados):
    return cen.get("suite") or dados.get("suite") or "legacy"


def _experiments_for(suite, dados, cli_experiments):
    """Experimentos da suite: --experiments > "experimentos" do arquivo >
    padrao da suite. Um cenario pode restringir os seus com
    "experimentos_cenario" (salvo quando --experiments e dado)."""
    if cli_experiments:
        return list(cli_experiments)
    if dados.get("experimentos"):
        return list(dados["experimentos"])
    return list(sim.SUITES.get(suite, sim.SUITES["all"])["experiments"])


def plan_scenarios(cenarios, run_dir, preset, suite, experiments, overrides,
                   force, protocols=None, git_info=None):
    """Separa o que roda do que e pulado; saida incompativel para a rodada
    (salvo --force). Devolve ([(cen, pedido)], [pulados])."""
    todo, skipped, erros = [], [], []
    for cen in cenarios:
        exps = cen.get("experimentos_cenario") or experiments
        req = scenario_request(cen, preset, suite, exps, overrides,
                               protocols, git_info)
        status, motivo = check_resume(Path(run_dir) / cen["nome"], req)
        if status == "pular" and not force:
            skipped.append((cen, req, motivo))
        elif status == "incompativel" and not force:
            erros.append(f"  - {cen['nome']}: {motivo}")
        else:
            todo.append((cen, req))
    if erros:
        raise SystemExit(
            f"Saidas incompativeis em {run_dir}:\n" + "\n".join(erros)
            + "\nNada foi sobrescrito. Use um novo --run-id (ex.: --run-id "
              f"{Path(run_dir).name}-2) ou --force para sobrescrever.")
    return todo, skipped


def _ler_resumo(out_dir):
    with open(os.path.join(out_dir, MARCADOR_EXPERIMENTO),
              encoding="utf-8") as f:
        return f.read()


def rodar_cenario(cen, idx, total, req=None, run_dir="outputs",
                  scenario_hash=None):
    """Executa um cenario num subprocesso. Retorna (nome, status, dur,
    conteudo, params)."""
    nome = cen.get("nome", f"cenario_{idx}")
    req = req or {"preset": None, "suite": None, "experiments": [],
                  "parametros": cen.get("parametros", {}), "protocols": []}
    params = req["parametros"]
    out_dir = os.path.join(str(run_dir), nome)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")
    resumo_path = os.path.join(out_dir, MARCADOR_EXPERIMENTO)
    # um resumo que ficou de outra execucao nunca pode passar por este
    if os.path.exists(resumo_path):
        os.remove(resumo_path)

    print(f"[{idx}/{total}] iniciando '{nome}' ...", flush=True)
    t0 = time.time()
    env = dict(os.environ, MPLBACKEND="Agg")   # backend sem GUI (seguro em paralelo)
    cmd = construir_comando(
        nome, params, out_dir,
        allow_clamp=cen.get("allow_clamp", False), preset=req["preset"],
        experiments=req["experiments"],
        suite=req["suite"], protocols=req["protocols"],
        hypothetical=cen.get("hardware_hipotetico", False),
        scenario_hash=scenario_hash)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                              text=True, env=env)
    dur = time.time() - t0

    if proc.returncode == 0 and os.path.exists(resumo_path):
        print(f"[{idx}/{total}] OK '{nome}' em {fmt_dur(dur)}", flush=True)
        return (nome, "OK", dur, _ler_resumo(out_dir), params)
    cauda = ""
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            cauda = "".join(f.readlines()[-25:])
    print(f"[{idx}/{total}] FALHOU '{nome}' (codigo {proc.returncode}) "
          f"apos {fmt_dur(dur)} -- veja {log_path}", flush=True)
    return (nome, "FALHOU", dur, cauda or "(sem saida)", params)


def main(argv=None):
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if not os.path.exists(a.arquivo):
        raise SystemExit(f"Arquivo nao encontrado: {a.arquivo}")
    dados, cenarios = carregar(a.arquivo)
    preset = dados.get("preset", "cloves")
    if preset not in sim.PRESETS:
        raise SystemExit(f"preset desconhecido '{preset}' em {a.arquivo}")
    selecionados = [c for c in cenarios
                    if a.suite == "all" or _suite_of(c, dados) == a.suite]
    if not selecionados:
        raise SystemExit(f"Nenhum cenario da suite '{a.suite}' em "
                         f"{a.arquivo}.")
    validar_truncamentos(selecionados)
    validar_hardware(selecionados)
    overrides = {}
    if a.epochs:
        overrides["EPOCHS"] = a.epochs
    if a.topologies:
        overrides["N_TOPOLOGIES"] = a.topologies
    run_dir = Path(a.outputs) / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    fhash = file_hash(a.arquivo)
    protocols = dados.get("protocolos")
    # [Fase 9] commit e worktree lidos UMA vez, no inicio da rodada: o
    # pedido de cada cenario e o manifesto da rodada sao desse estado
    git_info = sim._git_info()

    todo, skipped, registros = [], [], []
    for grupo in sorted({_suite_of(c, dados) for c in selecionados}):
        cens = [c for c in selecionados if _suite_of(c, dados) == grupo]
        if a.experiments:          # --experiments vence o do cenario
            cens = [{k: v for k, v in c.items()
                     if k != "experimentos_cenario"} for c in cens]
        t, s = plan_scenarios(cens, run_dir, preset, grupo,
                              _experiments_for(grupo, dados, a.experiments),
                              overrides, a.force, protocols, git_info)
        todo += t
        skipped += s
    ordem = {c["nome"]: i for i, c in enumerate(selecionados, 1)}
    total = len(selecionados)
    for cen, _, motivo in skipped:
        nome = cen["nome"]
        registros.append((nome, "PULADO", 0.0,
                          _ler_resumo(str(run_dir / nome)),
                          cen.get("parametros", {})))
        print(f"[{ordem[nome]}/{total}] '{nome}' {motivo} -> pulando.",
              flush=True)

    print(f"\nArquivo: {a.arquivo} | preset {preset} | suite {a.suite} | "
          f"run-id {a.run_id} | {total} cenario(s) | {len(todo)} a rodar | "
          f"{a.jobs} em paralelo\n", flush=True)
    t0_total = time.time()
    if todo:
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            futuros = [ex.submit(rodar_cenario, cen, ordem[cen["nome"]],
                                 total, req, run_dir, fhash)
                       for cen, req in todo]
            for fut in as_completed(futuros):
                registros.append(fut.result())
    total_dur = time.time() - t0_total
    registros.sort(key=lambda r: ordem.get(r[0], 999))

    ok = sum(1 for r in registros if r[1] in ("OK", "PULADO"))
    linhas = ["#" * 64, "  RESUMO GERAL DOS CENARIOS",
              f"  Arquivo: {a.arquivo} ({fhash}) | preset {preset} | "
              f"suite {a.suite} | run-id {a.run_id}",
              f"  Paralelismo: {a.jobs} processos | Tempo total: "
              f"{fmt_dur(total_dur)}",
              f"  Sobreposicoes da rodada: {overrides or 'nenhuma'}",
              f"  Cenarios: {len(registros)}  (OK/Pulados: {ok}, Falhas: "
              f"{len(registros) - ok})", "#" * 64]
    for nome, status, dur, conteudo, params in registros:
        linhas += ["", f"### {nome}  [{status}]  ({fmt_dur(dur)})"]
        if params:
            linhas.append(f"    parametros: "
                          f"{json.dumps(params, ensure_ascii=False)}")
        linhas += [f"    pasta: {run_dir / nome}", "", conteudo.rstrip()]
    (run_dir / "_RESUMO_GERAL.txt").write_text("\n".join(linhas) + "\n",
                                                encoding="utf-8")
    commit, dirty, diff_hash = git_info
    with open(run_dir / MANIFESTO, "w", encoding="utf-8") as f:
        json.dump({
            "schema": "cupid-suite-manifest/1", "arquivo": a.arquivo,
            "arquivo_hash": fhash, "preset": preset, "suite": a.suite,
            "run_id": a.run_id, "overrides": overrides,
            "experimentos_cli": a.experiments, "jobs": a.jobs,
            "model_version": sim.MODEL_VERSION, "git_commit": commit,
            "git_dirty": dirty, "git_diff_hash": diff_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "duracao_s": round(total_dur, 1),
            "cenarios": {r[0]: r[1] for r in registros}},
            f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("\n" + "#" * 60)
    print("  TODOS OS CENARIOS PROCESSADOS")
    print(f"  Tempo total: {fmt_dur(total_dur)}  |  OK/Pulados: {ok}/"
          f"{len(registros)}")
    print(f"  Resumo consolidado: {run_dir / '_RESUMO_GERAL.txt'}")
    print("#" * 60)
    return 1 if ok < len(registros) else 0   # [F20] falha nao passa calada


if __name__ == "__main__":
    sys.exit(main())
