# Simulador Cupid: convergecast por transmissão concorrente em UWB

Simulador de eventos discretos que compara protocolos de coleta de dados
(*convergecast*) por **transmissão concorrente** (CTX) sobre rádios
**UWB** multissalto:

- **Crystal** (Istomin et al., SenSys 2016): um relatório por vez, por inundação;
- **Weaver** (Trobinger et al., SenSys 2020): o salto inteiro transmite junto;
- **Cupid** (Park et al., DCOSS-IoT 2025): cada salto é dividido em dois
  grupos pela distância medida por *ranging*, que se revezam nos slots;
- **Cupid-FWB**: o Cupid com canais alocados em funil, seguindo o
  *Funneling Wider Bandwidth* (Tavares et al., Computer Communications 2019).

A regra central é **uma física só**: propagação, ruído, captura e
sincronização passam pela mesma função de recepção para todos os protocolos.
Cada protocolo decide apenas quem transmite em cada slot.

## Contexto

O simulador foi desenvolvido para o artigo *FWB-Cupid: Alocação de Canais em
Funil para Convergecast em UWB* (Lucas Emanuel Elias Alves, Hélio Martins de
Araújo Costa Neto, Luiz F. M. Vieira e Marcos A. M. Vieira, UFMG), aceito no
SBESC 2026.

Esta versão revisa o modelo físico e o dos protocolos em relação à versão
usada no artigo. Os números que ela produz podem, portanto, diferir dos
publicados. O preset padrão reproduz o rádio do testbed usado na avaliação do
Cupid, e cada parâmetro tem a fonte declarada nos comentários do código
(bloco de parâmetros no topo de `cupid_sim_v2.py`, `radio_profiles.py` e
`capture_models.py`).

## O que o modelo inclui

- **Rádio DW1000, canal 4**: potência, ruído, sensibilidade, preâmbulo, PAC
  e tempos de quadro derivados do datasheet e do user manual
  (`radio_profiles.py`).
- **Canal**: perda de percurso log-distância, sombreamento fixo por enlace,
  desvanecimento rápido por slot (opcionalmente com memória entre épocas) e
  atraso de propagação.
- **Recepção**: SINR contra toda a interferência somada mais ruído, com uma
  curva de erro suave. Trava no preâmbulo, com roubo de trava só entre
  preâmbulos alinhados.
- **Captura**: modelo empírico construído a partir de pontos medidos em
  DW1000 (EWSN 2020 e SECON 2019), para cópias do mesmo quadro e para quadros
  diferentes (`capture_models.py`). O regime que nenhuma medição cobre é
  tratado como faixa de sensibilidade.
- **Protocolos**:
  - bootstrap físico a cada época;
  - Cupid: agrupamento por *ranging* simulado e grade de slots da Fig. 5b do
    artigo do Cupid;
  - Weaver: TX-RX-RX, com ACK local e global;
  - Crystal: floods S, T e A com pares TA;
  - Cupid-FWB: planos de canais reais e número de rádios por nó.
- **Estatística**: a unidade independente é a topologia. Épocas incompletas
  são censuradas no prazo (RMST, Kaplan–Meier), não descartadas. O workload
  é pareado entre protocolos, as comparações também, e os IC usam o t de Student.
- **Reprodutibilidade**: sorteios por chave (época, slot, nó) e manifesto de
  proveniência em cada rodada. A retomada é segura: um cenário só é pulado
  se já foi rodado com o mesmo código e a mesma configuração.

## Instalação

Requer NumPy e Matplotlib. Testado com Python 3.13.

```bash
git clone https://github.com/heliomac99/fwb-cupid.git
cd fwb-cupid
python -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Uso rápido

```bash
# testes automaticos (~30 s)
python -m unittest discover -s tests -v

# um cenario avulso, pequeno: Cupid x Weaver, 50 epocas em 2 topologias
python cupid_sim_v2.py --name teste --out outputs/teste \
  --experiments cdf,pdr --protocols cupid,weaver --EPOCHS 50 --N_TOPOLOGIES 2

# piloto das duas suites (minutos)
python run_cenarios.py cenarios_cloves_core.json --suite core --epochs 50 --topologies 2 --jobs 2 --run-id piloto-core
python run_cenarios.py cenarios_cloves_fwb.json  --suite fwb  --epochs 50 --topologies 2 --jobs 2 --run-id piloto-fwb

# rodadas completas (horas; --jobs = cenarios em paralelo)
python run_cenarios.py cenarios_cloves_core.json --suite core --jobs 8 --run-id core
python run_cenarios.py cenarios_cloves_fwb.json  --suite fwb  --jobs 6 --run-id fwb
```

Qualquer parâmetro do modelo pode ser passado na linha de comando
(`--NOME valor`, por exemplo `--N_NODES 50 --FAST_FADE_DB 8`). A lista
completa, com comentários, está em `PARAM_NAMES`, no topo de
`cupid_sim_v2.py`. Nas suítes, `--epochs` e `--topologies` sobrepõem os
valores de todos os cenários sem editar o JSON.

Mais de um rádio por nó não existe no DW1000. Por isso, esse caso só roda
declarado como hardware hipotético (`--hypothetical-hardware` na linha de
comando, `"hardware_hipotetico": true` no cenário). Um pedido que o plano de
canais truncaria (por exemplo, 16 canais no plano `dw1000`, que tem 4) é
erro, salvo com `--allow-clamp`.

Todo script aceita `--help`. O formato dos arquivos de cenários está
descrito no campo `_ajuda` de cada JSON: em `parametros` entra só o que muda
em relação ao preset, e `descricao` é só documentação.

## Suítes e cenários

| Arquivo | Suíte | Protocolos | Conteúdo |
|---|---|---|---|
| `cenarios_cloves_core.json` | `core` | Cupid, Weaver, Crystal | 20 cenários: caso-base (36 nós, 83 × 33 m), famílias de topologias, densidade, profundidade, canal, sombreamento, carga, tamanho de quadro, modelo de captura e um pior caso combinado |
| `cenarios_cloves_fwb.json` | `fwb` | Cupid, Cupid-FWB | 7 cenários: referência de canal único, 1 e 2 rádios nos planos do DW1000, crosstalk e o funil ideal de 16 canais (hipotético) |
| `cenarios.json` | `legacy` | todos | suíte antiga, no preset `legacy_v2` |

Cada cenário muda uma dimensão do caso-base. Dentro dele rodam os
experimentos da suíte:

| Experimento | O que mede |
|---|---|
| `cdf` | distribuição da latência da época e por relatório |
| `carga` | latência em função do número de originadores |
| `pdr` | confiabilidade com prazo real em função da frequência de épocas |
| `micro` | micro-benchmark de captura por número de transmissores |
| `funil`, `crosstalk`, `radios`, `planos` | FWB: canais em funil, vazamento entre canais, rádios por nó e planos de canais reais |

## Presets

| Preset | Descrição |
|---|---|
| `cloves` (padrão) | DW1000 no canal 4 (3993,6 MHz), PRF 64, código 17, PAC 8, 6,8 Mbps, quadros de 126 B, slot de 813 µs; bootstrap físico, agrupamento por ranging e captura empírica |
| `legacy_v2` | motor anterior (canal 5, saltos por busca em largura, captura com peso assíncrono calibrado), reproduzido bit a bit |
| `etapa0`, `2026_09_10` | motores mais antigos, para comparação |

Use `--preset <nome>` num cenário avulso. `sensibilidade.py cloves` mostra o
efeito de cada passo do motor legado até o `cloves`.

## Saída

Cada rodada grava em `outputs/<run-id>/<cenário>/`:

| Arquivo | Conteúdo |
|---|---|
| `resumo_resultados.txt` | latência (mediana e RMST), épocas completas, PDR, custos de bootstrap e agrupamento, comparações pareadas por topologia |
| `amostras_epocas.csv` | uma linha por protocolo × topologia × época (gravado quando o experimento `cdf` roda) |
| `01_topologia.png` … `08_planos_de_canais.png` | gráficos de cada experimento |
| `config_efetiva.json` | parâmetros pedidos × efetivos e hash da configuração |
| `run_manifest.json` | proveniência: commit, worktree suja, versões, sementes, fontes do modelo físico e hipóteses em vigor |

A rodada inteira ganha também `_RESUMO_GERAL.txt` e um `run_manifest.json`.

## Bancadas

| Script | Para quê |
|---|---|
| `calibracao_captura.py` | confere o modelo de captura contra medições publicadas (1º flood T do Crystal na EWSN 2020, várias fontes no SECON 2019) |
| `microbenchmark_ranging.py` | erro do ranging simulado usado no agrupamento do Cupid |
| `sensibilidade.py` | conjuntos de configurações (captura, limiar, agrupamento, ACK, canal com memória etc.) com comparação pareada entre protocolos |

```bash
# captura: calibracao do peso legado + validacao dos dois modelos no radio CLOVES
python calibracao_captura.py --topologias 5 --tentativas 200 --jobs 8

# erro de ranging do agrupamento contra o 1,31 m do Cupid (referencia, nao alvo)
python microbenchmark_ranging.py --topologias 5

# do motor legado ao cloves, uma mudanca por passo
python sensibilidade.py cloves --topologias 5 --epocas 40 --jobs 8

# outros conjuntos (lista completa em --help)
python sensibilidade.py modelo_captura agrupamento intra_aquisicao canal_memoria --topologias 5 --epocas 40 --jobs 8
```

As saídas vão para `outputs/calibracao/`, `outputs/ranging/` e
`outputs/sensibilidade/`. São diagnóstico e validação parcial, não metas de
ajuste. Todo script aceita `--help`, que lista as opções e os valores
válidos.

## Estrutura

| Arquivo | Papel |
|---|---|
| `cupid_sim_v2.py` | o motor: parâmetros, topologia, física, protocolos, experimentos, gráficos e linha de comando |
| `radio_profiles.py` | números do rádio derivados do datasheet |
| `capture_models.py` | modelo de captura empírico (tabelas medidas) |
| `cupid_grouping.py` | bootstrap físico e período de agrupamento do Cupid |
| `rng_streams.py` | sorteios por chave estável |
| `experiment_stats.py` | censura, RMST, mediana identificável, IC e bootstrap pareado |
| `run_cenarios.py` | executa suítes em paralelo, com proveniência e retomada |
| `tests/` | testes automáticos (`unittest`) |

## Limitações

Os testes provam consistência de software (regras implementadas, reprodução
bit a bit dos motores legados, sorteios estáveis, paralelo igual a
sequencial), não a física. A camada física tem validação parcial nas regiões
medidas e extrapola fora delas. A captura é a parte menos validada: em
particular, falta medição para vários transmissores dentro da janela de
aquisição do preâmbulo, e o modelo trata esse regime como faixa de
sensibilidade. Nenhuma latência de protocolo foi usada para calibrar a
camada física. Os parâmetros sem fonte que fixe o valor (hipóteses) são
gravados, com o valor em vigor, no `run_manifest.json` de cada rodada.

Os comentários citam hashes de commits (por exemplo, o motor `5ef525b` do
preset `legacy_v2`) e rótulos de itens de revisão (`[F18]`, `[Fase 7]` etc.)
do desenvolvimento. Esse histórico não faz parte desta versão; os presets
reproduzem aqueles motores bit a bit (`tests/test_legacy_preset.py`).

## Referências

- Park, Lee, Paek, Bahk. *Cupid: Fast and Reliable Convergecast-over-UWB
  Protocol for Dense Internet of Things*. DCOSS-IoT 2025.
- Trobinger et al. *One Flood to Route Them All: Ultra-fast Convergecast of
  Concurrent Flows over UWB* (Weaver). SenSys 2020.
- Istomin et al. *Data Prediction + Synchronous Transmissions = Ultra-low
  Power Wireless Sensor Networks* (Crystal). SenSys 2016.
- Tavares et al. *FWB: Funneling Wider Bandwidth algorithm for high
  performance data collection in Wireless Sensor Networks*. Computer
  Communications 148, 2019. doi:10.1016/j.comcom.2019.09.015.
- Lobba et al. *Concurrent Transmissions for Multi-hop Communication on
  Ultra-wideband Radios*. EWSN 2020.
- Vecchia et al. *Playing with Fire: Exploring Concurrent Transmissions in
  Ultra-wideband Radios*. SECON 2019.
- Decawave/Qorvo. *DW1000 Datasheet* v2.23 e *DW1000 User Manual* v2.15.

## Como citar

```bibtex
@inproceedings{alves2026fwbcupid,
  title     = {{FWB-Cupid}: Aloca{\c{c}}{\~a}o de Canais em Funil para Convergecast em {UWB}},
  author    = {Lucas Emanuel Elias Alves and H{\'e}lio Martins de Ara{\'u}jo Costa Neto
               and Luiz F. M. Vieira and Marcos A. M. Vieira},
  booktitle = {Anais do Simp{\'o}sio Brasileiro de Engenharia de Sistemas Computacionais (SBESC)},
  year      = {2026}
}
```

Os mesmos dados estão em [CITATION.cff](CITATION.cff), que o GitHub mostra
como "Cite this repository".

## Licença

Distribuído sob a licença MIT. Veja [LICENSE](LICENSE).
