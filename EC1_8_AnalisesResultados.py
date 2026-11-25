# %% [markdown]
# # 8 — Análises dos Resultados da Tese (SECOM)
#
# Este notebook reúne **todas as análises** usadas no texto de Metodologia, Resultados e
# Discussão (arquivo `Metodologia_Resultados_Discussao.tex`). Ele não treina modelos novos: lê as
# saídas dos notebooks 1 a 5 e recalcula, com precisão completa, cada número citado no texto.
#
# **Arquivos de entrada** (todos na mesma pasta deste notebook):
#
# | Arquivo | Gerado por |
# |---|---|
# | `amostra.xlsx` | notebook 1 (amostragem) |
# | `Rankings_Consolidado.xlsx` | notebook `Consolidar_Rankings` (a partir dos `Ranking_*.xlsx` do notebook 2) |
# | `Modelos<Paradigma>_CV_Consolidado.xlsx` (7 arquivos) e `Mapa_Abas_Consolidado.xlsx` | notebook 4 |
# | `Resumo_Global_Dados_<data>.xlsx` e `CODE9_Escalabilidade_Por_Ranking_<data>.xlsx` | notebook 5 |
#
# **Saída**: `8_Analises_Tese.xlsx`, com uma aba por análise (valores com precisão completa).
# O notebook também imprime cada tabela já arredondada pela ABNT NBR 5891, como no texto.
#
# **Grupos comparados** (os mesmos do texto):
# - **Proposta**: 6 índices de CEQ (Cp, Cpk, Pp, Ppk, p, c), versões `_Original`. As versões
#   `_Normalizado`, `np` e `u` produzem as mesmas ordenações e ficam de fora.
# - **Literatura**: 19 métodos.
# - **Estatísticas descritivas**: Amplitude/σ, IQR/σ, |assimetria| e |curtose| (sem as versões
#   `_BaseCompleta`).
#
# | Seção | Análise | Onde aparece no texto |
# |---|---|---|
# | 1 | Amostra e sensores constantes | Metodologia (base de dados e amostragem) |
# | 2 | Reprodutibilidade dos 20 primeiros sensores e sensibilidade ao tamanho da lista | Organização da análise; Estabilidade; Descritivas |
# | 3 | Friedman/Nemenyi e correlação de Spearman entre rankings | Metodologia (equivalência); Equivalência entre rankings; Descritivas |
# | 4 | Relação entre os índices p e c e o IQR/σ | Comparação com as estatísticas descritivas |
# | 5 | Médias ± desvio-padrão, Mann-Whitney e tempo estimado dos modelos selecionados | Tabelas de qualidade e de tempo de cada paradigma |
# | 6 | Placar (maior média com duas casas decimais) | Placar por paradigma e síntese |
# | 7 | Fronteira de Pareto (Kneedle) e distância dos índices de CEQ até ela | Fronteira e distância por paradigma |
# | 8 | Melhor modelo individual de cada abordagem | Tabela de melhores modelos por paradigma |
# | 9 | Sensores dispensáveis e oscilação da curva | Metodologia (dispensáveis) e tabelas de dispensáveis |
# | 10 | Números da síntese | Síntese e discussão |
# | 11 | Estabilidade dos sensores escolhidos (índice de Nogueira) | Estabilidade de cada paradigma e síntese |
#
# **Diferenças em relação aos notebooks anteriores** (para não estranhar valores diferentes):
# - **Notebook 3**: lá entram também as versões `_Normalizado`, `np` e `u`. Como o Nemenyi
#   depende do número de métodos comparados, os valores-p dele diferem dos deste notebook (29
#   rankings). A correlação de Spearman de cada par é a mesma, porque só depende dos dois rankings.
# - **Aba `Reprodutibilidade` do `amostra.xlsx`**: os valores gravados ali (por exemplo, Cp = 0,29
#   e Pp = 0,08, quase igual à Amplitude/σ) correspondem à versão do notebook 1 com limites
#   mínimo/máximo. A seção 2 recalcula a reprodutibilidade com as funções atuais (limites P1/P99),
#   que são as que geraram os rankings usados na tese.
# - **Notebook 5**: as médias, os desvios e os valores-p são recalculados aqui a partir dos 5
#   valores por partição, com cada ranking como uma observação e os três grupos acima; por isso
#   podem diferir ligeiramente das tabelas consolidadas do notebook 5.
#
# Tempo de execução: cerca de 2 minutos (a maior parte na seção 2).

# %%
# %pip install pandas numpy scipy scikit-learn scikit-posthocs openpyxl

# %% [markdown]
# ### 0. Configuração, grupos e arredondamento ABNT

# %%
import os, glob, ast, itertools, warnings
from decimal import Decimal, ROUND_HALF_EVEN
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, friedmanchisquare, spearmanr, rankdata
import scikit_posthocs as sp

warnings.simplefilter("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

PASTA = os.getcwd()
ARQ_AMOSTRA = os.path.join(PASTA, "amostra.xlsx")
ARQ_RANKINGS = os.path.join(PASTA, "Rankings_Consolidado.xlsx")
ARQ_MAPA = os.path.join(PASTA, "Mapa_Abas_Consolidado.xlsx")
# notebook 5: usa o arquivo mais recente de cada tipo
ARQ_RESUMO = sorted(glob.glob(os.path.join(PASTA, "Resumo_Global_Dados_*.xlsx")))[-1]
ARQ_CODE9 = sorted(glob.glob(os.path.join(PASTA, "CODE9_Escalabilidade_Por_Ranking_*.xlsx")))[-1]
ARQ_SAIDA = os.path.join(PASTA, "8_Analises_Tese.xlsx")

PARADIGMAS = ["DT", "RF", "GB", "SVM", "MLP", "GNB", "LogReg_Lasso"]
PROP = ["Cp", "Cpk", "Pp", "Ppk", "p", "c"]
LIT = ["ANOVA", "ApproxMIC", "Chi2", "FScore_ANOVA", "FisherScore", "GainRatio", "GiniIndex",
       "HellingerDistance", "InformationGain", "KS", "Kendall", "LaplacianScore", "Pearson", "ReliefF",
       "SPEC", "SVDEntropy", "Spearman", "SymmetricalUncertainty", "TScore"]
DESC = ["AmplitudeSigma", "IQRSigma", "Assimetria", "Curtose"]
GRUPOS = ["Proposta", "Literatura", "Descritivas"]
CLASSE_INTERESSE = -1
TOP_X = 20
LIMIAR_CORR_FORTE = 0.70   # |rho| >= 0,70: correlação forte (mesma convenção em todo o texto)


def nome_metodo(ranking):
    """'Ranking_Cp_Original' ou 'Cp_Original' -> 'Cp'; 'Ranking_ANOVA' -> 'ANOVA'."""
    return str(ranking).replace("Ranking_", "").replace("_Original", "")


def grupo(metodo):
    if metodo in PROP: return "Proposta"
    if metodo in LIT: return "Literatura"
    if metodo in DESC: return "Descritivas"
    return None   # _Normalizado, np, u, _BaseCompleta: fora da análise


def arred(x, casas):
    """Arredondamento da ABNT NBR 5891 (5 seguido só de zeros -> último algarismo par)."""
    return float(Decimal(repr(round(float(x), 12))).quantize(Decimal(1).scaleb(-casas), rounding=ROUND_HALF_EVEN))


def abnt(x, casas):
    """Texto com vírgula decimal, como no texto da tese."""
    if x is None or (isinstance(x, float) and np.isnan(x)): return "--"
    v = Decimal(repr(round(float(x), 12))).quantize(Decimal(1).scaleb(-casas), rounding=ROUND_HALF_EVEN)
    v = abs(v) if v == 0 else v
    return str(v).replace(".", ",")


def p_abnt(p):
    return "<0,001" if p < 0.0005 else abnt(p, 3)


def mann_whitney(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.all(a == a[0]) and np.all(b == a[0]):
        return np.nan                     # grupos idênticos: teste não definido
    return mannwhitneyu(a, b, alternative="two-sided").pvalue


SAIDAS = {}   # nome da aba -> DataFrame, gravado no Excel ao final
print("Resumo do notebook 5:", os.path.basename(ARQ_RESUMO))
print("Escalabilidade do notebook 5:", os.path.basename(ARQ_CODE9))

# %% [markdown]
# ### 1. Amostra e sensores constantes
#
# Os índices de CEQ são calculados na classe de interesse (−1) da amostra. Sensores sem
# variação nessa classe (σ = 0) não têm índices de capacidade definidos e ficam no fim de
# todos os rankings.

# %%
abas_amostra = pd.read_excel(ARQ_AMOSTRA, sheet_name=None)
amostra = abas_amostra["Amostra"]
amostra.columns = amostra.columns.astype(str)
col_classe = amostra.columns[-1]
X_amostra = amostra.drop(columns=[col_classe]).select_dtypes("number")
y_amostra = amostra[col_classe].to_numpy()
A_int = X_amostra[y_amostra == CLASSE_INTERESSE]
constantes = A_int.columns[A_int.std(ddof=0) == 0].tolist()

print(f"Linhas na amostra: {len(amostra)} ({(y_amostra == CLASSE_INTERESSE).sum()} da classe -1 e "
      f"{(y_amostra != CLASSE_INTERESSE).sum()} falhas)")
print(f"Sensores: {X_amostra.shape[1]}; constantes na classe -1: {len(constantes)}; "
      f"com variação: {X_amostra.shape[1] - len(constantes)}")
print("\nResumo gravado pelo notebook 1:")
print(abas_amostra["Resumo"].iloc[:, :2].to_string(index=False))

# %% [markdown]
# ### 2. Reprodutibilidade dos 20 primeiros sensores
#
# Mesma medida do notebook 1: a classe de interesse da amostra é dividida 10 vezes em duas
# metades aleatórias, cada ordenação é calculada nas duas metades e registra-se a fração de
# sensores comuns aos dois top-20. As funções abaixo são **cópias exatas** das funções do
# notebook 1 (`indices_ceq_matriz`, `top_x`, `reprodutibilidade`,
# `reprodutibilidade_literatura`, `scores_literatura`). As estatísticas descritivas usam a
# mesma função de reprodutibilidade dos índices de CEQ.
#
# Valor esperado ao acaso: tamanho da lista dividido pelos 468 sensores com variação.

# %%
# ---- parâmetros e funções copiados do notebook 1 ----
PERCENTIL_LSL, PERCENTIL_USL = 1, 99
N_DIVISOES, SEMENTE = 10, 42


def indices_ceq_matriz(A):
    n = A.shape[0]
    lsl, usl = np.percentile(A, [PERCENTIL_LSL, PERCENTIL_USL], axis=0)
    media = A.mean(0)
    s_longo = A.std(0)                                               # ddof=0
    s_curto = (np.abs(np.diff(A, axis=0)).mean(0) / 1.128) if n >= 2 else np.full(A.shape[1], np.nan)
    ok = (n >= 2) & (s_longo > 0) & (s_curto > 0) & np.isfinite(s_curto)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = {
            "Cp": (usl - lsl) / (6 * s_curto),
            "Cpk": np.minimum(usl - media, media - lsl) / (3 * s_curto),
            "Pp": (usl - lsl) / (6 * s_longo),
            "Ppk": np.minimum(usl - media, media - lsl) / (3 * s_longo),
        }
    r = {k: np.where(ok, v, np.nan) for k, v in r.items()}
    ok_gc = (n >= 2) & (s_longo > 0)
    defeito = ((A < media - s_longo) | (A > media + s_longo)) & ok_gc   # fora de ±1σ
    d_amostra = defeito.sum(1)
    n_def = defeito.sum(0)
    c = (defeito * d_amostra[:, None]).sum(0)
    r.update({"p": np.where(ok_gc, n_def / n, np.nan), "np": np.where(ok_gc, n_def, np.nan),
              "c": np.where(ok_gc, c, np.nan), "u": np.where(ok_gc, c / n, np.nan)})
    return r


def top_x(scores, validas, x):
    sc = np.where(validas & np.isfinite(scores), scores, -np.inf)
    return frozenset(np.argsort(-sc, kind="mergesort")[:x].tolist())


def reprodutibilidade(A, validas, calc_scores, n_div=N_DIVISOES, x=TOP_X, semente=SEMENTE):
    r = np.random.default_rng(semente)
    n = len(A)
    soma = None
    for _ in range(n_div):
        p = r.permutation(n)
        sa, sb = calc_scores(A[np.sort(p[: n // 2])]), calc_scores(A[np.sort(p[n // 2:])])
        conc = {m: len(top_x(sa[m], validas, x) & top_x(sb[m], validas, x)) / x for m in sa}
        soma = conc if soma is None else {m: soma[m] + conc[m] for m in conc}
    return {m: v / n_div for m, v in soma.items()}


def reprodutibilidade_literatura(Xs, y, n_div=N_DIVISOES, x=TOP_X, semente=SEMENTE):
    r = np.random.default_rng(semente)
    i1, i0 = np.flatnonzero(y), np.flatnonzero(~y)
    soma = None
    for _ in range(n_div):
        p1, p0 = r.permutation(i1), r.permutation(i0)
        a = np.concatenate([p1[: len(p1) // 2], p0[: len(p0) // 2]])
        b = np.concatenate([p1[len(p1) // 2:], p0[len(p0) // 2:]])
        (sa, ia), (sb, ib) = scores_literatura(Xs[a], y[a]), scores_literatura(Xs[b], y[b])
        conc = {}
        for m in sa:
            ta = set(ia[np.argsort(-sa[m], kind="mergesort")[:x]].tolist())
            tb = set(ib[np.argsort(-sb[m], kind="mergesort")[:x]].tolist())
            conc[m] = len(ta & tb) / x
        soma = conc if soma is None else {m: soma[m] + conc[m] for m in conc}
    return {m: v / n_div for m, v in soma.items()}


def scores_literatura(Xs, y):
    from sklearn.feature_selection import f_classif, chi2
    med = np.nanmedian(Xs, axis=0)
    Xi = np.where(np.isnan(Xs), med, Xs)
    validas = np.isfinite(med) & (Xi.std(0) > 0)
    Xi = Xi[:, validas]
    g1, g0 = Xi[y], Xi[~y]
    yc = y.astype(float) - y.mean()
    def corr_abs(M):
        Mc = M - M.mean(0)
        return np.abs((Mc * yc[:, None]).sum(0) / (np.sqrt((Mc ** 2).sum(0) * (yc ** 2).sum()) + 1e-12))
    mn = (Xi - Xi.min(0)) / (Xi.max(0) - Xi.min(0))
    m1, m0, v1, v0 = g1.mean(0), g0.mean(0), g1.var(0, ddof=1), g0.var(0, ddof=1)
    mg = Xi.mean(0)
    out = {
        "ANOVA": f_classif(Xi, y)[0],
        "Pearson": corr_abs(Xi),
        "Spearman": corr_abs(np.apply_along_axis(rankdata, 0, Xi)),
        "Chi2": chi2(mn, y)[0],
        "TScore": np.abs(m1 - m0) / np.sqrt(v1 / len(g1) + v0 / len(g0) + 1e-12),
        "FisherScore": (len(g1) * (m1 - mg) ** 2 + len(g0) * (m0 - mg) ** 2)
                       / (len(g1) * v1 + len(g0) * v0 + 1e-9),
    }
    s1, s0 = np.sort(g1, 0), np.sort(g0, 0)
    ks = np.empty(Xi.shape[1])
    for j in range(Xi.shape[1]):
        grade = np.concatenate([s1[:, j], s0[:, j]])
        ks[j] = np.max(np.abs(np.searchsorted(s1[:, j], grade, "right") / len(g1)
                              - np.searchsorted(s0[:, j], grade, "right") / len(g0)))
    out["KS"] = ks
    idx_validas = np.flatnonzero(validas)
    return {m: np.nan_to_num(v, nan=-np.inf) for m, v in out.items()}, idx_validas


# ---- estatísticas descritivas (mesmas do notebook 2), na classe de interesse ----
def scores_descritivas(A):
    from scipy.stats import skew, kurtosis
    sd = A.std(0)
    q1, q3 = np.percentile(A, [25, 75], axis=0)
    with np.errstate(all="ignore"):
        return {"AmplitudeSigma": (A.max(0) - A.min(0)) / sd, "IQRSigma": (q3 - q1) / sd,
                "Assimetria": np.abs(skew(A, 0)), "Curtose": np.abs(kurtosis(A, 0))}


# A amostra já está imputada e em ordem temporal (saída do notebook 1)
X_np = X_amostra.to_numpy(float)
eh_int = (y_amostra == CLASSE_INTERESSE)   # como no notebook 1: True = classe de interesse (-1)
A_np = X_np[eh_int]
validas = A_np.std(0) > 0

linhas = []
for k in [10, 20, 30, 50]:
    rc = reprodutibilidade(A_np, validas, indices_ceq_matriz, x=k)
    rd = reprodutibilidade(A_np, validas, scores_descritivas, x=k)
    rl = reprodutibilidade_literatura(X_np, eh_int, x=k)
    for m in PROP: linhas.append({"k": k, "Grupo": "Proposta", "Metodo": m, "Reprodutibilidade": rc[m]})
    for m in DESC: linhas.append({"k": k, "Grupo": "Descritivas", "Metodo": m, "Reprodutibilidade": rd[m]})
    for m, v in rl.items(): linhas.append({"k": k, "Grupo": "Literatura", "Metodo": m, "Reprodutibilidade": v})
REPRO = pd.DataFrame(linhas)
SAIDAS["2_Reprodutibilidade"] = REPRO

r20 = REPRO[REPRO.k == TOP_X].sort_values(["Grupo", "Reprodutibilidade"], ascending=[True, False])
print("Reprodutibilidade do top-20 (2 casas decimais):")
print(r20.assign(Valor=r20.Reprodutibilidade.map(lambda v: abnt(v, 2)))[["Grupo", "Metodo", "Valor"]].to_string(index=False))

sens = (REPRO.groupby(["k", "Grupo"]).Reprodutibilidade.agg(["min", "max"]).unstack()
        .assign(Acaso=lambda d: d.index / validas.sum()))
print(f"\nSensibilidade ao tamanho da lista (acaso = k/{validas.sum()}):")
print(sens.map(lambda v: abnt(v, 2)).to_string())

# %% [markdown]
# ### 3. Equivalência entre rankings: Friedman/Nemenyi e correlação de Spearman
#
# **Como o Spearman funciona.** Para dois rankings dos mesmos *n* sensores, calcula-se, para
# cada sensor *j*, a diferença d_j entre suas posições nos dois rankings:
#
# ρ = 1 − 6·Σd_j² / [n·(n² − 1)]
#
# A soma Σd_j² é 0 quando as ordens são idênticas (ρ = 1) e máxima, n(n² − 1)/3, quando
# uma ordem é o inverso da outra (ρ = −1). Ordens sem relação dão ρ perto de 0.
#
# **Como interpretar** (convenção do texto): ρ ≥ 0,70 forte; 0,30 a 0,69 moderada;
# < 0,30 fraca; negativo, ordem invertida.
#
# A célula abaixo confere o exemplo da Metodologia (cinco sensores) e verifica que a
# fórmula dá o mesmo valor que `scipy.stats.spearmanr` nos rankings reais.

# %%
def rho_formula(pos_a, pos_b):
    d = np.asarray(pos_a, float) - np.asarray(pos_b, float)
    n = len(d)
    return 1 - 6 * (d ** 2).sum() / (n * (n ** 2 - 1))

A_ex, B_ex, C_ex = [1, 2, 3, 4, 5], [2, 1, 3, 5, 4], [5, 4, 3, 2, 1]
print(f"Exemplo da Metodologia: rho(A,B) = {abnt(rho_formula(A_ex, B_ex), 2)}; "
      f"rho(A,C) = {abnt(rho_formula(A_ex, C_ex), 2)}")

# ---- matriz de posições: 590 sensores x 29 rankings ----
abas_rk = pd.read_excel(ARQ_RANKINGS, sheet_name=None)
pos = {}
for aba, df in abas_rk.items():
    m = nome_metodo(aba)
    if grupo(m) is None:
        continue
    df = df.copy()
    df["Feature"] = df["Feature"].map(lambda v: str(int(v)) if isinstance(v, float) and v.is_integer() else str(v))
    pos[m] = df.set_index("Feature")["Ranking"].astype(float)
POS = pd.DataFrame(pos)[PROP + LIT + DESC].dropna()
assert POS.shape == (590, 29), POS.shape
assert (POS.nunique() == 590).all(), "há posições empatadas"

# Conferência: fórmula = scipy, nos rankings reais
a, b = "Cp", "ANOVA"
print(f"Cp x ANOVA: fórmula = {rho_formula(POS[a], POS[b]):.6f}; scipy = {spearmanr(POS[a], POS[b]).statistic:.6f}")

# ---- Friedman + Nemenyi (sensores como blocos, posição como medida) ----
chi2_f, p_f = friedmanchisquare(*[POS[c] for c in POS.columns])
NEMENYI = sp.posthoc_nemenyi_friedman(POS)
RHO = POS.corr(method="spearman")
postos = POS.rank(axis=1).mean()
print(f"\nFriedman: chi2 = {abnt(chi2_f, 2)}, gl = {POS.shape[1] - 1}, p = {p_abnt(p_f)}")
print(f"Posição média de cada ranking: {POS.mean().unique()} (sempre a mesma)")
print(f"Postos médios de Friedman: {abnt(postos.min(), 2)} a {abnt(postos.max(), 2)} (esperado: {abnt((POS.shape[1] + 1) / 2, 2)})")

G_rk = {m: grupo(m) for m in POS.columns}
PARES = pd.DataFrame([{"A": a, "B": b, "GA": G_rk[a], "GB": G_rk[b], "p_Nemenyi": NEMENYI.loc[a, b],
                       "rho": RHO.loc[a, b]} for a, b in itertools.combinations(POS.columns, 2)])
PARES["Concordancia"] = pd.cut(PARES.rho, [-1.01, 0, 0.30, 0.70, 1.01], right=False,
                               labels=["Inversa", "Fraca", "Moderada", "Forte"])
SAIDAS["3_Pares_Rankings"] = PARES

def resumo_grupos(df):
    lin = []
    for i, g1 in enumerate(GRUPOS):
        for g2 in GRUPOS[i:]:
            d = df[((df.GA == g1) & (df.GB == g2)) | ((df.GA == g2) & (df.GB == g1))]
            lin.append({"Par": f"{g1} x {g2}", "Pares": len(d),
                        "p_min": d.p_Nemenyi.min(), "p_max": d.p_Nemenyi.max(),
                        "Equivalentes": int((d.p_Nemenyi > 0.05).sum()),
                        "rho_min": d.rho.min(), "rho_mediana": d.rho.median(), "rho_max": d.rho.max(),
                        "Fraca": int((d.rho < 0.30).sum()), "Moderada": int(((d.rho >= 0.30) & (d.rho < 0.70)).sum()),
                        "Forte": int((d.rho >= 0.70).sum())})
    d = df
    lin.append({"Par": "Total", "Pares": len(d), "p_min": d.p_Nemenyi.min(), "p_max": d.p_Nemenyi.max(),
                "Equivalentes": int((d.p_Nemenyi > 0.05).sum()), "rho_min": d.rho.min(),
                "rho_mediana": d.rho.median(), "rho_max": d.rho.max(), "Fraca": int((d.rho < 0.30).sum()),
                "Moderada": int(((d.rho >= 0.30) & (d.rho < 0.70)).sum()), "Forte": int((d.rho >= 0.70).sum())})
    return pd.DataFrame(lin)

EQUIV = resumo_grupos(PARES)
SAIDAS["3_Equivalencia_Grupos"] = EQUIV
fmt_eq = EQUIV.copy()
for c in ["p_min", "p_max"]: fmt_eq[c] = fmt_eq[c].map(lambda v: abnt(v, 3))
for c in ["rho_min", "rho_mediana", "rho_max"]: fmt_eq[c] = fmt_eq[c].map(lambda v: abnt(v, 2))
print("\nTabela de equivalência por par de grupos:")
print(fmt_eq.to_string(index=False))
print(f"\nPares 'equivalentes' com rho < 0,50 (alerta do notebook 3): {((PARES.p_Nemenyi > 0.05) & (PARES.rho < 0.5)).sum()}")

# Índice de CEQ x literatura: método mais parecido com cada índice
print("\nMétodo da literatura mais parecido com cada índice de CEQ:")
for m in PROP:
    s = RHO.loc[m, LIT].sort_values(ascending=False)
    empatados = s[s.map(lambda v: abnt(v, 2)) == abnt(s.iloc[0], 2)].index.tolist()
    print(f"  {m}: {', '.join(empatados)} ({abnt(s.iloc[0], 2)})")

RHO_CEQ_DESC = RHO.loc[PROP, DESC].assign(Maior=lambda d: d.max(axis=1))
SAIDAS["3_Spearman_CEQ_x_Descritivas"] = RHO_CEQ_DESC
SAIDAS["3_Spearman_Matriz"] = RHO
print("\nSpearman: índices de CEQ x estatísticas descritivas")
print(RHO_CEQ_DESC.map(lambda v: abnt(v, 2)).to_string())
print("\nSpearman entre os índices de CEQ")
print(RHO.loc[PROP, PROP].map(lambda v: abnt(v, 2)).to_string())

# %% [markdown]
# #### 3.1 Verificação complementar: efeito dos 122 sensores constantes
#
# **Não é usada nas tabelas do texto** (que usam os 590 sensores, como o notebook 3). Os 122
# sensores constantes ficam nas posições 469 a 590 de todos os rankings, sempre na mesma
# ordem, e somam concordância a todos os pares. Esta célula mostra quanto o ρ muda quando
# eles são retirados — base da ressalva feita no texto (um ρ baixo com 590 sensores indica
# que as ordens dos sensores informativos não concordam, mas não em que direção).

# %%
fixos = POS.index[(POS > 590 - len(constantes)).all(axis=1)]
print(f"Sensores nas mesmas posições finais em todos os rankings: {len(fixos)} "
      f"(posições {int(POS.loc[fixos].min().min())} a {int(POS.loc[fixos].max().max())})")
RHO_INF = POS.drop(fixos).corr(method="spearman")
comp = pd.DataFrame({"rho_590": [RHO.loc[a, b] for a, b in itertools.combinations(POS.columns, 2)],
                     "rho_informativos": [RHO_INF.loc[a, b] for a, b in itertools.combinations(POS.columns, 2)]},
                    index=[f"{a} x {b}" for a, b in itertools.combinations(POS.columns, 2)])
SAIDAS["3_1_Spearman_590_vs_468"] = comp
print(comp.loc[["c x Assimetria", "p x IQRSigma", "Cp x ANOVA", "Cpk x Ppk"]].map(lambda v: abnt(v, 2)).to_string())

# %% [markdown]
# ### 4. Relação entre os índices p e c e o IQR/σ
#
# O índice c de um sensor soma, nas leituras em que ele está fora de ±1σ, quantos sensores
# estavam fora ao mesmo tempo. Logo, c_j = (n·p_j)·D̄_j, em que D̄_j é o número médio de
# sensores fora nessas leituras. Se D̄_j varia pouco entre os sensores, c e p ordenam os
# sensores quase da mesma forma. A segunda parte compara o IQR/σ e a fração fora de ±1σ em
# três distribuições conhecidas (valores teóricos).

# %%
A_v = A_np[:, validas]
media, desvio = A_v.mean(0), A_v.std(0)
fora = (A_v < media - desvio) | (A_v > media + desvio)
n_lin = len(A_v)
n_fora = fora.sum(0)
d_linha = fora.sum(1)
c_idx = (fora * d_linha[:, None]).sum(0)
D_bar = np.where(n_fora > 0, c_idx / np.maximum(n_fora, 1), np.nan)
assert np.allclose(c_idx, n_fora * np.nan_to_num(D_bar))       # c = (n·p)·D̄
print(f"Leituras da classe -1: {n_lin}")
print(f"Leituras fora de ±1σ por sensor: {n_fora.min()} a {n_fora.max()}")
print(f"D̄ (sensores fora ao mesmo tempo, média por sensor): {abnt(np.nanmin(D_bar), 0)} a {abnt(np.nanmax(D_bar), 0)}")
print(f"Spearman entre p e c: {RHO.loc['p', 'c']:.4f}")

DIST = pd.DataFrame({
    "Distribuição": ["Laplace", "Normal", "Uniforme"],
    # Laplace(b): sigma = b*sqrt(2), IQR = 2*b*ln2, P(|X| > sigma) = exp(-sqrt(2))
    # Normal: IQR = 1,349*sigma, P(|Z| > 1) = 0,3173
    # Uniforme(largura w): sigma = w/sqrt(12), IQR = w/2, P(fora) = 1 - 2/sqrt(12)
    "IQR/sigma": [2 * np.log(2) / np.sqrt(2), 2 * 0.6744897501960817, np.sqrt(12) / 2],
    "Fracao_fora_1sigma": [np.exp(-np.sqrt(2)), 0.31731050786291415, 1 - 2 / np.sqrt(12)],
})
SAIDAS["4_Distribuicoes"] = DIST
print(DIST.assign(**{c: DIST[c].map(lambda v: abnt(v, 2)) for c in ["IQR/sigma", "Fracao_fora_1sigma"]}).to_string(index=False))

# %% [markdown]
# ### 5. Modelos selecionados: médias ± desvio-padrão, Mann-Whitney e tempo estimado
#
# Cada ranking tem dois modelos selecionados por paradigma (Kneedle e estatístico), lidos do
# `Resumo_Global_Dados` do notebook 5. As métricas são recalculadas a partir dos 5 valores por
# partição (`_Por_Fold`), com precisão completa. O tempo do modelo selecionado é estimado pela
# lei de potência T = c·N^α, com c = T̄ / média(N^α, N = 1..590), em que α e T̄ vêm do
# `CODE9_Escalabilidade_Por_Ranking` do notebook 5.

# %%
SEL = pd.read_excel(ARQ_RESUMO)
ESC = pd.read_excel(ARQ_CODE9)
for d in (SEL, ESC):
    d["Metodo"] = d.Ranking.map(nome_metodo)
    d["G"] = d.Metodo.map(grupo)
SEL = SEL[SEL.G.notna()].copy()
ESC = ESC[ESC.G.notna()].copy()
assert SEL.groupby("G").Metodo.nunique().to_dict() == {"Proposta": 6, "Literatura": 19, "Descritivas": 4}

COLS_FOLD = {"BA": "Acurácia Balanceada_Por_Fold", "Se": "Sensibilidade_Por_Fold",
             "Sp": "Especificidade_Por_Fold", "AUC": "AUC_Por_Fold"}
for k, c in COLS_FOLD.items():
    SEL[k] = SEL[c].map(lambda s: float(np.mean(ast.literal_eval(s))))
SEL["N"] = SEL["Nº de Atributos"].astype(float)
SEL["T"] = SEL.Tipo
Ns = np.arange(1, 591)
ESC["cte"] = ESC["Mean Time (s)"] / ESC["Scalability (alpha)"].map(lambda a: np.mean(Ns ** a))
SEL = SEL.merge(ESC[["Paradigma", "Metodo", "cte", "Scalability (alpha)"]], on=["Paradigma", "Metodo"])
SEL["Test"] = SEL.cte * SEL.N ** SEL["Scalability (alpha)"]
POR_RANKING = SEL[["Paradigma", "T", "G", "Metodo", "N", "BA", "Se", "Sp", "AUC", "Test"]].copy()
SAIDAS["5_Modelos_Selecionados"] = POR_RANKING

lin = []
for (p, t), d in POR_RANKING.groupby(["Paradigma", "T"]):
    r = {"P": p, "T": t}
    for k in ["N", "BA", "Se", "Sp", "AUC", "Test"]:
        A, L, C = (d[d.G == g][k] for g in GRUPOS)
        r[k + "_prop"], r[k + "_lit"], r[k + "_desc"] = A.mean(), L.mean(), C.mean()
        r[k + "_sd_prop"], r[k + "_sd_lit"], r[k + "_sd_desc"] = A.std(ddof=1), L.std(ddof=1), C.std(ddof=1)
        r[k + "_p_lit"], r[k + "_p_desc"] = mann_whitney(A, L), mann_whitney(A, C)
    lin.append(r)
QUALIDADE = pd.DataFrame(lin)
SAIDAS["5_Qualidade_e_Tempo_Selec"] = QUALIDADE

lin = []
for p, d in ESC.groupby("Paradigma"):
    for m in ["Mean Time (s)", "Efficiency (s/feat.)", "Scalability (alpha)", "Reliability (r)"]:
        A, L, C = (d[d.G == g][m] for g in GRUPOS)
        lin.append({"P": p, "Indicador": m, "prop": A.mean(), "lit": L.mean(), "desc": C.mean(),
                    "sd_prop": A.std(ddof=1), "sd_lit": L.std(ddof=1), "sd_desc": C.std(ddof=1),
                    "p_lit": mann_whitney(A, L), "p_desc": mann_whitney(A, C)})
TEMPO = pd.DataFrame(lin)
SAIDAS["5_Tempo_Curva"] = TEMPO


def tabela_paradigma(p):
    """Tabela de qualidade de um paradigma, como no texto (média ± DP; p da proposta x literatura)."""
    lin = []
    for t in ["Estatístico", "Kneedle"]:
        r = QUALIDADE[(QUALIDADE.P == p) & (QUALIDADE["T"] == t)].iloc[0]
        for k, rot, c in [("N", "Nº de atributos", 1), ("BA", "Acurácia balanceada", 2), ("Se", "Sensibilidade", 2),
                          ("Sp", "Especificidade", 2), ("AUC", "AUC", 2), ("Test", "Tempo estimado (s)", 2)]:
            lin.append({"Critério": t, "Indicador": rot,
                        "Proposta": f"{abnt(r[k + '_prop'], c)} ± {abnt(r[k + '_sd_prop'], c)}",
                        "Literatura": f"{abnt(r[k + '_lit'], c)} ± {abnt(r[k + '_sd_lit'], c)}",
                        "Descritivas": f"{abnt(r[k + '_desc'], c)} ± {abnt(r[k + '_sd_desc'], c)}",
                        "p": p_abnt(r[k + "_p_lit"]) if not np.isnan(r[k + "_p_lit"]) else "--"})
    return pd.DataFrame(lin)


def tabela_tempo(p):
    lin = []
    for m, rot, mult in [("Mean Time (s)", "Tempo médio por modelo (s)", 1), ("Efficiency (s/feat.)", "Tempo por atributo (ms)", 1000),
                         ("Scalability (alpha)", "Escalabilidade (alpha)", 1), ("Reliability (r)", "Confiabilidade (r)", 1)]:
        r = TEMPO[(TEMPO.P == p) & (TEMPO.Indicador == m)].iloc[0]
        lin.append({"Indicador": rot, "Proposta": f"{abnt(r.prop * mult, 2)} ± {abnt(r.sd_prop * mult, 2)}",
                    "Literatura": f"{abnt(r.lit * mult, 2)} ± {abnt(r.sd_lit * mult, 2)}",
                    "Descritivas": f"{abnt(r.desc * mult, 2)} ± {abnt(r.sd_desc * mult, 2)}", "p": p_abnt(r.p_lit)})
    return pd.DataFrame(lin)


for p in PARADIGMAS:
    print(f"\n===== {p}: qualidade e tempo estimado do modelo selecionado")
    print(tabela_paradigma(p).to_string(index=False))
    print(f"----- {p}: tempo ao longo da curva completa")
    print(tabela_tempo(p).to_string(index=False))

# %% [markdown]
# ### 6. Placar: qual abordagem teve a maior média
#
# Em cada paradigma há 8 comparações (4 métricas × 2 critérios). As médias são comparadas
# **com duas casas decimais** (ABNT); médias iguais nessa precisão são empate. Uma vitória é
# significativa quando o Mann-Whitney dá p ≤ 0,05. Para o número de atributos, vence a menor
# média (uma casa decimal).

# %%
NOME_MET = {"BA": "Acurácia balanceada", "Se": "Sensibilidade", "Sp": "Especificidade", "AUC": "AUC", "N": "Nº atributos"}
CASAS_PLACAR = {"BA": 2, "Se": 2, "Sp": 2, "AUC": 2, "N": 1}


def vencedor(a, b, k, nome_b):
    ra, rb = arred(a, CASAS_PLACAR[k]), arred(b, CASAS_PLACAR[k])
    if ra == rb: return "empate"
    melhor_a = (ra < rb) if k == "N" else (ra > rb)
    return "Proposta" if melhor_a else nome_b


lin = []
for _, r in QUALIDADE.iterrows():
    for k in ["BA", "Se", "Sp", "AUC", "N"]:
        for outro, suf in [("Literatura", "lit"), ("Descritivas", "desc")]:
            w = vencedor(r[k + "_prop"], r[k + "_" + suf], k, outro)
            lin.append({"P": r.P, "T": r["T"], "Metrica": NOME_MET[k], "Contra": outro,
                        "Proposta": r[k + "_prop"], "Outro": r[k + "_" + suf], "p": r[k + "_p_" + suf],
                        "Vencedor": w, "Significativo": bool(w != "empate" and r[k + "_p_" + suf] <= 0.05)})
PLACAR = pd.DataFrame(lin)
SAIDAS["6_Placar"] = PLACAR

q = PLACAR[PLACAR.Metrica != "Nº atributos"]
for outro in ["Literatura", "Descritivas"]:
    d = q[q.Contra == outro]
    tab = d.groupby("P").Vencedor.value_counts().unstack(fill_value=0).reindex(PARADIGMAS)
    print(f"\nPlacar Proposta x {outro} (8 comparações por paradigma):")
    print(tab.to_string())
    print("Total:", d.Vencedor.value_counts().to_dict(),
          "| significativas:", d[d.Significativo].Vencedor.value_counts().to_dict())
n_atr = PLACAR[(PLACAR.Metrica == "Nº atributos") & (PLACAR.Contra == "Descritivas")]
print("\nProposta x Descritivas, quem usou menos atributos (14 situações):", n_atr.Vencedor.value_counts().to_dict())

# %% [markdown]
# ### 7. Fronteira de Pareto e distância dos índices de CEQ até ela (critério Kneedle)
#
# Fronteira entre os 29 modelos escolhidos pelo Kneedle em cada paradigma (um por ranking),
# com a acurácia balanceada arredondada a duas casas: um modelo está na fronteira se nenhum
# outro tem N menor ou igual e acurácia balanceada maior ou igual, sendo melhor em pelo menos
# um dos dois. Para cada índice de CEQ fora da fronteira, mede-se: (A) a maior acurácia da
# fronteira com até o mesmo N e a diferença; (B) o menor N com que a fronteira alcança a
# acurácia do índice, e quantos atributos o índice usa a mais.

# %%
POR_RANKING["BAr"] = POR_RANKING.BA.map(lambda x: arred(x, 2))
KNEE = POR_RANKING[POR_RANKING["T"] == "Kneedle"]
# Por que só o Kneedle: o critério estatístico escolhe muitos modelos degenerados
sel_est = POR_RANKING[POR_RANKING["T"] == "Estatístico"]
print(f"Estatístico: {(sel_est.N == 1).sum()} de {len(sel_est)} modelos com 1 sensor; "
      f"{(sel_est.BAr <= 0.52).sum()} com acurácia balanceada <= 0,52")
print(f"Kneedle:     {(KNEE.N == 1).sum()} de {len(KNEE)} modelos com 1 sensor; "
      f"{(KNEE.BAr <= 0.52).sum()} com acurácia balanceada <= 0,52")
fr = []
for p, d in KNEE.groupby("Paradigma"):
    for i, r in d.iterrows():
        dominado = ((d.N <= r.N) & (d.BAr >= r.BAr) & ((d.N < r.N) | (d.BAr > r.BAr))).any()
        if not dominado: fr.append(r)
FRONTEIRA = pd.DataFrame(fr)
SAIDAS["7_Fronteira"] = FRONTEIRA

lin = []
for p in PARADIGMAS:
    f_p = FRONTEIRA[FRONTEIRA.Paradigma == p]
    for _, r in KNEE[(KNEE.Paradigma == p) & (KNEE.G == "Proposta")].iterrows():
        a = f_p[f_p.N <= r.N].sort_values(["BAr", "N"], ascending=[False, True]).iloc[0]
        b = f_p[f_p.BAr >= r.BAr].sort_values(["N", "BAr"], ascending=[True, False]).iloc[0]
        lin.append({"P": p, "Indice": r.Metodo, "T": r["T"], "N": int(r.N), "BA": r.BAr,
                    "A_N": int(a.N), "A_BA": a.BAr, "A_metodo": f"{a.Metodo} ({a['T']})", "Diferenca": r.BAr - a.BAr,
                    "B_N": int(b.N), "B_BA": b.BAr, "B_metodo": f"{b.Metodo} ({b['T']})", "A_mais": int(r.N - b.N),
                    "Na_fronteira": bool(((f_p.Metodo == r.Metodo) & (f_p["T"] == r["T"])).any())})
DISTANCIA = pd.DataFrame(lin)
SAIDAS["7_Distancia"] = DISTANCIA

for p in PARADIGMAS:
    f_p = FRONTEIRA[FRONTEIRA.Paradigma == p]
    print(f"\n===== {p}: fronteira ({len(f_p)} modelos; por grupo: {f_p.G.value_counts().to_dict()})")
    print(f_p.sort_values(["N", "BAr"])[["G", "Metodo", "T", "N", "BAr"]].to_string(index=False))

# %% [markdown]
# ### 8. Melhor modelo individual de cada abordagem, em cada métrica (critério Kneedle)
#
# Maior valor (duas casas) entre os modelos escolhidos pelo Kneedle em cada grupo; em caso de
# empate, o de menos sensores.

# %%
lin = []
for p in PARADIGMAS:
    d = KNEE[KNEE.Paradigma == p]
    for k in ["BA", "Se", "Sp", "AUC"]:
        for g in GRUPOS:
            x = d[d.G == g].assign(v=lambda z: z[k].map(lambda v: arred(v, 2))).sort_values(["v", "N"], ascending=[False, True])
            top = x[(x.v == x.v.iloc[0]) & (x.N == x[x.v == x.v.iloc[0]].N.min())]
            lin.append({"P": p, "Metrica": NOME_MET[k], "Grupo": g, "Valor": x.v.iloc[0], "N": int(top.N.iloc[0]),
                        "Modelos": ", ".join(f"{m} ({t})" for m, t in zip(top.Metodo, top["T"]))})
MELHORES = pd.DataFrame(lin)
SAIDAS["8_Melhores_Modelos"] = MELHORES
print(MELHORES.to_string(index=False))

# %% [markdown]
# ### 9. Sensores dispensáveis e oscilação da curva
#
# Dispensáveis = sensores retirados do fim do ranking sem que a acurácia balanceada caia mais
# que um desvio-padrão abaixo da melhor da curva, isto é, 590 − N do critério estatístico.
# Para os dispensáveis informativos (entre as posições N+1 e 468), verifica-se se têm
# correlação de Spearman |ρ| ≥ 0,70 (na amostra) com pelo menos um sensor mantido.
#
# A curva completa de cada ranking é lida dos `Modelos<Paradigma>_CV_Consolidado.xlsx`. A
# oscilação (quantas remoções aumentam, diminuem ou mantêm a acurácia balanceada) usa a coluna
# `Acurácia Balanceada Média`, gravada com três casas pelo notebook 4.

# %%
mapa = pd.read_excel(ARQ_MAPA)
mapa = {(r.Paradigma, r["Nome da Aba"]): r.Ranking for _, r in mapa.iterrows()}
curvas = []
for p in PARADIGMAS:
    abas = pd.read_excel(os.path.join(PASTA, f"Modelos{p}_CV_Consolidado.xlsx"), sheet_name=None)
    for aba, df in abas.items():
        m = nome_metodo(mapa[(p, aba)])
        if grupo(m) is None: continue
        d = df[["Nº de Atributos", "Acurácia Balanceada Média", "Acurácia Balanceada_Por_Fold"]].copy()
        d["BA"] = d["Acurácia Balanceada_Por_Fold"].map(lambda s: float(np.mean(ast.literal_eval(s))))
        d["P"], d["Metodo"], d["G"] = p, m, grupo(m)
        curvas.append(d.drop(columns="Acurácia Balanceada_Por_Fold"))
CURVAS = pd.concat(curvas, ignore_index=True)
n_inf = 590 - len(constantes)

# ---- oscilação da acurácia balanceada a cada remoção de sensor informativo ----
lin = []
for (p, m), d in CURVAS.groupby(["P", "Metodo"]):
    v = d[d["Nº de Atributos"] <= n_inf].sort_values("Nº de Atributos", ascending=False)["Acurácia Balanceada Média"].to_numpy()
    dif = np.diff(v)
    lin.append({"P": p, "Metodo": m, "Diminuiu": (dif < 0).sum(), "Aumentou": (dif > 0).sum(), "Igual": (dif == 0).sum()})
OSC = pd.DataFrame(lin)
SAIDAS["9_Oscilacao"] = OSC
print(f"Oscilação média por ranking ({n_inf - 1} remoções de sensores informativos):")
print(OSC.groupby("P")[["Diminuiu", "Aumentou", "Igual"]].mean().reindex(PARADIGMAS).map(lambda v: abnt(v, 1)).to_string())

# ---- dispensáveis ----
melhor_curva = CURVAS.groupby(["P", "Metodo"]).BA.max().rename("BA_melhor")
X_rank = X_amostra.rank()
CORR_SENS = X_rank.corr().abs()                      # Spearman entre sensores, na amostra
ordem = {m: POS[m].sort_values().index.tolist() for m in POS.columns}
lin = []
for _, r in POR_RANKING[POR_RANKING["T"] == "Estatístico"].iterrows():
    N = int(r.N)
    mantidos, disp_inf = ordem[r.Metodo][:N], ordem[r.Metodo][N:n_inf]
    forte = (CORR_SENS.loc[disp_inf, mantidos].fillna(0).max(axis=1) >= LIMIAR_CORR_FORTE).mean() if disp_inf else np.nan
    lin.append({"P": r.Paradigma, "Metodo": r.Metodo, "G": r.G, "N_mantidos": N, "Dispensaveis": 590 - N,
                "Dispensaveis_informativos": max(n_inf - N, 0), "Pct_de_590": (590 - N) / 590,
                "BA_mantidos": r.BA, "BA_melhor_curva": melhor_curva.loc[(r.Paradigma, r.Metodo)],
                "Frac_corr_forte": forte})
DISP = pd.DataFrame(lin)
SAIDAS["9_Dispensaveis"] = DISP
res = DISP.groupby(["P", "G"])[["Dispensaveis", "BA_melhor_curva", "Frac_corr_forte"]].mean().unstack()
print("\nDispensáveis (média por grupo), melhor AB da curva e fração com correlação forte:")
print(res.reindex(PARADIGMAS).round(3).to_string())

# %% [markdown]
# ### 10. Números da síntese
#
# Razões entre a proposta e a literatura (calculadas com os valores não arredondados e
# arredondadas ao final), faixas e contagens citadas na Síntese e discussão.

# %%
lin = []
for p in PARADIGMAS:
    rs = QUALIDADE[(QUALIDADE.P == p) & (QUALIDADE["T"] == "Estatístico")].iloc[0]
    rk = QUALIDADE[(QUALIDADE.P == p) & (QUALIDADE["T"] == "Kneedle")].iloc[0]
    mt = TEMPO[(TEMPO.P == p) & (TEMPO.Indicador == "Mean Time (s)")].iloc[0]
    al = TEMPO[(TEMPO.P == p) & (TEMPO.Indicador == "Scalability (alpha)")].iloc[0]
    f_p = FRONTEIRA[FRONTEIRA.Paradigma == p]
    dp = DISP[DISP.P == p]
    lin.append({"P": p,
                "Razao_atrib_Est": rs.N_prop / rs.N_lit, "Razao_atrib_Kne": rk.N_prop / rk.N_lit,
                "Razao_tempo_Kne": rk.Test_prop / rk.Test_lit, "p_tempo_Kne": rk.Test_p_lit,
                "Dif_tempo_medio_pct": (mt.prop - mt.lit) / mt.lit * 100, "p_tempo_medio": mt.p_lit,
                "alpha_prop": al.prop, "alpha_lit": al.lit, "p_alpha": al.p_lit,
                "Fronteira_prop": int((f_p.G == "Proposta").sum()), "Fronteira_lit": int((f_p.G == "Literatura").sum()),
                "Fronteira_desc": int((f_p.G == "Descritivas").sum()),
                "Disp_prop": dp[dp.G == "Proposta"].Dispensaveis.mean(), "Disp_lit": dp[dp.G == "Literatura"].Dispensaveis.mean(),
                "Disp_desc": dp[dp.G == "Descritivas"].Dispensaveis.mean(),
                "Corr_forte_prop": dp[dp.G == "Proposta"].Frac_corr_forte.mean()})
SINTESE = pd.DataFrame(lin)
SAIDAS["10_Sintese"] = SINTESE
print(SINTESE.round(3).to_string(index=False))
print(f"\nMaior diferença no tempo médio por modelo: {abnt(SINTESE.Dif_tempo_medio_pct.abs().max(), 1)}%")
print(f"Alpha: de {abnt(TEMPO[TEMPO.Indicador == 'Scalability (alpha)'][['prop', 'lit']].min().min(), 2)} "
      f"a {abnt(TEMPO[TEMPO.Indicador == 'Scalability (alpha)'][['prop', 'lit']].max().max(), 2)}")
print(f"Acurácia balanceada dos modelos selecionados: {abnt(POR_RANKING.BA.min(), 2)} a {abnt(POR_RANKING.BA.max(), 2)}")
dt = SEL[SEL.Paradigma == "DT"]
print(f"DT: AUC igual à acurácia balanceada (3 casas) em "
      f"{(dt['AUC Média'] == dt['Acurácia Balanceada Média']).sum()} de {len(dt)} modelos selecionados")

# %% [markdown]
# ### 11. Estabilidade dos sensores escolhidos (índice de Nogueira)
#
# Os 29 rankings são recalculados em `M_SUB = 100` subamostras estratificadas de 50% da
# amostra (sem reposição, mantendo a ordem temporal), com as **mesmas fórmulas do notebook 2**
# (a função `rankings_nb2` abaixo reproduz os rankings do `Rankings_Consolidado.xlsx` na
# amostra completa; a primeira célula confere isso). Em cada paradigma, cada ranking escolhe,
# em cada subamostra, os seus k primeiros sensores, com k = número de sensores do Kneedle
# naquele ranking e paradigma. A estabilidade é o índice de Nogueira, Sechidis e Brown (2018),
# com IC 95%:
#
# Φ = 1 − [ (1/d) Σ_f s_f² ] / [ (k̄/d)(1 − k̄/d) ],  s_f² = M/(M−1) · p̂_f (1 − p̂_f)
#
# Classificação (Fleiss, 1981): excelente Φ > 0,75; razoável a boa 0,40 a 0,75; fraca < 0,40.
# Um ranking é estável quando Φ ≥ 0,40.
#
# ⏱ É a parte mais lenta do notebook (cerca de 15 a 25 minutos com 2 núcleos; o ReliefF, o
# SVD Entropy e o MIC dominam o tempo). O resultado é guardado em `8_ordens_subamostras.pkl`
# e reaproveitado se o notebook for rodado de novo.
#
# Na conferência, todos os rankings têm ρ = 1,00 ou muito próximo e os mesmos 20 primeiros
# sensores (o SPEC, 19 de 20, por diferença numérica no cálculo dos autovetores). As pequenas
# diferenças no InformationGain, GainRatio e SymmetricalUncertainty (ρ ≈ 0,94 a 0,97) vêm do
# estimador de informação mútua do scikit-learn e ficam nos sensores do fim da lista.

# %%
# %pip install skrebate
import warnings as _w
from scipy.stats import skew, kurtosis, spearmanr as _spr, kendalltau, ttest_ind, ks_2samp, entropy, norm
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
from sklearn.metrics import mutual_info_score
from joblib import Parallel, delayed

PERCENTIL_LSL, PERCENTIL_USL = 1, 99
PROP = ["Cp", "Cpk", "Pp", "Ppk", "p", "c"]
LIT = ["ANOVA", "ApproxMIC", "Chi2", "FScore_ANOVA", "FisherScore", "GainRatio", "GiniIndex",
       "HellingerDistance", "InformationGain", "KS", "Kendall", "LaplacianScore", "Pearson", "ReliefF",
       "SPEC", "SVDEntropy", "Spearman", "SymmetricalUncertainty", "TScore"]
DESC = ["AmplitudeSigma", "IQRSigma", "Assimetria", "Curtose"]


def _ordem(features, scores, constantes, ascending=False):
    """montar_ranking do notebook 2: constantes -> NaN -> fim; retorna a lista ordenada."""
    d = pd.DataFrame({"Feature": list(features), "Score": np.asarray(scores, dtype=float)})
    d.loc[d.Feature.isin(constantes), "Score"] = np.nan
    return d.sort_values("Score", ascending=ascending, na_position="last").Feature.tolist()


def _ordem_ceq(features, scores):
    """salvar_ranking do bloco CEQ do notebook 2 (sem a regra de CONSTANTES; NaN no fim)."""
    d = pd.DataFrame({"Feature": list(features), "Score": np.asarray(scores, dtype=float)})
    return d.sort_values("Score", ascending=False, na_position="last").Feature.tolist()


def rankings_nb2(df, classe=-1, lentos=True):
    """df: amostra (ou subamostra) com a classe na última coluna, já imputada e em ordem
    temporal. Retorna {metodo: lista de sensores do melhor para o pior}."""
    df = df.reset_index(drop=True)
    alvo = df.columns[-1]
    X_full = df.drop(columns=[alvo]).select_dtypes("number")
    X_full = X_full.fillna(X_full.median())
    cols = X_full.columns
    eh = (df[alvo].astype(str) == str(classe)).to_numpy()
    y = eh.astype(int)
    X_classe = X_full[eh].reset_index(drop=True)
    CONST = set(cols[X_full.std(ddof=0) == 0])
    out = {}

    # ---- Proposta (calcular_ceq e calcular_graficos_controle, na classe escolhida) ----
    A = X_classe.to_numpy(float)
    n = len(A)
    lsl, usl = np.percentile(A, [PERCENTIL_LSL, PERCENTIL_USL], axis=0)
    media = A.mean(0)
    s_l = A.std(0)
    s_c = np.abs(np.diff(A, axis=0)).mean(0) / 1.128
    ok = (s_l > 0) & (s_c > 0)
    with np.errstate(all="ignore"):
        ceq = {"Cp": (usl - lsl) / (6 * s_c), "Cpk": np.minimum(usl - media, media - lsl) / (3 * s_c),
               "Pp": (usl - lsl) / (6 * s_l), "Ppk": np.minimum(usl - media, media - lsl) / (3 * s_l)}
    for m, v in ceq.items():
        out[m] = _ordem_ceq(cols, np.where(ok, v, np.nan))
    okg = s_l > 0
    defeito = ((A < media - s_l) | (A > media + s_l)) & okg
    d_lin = defeito.sum(1)
    out["p"] = _ordem_ceq(cols, np.where(okg, defeito.sum(0) / n, np.nan))
    out["c"] = _ordem_ceq(cols, np.where(okg, (defeito * d_lin[:, None]).sum(0), np.nan))

    # ---- Estatísticas descritivas (classe escolhida) ----
    sd = A.std(0)
    q1, q3 = np.percentile(A, [25, 75], axis=0)
    with np.errstate(all="ignore"):
        desc = {"AmplitudeSigma": np.where(sd > 0, (A.max(0) - A.min(0)) / sd, np.nan),
                "IQRSigma": np.where(sd > 0, (q3 - q1) / sd, np.nan),
                "Assimetria": np.abs(skew(A, axis=0, bias=True)),
                "Curtose": np.abs(kurtosis(A, axis=0, fisher=True, bias=True))}
    for m, v in desc.items():
        out[m] = _ordem(cols, v, CONST)

    # ---- Literatura, supervisionados (base binária: classe escolhida = 1) ----
    X = X_full.to_numpy(float)
    Xs = MinMaxScaler().fit_transform(X_full)
    # ANOVA manual
    om = X.mean(0)
    between = np.zeros(X.shape[1]); within = np.zeros(X.shape[1])
    for c in np.unique(y):
        Xc = X[y == c]; mc = Xc.mean(0)
        between += len(Xc) * (mc - om) ** 2; within += ((Xc - mc) ** 2).sum(0)
    out["ANOVA"] = _ordem(cols, np.where(within != 0, between / np.where(within == 0, 1, within), 0), CONST)
    out["Pearson"] = _ordem(cols, [abs(np.corrcoef(Xs[:, i], y)[0, 1]) for i in range(Xs.shape[1])], CONST)
    out["Spearman"] = _ordem(cols, [abs(spearmanr(Xs[:, i], y)[0]) for i in range(Xs.shape[1])], CONST)
    out["Kendall"] = _ordem(cols, [abs(kendalltau(Xs[:, i], y)[0]) for i in range(Xs.shape[1])], CONST)
    out["Chi2"] = _ordem(cols, chi2(Xs, y)[0], CONST)
    mi = mutual_info_classif(X_full, y, discrete_features=False, random_state=42)
    out["InformationGain"] = _ordem(cols, mi, CONST)

    def iv(col):
        probs = pd.qcut(col, q=10, duplicates="drop").value_counts(normalize=True)
        return -np.sum(probs * np.log2(probs + 1e-9)), entropy(probs, base=2)
    ivs = [iv(X_full[c]) for c in cols]
    out["GainRatio"] = _ordem(cols, [g / v[0] if v[0] != 0 else 0 for g, v in zip(mi, ivs)], CONST)
    H_y = entropy(np.bincount(y) / len(y), base=2)
    out["SymmetricalUncertainty"] = _ordem(cols, [2 * g / (v[1] + H_y + 1e-9) for g, v in zip(mi, ivs)], CONST)
    out["FScore_ANOVA"] = _ordem(cols, f_classif(X_full, y)[0], CONST)
    g0, g1 = X[y == 0], X[y == 1]
    out["TScore"] = _ordem(cols, [abs(ttest_ind(g0[:, i], g1[:, i], equal_var=False)[0]) for i in range(X.shape[1])], CONST)
    num = len(g0) * (g0.mean(0) - om) ** 2 + len(g1) * (g1.mean(0) - om) ** 2
    den = len(g0) * g0.var(0, ddof=1) + len(g1) * g1.var(0, ddof=1)
    out["FisherScore"] = _ordem(cols, num / (den + 1e-9), CONST)

    def gini(att):
        bins = pd.Series(pd.qcut(att, q=10, duplicates="drop", labels=False)).fillna(-1).to_numpy()
        g = 0
        for b in np.unique(bins):
            sub = y[bins == b]
            pr = np.bincount(sub, minlength=2) / len(sub)
            g += (len(sub) / len(y)) * (1 - np.sum(pr ** 2))
        return g
    out["GiniIndex"] = _ordem(cols, [gini(X_full[c]) for c in cols], CONST, ascending=True)

    def mic(x):
        try:
            xb = pd.qcut(x, q=10, duplicates="drop", labels=False).fillna(-1)
        except ValueError:
            xb = pd.cut(x, bins=10, labels=False).fillna(-1)
        m = mutual_info_score(xb, y)
        nf = np.log2(min(len(np.unique(xb)), len(np.unique(y))) + 1e-9)
        return m / nf if nf > 0 else 0
    out["ApproxMIC"] = _ordem(cols, [mic(X_full[c]) for c in cols], CONST)
    out["KS"] = _ordem(cols, [ks_2samp(g0[:, i], g1[:, i])[0] for i in range(X.shape[1])], CONST)

    def hell(i):
        f = X[:, i]
        h1, _ = np.histogram(f[y == 0], bins=10, range=(f.min(), f.max()), density=True)
        h2, _ = np.histogram(f[y == 1], bins=10, range=(f.min(), f.max()), density=True)
        p, q = h1 / np.sum(h1), h2 / np.sum(h2)
        return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
    out["HellingerDistance"] = _ordem(cols, [hell(i) for i in range(X.shape[1])], CONST)

    # ---- Literatura, não supervisionados (só a classe escolhida) ----
    Xc = X_classe.to_numpy(float)

    def ent(v):
        v = np.clip(v, 0, None); v = v[v > 1e-12]; p = v / v.sum()
        return -np.sum(p * np.log(p)) / np.log(len(p)) if len(p) > 1 else 0.0
    var = [i for i in range(Xc.shape[1]) if Xc[:, i].std() > 0]
    Z = Xc[:, var]; Z = (Z - Z.mean(0)) / Z.std(0); G = Z.T @ Z
    Et = ent(np.linalg.eigvalsh(G))
    ce = np.full(Xc.shape[1], np.nan)
    if lentos:
        for j, i in enumerate(var):
            keep = np.r_[0:j, j + 1:len(var)]
            ce[i] = Et - ent(np.linalg.eigvalsh(G[np.ix_(keep, keep)]))
    out["SVDEntropy"] = _ordem(cols, ce, CONST)

    dist = squareform(pdist(Xc, metric="euclidean"))
    W = np.zeros((len(Xc), len(Xc)))
    for i in range(len(Xc)):
        idx = np.argsort(dist[i])[1:6]
        W[i, idx] = 1; W[idx, i] = 1
    D = np.diag(W.sum(1)); L = D - W
    F = Xc - Xc.mean(0)
    lap = np.einsum("ij,ij->j", F, L @ F) / (np.einsum("ij,ij->j", F, D @ F) + 1e-12)
    out["LaplacianScore"] = _ordem(cols, lap, CONST, ascending=True)
    _, vecs = eigh(L)
    E = vecs[:, 1:6]
    Fs = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-12)
    out["SPEC"] = _ordem(cols, ((E.T @ Fs) ** 2).sum(0), CONST)

    if lentos:
        from skrebate import ReliefF
        r = ReliefF(n_neighbors=10, n_features_to_select=X_full.shape[1])
        r.fit(Xs, y)
        out["ReliefF"] = _ordem(cols, r.feature_importances_, CONST)
    return out


# Conferência: os rankings reimplementados reproduzem os do notebook 2 na amostra completa
_R = rankings_nb2(amostra)
for m in PROP + LIT + DESC:
    ref = POS[m].sort_values().index.tolist()
    rho = spearmanr(POS[m].loc[ref], pd.Series({f: i + 1 for i, f in enumerate(_R[m])}).loc[ref]).statistic
    top20 = len(set(ref[:20]) & set(_R[m][:20]))
    print(f"{m:24s} rho = {rho:.4f} | top-20 em comum: {top20}")

# %%
M_SUB, FRACAO_SUB, N_JOBS = 100, 0.5, 2
ARQ_ORDENS = os.path.join(PASTA, "8_ordens_subamostras.pkl")
cols_am = list(amostra.columns[:-1]); pos_col = {c: i for i, c in enumerate(cols_am)}
if os.path.exists(ARQ_ORDENS):
    ORDENS = pd.read_pickle(ARQ_ORDENS)
    print("Ordens das subamostras lidas de", ARQ_ORDENS)
else:
    rng = np.random.default_rng(SEMENTE)
    idx = []
    for _ in range(M_SUB):
        sel = []
        for c, g in amostra.groupby(col_classe).groups.items():
            g = np.asarray(g); sel.append(rng.choice(g, size=int(round(FRACAO_SUB * len(g))), replace=False))
        idx.append(np.sort(np.concatenate(sel)))              # ordem temporal preservada
    def _um(i):
        import warnings; warnings.simplefilter("ignore")   # os processos paralelos não herdam o filtro
        R_ = rankings_nb2(amostra.iloc[idx[i]])
        return {m: np.array([pos_col[f] for f in R_[m]], dtype=np.int16) for m in PROP + LIT + DESC}
    res = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(_um)(i) for i in range(M_SUB))
    ORDENS = {m: np.vstack([r[m] for r in res]) for m in PROP + LIT + DESC}
    pd.to_pickle(ORDENS, ARQ_ORDENS)


def nogueira(Z, alpha=0.05):
    """Índice de estabilidade de Nogueira et al. (2018), com variância e IC (código dos autores)."""
    Z = np.asarray(Z, float); M, d = Z.shape
    pf = Z.mean(0); kbar = pf.sum(); den = (kbar / d) * (1 - kbar / d)
    phi = 1 - (M / (M - 1)) * np.mean(pf * (1 - pf)) / den
    k = Z.sum(1)
    ph = (1 / den) * ((Z * pf).mean(1) - k * kbar / d ** 2 + (phi / 2) * (2 * k * kbar / d ** 2 - k / d - kbar / d + 1))
    var = 4 / M ** 2 * np.sum((ph - ph.mean()) ** 2)
    z = norm.ppf(1 - alpha / 2)
    return phi, var, phi - z * np.sqrt(var), phi + z * np.sqrt(var)


lin = []
for _, r in KNEE.iterrows():
    o = ORDENS[r.Metodo]; M, d = o.shape; k = int(r.N)
    Z = np.zeros((M, d), dtype=np.int8)
    for i in range(M): Z[i, o[i, :k]] = 1
    phi, var, li, ls = nogueira(Z)
    lin.append({"P": r.Paradigma, "Metodo": r.Metodo, "G": r.G, "N": k, "Phi": phi, "IC_inf": li, "IC_sup": ls,
                "Classificacao": "Excelente" if round(phi, 2) > 0.75 else ("Razoável a boa" if round(phi, 2) >= 0.40 else "Fraca"),
                "Trivial_N_ge_468": k >= 468})
ESTAB = pd.DataFrame(lin)
SAIDAS["11_Estabilidade"] = ESTAB

lin = []
for p in PARADIGMAS:
    e = ESTAB[ESTAB.P == p]
    linha = {"P": p, "p_Prop_x_Lit": mann_whitney(e[e.G == "Proposta"].Phi, e[e.G == "Literatura"].Phi)}
    for g in GRUPOS:
        x = e[e.G == g]
        linha[f"Estaveis_{g}"] = f"{(x.Phi.round(2) >= 0.40).sum()}/{len(x)}"
        linha[f"Pct_{g}"] = 100 * (x.Phi.round(2) >= 0.40).mean()
        linha[f"Phi_medio_{g}"] = x.Phi.mean()
    lin.append(linha)
ESTAB_RES = pd.DataFrame(lin)
SAIDAS["11_Estabilidade_Resumo"] = ESTAB_RES
print(ESTAB_RES.round(3).to_string(index=False))

# %% [markdown]
# ### 12. Gravar todas as tabelas

# %%
with pd.ExcelWriter(ARQ_SAIDA, engine="openpyxl") as w:
    for aba, df in SAIDAS.items():
        df.to_excel(w, sheet_name=aba[:31], index=not isinstance(df.index, pd.RangeIndex))
print(f"✅ {len(SAIDAS)} tabelas gravadas em {ARQ_SAIDA}")
