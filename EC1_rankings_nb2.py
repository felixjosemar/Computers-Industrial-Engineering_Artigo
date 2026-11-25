# Reimplementação dos 29 rankings do notebook 2 (mesmas fórmulas), para recalcular as
# ordenações em subamostras da amostra (estabilidade de Nogueira). Conferida contra o
# Rankings_Consolidado.xlsx na amostra completa (ver verificar_rankings()).
import warnings
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, spearmanr, kendalltau, ttest_ind, ks_2samp, entropy
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
from sklearn.metrics import mutual_info_score

warnings.simplefilter("ignore")
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


def rankings(df, classe=-1, lentos=True, limites="percentis"):
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
    if limites == "percentis":
        lsl, usl = np.percentile(A, [PERCENTIL_LSL, PERCENTIL_USL], axis=0)
    else:                                   # versão anterior: mínimo e máximo de cada sensor
        lsl, usl = A.min(0), A.max(0)
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
