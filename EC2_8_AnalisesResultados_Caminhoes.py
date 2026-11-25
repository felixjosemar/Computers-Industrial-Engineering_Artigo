# # 8 - Análises do estudo de caso 2 (caminhões fora de estrada)
# Reúne as análises usadas na tese para os cinco cenários (turmas A a E):
# seleção dos modelos (Kneedle e estatístico), estatísticas por grupo, placar, atributos
# dispensáveis, tempo, equivalência entre rankings (Friedman/Nemenyi e Spearman) e
# estabilidade de Nogueira em 100 subamostras. Coloque este caderno na pasta que contém
# `amostra.xlsx` e as subpastas `Classe_A` ... `Classe_E`.

import os
PASTA = os.getcwd() + "/"
SAIDA = os.path.join(PASTA, "8_Resultados_Caminhoes") + "/"
os.makedirs(SAIDA, exist_ok=True)

# ## 1. Rankings (cópia literal das funções do caderno 2), usados nas subamostras

# Reprodução dos rankings do notebook 2 do estudo de caso dos caminhões
# (2_CriacaoRanking_PorClasse.ipynb: limites LSL/USL = percentis 1 e 99, atributo constante
# com índice indefinido = NaN, SVD Entropy por coluna e informação mútua sem semente).
# As funções abaixo são cópias literais das células do notebook; só os "gerar_*"
# foram substituídos por uma função que devolve as ordenações em memória.
import os, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, spearmanr, kendalltau, ttest_ind, ks_2samp, entropy
from scipy.linalg import svd, eigh
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
from sklearn.metrics import mutual_info_score
warnings.simplefilter("ignore")

def montar_ranking(features, scores, ascending=False):
    """Padroniza a saída em 3 colunas: Feature, Score, Ranking (1 = melhor atributo)."""
    ranking_df = pd.DataFrame({"Feature": features, "Score": scores})
    ranking_df = ranking_df.sort_values(by="Score", ascending=ascending, na_position="last").reset_index(drop=True)
    ranking_df["Ranking"] = ranking_df.index + 1
    return ranking_df


def salvar_ranking(ranking_df, nome_arquivo, tempo_execucao, out_dir, lista_tempos, mensagem=None):
    """Salva o ranking (já no formato Feature/Score/Ranking) dentro de `out_dir`
    (a subpasta Classe_<rótulo>/ sendo processada) e registra o tempo em memória."""
    caminho = os.path.join(out_dir, nome_arquivo)
    ranking_df.to_excel(caminho, index=False)
    print(mensagem or f"✅ Ranking salvo em '{caminho}'")
    lista_tempos.append((nome_arquivo, tempo_execucao))



PERCENTIL_LSL = 1   # mesmos percentis do notebook 2 (limites LSL/USL)
PERCENTIL_USL = 99


def sigma_curto_prazo_mr(dados):
    """Estima o sigma de CURTO PRAZO (variação inerente) via amplitude móvel (moving
    range) entre observações consecutivas -- técnica padrão de um gráfico I-MR quando
    não há subgrupos racionais definidos (Montgomery, 2020). PREMISSA: assume que a
    ordem das linhas reflete a ordem real de coleta/processo."""
    valores = dados.values
    if len(valores) < 2:
        return np.nan
    mr = np.abs(np.diff(valores))
    return np.mean(mr) / 1.128  # d2 para subgrupo de tamanho 2


def calcular_ceq(df_classe):
    """Cp/Cpk usam sigma de CURTO prazo (moving range) -- 'capacidade potencial'.
    Pp/Ppk usam sigma de LONGO prazo (desvio-padrão amostral geral) -- 'desempenho
    real'. Diferente de antes, as quatro fórmulas NÃO são mais idênticas entre si:
    a diferença Cp vs. Pp (e Cpk vs. Ppk) é justamente o que permite avaliar
    estabilidade -- Cp ≈ Pp indica processo estável; Cp >> Pp indica variação extra
    de longo prazo (deriva, instabilidade) não capturada pela variação ponto-a-ponto.
    """
    resultados = []
    for col in df_classe.columns:
        dados = df_classe[col].dropna()
        LSL, USL = np.percentile(dados, [PERCENTIL_LSL, PERCENTIL_USL]) if len(dados) else (np.nan, np.nan)
        media = dados.mean()

        std_longo = np.std(dados, ddof=0)          # usado por Pp/Ppk
        std_curto = sigma_curto_prazo_mr(dados)     # usado por Cp/Cpk

        invalido = (
            len(dados) < 2 or std_longo == 0
            or std_curto == 0 or pd.isna(std_curto)
        )
        if invalido:
            # Índice indefinido (sensor constante): NaN -> fim do ranking. Antes era 0,
            # o que o deixava ACIMA de sensores com Cpk/Ppk negativo (média fora de [P1, P99]).
            valores = {m: np.nan for m in ["Cp", "Cpk", "Pp", "Ppk"]}
        else:
            Cp = (USL - LSL) / (6 * std_curto)
            Cpu = (USL - media) / (3 * std_curto)
            Cpl = (media - LSL) / (3 * std_curto)
            Cpk = min(Cpu, Cpl)
            Pp = (USL - LSL) / (6 * std_longo)
            Ppu = (USL - media) / (3 * std_longo)
            Ppl = (media - LSL) / (3 * std_longo)
            Ppk = min(Ppu, Ppl)
            valores = {"Cp": Cp, "Cpk": Cpk, "Pp": Pp, "Ppk": Ppk}

        resultados.append({"Feature": col, **valores})
    return pd.DataFrame(resultados)


def calcular_graficos_controle(df_classe):
    n = len(df_classe)
    colunas = df_classe.columns

    # Matriz binária de "defeito": 1 onde o valor sai de ±1σ da média da
    # PRÓPRIA coluna -- calculada de uma vez para TODOS os atributos, pois
    # c/u (abaixo) precisam saber quantos atributos estão fora de controle
    # simultaneamente em cada amostra, não só o atributo atual.
    matriz_defeito = pd.DataFrame(0, index=df_classe.index, columns=colunas, dtype=int)
    for col in colunas:
        dados_col = df_classe[col]
        dados_validos = dados_col.dropna()
        media = dados_validos.mean()
        std = np.std(dados_validos, ddof=0)  # consistente com o ddof=0 de calcular_ceq(), acima
        if len(dados_validos) >= 2 and std > 0:
            fora_de_controle = (dados_col < media - std) | (dados_col > media + std)
            matriz_defeito[col] = fora_de_controle.fillna(False).astype(int)

    # D_i: nº TOTAL de atributos fora de controle, simultaneamente, na amostra i
    # (conceito clássico de "defeitos por unidade" -- uma unidade pode ter
    # vários defeitos ao mesmo tempo, em características/atributos diferentes).
    defeitos_simultaneos_por_amostra = matriz_defeito.sum(axis=1)

    resultados = []
    for col in colunas:
        dados = df_classe[col].dropna()
        std = np.std(dados, ddof=0)

        if len(dados) < 2 or std == 0:
            valores = {m: np.nan for m in ["p", "np", "c", "u"]}   # indefinido -> fim do ranking
        else:
            amostras_defeituosas_nesta_coluna = matriz_defeito[col].astype(bool)
            n_defeituosas = int(amostras_defeituosas_nesta_coluna.sum())

            p = n_defeituosas / n
            np_val = n_defeituosas
            c = int(defeitos_simultaneos_por_amostra[amostras_defeituosas_nesta_coluna].sum())
            u = c / n

            valores = {"p": p, "np": np_val, "c": c, "u": u}

        resultados.append({"Feature": col, **valores})
    return pd.DataFrame(resultados)


def gerar_bloco1_ceq_controle(base_classe1_orig, base_classe1_norm, out_dir, lista_tempos):
    for normalizado, base_classe1 in zip([False, True], [base_classe1_orig, base_classe1_norm]):
        df_ceq = calcular_ceq(base_classe1)
        for metrica in ["Cp", "Cpk", "Pp", "Ppk"]:
            t0 = time.time()
            df_rank = montar_ranking(df_ceq["Feature"], df_ceq[metrica], ascending=False)
            nome = f"Ranking_{metrica}_{'Normalizado' if normalizado else 'Original'}.xlsx"
            salvar_ranking(df_rank, nome, time.time() - t0, out_dir, lista_tempos)
        df_gc = calcular_graficos_controle(base_classe1)
        for metrica in ["p", "np", "c", "u"]:
            t0 = time.time()
            df_rank = montar_ranking(df_gc["Feature"], df_gc[metrica], ascending=False)
            nome = f"Ranking_{metrica}_{'Normalizado' if normalizado else 'Original'}.xlsx"
            salvar_ranking(df_rank, nome, time.time() - t0, out_dir, lista_tempos)



def amplitude_sigma(coluna):
    x = coluna.dropna().values
    if len(x) < 2:
        return np.nan
    std = np.std(x, ddof=0)
    return (np.max(x) - np.min(x)) / std if std > 0 else np.nan


def iqr_sigma(coluna):
    x = coluna.dropna().values
    if len(x) < 2:
        return np.nan
    std = np.std(x, ddof=0)
    q1, q3 = np.percentile(x, [25, 75])
    return (q3 - q1) / std if std > 0 else np.nan


def assimetria_abs(coluna):
    x = coluna.dropna().values
    if len(x) < 3:
        return np.nan
    return abs(skew(x, bias=True))


def curtose_abs(coluna):
    x = coluna.dropna().values
    if len(x) < 3:
        return np.nan
    return abs(kurtosis(x, fisher=True, bias=True))


metricas_comparacao = {
    "AmplitudeSigma": amplitude_sigma, "IQRSigma": iqr_sigma,
    "Assimetria": assimetria_abs, "Curtose": curtose_abs,
}


def gerar_descritivas(df_numerico_comp, df_numerico_completo, out_dir, lista_tempos):
    versoes = [("", df_numerico_comp), ("_BaseCompleta", df_numerico_completo)]
    for nome_metrica, func in metricas_comparacao.items():
        for sufixo, base_calc in versoes:
            t0 = time.time()
            scores = [func(base_calc[col]) for col in base_calc.columns]
            ranking_df = montar_ranking(base_calc.columns, scores, ascending=False)
            nome_arq = f"Ranking_{nome_metrica}{sufixo}.xlsx"
            salvar_ranking(ranking_df, nome_arq, time.time() - t0, out_dir, lista_tempos)



def anova_f_score(X, y):
    scores = []
    classes = np.unique(y)
    overall_means = np.mean(X, axis=0)
    for i in range(X.shape[1]):
        between = within = 0.0
        for c in classes:
            X_c = X[y == c, i]
            mean_c = np.mean(X_c)
            between += len(X_c) * (mean_c - overall_means[i]) ** 2
            within += np.sum((X_c - mean_c) ** 2)
        scores.append(between / within if within != 0 else 0)
    return scores


def gerar_anova(X_full, y_bin, out_dir, lista_tempos):
    t0 = time.time()
    f_scores = anova_f_score(X_full.values, y_bin.values)
    ranking_df = montar_ranking(X_full.columns, f_scores, ascending=False)
    salvar_ranking(ranking_df, "Ranking_ANOVA.xlsx", time.time() - t0, out_dir, lista_tempos)



def svd_entropy(vetor):
    vetor = np.array(vetor)
    if vetor.ndim == 1:
        vetor = vetor.reshape(1, -1)
    U, S, Vh = svd(vetor, full_matrices=False)
    P = S / np.sum(S)
    return entropy(P)


def gerar_svd_entropy(df_classe, out_dir, lista_tempos):
    t0 = time.time()
    df_numerico = df_classe.select_dtypes(include=[np.number])
    entropias = [svd_entropy(df_numerico[col].values) for col in df_numerico.columns]
    ranking_df = montar_ranking(df_numerico.columns, entropias, ascending=False)
    salvar_ranking(ranking_df, "Ranking_SVDEntropy.xlsx", time.time() - t0, out_dir, lista_tempos)



def _grafo_vizinhanca(X, k=5):
    n_samples = X.shape[0]
    dist_matrix = squareform(pdist(X, metric="euclidean"))
    W = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        idx = np.argsort(dist_matrix[i])[1:k + 1]
        W[i, idx] = 1
        W[idx, i] = 1
    D = np.diag(W.sum(axis=1))
    L = D - W
    return D, L


def laplacian_score(X, k=5):
    D, L = _grafo_vizinhanca(X, k)
    scores = []
    for i in range(X.shape[1]):
        f = X[:, i]
        f = f - np.mean(f)
        numerator = f.T @ L @ f
        denominator = f.T @ D @ f + 1e-12
        scores.append(numerator / denominator)
    return scores


def gerar_laplacian(df_classe, out_dir, lista_tempos):
    t0 = time.time()
    Xc = df_classe.select_dtypes(include=[np.number]).values
    atributos = df_classe.select_dtypes(include=[np.number]).columns
    scores = laplacian_score(Xc)
    ranking_df = montar_ranking(atributos, scores, ascending=True)
    salvar_ranking(ranking_df, "Ranking_LaplacianScore.xlsx", time.time() - t0, out_dir, lista_tempos)



def spec_score(X, k=5, n_eig=5):
    D, L = _grafo_vizinhanca(X, k)
    eigvals, eigvecs = eigh(L)
    eigvecs_selected = eigvecs[:, 1:n_eig + 1]
    scores = []
    for i in range(X.shape[1]):
        f = X[:, i]
        f = (f - np.mean(f)) / (np.std(f) + 1e-12)
        proj = eigvecs_selected.T @ f
        scores.append(np.sum(proj ** 2))
    return scores


def gerar_spec(df_classe, out_dir, lista_tempos):
    t0 = time.time()
    Xc = df_classe.select_dtypes(include=[np.number]).values
    atributos = df_classe.select_dtypes(include=[np.number]).columns
    scores = spec_score(Xc)
    ranking_df = montar_ranking(atributos, scores, ascending=False)
    salvar_ranking(ranking_df, "Ranking_SPEC.xlsx", time.time() - t0, out_dir, lista_tempos)



def gerar_relieff(X_full, X_scaled_full, y_bin, out_dir, lista_tempos):
    t0 = time.time()
    relieff = ReliefF(n_neighbors=10, n_features_to_select=X_full.shape[1])
    relieff.fit(X_scaled_full, y_bin.values)
    ranking_df = montar_ranking(X_full.columns, relieff.feature_importances_, ascending=False)
    salvar_ranking(ranking_df, "Ranking_ReliefF.xlsx", time.time() - t0, out_dir, lista_tempos)



def gerar_pearson(X_full, X_scaled_full, y_bin, out_dir, lista_tempos):
    t0 = time.time()
    correlacoes = [abs(np.corrcoef(X_scaled_full[:, i], y_bin.values)[0, 1]) for i in range(X_full.shape[1])]
    ranking_df = montar_ranking(X_full.columns, correlacoes, ascending=False)
    salvar_ranking(ranking_df, "Ranking_Pearson.xlsx", time.time() - t0, out_dir, lista_tempos)



def gerar_spearman(X_full, X_scaled_full, y_bin, out_dir, lista_tempos):
    t0 = time.time()
    correlacoes = []
    for i in range(X_full.shape[1]):
        rho, _ = spearmanr(X_scaled_full[:, i], y_bin.values)
        correlacoes.append(abs(rho))
    ranking_df = montar_ranking(X_full.columns, correlacoes, ascending=False)
    salvar_ranking(ranking_df, "Ranking_Spearman.xlsx", time.time() - t0, out_dir, lista_tempos)



def gerar_kendall(X_full, X_scaled_full, y_bin, out_dir, lista_tempos):
    t0 = time.time()
    correlacoes = []
    for i in range(X_full.shape[1]):
        tau, _ = kendalltau(X_scaled_full[:, i], y_bin.values)
        correlacoes.append(abs(tau))
    ranking_df = montar_ranking(X_full.columns, correlacoes, ascending=False)
    salvar_ranking(ranking_df, "Ranking_Kendall.xlsx", time.time() - t0, out_dir, lista_tempos)



def gerar_chi2(X_full, X_scaled_full, y_bin, out_dir, lista_tempos):
    t0 = time.time()
    chi2_scores, p_values = chi2(X_scaled_full, y_bin.values)
    ranking_df = montar_ranking(X_full.columns, chi2_scores, ascending=False)
    salvar_ranking(ranking_df, "Ranking_Chi2.xlsx", time.time() - t0, out_dir, lista_tempos)



def gerar_infogain(X_full, y_bin, out_dir, lista_tempos):
    t0 = time.time()
    info_gain_scores = mutual_info_classif(X_full, y_bin.values, discrete_features=False)
    ranking_df = montar_ranking(X_full.columns, info_gain_scores, ascending=False)
    salvar_ranking(ranking_df, "Ranking_InformationGain.xlsx", time.time() - t0, out_dir, lista_tempos)



def _intrinsic_value(column):
    bins = pd.qcut(column, q=10, duplicates="drop")
    probs = bins.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs + 1e-9))


def gerar_gainratio(X_full, y_bin, out_dir, lista_tempos):
    t0 = time.time()
    info_gain_scores = mutual_info_classif(X_full, y_bin.values, discrete_features=False)
    intrinsic_values = [_intrinsic_value(X_full[col]) for col in X_full.columns]
    gain_ratios = [ig / iv if iv != 0 else 0 for ig, iv in zip(info_gain_scores, intrinsic_values)]
    ranking_df = montar_ranking(X_full.columns, gain_ratios, ascending=False)
    salvar_ranking(ranking_df, "Ranking_GainRatio.xlsx", time.time() - t0, out_dir, lista_tempos)



def gerar_fscore_anova(X_full, y_bin, out_dir, lista_tempos):
    t0 = time.time()
    f_scores, p_values = f_classif(X_full, y_bin.values)
    ranking_df = montar_ranking(X_full.columns, f_scores, ascending=False)
    salvar_ranking(ranking_df, "Ranking_FScore_ANOVA.xlsx", time.time() - t0, out_dir, lista_tempos)



def gerar_tscore(X_full, y_bin, out_dir, lista_tempos):
    t0 = time.time()
    classes = np.unique(y_bin.values)  # sempre [0, 1] com a base binária (classe vs. resto)
    t_scores = []
    for col in X_full.columns:
        g1 = X_full[y_bin.values == classes[0]][col]
        g2 = X_full[y_bin.values == classes[1]][col]
        t_stat, p_val = ttest_ind(g1, g2, equal_var=False)
        t_scores.append(abs(t_stat))
    ranking_df = montar_ranking(X_full.columns, t_scores, ascending=False)
    salvar_ranking(ranking_df, "Ranking_TScore.xlsx", time.time() - t0, out_dir, lista_tempos)



def gerar_fisherscore(X_full, y_bin, out_dir, lista_tempos):
    t0 = time.time()
    classes = np.unique(y_bin.values)
    fisher_scores = []
    for col in X_full.columns:
        overall_mean = X_full[col].mean()
        numerator = denominator = 0
        for c in classes:
            class_data = X_full[y_bin.values == c][col]
            class_mean = class_data.mean()
            class_var = class_data.var(ddof=1)
            numerator += len(class_data) * (class_mean - overall_mean) ** 2
            denominator += len(class_data) * class_var
        fisher_scores.append(numerator / (denominator + 1e-9))
    ranking_df = montar_ranking(X_full.columns, fisher_scores, ascending=False)
    salvar_ranking(ranking_df, "Ranking_FisherScore.xlsx", time.time() - t0, out_dir, lista_tempos)



def gerar_giniindex(X_full, y_bin, out_dir, lista_tempos):
    t0 = time.time()
    classes = np.unique(y_bin.values)

    def gini_index(attribute, target):
        bins = pd.qcut(attribute, q=10, duplicates="drop")
        gini = 0
        for b in bins.unique():
            subset = target[bins == b]
            if len(subset) == 0:
                continue
            probs = np.bincount(subset, minlength=len(classes)) / len(subset)
            gini_bin = 1 - np.sum(probs ** 2)
            gini += (len(subset) / len(target)) * gini_bin
        return gini

    gini_scores = [gini_index(X_full[col], y_bin.values) for col in X_full.columns]
    ranking_df = montar_ranking(X_full.columns, gini_scores, ascending=True)
    salvar_ranking(ranking_df, "Ranking_GiniIndex.xlsx", time.time() - t0, out_dir, lista_tempos)



def _entropy_discrete(values):
    probs = np.bincount(values) / len(values)
    return entropy(probs, base=2)


def _entropy_continuous(column):
    bins = pd.qcut(column, q=10, duplicates="drop")
    probs = bins.value_counts(normalize=True)
    return entropy(probs, base=2)


def gerar_symmetricaluncertainty(X_full, y_bin, out_dir, lista_tempos):
    t0 = time.time()
    info_gain = mutual_info_classif(X_full, y_bin.values, discrete_features=False)
    H_y = _entropy_discrete(y_bin.values)
    H_x = [_entropy_continuous(X_full[col]) for col in X_full.columns]
    su_scores = [2 * ig / (hx + H_y + 1e-9) for ig, hx in zip(info_gain, H_x)]
    ranking_df = montar_ranking(X_full.columns, su_scores, ascending=False)
    salvar_ranking(ranking_df, "Ranking_SymmetricalUncertainty.xlsx", time.time() - t0, out_dir, lista_tempos)



def _approximate_mic(x, y, bins=10):
    try:
        x_binned = pd.qcut(x, q=bins, duplicates="drop", labels=False).fillna(-1)
    except ValueError:
        x_binned = pd.cut(x, bins=bins, labels=False).fillna(-1)
    if len(np.unique(y)) > bins:
        try:
            y_binned = pd.qcut(y, q=bins, duplicates="drop", labels=False).fillna(-1)
        except ValueError:
            y_binned = pd.cut(y, bins=bins, labels=False).fillna(-1)
    else:
        y_binned = y
    mi = mutual_info_score(x_binned, y_binned)
    norm_factor = np.log2(min(len(np.unique(x_binned)), len(np.unique(y_binned))) + 1e-9)
    return mi / norm_factor if norm_factor > 0 else 0


def gerar_approxmic(X_full, y_bin, out_dir, lista_tempos):
    t0 = time.time()
    mic_scores = [_approximate_mic(X_full[col], y_bin.values) for col in X_full.columns]
    ranking_df = montar_ranking(X_full.columns, mic_scores, ascending=False)
    salvar_ranking(ranking_df, "Ranking_ApproxMIC.xlsx", time.time() - t0, out_dir, lista_tempos)



def gerar_ks(X_full, y_bin, out_dir, lista_tempos):
    t0 = time.time()
    classes = np.unique(y_bin.values)
    ks_scores = []
    for col in X_full.columns:
        g1 = X_full[y_bin.values == classes[0]][col]
        g2 = X_full[y_bin.values == classes[1]][col]
        ks_stat, p_val = ks_2samp(g1, g2)
        ks_scores.append(ks_stat)
    ranking_df = montar_ranking(X_full.columns, ks_scores, ascending=False)
    salvar_ranking(ranking_df, "Ranking_KS.xlsx", time.time() - t0, out_dir, lista_tempos)



def _hellinger(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)


def gerar_hellinger(X_full, y_bin, out_dir, lista_tempos):
    t0 = time.time()
    classes = np.unique(y_bin.values)

    def compute_hellinger(feature, yv, bins=10):
        g1 = feature[yv == classes[0]]
        g2 = feature[yv == classes[1]]
        hist1, _ = np.histogram(g1, bins=bins, range=(feature.min(), feature.max()), density=True)
        hist2, _ = np.histogram(g2, bins=bins, range=(feature.min(), feature.max()), density=True)
        p = hist1 / np.sum(hist1)
        q = hist2 / np.sum(hist2)
        return _hellinger(p, q)

    hellinger_scores = [compute_hellinger(X_full[col], y_bin.values) for col in X_full.columns]
    ranking_df = montar_ranking(X_full.columns, hellinger_scores, ascending=False)
    salvar_ranking(ranking_df, "Ranking_HellingerDistance.xlsx", time.time() - t0, out_dir, lista_tempos)



def _ordem(rk):
    return [str(f) for f in rk["Feature"]]


def rankings_cam(df, classe, semente_mi=None):
    """df: amostra (ou subamostra), classe na última coluna. Devolve {método: lista de atributos}."""
    df = df.reset_index(drop=True).copy()
    df.columns = df.columns.astype(str).str.strip()
    coluna_alvo = df.columns[-1]
    X = df.iloc[:, :-1]; y = df.iloc[:, -1]
    X_quant = X.select_dtypes(include=[np.number]).copy()
    X_quant = X_quant.fillna(X_quant.median())
    X_full = df.drop(columns=[coluna_alvo]).select_dtypes(include=[np.number]).copy()
    X_full = X_full.fillna(X_full.median())
    X_scaled_full = MinMaxScaler().fit_transform(X_full)
    df_classe = df[df.iloc[:, -1].astype(str) == str(classe)].reset_index(drop=True)
    num = df_classe.select_dtypes(include=[np.number]).columns
    df_classe[num] = df_classe[num].fillna(df_classe[num].median().fillna(df[num].median()))
    y_bin = (df.iloc[:, -1].astype(str) == str(classe)).astype(int).reset_index(drop=True)
    out = {}
    base = X_quant[y.astype(str) == str(classe)].reset_index(drop=True)
    ceq = calcular_ceq(base)
    for m in ["Cp", "Cpk", "Pp", "Ppk"]:
        out[m] = _ordem(montar_ranking(ceq["Feature"], ceq[m], ascending=False))
    gc = calcular_graficos_controle(base)
    for m in ["p", "c"]:
        out[m] = _ordem(montar_ranking(gc["Feature"], gc[m], ascending=False))
    comp = df_classe.drop(columns=[coluna_alvo], errors="ignore").select_dtypes(include=[np.number])
    for nome, func in metricas_comparacao.items():
        out[nome] = _ordem(montar_ranking(comp.columns, [func(comp[c]) for c in comp.columns], ascending=False))
    Xv, yv = X_full.values, y_bin.values
    out["ANOVA"] = _ordem(montar_ranking(X_full.columns, anova_f_score(Xv, yv), ascending=False))
    dn = df_classe.select_dtypes(include=[np.number])
    out["SVDEntropy"] = _ordem(montar_ranking(dn.columns, [svd_entropy(dn[c].values) for c in dn.columns], ascending=False))
    Xc = dn.values
    out["LaplacianScore"] = _ordem(montar_ranking(dn.columns, laplacian_score(Xc), ascending=True))
    out["SPEC"] = _ordem(montar_ranking(dn.columns, spec_score(Xc), ascending=False))
    from skrebate import ReliefF
    r = ReliefF(n_neighbors=10, n_features_to_select=X_full.shape[1]); r.fit(X_scaled_full, yv)
    out["ReliefF"] = _ordem(montar_ranking(X_full.columns, r.feature_importances_, ascending=False))
    out["Pearson"] = _ordem(montar_ranking(X_full.columns, [abs(np.corrcoef(X_scaled_full[:, i], yv)[0, 1]) for i in range(X_full.shape[1])], ascending=False))
    out["Spearman"] = _ordem(montar_ranking(X_full.columns, [abs(spearmanr(X_scaled_full[:, i], yv)[0]) for i in range(X_full.shape[1])], ascending=False))
    out["Kendall"] = _ordem(montar_ranking(X_full.columns, [abs(kendalltau(X_scaled_full[:, i], yv)[0]) for i in range(X_full.shape[1])], ascending=False))
    out["Chi2"] = _ordem(montar_ranking(X_full.columns, chi2(X_scaled_full, yv)[0], ascending=False))
    mi = lambda: mutual_info_classif(X_full, yv, discrete_features=False, random_state=semente_mi)
    out["InformationGain"] = _ordem(montar_ranking(X_full.columns, mi(), ascending=False))
    iv = [_intrinsic_value(X_full[c]) for c in X_full.columns]
    out["GainRatio"] = _ordem(montar_ranking(X_full.columns, [g / v if v != 0 else 0 for g, v in zip(mi(), iv)], ascending=False))
    out["FScore_ANOVA"] = _ordem(montar_ranking(X_full.columns, f_classif(X_full, yv)[0], ascending=False))
    cl = np.unique(yv)
    out["TScore"] = _ordem(montar_ranking(X_full.columns, [abs(ttest_ind(X_full[yv == cl[0]][c], X_full[yv == cl[1]][c], equal_var=False)[0]) for c in X_full.columns], ascending=False))
    fs = []
    for c in X_full.columns:
        om = X_full[c].mean(); num_ = den_ = 0
        for k in cl:
            cd = X_full[yv == k][c]; num_ += len(cd) * (cd.mean() - om) ** 2; den_ += len(cd) * cd.var(ddof=1)
        fs.append(num_ / (den_ + 1e-9))
    out["FisherScore"] = _ordem(montar_ranking(X_full.columns, fs, ascending=False))
    def gini_index(attribute, target):
        bins = pd.qcut(attribute, q=10, duplicates="drop")
        g = 0
        for b in bins.unique():
            sub = target[bins == b]
            if len(sub) == 0: continue
            pr = np.bincount(sub, minlength=len(cl)) / len(sub)
            g += (len(sub) / len(target)) * (1 - np.sum(pr ** 2))
        return g
    out["GiniIndex"] = _ordem(montar_ranking(X_full.columns, [gini_index(X_full[c], yv) for c in X_full.columns], ascending=True))
    H_y = _entropy_discrete(yv); H_x = [_entropy_continuous(X_full[c]) for c in X_full.columns]
    out["SymmetricalUncertainty"] = _ordem(montar_ranking(X_full.columns, [2 * g / (h + H_y + 1e-9) for g, h in zip(mi(), H_x)], ascending=False))
    out["ApproxMIC"] = _ordem(montar_ranking(X_full.columns, [_approximate_mic(X_full[c], yv) for c in X_full.columns], ascending=False))
    out["KS"] = _ordem(montar_ranking(X_full.columns, [ks_2samp(X_full[yv == cl[0]][c], X_full[yv == cl[1]][c])[0] for c in X_full.columns], ascending=False))
    def hell(feature):
        g1, g2 = feature[yv == cl[0]], feature[yv == cl[1]]
        h1, _ = np.histogram(g1, bins=10, range=(feature.min(), feature.max()), density=True)
        h2, _ = np.histogram(g2, bins=10, range=(feature.min(), feature.max()), density=True)
        return _hellinger(h1 / np.sum(h1), h2 / np.sum(h2))
    out["HellingerDistance"] = _ordem(montar_ranking(X_full.columns, [hell(X_full[c]) for c in X_full.columns], ascending=False))
    return out

# ## 2. Índice de estabilidade de Nogueira

# Índice de estabilidade de Nogueira, Sechidis e Brown (2018, JMLR), com variância
# assintótica e IC 95% (mesmas fórmulas do código publicado pelos autores).
import numpy as np, pandas as pd
from scipy.stats import norm

def nogueira(Z, alpha=0.05):
    Z = np.asarray(Z, float); M, d = Z.shape
    pf = Z.mean(0); kbar = pf.sum(); den = (kbar / d) * (1 - kbar / d)
    phi = 1 - (M / (M - 1)) * np.mean(pf * (1 - pf)) / den
    k = Z.sum(1)
    ph = (1 / den) * ((Z * pf).mean(1) - k * kbar / d ** 2 + (phi / 2) * (2 * k * kbar / d ** 2 - k / d - kbar / d + 1))
    var = 4 / M ** 2 * np.sum((ph - ph.mean()) ** 2)
    z = norm.ppf(1 - alpha / 2)
    return phi, var, phi - z * np.sqrt(var), phi + z * np.sqrt(var)

# ## 3. Seleção dos modelos, grupos, placar, dispensáveis, tempo e equivalência

# Estudo de caso 2 (caminhões fora de estrada): seleção dos modelos, estatísticas por grupo,
# placar, fronteira, dispensáveis, tempo e equivalência entre rankings, por cenário (classe).
import sys, re, itertools, warnings
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, linregress, friedmanchisquare
import scikit_posthocs as sp
warnings.simplefilter("ignore")
from decimal import Decimal, ROUND_HALF_EVEN
def arred(x, d):  # ABNT NBR 5891
    if x is None or (isinstance(x, float) and np.isnan(x)): return None
    return Decimal(repr(round(float(x), 12))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_EVEN)
B = PASTA
O = SAIDA
CLS = ["A", "B", "C", "D", "E"]
PAR = ["DT", "RF", "GB", "SVM", "MLP", "GNB", "LogReg_Lasso"]
PROP = ["Cp", "Cpk", "Pp", "Ppk", "p", "c"]
LIT = ["ANOVA", "ApproxMIC", "Chi2", "FScore_ANOVA", "FisherScore", "GainRatio", "GiniIndex",
       "HellingerDistance", "InformationGain", "KS", "Kendall", "LaplacianScore", "Pearson", "ReliefF",
       "SPEC", "SVDEntropy", "Spearman", "SymmetricalUncertainty", "TScore"]
DESC = ["AmplitudeSigma", "IQRSigma", "Assimetria", "Curtose"]
GRP = {**{m: "Proposta" for m in PROP}, **{m: "Literatura" for m in LIT}, **{m: "Descritivas" for m in DESC}}
nome = lambda r: str(r).replace("Ranking_", "").replace("_Original", "")
fold = lambda s: float(np.nanmean([float(v) for v in re.findall(r"[-+0-9.eE]+|nan", str(s))])) if str(s).startswith("[") else float(s)


def kneedle(df):  # notebook 5
    d = df.sort_values("Nº de Atributos")
    x, y = d["Nº de Atributos"].values.astype(float), d["Acurácia Balanceada Média"].values.astype(float)
    if x.max() == x.min() or y.max() == y.min(): return d.index[0]
    xn, yn = (x - x.min()) / (x.max() - x.min()), (y - y.min()) / (y.max() - y.min())
    c = np.vstack((xn, yn)).T; vl = c[-1] - c[0]; vp = c - c[0]; nr = np.linalg.norm(vl)
    if nr == 0: return d.index[0]
    return d.iloc[np.argmax(np.abs((vl[0] * vp[:, 1] - vl[1] * vp[:, 0]) / nr))].name


def estatistico(df):  # notebook 5 (regra de 1 desvio-padrão)
    d = df.sort_values("Nº de Atributos")
    b = d["Acurácia Balanceada Média"].idxmax()
    lim = d.loc[b, "Acurácia Balanceada Média"] - d.loc[b, "Acurácia Balanceada Desvio"]
    c = d[d["Acurácia Balanceada Média"] >= lim]
    return c.iloc[0].name if not c.empty else b


def mw(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.all(a == a[0]) and np.all(b == a[0]): return np.nan
    return mannwhitneyu(a, b, alternative="two-sided").pvalue


sel, tempo, curv = [], [], []
for c in CLS:
    mapa = pd.read_excel(B + f"Classe_{c}/Mapa_Abas_Consolidado.xlsx")
    mapa = {(r.Paradigma, r["Nome da Aba"]): r.Ranking for _, r in mapa.iterrows()}
    for p in PAR:
        for aba, df in pd.read_excel(B + f"Classe_{c}/Modelos{p}_CV_Consolidado.xlsx", sheet_name=None).items():
            m = nome(mapa[(p, aba)])
            if m not in GRP: continue
            d = df[(df["Nº de Atributos"] > 0) & (df["Tempo de Execução (s)"] > 0)]
            lr = linregress(np.log(d["Nº de Atributos"]), np.log(d["Tempo de Execução (s)"]))
            Tm = d["Tempo de Execução (s)"].mean()
            tempo.append(dict(C=c, P=p, Metodo=m, G=GRP[m], alpha=lr.slope, r=lr.rvalue, Tm=Tm,
                              Ef=(d["Tempo de Execução (s)"] / d["Nº de Atributos"]).mean()))
            cte = Tm / np.mean(np.arange(1, 107) ** lr.slope)
            cv = df[["Nº de Atributos", "Acurácia Balanceada Média"]].copy()
            cv["BA"] = df["Acurácia Balanceada_Por_Fold"].map(fold); cv["C"], cv["P"], cv["Metodo"], cv["G"] = c, p, m, GRP[m]
            curv.append(cv)
            for t, i in [("Kneedle", kneedle(df)), ("Estatístico", estatistico(df))]:
                r = df.loc[i]; N = int(r["Nº de Atributos"])
                sel.append(dict(C=c, P=p, T=t, Metodo=m, G=GRP[m], N=N, BA=fold(r["Acurácia Balanceada_Por_Fold"]),
                                AUC=fold(r["AUC_Por_Fold"]), F1=fold(r["F1-Score (Macro Avg)_Por_Fold"]),
                                Test=cte * N ** lr.slope, BA3=r["Acurácia Balanceada Média"]))
SEL, TEMPO, CURV = pd.DataFrame(sel), pd.DataFrame(tempo), pd.concat(curv, ignore_index=True)
assert SEL.groupby(["C", "P", "T"]).size().eq(29).all()
SEL.to_pickle(O + "sel.pkl"); TEMPO.to_pickle(O + "tempo_rk.pkl"); CURV.to_pickle(O + "curvas.pkl")

# --- estatísticas por grupo ---
lin = []
for (c, p, t), d in SEL.groupby(["C", "P", "T"]):
    r = dict(C=c, P=p, T=t)
    for k in ["N", "BA", "AUC", "F1", "Test"]:
        A, L, D = (d[d.G == g][k] for g in ["Proposta", "Literatura", "Descritivas"])
        r.update({k + "_ceq": A.mean(), k + "_lit": L.mean(), k + "_ctl": D.mean(), k + "_sd_ceq": A.std(ddof=1),
                  k + "_sd_lit": L.std(ddof=1), k + "_sd_ctl": D.std(ddof=1), k + "_p": mw(A, L), k + "_pctl": mw(A, D)})
    lin.append(r)
RC = pd.DataFrame(lin); RC.to_pickle(O + "recalc.pkl")
lin = []
for (c, p), d in TEMPO.groupby(["C", "P"]):
    for k in ["Tm", "Ef", "alpha", "r"]:
        A, L, D = (d[d.G == g][k] for g in ["Proposta", "Literatura", "Descritivas"])
        lin.append(dict(C=c, P=p, M=k, ceq=A.mean(), lit=L.mean(), ctl=D.mean(), sd_ceq=A.std(ddof=1), sd_lit=L.std(ddof=1),
                        sd_ctl=D.std(ddof=1), p=mw(A, L), pctl=mw(A, D)))
pd.DataFrame(lin).to_pickle(O + "tempo.pkl")

# --- placar ---
DEC = {"N": 1, "BA": 2, "AUC": 2, "F1": 2}
lin = []
for _, r in RC.iterrows():
    for k in DEC:
        for outro, suf, pk in [("Literatura", "_lit", "_p"), ("Descritivas", "_ctl", "_pctl")]:
            ra, rb = arred(r[k + "_ceq"], DEC[k]), arred(r[k + suf], DEC[k])
            w = "empate" if ra == rb else ("Proposta" if ((ra < rb) if k == "N" else (ra > rb)) else outro)
            lin.append(dict(C=r.C, P=r.P, T=r["T"], M=k, Contra=outro, W=w, p=r[k + pk], sig=bool(w != "empate" and r[k + pk] <= 0.05)))
pd.DataFrame(lin).to_pickle(O + "placar.pkl")

# --- dispensáveis (critério estatístico) ---
am = pd.read_excel(B + "amostra.xlsx", sheet_name="Amostra"); am.columns = am.columns.astype(str)
X = am.iloc[:, :-1]; Rk = X.rank().corr().abs()
best = CURV.groupby(["C", "P", "Metodo"]).BA.max()
rk_files = {}
lin = []
for c in CLS:
    n_inf = int((am[am.iloc[:, -1].astype(str) == c].iloc[:, :-1].std(ddof=0) > 0).sum())
    for m in PROP + LIT + DESC:
        f = B + f"Classe_{c}/Ranking_{m}_Original.xlsx" if m in PROP else B + f"Classe_{c}/Ranking_{m}.xlsx"
        rk_files[(c, m)] = [str(v) for v in pd.read_excel(f).sort_values("Ranking").Feature]
    for _, r in SEL[(SEL.C == c) & (SEL["T"] == "Estatístico")].iterrows():
        o = rk_files[(c, r.Metodo)]; N = r.N
        mant, disp = o[:N], o[N:]
        disp_inf = [f for f in disp if X.loc[am.iloc[:, -1].astype(str) == c, f].std(ddof=0) > 0]
        forte = (Rk.loc[disp_inf, mant].fillna(0).max(axis=1) >= 0.7).mean() if disp_inf else np.nan
        lin.append(dict(C=c, P=r.P, M=r.Metodo, G=r.G, N=N, disp=106 - N, pct=(106 - N) / 106, BA=r.BA,
                        BAmax=best.loc[(c, r.P, r.Metodo)], forte=forte, n_inf=n_inf, disp_inf=len(disp_inf)))
pd.DataFrame(lin).to_pickle(O + "dispensaveis.pkl")
pd.to_pickle(rk_files, O + "rankings.pkl")

# --- equivalência entre rankings (Friedman/Nemenyi e Spearman, 106 atributos) ---
lin, glob_ = [], []
for c in CLS:
    pos = pd.DataFrame({m: pd.Series({f: i + 1 for i, f in enumerate(rk_files[(c, m)])}) for m in PROP + LIT + DESC})
    chi, pf = friedmanchisquare(*[pos[m] for m in pos.columns])
    P_ = sp.posthoc_nemenyi_friedman(pos); R_ = pos.corr(method="spearman")
    glob_.append(dict(C=c, chi2=chi, pF=pf, postos_min=pos.rank(axis=1).mean().min(), postos_max=pos.rank(axis=1).mean().max()))
    for a, b in itertools.combinations(pos.columns, 2):
        lin.append(dict(C=c, A=a, B=b, GA=GRP[a], GB=GRP[b], p=P_.loc[a, b], rho=R_.loc[a, b]))
pd.DataFrame(lin).to_pickle(O + "pares.pkl"); pd.DataFrame(glob_).to_pickle(O + "friedman.pkl")
print("ok", SEL.shape)

# ## 4. Rankings em 100 subamostras estratificadas de 50% (cerca de 25 min)

# 100 subamostras estratificadas (50 %) da amostra de 197 registros; rankings de cada cenário (classe).
import sys, time, pickle
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
B = PASTA
am = pd.read_excel(B + "amostra.xlsx", sheet_name="Amostra"); am.columns = am.columns.astype(str)
y = am.iloc[:, -1].astype(str)
sss = StratifiedShuffleSplit(n_splits=100, train_size=0.5, random_state=42)
ARQ = SAIDA + "ordens_subamostras.pkl"
out = {c: [] for c in "ABCDE"}
t0 = time.time()
for k, (idx, _) in enumerate(sss.split(am, y)):
    sub = am.iloc[np.sort(idx)]           # mantém a ordem temporal
    for c in "ABCDE":
        out[c].append(rankings_cam(sub, c, semente_mi=42))
    if k % 10 == 0: print(k, round(time.time() - t0), flush=True)
pickle.dump(out, open(SAIDA + "ordens_subamostras.pkl", "wb"))
print("fim", round(time.time() - t0))

# ## 5. Estabilidade por cenário, paradigma e ranking

# Estabilidade de Nogueira por cenário, paradigma e ranking (N = escolha do Kneedle), d = 106 atributos.
import sys, pickle, numpy as np, pandas as pd
O = pickle.load(open(SAIDA + "ordens_subamostras.pkl", "rb"))
SEL = pd.read_pickle(SAIDA + "sel.pkl"); K = SEL[SEL["T"] == "Kneedle"]
DISP = pd.read_pickle(SAIDA + "dispensaveis.pkl")
ninf = DISP.drop_duplicates("C").set_index("C").n_inf
cols = list(O["A"][0]["Cp"]); idx = {f: i for i, f in enumerate(cols)}; d = len(cols)
lin = []
for _, r in K.iterrows():
    Z = np.zeros((len(O[r.C]), d), dtype=np.int8)
    for i, o in enumerate(O[r.C]):
        Z[i, [idx[f] for f in o[r.Metodo][:r.N]]] = 1
    phi, var, li, ls = nogueira(Z)
    lin.append(dict(C=r.C, P=r.P, Metodo=r.Metodo, G=r.G, N=r.N, phi=phi, var=var, li=li, ls=ls,
                    triv=bool((r.G != "Literatura" or r.Metodo in ("LaplacianScore", "SPEC", "SVDEntropy")) and r.N >= ninf[r.C])))
E = pd.DataFrame(lin); E.to_pickle(SAIDA + "estabilidade.pkl")
print(d, E.groupby(["P", "G"]).apply(lambda x: f"{(x.phi>=0.4).sum()}/{len(x)} m {x.phi.mean():.2f} t{x.triv.sum()}").unstack())

# ## 6. Exportação para Excel

import pandas as pd
with pd.ExcelWriter(SAIDA + "8_Analises_Caminhoes.xlsx") as w:
    for nome in ["sel", "recalc", "tempo", "placar", "dispensaveis", "pares", "friedman", "estabilidade"]:
        pd.read_pickle(SAIDA + nome + ".pkl").to_excel(w, sheet_name=nome[:31], index=False)
print("Excel gravado em", SAIDA)
