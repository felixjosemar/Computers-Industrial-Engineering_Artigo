# Estudo de caso 1: SECOM (fabricação de semicondutores)

Base pública UCI SECOM: **1.567 unidades** (1.463 conformes e 104 falhas) e **590 sensores**. O problema é binário e a classe de referência dos índices de CEQ é a das **unidades conformes** (`-1`).

Este estudo de caso é a **referência metodológica**: o EC2 aplica exatamente o mesmo método.

## Conteúdo

| Pasta | O que tem |
|---|---|
| `codigo/` | Cadernos 1 a 8 (e `EC1_rankings_nb2.py`, função de apoio usada no caderno 8) |
| `dados/` | `uci-secom.xlsx`, a base de entrada |
| `resultados/` | Planilhas, tabelas em LaTeX e figuras geradas pela execução do artigo |

## Cadernos

| Caderno | O que faz | Principais saídas |
|---|---|---|
| `EC1_1_Amostras.ipynb` | Amostragem adaptativa estratificada: separa a *amostra* (para os rankings) da *base restante* (para os modelos) | `amostra.xlsx`, `base_restante.xlsx` |
| `EC1_2_CriacaoRanking.ipynb` | Calcula os índices de CEQ (Cp, Cpk, Pp, Ppk, p, np, c, u; o artigo usa seis: Cp, Cpk, Pp, Ppk, p e c) e os rankings das estatísticas descritivas e dos 19 métodos da literatura | `Ranking_*.xlsx`, `Tempos_Execucao.xlsx`, `amostra_classe_-1.xlsx` |
| `EC1_2_2_Consolidar_Rankings.ipynb` | Reúne os rankings em uma planilha só | `Rankings_Consolidado.xlsx` |
| `EC1_3_AvaliacaoDeEquivalenciaDosRankings.ipynb` | Friedman, Nemenyi e correlação de Spearman entre os rankings | `Resultados_Completos_Rankings/` |
| `EC1_4_CriacaoModelos_BackwardElimination.ipynb` | Eliminação regressiva com validação cruzada estratificada de 5 partições, em 7 paradigmas, **sobre a base restante** | `Modelos<Paradigma>_CV_Consolidado.xlsx`, `Resumo_Geral_Modelos<Paradigma>_CV.xlsx`, `Mapa_Abas_Consolidado.xlsx`, `Redundancia_*.xlsx`, `Alerta_AUC_Abaixo_0.5.xlsx` |
| `EC1_5_AnaliseParetoGlobalEPorParadigma.ipynb` | Escolha do modelo em cada curva (*Kneedle* e regra de 1 desvio-padrão), comparação CEQ × literatura, Mann-Whitney, custo computacional e fronteira de Pareto | `Resumo_Global_Dados_*.xlsx`, `Tabela_*.tex/.xlsx`, `CODE5_*.xlsx`, `CODE9_*.xlsx`, `Analise_*_Pareto_Destaque_*.png` |
| `EC1_6_FronteiraParetoSeparada.ipynb` | Fronteira de Pareto da proposta e dos demais métodos em separado | `Resultados_DuplaFronteira/` |
| `EC1_7_FronteiraParetoPorIndicador.ipynb` | Fronteira de Pareto por indicador | `Resultados_DuplaFronteira_PorIndicador/` |
| `EC1_8_AnalisesResultados.ipynb` (e `.py`) | Reúne as análises usadas no artigo (estabilidade Φ de Nogueira em 100 subamostras, placar, atributos dispensáveis, tempo) | `8_Analises_Tese.xlsx` |
| `EC1_9_Custo_e_Estabilidade.ipynb` | Tabelas de custo computacional, atributos selecionados e estabilidade, lidas de `8_Analises_Tese.xlsx` (não recalcula) | `9_Custo_e_Estabilidade/*.xlsx` |

## Pontos de método

- **Limites dos índices de capacidade**: LSL e USL são os percentis 1 e 99 de cada sensor na classe de referência (não o mínimo e o máximo). O caderno 1 usa os mesmos percentis na regra de parada; se mudar um, mude o outro.
- **Σ de curto e longo prazo**: Cp e Cpk usam o desvio-padrão de curto prazo (amplitude móvel); Pp e Ppk usam o de longo prazo. Por isso a amostra é mantida em ordem temporal.
- **Regra de parada da amostra**: parte de 100 registros e acrescenta 1 por iteração. Para quando a reprodutibilidade da lista dos 20 primeiros sensores de cada índice de CEQ deixa de melhorar, com piso de 197 registros da classe de referência (critério do intervalo de confiança do σ) e teto de 50% da base. Na execução do artigo o teto foi atingido: 783 registros na amostra (731 conformes e 52 falhas) e 784 na base restante.
- **Base de modelagem**: os classificadores usam só a `base_restante.xlsx`, nunca a amostra.
- **Validação**: 5 partições estratificadas, com embaralhamento e semente 42, sem balanceamento de classes. Hiperparâmetros padrão do scikit-learn (o SVM usa `probability=True`, necessário para calcular a AUC).
- **Grupos comparados**: proposta = os seis índices `_Original` (Cp, Cpk, Pp, Ppk, p, c); descritivas = AmplitudeSigma, IQRSigma, Assimetria e Curtose; literatura = 19 métodos. As versões `_Normalizado` e `_BaseCompleta` e os índices np e u são gerados, mas não entram nas comparações do artigo.
- **Estabilidade**: índice Φ de Nogueira, 100 subamostras estratificadas de 50%; lista estável se Φ ≥ 0,40.

## Números do artigo para este estudo de caso

| Indicador | Proposta (CEQ) | Literatura |
|---|---|---|
| *Rankings* estáveis (Φ ≥ 0,40) | 35 de 42 (83,3%) | 33 de 133 (24,8%) |
| Comparações de qualidade com maior média (de 28) | 7 | 10 (11 empates) |
| Diferenças significativas (Mann-Whitney, p ≤ 0,05) | 1 | 4 |

As 28 comparações são 4 métricas (acurácia balanceada, sensibilidade, especificidade e AUC) em 7 paradigmas.

## Como reproduzir

Veja "Como executar" no [README principal](../README.md). Resumo: copie `codigo/*` e `dados/*` para uma pasta de trabalho, abra os cadernos e rode na ordem 1, 2, 2_2, 3, 4, 5, 6, 7, 8. O caderno 1 abre uma janela para escolher `uci-secom.xlsx`.
