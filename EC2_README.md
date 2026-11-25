# Estudo de caso 2: caminhões fora de estrada (5 turmas)

> **Dados confidenciais.** A base deste estudo de caso pertence à empresa que a cedeu e **não é pública**. Para obtê-la, contate o autor correspondente (josemar.felix@aluno.ufop.edu.br) e solicite o início do processo de autorização de liberação dos dados pela empresa. Veja "Disponibilidade dos dados" abaixo.

Dados de operação de caminhões fora de estrada em uma mina a céu aberto, referentes ao mês de janeiro: **7.936 registros** (turnos de 6 horas) e **106 atributos**, que são os estados operacionais em que um caminhão pode estar (viajando cheio, viajando vazio, carregando, basculando, em fila etc.). O valor de cada atributo é o tempo, em segundos, em que o caminhão ficou naquele estado no turno. A classe é a **turma de operação** responsável pelo turno: A (1.600 registros), B (1.600), C (1.536), D (1.536) e E (1.664).

O método é **o mesmo do [estudo de caso 1](../EC1_SECOM/EC1_README.md)**. A única diferença é a estrutura das classes: não há classe conforme, então **cada turma é usada, em separado, como classe de referência**, e o método é aplicado cinco vezes (5 cenários, um por turma). Os classificadores continuam prevendo as cinco turmas; o que muda entre os cenários é a ordem de eliminação dos atributos, que vem do ranking calculado com a turma de referência. Sensibilidade e especificidade não se aplicam a cinco classes, e as métricas são acurácia balanceada, AUC macro e F1 macro.

## Conteúdo

| Pasta | O que tem |
|---|---|
| `codigo/` | Cadernos 1 a 8, adaptados do EC1 |
| `dados/` | Apenas instruções de acesso. A base de entrada é confidencial e não é publicada |
| `resultados/` | Uma subpasta por turma (`Classe_A` a `Classe_E`) com rankings, curvas de eliminação regressiva, resumos, tabelas e figuras |

## Cadernos e o que mudou em relação ao EC1

| Caderno | O que faz | Adaptação para o EC2 |
|---|---|---|
| `EC2_1_Amostras.ipynb` | Amostra e base restante | A regra de parada é a do EC1, verificada nas **cinco turmas ao mesmo tempo** (cada turma é a classe de referência do seu cenário). O piso de 197 registros vale para a amostra inteira. |
| `EC2_2_CriacaoRanking_PorClasse.ipynb` | Rankings de cada turma | Limites LSL/USL = percentis 1 e 99; índice indefinido (atributo constante) = NaN, que vai para o fim do ranking, como no EC1. Gera uma subpasta `Classe_<turma>/` por cenário. |
| `EC2_2_2_consolidar_rankings.py` | Reúne os rankings | Roda dentro de cada `Classe_<turma>/`. |
| `EC2_3_AvaliacaoDeEquivalenciaDosRankings_PorClasse.ipynb` | Friedman, Nemenyi, Spearman | Repete a análise do EC1 em cada subpasta de turma. |
| `EC2_4_CriacaoModelos_BackwardElimination_PorClasse.ipynb` | Eliminação regressiva, 7 paradigmas | Treina **sobre `base_restante.xlsx`**, como no EC1. Alvo multiclasse (A a E). |
| `EC2_4b_CriacaoModelos_LogRegLasso_PorClasse.ipynb` | Só a Regressão Logística (L1) | Mesma regra do caderno 4, para rodar esse paradigma separadamente. |
| `EC2_5_AnaliseParetoGlobalEPorParadigma_PorClasse.ipynb` | *Kneedle*, regra de 1 desvio-padrão, comparações, Pareto | Roda em cada subpasta de turma. |
| `6_...` e `EC2_7_FronteiraParetoPorIndicador_PorClasse.ipynb` | Fronteiras de Pareto | Idem. |
| `EC2_8_AnalisesResultados_Caminhoes.ipynb` (e `.py`) | Análises finais para os cinco cenários: seleção dos modelos, comparação por grupo, placar, atributos dispensáveis, tempo, equivalência e estabilidade de Nogueira em 100 subamostras | Usa as mesmas funções de ranking do caderno 2. |

## Como reproduzir

Veja "Como executar" no [README principal](../README.md). Obtenha a base conforme a seção "Disponibilidade dos dados", copie `codigo/*` e a base para uma pasta de trabalho e rode na ordem 1, 2, 2_2 (dentro de cada `Classe_<turma>/`), 3, 4, 4b, 5, 6, 7, 8. O caderno 1 abre uma janela para escolher o arquivo de entrada. O caderno 4 é o mais demorado: o número de modelos é `5 turmas × 29 rankings × 7 paradigmas × 106 atributos` (cada um com 5 partições); ajuste `n_jobs`.

## Sobre a pasta `resultados/` (leia antes de comparar com o código)

As planilhas de `resultados/` vêm de uma **execução anterior** do estudo de caso 2, feita com versões anteriores dos cadernos 1, 2 e 4. Pelo código que as gerou, essa execução diferia do EC1 em três pontos:

1. a amostra tem **197 registros** (`amostra_execucao_anterior.xlsx`), sem separar uma base restante;
2. os limites LSL/USL dos rankings eram o **mínimo e o máximo** de cada atributo, e não os percentis 1 e 99;
3. os modelos foram treinados sobre **essa mesma amostra de 197 registros**, e não sobre uma base restante.

O código desta pasta `codigo/` já segue a metodologia do EC1 nesses três pontos. Por isso, **reexecutar os cadernos não reproduz exatamente as planilhas de `resultados/`**. Num teste do caderno 1 sobre a base completa, a regra de parada do EC1 não convergiu e a amostra chegou ao teto de 50% da base (3.968 registros na amostra e 3.968 na base restante), como também ocorreu no EC1 (783 e 784 registros). Para alinhar o artigo e as planilhas à metodologia do EC1, é preciso reexecutar os cadernos 1 a 8.

Os números abaixo, usados no artigo, são reproduzidos pelo caderno 8 a partir dos arquivos desta pasta (ele lê as curvas `Modelos*_CV_Consolidado.xlsx`, escolhe o modelo de cada curva pelo *Kneedle* e compara os grupos):

| Indicador | Proposta (CEQ) | Literatura |
|---|---|---|
| *Rankings* estáveis (Φ ≥ 0,40; Φ arredondado a 2 casas) | 206 de 210 (98,1%) | 479 de 665 (72,0%) |
| Estabilidade média (Φ) | 0,75 | 0,53 |
| Comparações de qualidade com maior média (de 105) | 18 | 75 (12 empates) |
| Diferenças significativas (Mann-Whitney, p ≤ 0,05) | 3 | 27 |
| Combinações em que usou menos atributos (de 35) | 23 | 12 |

As estatísticas descritivas tiveram 134 de 140 *rankings* estáveis. As 105 comparações são 3 métricas (acurácia balanceada, AUC macro e F1 macro) em 35 combinações de paradigma e cenário.

Também estão em cada `Classe_<turma>/` os arquivos `Resumo_Global_Dados_*.xlsx`, `Tabela_*.tex` e as figuras de Pareto, gerados pelo caderno 5 a partir das curvas dessa execução.


A base pertence à empresa que a cedeu e é confidencial. Ela não é pública, e os arquivos com registros individuais (a base e as amostras) foram removidos. **Esta base não é pública.** Para obter acesso, entre em contato com o autor correspondente (josemar.felix@aluno.ufop.edu.br) e solicite a abertura do processo de autorização de liberação dos dados junto à empresa. As planilhas em `resultados/` são agregadas e não contêm registros individuais.
