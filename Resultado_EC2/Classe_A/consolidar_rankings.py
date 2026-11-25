"""
Consolida todos os arquivos "Ranking_*.xlsx" de uma pasta em um único arquivo Excel,
onde cada arquivo original vira uma aba.

Uso:
    python3 consolidar_rankings.py [pasta_de_entrada] [arquivo_de_saida.xlsx]

Se nenhum argumento for passado, usa a pasta atual como entrada e grava
"Rankings_Consolidado.xlsx" na mesma pasta.
"""

import sys
import re
import glob
import os
import pandas as pd

# Caracteres que o Excel não aceita em nome de aba, e o limite de 31 caracteres.
CARACTERES_INVALIDOS = re.compile(r'[\[\]:*?/\\]')
LIMITE_NOME_ABA = 31


def nome_aba_valido(nome, nomes_ja_usados):
    """Limpa o nome do arquivo para virar um nome de aba válido no Excel (sem
    caracteres proibidos, até 31 caracteres, sem repetir um nome já usado)."""
    nome = CARACTERES_INVALIDOS.sub("_", nome)
    nome = nome[:LIMITE_NOME_ABA]

    candidato = nome
    sufixo = 1
    while candidato in nomes_ja_usados:
        sufixo += 1
        corte = LIMITE_NOME_ABA - len(f"_{sufixo}")
        candidato = f"{nome[:corte]}_{sufixo}"
    return candidato


def consolidar(pasta_entrada: str, caminho_saida: str) -> None:
    padrao = os.path.join(pasta_entrada, "Ranking_*.xlsx")
    arquivos = sorted(glob.glob(padrao))

    if not arquivos:
        print(f"Nenhum arquivo 'Ranking_*.xlsx' encontrado em: {pasta_entrada}")
        return

    print(f"{len(arquivos)} arquivo(s) encontrado(s) em: {pasta_entrada}")

    nomes_usados = set()
    abas_gravadas = 0

    with pd.ExcelWriter(caminho_saida, engine="openpyxl") as writer:
        for caminho in arquivos:
            nome_arquivo = os.path.splitext(os.path.basename(caminho))[0]
            # "Ranking_Cp_Original.xlsx" -> aba "Cp_Original"
            nome_base = nome_arquivo[len("Ranking_"):] if nome_arquivo.startswith("Ranking_") else nome_arquivo

            try:
                df = pd.read_excel(caminho)
            except Exception as e:
                print(f"  ⚠️  Erro ao ler '{os.path.basename(caminho)}': {e} -- pulando.")
                continue

            nome_aba = nome_aba_valido(nome_base, nomes_usados)
            nomes_usados.add(nome_aba)

            df.to_excel(writer, sheet_name=nome_aba, index=False)
            abas_gravadas += 1
            marca = f" (aba: '{nome_aba}')" if nome_aba != nome_base else ""
            print(f"  ✅ {os.path.basename(caminho)}{marca}")

    print(f"\n🎉 {abas_gravadas} aba(s) gravada(s) em: {caminho_saida}")


if __name__ == "__main__":
    pasta_entrada = sys.argv[1] if len(sys.argv) > 1 else "."
    caminho_saida = sys.argv[2] if len(sys.argv) > 2 else os.path.join(pasta_entrada, "Rankings_Consolidado.xlsx")
    consolidar(pasta_entrada, caminho_saida)
