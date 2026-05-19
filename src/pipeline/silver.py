"""
src/pipeline/silver.py
───────────────────────────────────────────────────────────────
Camada Silver — limpa, normaliza e valida os dados da camada bronze.

O que faz:
    1. Lê os Parquets da camada bronze
    2. Normaliza nomes de seleções (inglês → português)
    3. Trata valores nulos
    4. Remove duplicatas
    5. Valida qualidade dos dados
    6. Salva em Parquet na pasta data/silver/

Arquivos gerados:
    data/silver/matches.parquet  → partidas limpas e normalizadas
    data/silver/shots.parquet    → chutes limpos para o modelo xG
    data/silver/stats.parquet    → estatísticas limpas por time/partida

Como rodar:
    python src/pipeline/silver.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Caminhos ──────────────────────────────────────────────────
BRONZE_DIR = Path("data/bronze")
SILVER_DIR = Path("data/silver")
SILVER_DIR.mkdir(parents=True, exist_ok=True)


def print_header():
    print("\n" + "="*55)
    print("  Pipeline Silver — Limpeza e Normalização")
    print("="*55 + "\n")


# ─────────────────────────────────────────────────────────────
# DICIONÁRIO DE NORMALIZAÇÃO DE NOMES
# ─────────────────────────────────────────────────────────────

NOMES_SELECOES = {
    # América do Sul
    "Brazil":             "Brasil",
    "Argentina":          "Argentina",
    "Uruguay":            "Uruguai",
    "Colombia":           "Colômbia",
    "Chile":              "Chile",
    "Peru":               "Peru",
    "Ecuador":            "Equador",
    "Paraguay":           "Paraguai",
    "Bolivia":            "Bolívia",
    "Venezuela":          "Venezuela",

    # Europa
    "France":             "França",
    "Germany":            "Alemanha",
    "Spain":              "Espanha",
    "England":            "Inglaterra",
    "Portugal":           "Portugal",
    "Netherlands":        "Holanda",
    "Belgium":            "Bélgica",
    "Croatia":            "Croácia",
    "Switzerland":        "Suíça",
    "Denmark":            "Dinamarca",
    "Sweden":             "Suécia",
    "Poland":             "Polônia",
    "Serbia":             "Sérvia",
    "Russia":             "Rússia",
    "Ukraine":            "Ucrânia",
    "Czech Republic":     "República Tcheca",
    "Slovakia":           "Eslováquia",
    "Hungary":            "Hungria",
    "Romania":            "Romênia",
    "Greece":             "Grécia",
    "Turkey":             "Turquia",
    "Austria":            "Áustria",
    "Wales":              "País de Gales",
    "Scotland":           "Escócia",
    "Norway":             "Noruega",
    "Finland":            "Finlândia",
    "Iceland":            "Islândia",
    "Albania":            "Albânia",
    "Slovenia":           "Eslovênia",
    "Bosnia Herzegovina": "Bósnia e Herzegovina",
    "North Macedonia":    "Macedônia do Norte",
    "Montenegro":         "Montenegro",
    "Kosovo":             "Kosovo",

    # América do Norte e Central
    "United States":      "Estados Unidos",
    "USA":                "Estados Unidos",
    "Mexico":             "México",
    "Canada":             "Canadá",
    "Costa Rica":         "Costa Rica",
    "Panama":             "Panamá",
    "Honduras":           "Honduras",
    "Jamaica":            "Jamaica",
    "Trinidad and Tobago":"Trinidad e Tobago",
    "El Salvador":        "El Salvador",
    "Haiti":              "Haiti",
    "Cuba":               "Cuba",

    # África
    "Morocco":            "Marrocos",
    "Senegal":            "Senegal",
    "Ghana":              "Gana",
    "Nigeria":            "Nigéria",
    "Cameroon":           "Camarões",
    "Egypt":              "Egito",
    "Algeria":            "Argélia",
    "Tunisia":            "Tunísia",
    "South Africa":       "África do Sul",
    "Ivory Coast":        "Costa do Marfim",
    "Mali":               "Mali",
    "Burkina Faso":       "Burkina Faso",
    "Guinea":             "Guiné",
    "Congo DR":           "República Democrática do Congo",
    "Zambia":             "Zâmbia",
    "Tanzania":           "Tanzânia",
    "Uganda":             "Uganda",
    "Kenya":              "Quênia",
    "Angola":             "Angola",
    "Mozambique":         "Moçambique",

    # Ásia
    "Japan":              "Japão",
    "South Korea":        "Coreia do Sul",
    "Iran":               "Irã",
    "Saudi Arabia":       "Arábia Saudita",
    "Australia":          "Austrália",
    "Qatar":              "Catar",
    "China":              "China",
    "Iraq":               "Iraque",
    "United Arab Emirates":"Emirados Árabes",
    "Uzbekistan":         "Uzbequistão",
    "Thailand":           "Tailândia",
    "Vietnam":            "Vietnã",
    "India":              "Índia",
    "Indonesia":          "Indonésia",

    # Oceania
    "New Zealand":        "Nova Zelândia",
    "Fiji":               "Fiji",

    # Outros
    "Panama":             "Panamá",
}

# Normalização de fases do torneio
NOMES_FASES = {
    "Group Stage":        "Fase de Grupos",
    "Round of 16":        "Oitavas de Final",
    "Quarter-finals":     "Quartas de Final",
    "Semi-finals":        "Semifinal",
    "3rd Place Final":    "Disputa de 3º Lugar",
    "Final":              "Final",
}

# Normalização de partes do corpo
PARTES_CORPO = {
    "Right Foot":         "Pé Direito",
    "Left Foot":          "Pé Esquerdo",
    "Head":               "Cabeça",
    "No Touch":           "Sem Toque",
}

# Normalização de tipos de jogada
TIPOS_JOGADA = {
    "Open Play":          "Jogo Aberto",
    "From Corner":        "Escanteio",
    "Free Kick":          "Falta",
    "Penalty":            "Pênalti",
    "Kick Off":           "Chute Inicial",
}


def normalizar_nome(nome: str, dicionario: dict) -> str:
    """Normaliza um nome usando o dicionário fornecido."""
    if pd.isna(nome):
        return nome
    return dicionario.get(nome, nome)


# ─────────────────────────────────────────────────────────────
# 1. SILVER MATCHES
# ─────────────────────────────────────────────────────────────

def processar_silver_matches() -> pd.DataFrame:
    """
    Limpa e normaliza a tabela de partidas.
    """
    print("📋 Processando silver matches...")

    path = BRONZE_DIR / "matches.parquet"
    if not path.exists():
        print("   ❌ bronze/matches.parquet não encontrado!")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    total_original = len(df)

    # 1. Remove duplicatas
    df = df.drop_duplicates(subset=["match_id"])
    removidas = total_original - len(df)
    if removidas:
        print(f"   🗑️  {removidas} duplicatas removidas")

    # 2. Normaliza nomes das seleções
    for col in ["time_casa", "time_fora"]:
        df[col] = df[col].apply(lambda x: normalizar_nome(x, NOMES_SELECOES))

    # 3. Normaliza fases
    df["fase"] = df["fase"].apply(lambda x: normalizar_nome(x, NOMES_FASES))

    # 4. Trata nulos
    df["estadio"]      = df["estadio"].fillna("Não informado")
    df["pais_estadio"] = df["pais_estadio"].fillna("Não informado")

    # 5. Adiciona coluna de saldo de gols
    df["saldo_gols"] = df["gols_casa"] - df["gols_fora"]

    # 6. Ordena por data
    df = df.sort_values(["ano", "data"]).reset_index(drop=True)

    # Validações
    assert df["match_id"].nunique() == len(df), "match_id não é único!"
    assert df["gols_casa"].min() >= 0, "Gols negativos encontrados!"
    assert df["gols_fora"].min() >= 0, "Gols negativos encontrados!"

    print(f"   ✅ {len(df)} partidas | {df['time_casa'].nunique()} seleções únicas")
    return df


# ─────────────────────────────────────────────────────────────
# 2. SILVER SHOTS
# ─────────────────────────────────────────────────────────────

def processar_silver_shots() -> pd.DataFrame:
    """
    Limpa e normaliza a tabela de chutes para o modelo xG.
    """
    print("\n⚽ Processando silver shots...")

    path = BRONZE_DIR / "shots.parquet"
    if not path.exists():
        print("   ❌ bronze/shots.parquet não encontrado!")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    total_original = len(df)

    # 1. Remove chutes sem localização (não servem para o modelo)
    df = df.dropna(subset=["loc_x", "loc_y"])
    removidos_loc = total_original - len(df)
    if removidos_loc:
        print(f"   🗑️  {removidos_loc} chutes sem localização removidos")

    # 2. Remove pênaltis do modelo xG
    # (pênaltis têm xG fixo ~0.76 e distorcem o modelo)
    df_sem_pen = df[df["tipo_jogada"] != "Penalty"].copy()
    removidos_pen = len(df) - len(df_sem_pen)
    if removidos_pen:
        print(f"   🗑️  {removidos_pen} pênaltis removidos do modelo xG")
    df = df_sem_pen

    # 3. Normaliza nomes
    for col in ["time_casa", "time_fora", "time_nome"]:
        df[col] = df[col].apply(lambda x: normalizar_nome(x, NOMES_SELECOES))

    df["fase"]        = df["fase"].apply(lambda x: normalizar_nome(x, NOMES_FASES))
    df["parte_corpo"] = df["parte_corpo"].apply(lambda x: normalizar_nome(x, PARTES_CORPO))
    df["tipo_jogada"] = df["tipo_jogada"].apply(lambda x: normalizar_nome(x, TIPOS_JOGADA))

    # 4. Trata nulos
    df["parte_corpo"] = df["parte_corpo"].fillna("Não informado")
    df["tipo_jogada"] = df["tipo_jogada"].fillna("Jogo Aberto")
    df["tecnica"]     = df["tecnica"].fillna("Normal")
    df["sob_pressao"] = df["sob_pressao"].fillna(False)

    # 5. Remove distâncias impossíveis
    # Campo da StatsBomb tem 120x80 — distância máxima possível ~130m
    df = df[df["distancia_gol"] <= 130]

    # 6. Adiciona faixas de distância para análise
    df["faixa_distancia"] = pd.cut(
        df["distancia_gol"],
        bins=[0, 6, 11, 18, 25, 35, 130],
        labels=["0-6m", "6-11m", "11-18m", "18-25m", "25-35m", "35m+"]
    )

    # 7. Flag se o chute foi no alvo
    df["no_alvo"] = df["resultado"].isin(["Goal", "Saved"]).astype(int)

    # Validações
    assert df["foi_gol"].isin([0, 1]).all(), "foi_gol contém valores inválidos!"
    assert df["distancia_gol"].min() >= 0, "Distâncias negativas encontradas!"

    gols = df["foi_gol"].sum()
    taxa = round(gols / len(df) * 100, 1)
    print(f"   ✅ {len(df):,} chutes válidos | {gols:,} gols | conversão: {taxa}%")
    print(f"   📊 Distância média: {df['distancia_gol'].mean():.1f}m")
    return df


# ─────────────────────────────────────────────────────────────
# 3. SILVER STATS
# ─────────────────────────────────────────────────────────────

def processar_silver_stats() -> pd.DataFrame:
    """
    Limpa e normaliza as estatísticas por time por partida.
    """
    print("\n📊 Processando silver stats...")

    path = BRONZE_DIR / "stats.parquet"
    if not path.exists():
        print("   ❌ bronze/stats.parquet não encontrado!")
        return pd.DataFrame()

    df = pd.read_parquet(path)

    # 1. Normaliza nomes
    for col in ["time_nome", "adversario"]:
        df[col] = df[col].apply(lambda x: normalizar_nome(x, NOMES_SELECOES))

    df["fase"] = df["fase"].apply(lambda x: normalizar_nome(x, NOMES_FASES))

    # 2. Trata nulos numéricos com 0
    cols_numericas = [
        "passes_total", "passes_completos", "precisao_passes",
        "chutes_total", "chutes_alvo", "pressoes",
        "duelos_total", "duelos_ganhos"
    ]
    df[cols_numericas] = df[cols_numericas].fillna(0)

    # 3. Adiciona métricas derivadas
    df["chutes_fora_alvo"]   = df["chutes_total"] - df["chutes_alvo"]
    df["duelos_perdidos"]    = df["duelos_total"] - df["duelos_ganhos"]
    df["pct_duelos_ganhos"]  = (
        df["duelos_ganhos"] / df["duelos_total"].replace(0, np.nan) * 100
    ).round(1).fillna(0)

    # 4. Remove duplicatas
    df = df.drop_duplicates(subset=["match_id", "time_id"])

    # Validações
    assert df["precisao_passes"].between(0, 100).all(), "Precisão de passes fora de 0-100!"

    print(f"   ✅ {len(df):,} registros | {df['time_nome'].nunique()} seleções")
    print(f"   📊 Precisão média de passes: {df['precisao_passes'].mean():.1f}%")
    return df


# ─────────────────────────────────────────────────────────────
# 4. SALVAR
# ─────────────────────────────────────────────────────────────

def salvar_parquet(df: pd.DataFrame, nome: str):
    if df.empty:
        print(f"   ⚠️  {nome}.parquet — DataFrame vazio, pulando")
        return
    path = SILVER_DIR / f"{nome}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")
    size_kb = path.stat().st_size / 1024
    print(f"   💾 Salvo: {path} ({size_kb:.1f} KB | {len(df):,} linhas | {len(df.columns)} colunas)")


# ─────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────

def run():
    print_header()

    matches_df = processar_silver_matches()
    shots_df   = processar_silver_shots()
    stats_df   = processar_silver_stats()

    print("\n💾 Salvando camada silver...")
    salvar_parquet(matches_df, "matches")
    salvar_parquet(shots_df,   "shots")
    salvar_parquet(stats_df,   "stats")

    print("\n" + "="*55)
    print("  Silver concluído!")
    print("="*55)
    print(f"""
  Arquivos gerados:
  ─────────────────────────────────────────────
  data/silver/
  ├── matches.parquet  → {len(matches_df):>5,} partidas normalizadas
  ├── shots.parquet    → {len(shots_df):>5,} chutes válidos
  └── stats.parquet    → {len(stats_df):>5,} registros time×partida

  Próximo passo:
  ─────────────────────────────────────────────
  python src/pipeline/gold.py
    """)


if __name__ == "__main__":
    run()
