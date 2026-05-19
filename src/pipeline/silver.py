"""
src/pipeline/silver.py
───────────────────────────────────────────────────────────────
Camada Silver — limpa, normaliza e valida dados de todas as fontes.

Fontes tratadas:
    - StatsBomb (Copas 2018 e 2022)
    - Kaggle (Eliminatórias 2026)
    - API-Football (UEFA e CAF)

O que faz:
    1. Normaliza nomes de seleções (inglês → português)
    2. Trata valores nulos
    3. Remove duplicatas
    4. Adiciona peso por fonte (para o modelo ML)
    5. Valida qualidade dos dados

Arquivos gerados:
    data/silver/matches.parquet     → todas as partidas normalizadas
    data/silver/shots.parquet       → chutes limpos para modelo xG
    data/silver/stats.parquet       → estatísticas limpas
    data/silver/qualifiers.parquet  → eliminatórias normalizadas

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
# DICIONÁRIOS DE NORMALIZAÇÃO
# ─────────────────────────────────────────────────────────────

NOMES_SELECOES = {
    # América do Sul
    "Brazil":               "Brasil",
    "Argentina":            "Argentina",
    "Uruguay":              "Uruguai",
    "Colombia":             "Colômbia",
    "Chile":                "Chile",
    "Peru":                 "Peru",
    "Ecuador":              "Equador",
    "Paraguay":             "Paraguai",
    "Bolivia":              "Bolívia",
    "Venezuela":            "Venezuela",

    # Europa
    "France":               "França",
    "Germany":              "Alemanha",
    "Spain":                "Espanha",
    "England":              "Inglaterra",
    "Portugal":             "Portugal",
    "Netherlands":          "Holanda",
    "Belgium":              "Bélgica",
    "Croatia":              "Croácia",
    "Switzerland":          "Suíça",
    "Denmark":              "Dinamarca",
    "Sweden":               "Suécia",
    "Poland":               "Polônia",
    "Serbia":               "Sérvia",
    "Russia":               "Rússia",
    "Ukraine":              "Ucrânia",
    "Czech Republic":       "República Tcheca",
    "Czechia":              "República Tcheca",
    "Slovakia":             "Eslováquia",
    "Hungary":              "Hungria",
    "Romania":              "Romênia",
    "Greece":               "Grécia",
    "Turkey":               "Turquia",
    "Austria":              "Áustria",
    "Wales":                "País de Gales",
    "Scotland":             "Escócia",
    "Norway":               "Noruega",
    "Finland":              "Finlândia",
    "Iceland":              "Islândia",
    "Albania":              "Albânia",
    "Slovenia":             "Eslovênia",
    "Bosnia Herzegovina":   "Bósnia e Herzegovina",
    "Bosnia & Herzegovina": "Bósnia e Herzegovina",
    "North Macedonia":      "Macedônia do Norte",
    "Montenegro":           "Montenegro",
    "Kosovo":               "Kosovo",
    "Bulgaria":             "Bulgária",
    "Belarus":              "Bielorrússia",
    "Georgia":              "Geórgia",
    "Armenia":              "Armênia",
    "Azerbaijan":           "Azerbaijão",
    "Kazakhstan":           "Cazaquistão",
    "Luxembourg":           "Luxemburgo",
    "Lithuania":            "Lituânia",
    "Latvia":               "Letônia",
    "Estonia":              "Estônia",
    "Moldova":              "Moldávia",
    "Gibraltar":            "Gibraltar",
    "Andorra":              "Andorra",
    "San Marino":           "San Marino",
    "Liechtenstein":        "Liechtenstein",
    "Malta":                "Malta",
    "Cyprus":               "Chipre",
    "Northern Ireland":     "Irlanda do Norte",
    "Republic of Ireland":  "República da Irlanda",
    "Ireland":              "Irlanda",
    "Israel":               "Israel",
    "Faroe Islands":        "Ilhas Faroé",

    # América do Norte e Central
    "United States":        "Estados Unidos",
    "USA":                  "Estados Unidos",
    "Mexico":               "México",
    "Canada":               "Canadá",
    "Costa Rica":           "Costa Rica",
    "Panama":               "Panamá",
    "Honduras":             "Honduras",
    "Jamaica":              "Jamaica",
    "Trinidad and Tobago":  "Trinidad e Tobago",
    "Trinidad & Tobago":    "Trinidad e Tobago",
    "El Salvador":          "El Salvador",
    "Haiti":                "Haiti",
    "Cuba":                 "Cuba",
    "Guatemala":            "Guatemala",
    "Nicaragua":            "Nicarágua",
    "Belize":               "Belize",
    "Barbados":             "Barbados",
    "Antigua and Barbuda":  "Antígua e Barbuda",
    "Antigua & Barbuda":    "Antígua e Barbuda",
    "Grenada":              "Granada",
    "Saint Lucia":          "Santa Lúcia",
    "Saint Kitts and Nevis":"São Cristóvão e Névis",
    "Dominica":             "Dominica",
    "Dominican Republic":   "República Dominicana",
    "Puerto Rico":          "Porto Rico",
    "Bermuda":              "Bermudas",
    "Aruba":                "Aruba",
    "Curacao":              "Curaçao",
    "Curaçao":              "Curaçao",
    "Guyana":               "Guiana",
    "Suriname":             "Suriname",
    "Anguilla":             "Anguila",
    "Bahamas":              "Bahamas",

    # África
    "Morocco":              "Marrocos",
    "Senegal":              "Senegal",
    "Ghana":                "Gana",
    "Nigeria":              "Nigéria",
    "Cameroon":             "Camarões",
    "Egypt":                "Egito",
    "Algeria":              "Argélia",
    "Tunisia":              "Tunísia",
    "South Africa":         "África do Sul",
    "Ivory Coast":          "Costa do Marfim",
    "Mali":                 "Mali",
    "Burkina Faso":         "Burkina Faso",
    "Guinea":               "Guiné",
    "Congo DR":             "República Democrática do Congo",
    "DR Congo":             "República Democrática do Congo",
    "Congo":                "Congo",
    "Zambia":               "Zâmbia",
    "Tanzania":             "Tanzânia",
    "Uganda":               "Uganda",
    "Kenya":                "Quênia",
    "Angola":               "Angola",
    "Mozambique":           "Moçambique",
    "Zimbabwe":             "Zimbábue",
    "Botswana":             "Botsuana",
    "Namibia":              "Namíbia",
    "Rwanda":               "Ruanda",
    "Ethiopia":             "Etiópia",
    "Benin":                "Benin",
    "Gabon":                "Gabão",
    "Sudan":                "Sudão",
    "Libya":                "Líbia",
    "Madagascar":           "Madagáscar",
    "Mauritania":           "Mauritânia",
    "Niger":                "Níger",
    "Togo":                 "Togo",
    "Cape Verde":           "Cabo Verde",
    "Equatorial Guinea":    "Guiné Equatorial",
    "Guinea-Bissau":        "Guiné-Bissau",
    "Sierra Leone":         "Serra Leoa",
    "Liberia":              "Libéria",
    "Central African Rep.": "República Centro-Africana",
    "Comoros":              "Comores",
    "Eswatini":             "Essuatíni",
    "Lesotho":              "Lesoto",
    "Malawi":               "Maláui",
    "Burundi":              "Burundi",
    "South Sudan":          "Sudão do Sul",
    "Somalia":              "Somália",
    "Djibouti":             "Djibuti",
    "Eritrea":              "Eritreia",

    # Ásia
    "Japan":                "Japão",
    "South Korea":          "Coreia do Sul",
    "Iran":                 "Irã",
    "Saudi Arabia":         "Arábia Saudita",
    "Australia":            "Austrália",
    "Qatar":                "Catar",
    "China":                "China",
    "Iraq":                 "Iraque",
    "United Arab Emirates": "Emirados Árabes",
    "UAE":                  "Emirados Árabes",
    "Uzbekistan":           "Uzbequistão",
    "Thailand":             "Tailândia",
    "Vietnam":              "Vietnã",
    "India":                "Índia",
    "Indonesia":            "Indonésia",
    "Bahrain":              "Bahrein",
    "Jordan":               "Jordânia",
    "Oman":                 "Omã",
    "Kuwait":               "Kuwait",
    "Syria":                "Síria",
    "Lebanon":              "Líbano",
    "Palestine":            "Palestina",
    "Yemen":                "Iêmen",
    "Afghanistan":          "Afeganistão",
    "Bangladesh":           "Bangladesh",
    "Pakistan":             "Paquistão",
    "Nepal":                "Nepal",
    "Sri Lanka":            "Sri Lanka",
    "Myanmar":              "Mianmar",
    "Cambodia":             "Camboja",
    "Malaysia":             "Malásia",
    "Philippines":          "Filipinas",
    "Singapore":            "Singapura",
    "Taiwan":               "Taiwan",
    "Hong Kong":            "Hong Kong",
    "Macau":                "Macau",
    "Mongolia":             "Mongólia",
    "North Korea":          "Coreia do Norte",
    "Kyrgyzstan":           "Quirguistão",
    "Tajikistan":           "Tajiquistão",
    "Turkmenistan":         "Turcomenistão",
    "Maldives":             "Maldivas",

    # Oceania
    "New Zealand":          "Nova Zelândia",
    "Fiji":                 "Fiji",
    "Papua New Guinea":     "Papua Nova Guiné",
    "Solomon Islands":      "Ilhas Salomão",
    "Vanuatu":              "Vanuatu",
    "Samoa":                "Samoa",
    "Tahiti":               "Taiti",
    "New Caledonia":        "Nova Caledônia",
}

NOMES_FASES = {
    "Group Stage":              "Fase de Grupos",
    "Round of 16":              "Oitavas de Final",
    "Quarter-finals":           "Quartas de Final",
    "Semi-finals":              "Semifinal",
    "3rd Place Final":          "Disputa de 3º Lugar",
    "Final":                    "Final",
    "Qualifying Round 1":       "Qualificação - Rodada 1",
    "Qualifying Round 2":       "Qualificação - Rodada 2",
    "Qualifying Round 3":       "Qualificação - Rodada 3",
    "Qualifying Round 4":       "Qualificação - Rodada 4",
    "Qualificação":             "Qualificação",
}

PARTES_CORPO = {
    "Right Foot": "Pé Direito",
    "Left Foot":  "Pé Esquerdo",
    "Head":       "Cabeça",
    "No Touch":   "Sem Toque",
}

TIPOS_JOGADA = {
    "Open Play":  "Jogo Aberto",
    "From Corner":"Escanteio",
    "Free Kick":  "Falta",
    "Penalty":    "Pênalti",
    "Kick Off":   "Chute Inicial",
}

# Peso por fonte para o modelo ML
PESOS_FONTE = {
    "statsbomb":   0.8,   # Copa do Mundo histórica
    "kaggle":      0.5,   # Eliminatórias 2026
    "api_football":0.5,   # Eliminatórias UEFA/CAF
}


def normalizar_nome(nome, dicionario: dict):
    if pd.isna(nome) or nome == "nan":
        return nome
    return dicionario.get(str(nome), str(nome))


# ─────────────────────────────────────────────────────────────
# 1. SILVER MATCHES
# ─────────────────────────────────────────────────────────────

def processar_silver_matches() -> pd.DataFrame:
    print("📋 Processando silver matches...")

    path = BRONZE_DIR / "matches.parquet"
    if not path.exists():
        print("   ❌ bronze/matches.parquet não encontrado!")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    total_original = len(df)

    # Remove duplicatas
    df = df.drop_duplicates(subset=["match_id"])
    removidas = total_original - len(df)
    if removidas:
        print(f"   🗑️  {removidas} duplicatas removidas")

    # Normaliza nomes das seleções
    for col in ["time_casa", "time_fora"]:
        df[col] = df[col].apply(lambda x: normalizar_nome(x, NOMES_SELECOES))

    # Normaliza fases
    df["fase"] = df["fase"].apply(lambda x: normalizar_nome(x, NOMES_FASES))

    # Trata nulos
    df["estadio"] = df["estadio"].fillna("Não informado")

    # Adiciona saldo de gols
    df["saldo_gols"] = df["gols_casa"] - df["gols_fora"]

    # Adiciona peso por fonte
    df["peso_fonte"] = df["fonte"].map(PESOS_FONTE).fillna(0.3)

    # Ordena
    df = df.sort_values(["ano", "data"]).reset_index(drop=True)

    selecoes = pd.concat([df["time_casa"], df["time_fora"]]).nunique()
    print(f"   ✅ {len(df):,} partidas | {selecoes} seleções únicas")
    print(f"   📊 Copas: {len(df[df['fonte']=='statsbomb'])} | Eliminatórias: {len(df[df['fonte']!='statsbomb'])}")
    return df


# ─────────────────────────────────────────────────────────────
# 2. SILVER SHOTS
# ─────────────────────────────────────────────────────────────

def processar_silver_shots() -> pd.DataFrame:
    print("\n⚽ Processando silver shots...")

    path = BRONZE_DIR / "shots.parquet"
    if not path.exists():
        print("   ❌ bronze/shots.parquet não encontrado!")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    total_original = len(df)

    # Remove sem localização
    df = df.dropna(subset=["loc_x", "loc_y"])
    removidos_loc = total_original - len(df)
    if removidos_loc:
        print(f"   🗑️  {removidos_loc} chutes sem localização removidos")

    # Remove pênaltis
    df_sem_pen = df[df["tipo_jogada"] != "Penalty"].copy()
    removidos_pen = len(df) - len(df_sem_pen)
    if removidos_pen:
        print(f"   🗑️  {removidos_pen} pênaltis removidos do modelo xG")
    df = df_sem_pen

    # Normaliza nomes
    for col in ["time_casa", "time_fora", "time_nome"]:
        df[col] = df[col].apply(lambda x: normalizar_nome(x, NOMES_SELECOES))

    df["fase"]        = df["fase"].apply(lambda x: normalizar_nome(x, NOMES_FASES))
    df["parte_corpo"] = df["parte_corpo"].apply(lambda x: normalizar_nome(x, PARTES_CORPO))
    df["tipo_jogada"] = df["tipo_jogada"].apply(lambda x: normalizar_nome(x, TIPOS_JOGADA))

    # Trata nulos
    df["parte_corpo"] = df["parte_corpo"].fillna("Não informado")
    df["tipo_jogada"] = df["tipo_jogada"].fillna("Jogo Aberto")
    df["tecnica"]     = df["tecnica"].fillna("Normal")
    df["sob_pressao"] = df["sob_pressao"].fillna(False)

    # Remove distâncias impossíveis
    df["distancia_gol"] = pd.to_numeric(df["distancia_gol"], errors="coerce")
    df["angulo_gol"]    = pd.to_numeric(df["angulo_gol"], errors="coerce")
    df = df[df["distancia_gol"] <= 130]

    # Faixas de distância
    df["faixa_distancia"] = pd.cut(
        df["distancia_gol"],
        bins=[0, 6, 11, 18, 25, 35, 130],
        labels=["0-6m", "6-11m", "11-18m", "18-25m", "25-35m", "35m+"]
    )

    # Flag no alvo
    df["no_alvo"] = df["resultado"].isin(["Goal", "Saved"]).astype(int)
    df["foi_gol"] = pd.to_numeric(df["foi_gol"], errors="coerce").fillna(0).astype(int)

    gols = df["foi_gol"].sum()
    taxa = round(gols / len(df) * 100, 1)
    print(f"   ✅ {len(df):,} chutes válidos | {gols:,} gols | conversão: {taxa}%")
    print(f"   📊 Distância média: {df['distancia_gol'].mean():.1f}m")
    return df


# ─────────────────────────────────────────────────────────────
# 3. SILVER STATS
# ─────────────────────────────────────────────────────────────

def processar_silver_stats() -> pd.DataFrame:
    print("\n📊 Processando silver stats...")

    path = BRONZE_DIR / "stats.parquet"
    if not path.exists():
        print("   ❌ bronze/stats.parquet não encontrado!")
        return pd.DataFrame()

    df = pd.read_parquet(path)

    # Normaliza nomes
    for col in ["time_nome", "adversario"]:
        df[col] = df[col].apply(lambda x: normalizar_nome(x, NOMES_SELECOES))

    df["fase"] = df["fase"].apply(lambda x: normalizar_nome(x, NOMES_FASES))

    # Converte numéricos
    cols_numericas = [
        "passes_total", "passes_completos", "precisao_passes",
        "chutes_total", "chutes_alvo", "duelos_total", "duelos_ganhos",
        "posse_pct", "escanteios", "cartoes_amarelos",
        "cartoes_vermelhos", "faltas"
    ]
    for col in cols_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Métricas derivadas
    df["chutes_fora_alvo"]  = df["chutes_total"] - df["chutes_alvo"]
    df["duelos_perdidos"]   = df["duelos_total"] - df["duelos_ganhos"]
    df["pct_duelos_ganhos"] = (
        df["duelos_ganhos"] / df["duelos_total"].replace(0, np.nan) * 100
    ).round(1).fillna(0)

    # Peso por fonte
    df["peso_fonte"] = df["fonte"].map(PESOS_FONTE).fillna(0.3)

    # Remove duplicatas
    df = df.drop_duplicates(subset=["match_id", "time_nome"])

    print(f"   ✅ {len(df):,} registros | {df['time_nome'].nunique()} seleções")
    print(f"   📊 Copas: {len(df[df['fonte']=='statsbomb'])} | Eliminatórias: {len(df[df['fonte']!='statsbomb'])}")
    return df


# ─────────────────────────────────────────────────────────────
# 4. SILVER QUALIFIERS
# ─────────────────────────────────────────────────────────────

def processar_silver_qualifiers() -> pd.DataFrame:
    print("\n🏟️  Processando silver qualifiers...")

    path = BRONZE_DIR / "qualifiers.parquet"
    if not path.exists():
        print("   ⚠️  bronze/qualifiers.parquet não encontrado")
        return pd.DataFrame()

    df = pd.read_parquet(path)

    # Normaliza nomes
    for col in ["time_casa", "time_fora"]:
        df[col] = df[col].apply(lambda x: normalizar_nome(x, NOMES_SELECOES))

    # Trata nulos
    df["estadio"] = df["estadio"].fillna("Não informado")
    df["saldo_gols"] = df["gols_casa"] - df["gols_fora"]

    # Peso fixo para eliminatórias
    df["peso_fonte"] = 0.5

    # Calcula métricas por seleção nas eliminatórias
    times = pd.concat([
        df[["time_casa", "gols_casa", "gols_fora", "resultado_casa"]].rename(
            columns={"time_casa": "time", "gols_casa": "gols_pro", "gols_fora": "gols_contra", "resultado_casa": "resultado"}
        ),
        df[["time_fora", "gols_fora", "gols_casa"]].assign(
            resultado=df["resultado_casa"].map({"V": "D", "D": "V", "E": "E"})
        ).rename(
            columns={"time_fora": "time", "gols_fora": "gols_pro", "gols_casa": "gols_contra"}
        )
    ])

    selecoes = df["time_casa"].nunique() + df["time_fora"].nunique()
    print(f"   ✅ {len(df):,} partidas | {pd.concat([df['time_casa'], df['time_fora']]).nunique()} seleções")
    return df


# ─────────────────────────────────────────────────────────────
# 5. SALVAR
# ─────────────────────────────────────────────────────────────

def salvar_parquet(df: pd.DataFrame, nome: str):
    if df.empty:
        print(f"   ⚠️  {nome}.parquet — vazio, pulando")
        return
    path = SILVER_DIR / f"{nome}.parquet"

    # Converte object para string
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).replace("nan", None)

    df.to_parquet(path, index=False, engine="pyarrow")
    size_kb = path.stat().st_size / 1024
    print(f"   💾 {path} ({size_kb:.1f} KB | {len(df):,} linhas | {len(df.columns)} colunas)")


# ─────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────

def run():
    print_header()

    matches_df    = processar_silver_matches()
    shots_df      = processar_silver_shots()
    stats_df      = processar_silver_stats()
    qualifiers_df = processar_silver_qualifiers()

    print("\n💾 Salvando camada silver...")
    salvar_parquet(matches_df,    "matches")
    salvar_parquet(shots_df,      "shots")
    salvar_parquet(stats_df,      "stats")
    salvar_parquet(qualifiers_df, "qualifiers")

    print("\n" + "="*55)
    print("  Silver concluído!")
    print("="*55)
    print(f"""
  Arquivos gerados:
  ─────────────────────────────────────────────
  data/silver/
  ├── matches.parquet     → {len(matches_df):>6,} partidas normalizadas
  ├── shots.parquet       → {len(shots_df):>6,} chutes válidos
  ├── stats.parquet       → {len(stats_df):>6,} registros time×partida
  └── qualifiers.parquet  → {len(qualifiers_df):>6,} eliminatórias

  Próximo passo:
  ─────────────────────────────────────────────
  python src/pipeline/gold.py
    """)


if __name__ == "__main__":
    run()