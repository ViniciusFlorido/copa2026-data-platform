"""
src/pipeline/gold.py
───────────────────────────────────────────────────────────────
Camada Gold — cria tabelas analíticas e features de ML
a partir dos dados da camada silver.

Fontes incluídas:
    - StatsBomb (Copas 2018 e 2022)
    - Kaggle (Eliminatórias 2026)
    - API-Football (UEFA e CAF)

Arquivos gerados:
    data/gold/dim_selecoes.parquet      → dimensão de seleções
    data/gold/fct_partidas.parquet      → fatos das partidas
    data/gold/agg_selecao_copa.parquet  → métricas por seleção nas Copas
    data/gold/agg_selecao_quali.parquet → métricas por seleção nas eliminatórias
    data/gold/feat_xg.parquet           → features para modelo xG
    data/gold/feat_predicao.parquet     → features para modelo de predição

Como rodar:
    python src/pipeline/gold.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Caminhos ──────────────────────────────────────────────────
SILVER_DIR = Path("data/silver")
GOLD_DIR   = Path("data/gold")
GOLD_DIR.mkdir(parents=True, exist_ok=True)


def print_header():
    print("\n" + "="*55)
    print("  Pipeline Gold — Métricas e Features ML")
    print("="*55 + "\n")


# ─────────────────────────────────────────────────────────────
# 1. DIM_SELECOES
# ─────────────────────────────────────────────────────────────

def criar_dim_selecoes(matches_df: pd.DataFrame, qualifiers_df: pd.DataFrame) -> pd.DataFrame:
    """Dimensão de todas as seleções com continente e fonte."""
    print("🏳️  Criando dim_selecoes...")

    # Coleta times de todas as fontes
    times_matches = pd.concat([
        matches_df[["time_casa"]].rename(columns={"time_casa": "time_nome"}),
        matches_df[["time_fora"]].rename(columns={"time_fora": "time_nome"}),
    ])

    times_quali = pd.concat([
        qualifiers_df[["time_casa"]].rename(columns={"time_casa": "time_nome"}),
        qualifiers_df[["time_fora"]].rename(columns={"time_fora": "time_nome"}),
    ]) if not qualifiers_df.empty else pd.DataFrame(columns=["time_nome"])

    dim = pd.concat([times_matches, times_quali]).drop_duplicates()
    dim = dim[dim["time_nome"].notna() & (dim["time_nome"] != "nan")]
    dim = dim.sort_values("time_nome").reset_index(drop=True)

    continentes = {
        # América do Sul
        "Brasil": "América do Sul", "Argentina": "América do Sul",
        "Uruguai": "América do Sul", "Colômbia": "América do Sul",
        "Chile": "América do Sul", "Peru": "América do Sul",
        "Equador": "América do Sul", "Paraguai": "América do Sul",
        "Bolívia": "América do Sul", "Venezuela": "América do Sul",
        "Guiana": "América do Sul", "Suriname": "América do Sul",

        # Europa
        "França": "Europa", "Alemanha": "Europa", "Espanha": "Europa",
        "Inglaterra": "Europa", "Portugal": "Europa", "Holanda": "Europa",
        "Bélgica": "Europa", "Croácia": "Europa", "Suíça": "Europa",
        "Dinamarca": "Europa", "Suécia": "Europa", "Polônia": "Europa",
        "Sérvia": "Europa", "Rússia": "Europa", "Ucrânia": "Europa",
        "República Tcheca": "Europa", "Eslováquia": "Europa",
        "Hungria": "Europa", "Romênia": "Europa", "Grécia": "Europa",
        "Turquia": "Europa", "Áustria": "Europa", "País de Gales": "Europa",
        "Escócia": "Europa", "Noruega": "Europa", "Finlândia": "Europa",
        "Islândia": "Europa", "Albânia": "Europa", "Eslovênia": "Europa",
        "Bósnia e Herzegovina": "Europa", "Macedônia do Norte": "Europa",
        "Montenegro": "Europa", "Kosovo": "Europa", "Bulgária": "Europa",
        "Bielorrússia": "Europa", "Geórgia": "Europa", "Armênia": "Europa",
        "Azerbaijão": "Europa", "Cazaquistão": "Europa",
        "Irlanda": "Europa", "República da Irlanda": "Europa",
        "Irlanda do Norte": "Europa", "Israel": "Europa",
        "Luxemburgo": "Europa", "Lituânia": "Europa",
        "Letônia": "Europa", "Estônia": "Europa", "Moldávia": "Europa",
        "Gibraltar": "Europa", "Andorra": "Europa", "Chipre": "Europa",
        "Malta": "Europa", "Ilhas Faroé": "Europa",

        # América do Norte e Central
        "Estados Unidos": "América do Norte", "México": "América do Norte",
        "Canadá": "América do Norte", "Costa Rica": "América Central",
        "Panamá": "América Central", "Honduras": "América Central",
        "Jamaica": "Caribe", "Trinidad e Tobago": "Caribe",
        "El Salvador": "América Central", "Haiti": "Caribe",
        "Cuba": "Caribe", "Guatemala": "América Central",
        "Nicarágua": "América Central", "Barbados": "Caribe",
        "República Dominicana": "Caribe", "Bermudas": "Caribe",
        "Curaçao": "Caribe", "Bahamas": "Caribe",

        # África
        "Marrocos": "África", "Senegal": "África", "Gana": "África",
        "Nigéria": "África", "Camarões": "África", "Egito": "África",
        "Argélia": "África", "Tunísia": "África", "África do Sul": "África",
        "Costa do Marfim": "África", "Mali": "África",
        "Burkina Faso": "África", "Guiné": "África",
        "República Democrática do Congo": "África", "Congo": "África",
        "Zâmbia": "África", "Tanzânia": "África", "Uganda": "África",
        "Quênia": "África", "Angola": "África", "Moçambique": "África",
        "Zimbábue": "África", "Botsuana": "África", "Namíbia": "África",
        "Ruanda": "África", "Etiópia": "África", "Benin": "África",
        "Gabão": "África", "Sudão": "África", "Líbia": "África",
        "Togo": "África", "Cabo Verde": "África",
        "Guiné Equatorial": "África", "Guiné-Bissau": "África",
        "Serra Leoa": "África", "Libéria": "África",
        "Comores": "África", "Burundi": "África",

        # Ásia
        "Japão": "Ásia", "Coreia do Sul": "Ásia", "Irã": "Ásia",
        "Arábia Saudita": "Ásia", "Catar": "Ásia", "China": "Ásia",
        "Iraque": "Ásia", "Emirados Árabes": "Ásia",
        "Uzbequistão": "Ásia", "Tailândia": "Ásia", "Vietnã": "Ásia",
        "Índia": "Ásia", "Indonésia": "Ásia", "Bahrein": "Ásia",
        "Jordânia": "Ásia", "Omã": "Ásia", "Kuwait": "Ásia",
        "Síria": "Ásia", "Líbano": "Ásia", "Palestina": "Ásia",
        "Afeganistão": "Ásia", "Bangladesh": "Ásia",
        "Paquistão": "Ásia", "Nepal": "Ásia", "Sri Lanka": "Ásia",
        "Mianmar": "Ásia", "Camboja": "Ásia", "Malásia": "Ásia",
        "Filipinas": "Ásia", "Singapura": "Ásia",
        "Mongólia": "Ásia", "Coreia do Norte": "Ásia",
        "Quirguistão": "Ásia", "Tajiquistão": "Ásia",
        "Turcomenistão": "Ásia", "Maldivas": "Ásia",

        # Oceania
        "Austrália": "Oceania", "Nova Zelândia": "Oceania",
        "Fiji": "Oceania", "Papua Nova Guiné": "Oceania",
        "Ilhas Salomão": "Oceania", "Vanuatu": "Oceania",
        "Samoa": "Oceania", "Taiti": "Oceania",
        "Nova Caledônia": "Oceania",
    }

    dim["continente"] = dim["time_nome"].map(continentes).fillna("Outro")

    print(f"   ✅ {len(dim)} seleções | {dim['continente'].nunique()} continentes")
    return dim


# ─────────────────────────────────────────────────────────────
# 2. FCT_PARTIDAS
# ─────────────────────────────────────────────────────────────

def criar_fct_partidas(matches_df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    """Fatos das partidas combinando informações gerais com estatísticas."""
    print("\n📋 Criando fct_partidas...")

    stats_casa = stats_df[stats_df["eh_casa"] == True].copy()
    stats_fora = stats_df[stats_df["eh_casa"] == False].copy()

    fct = matches_df.merge(
        stats_casa[["match_id", "passes_total", "passes_completos",
                    "precisao_passes", "chutes_total", "chutes_alvo",
                    "pressoes", "duelos_ganhos", "posse_pct"]].rename(
            columns={c: f"{c}_casa" for c in ["passes_total", "passes_completos",
                    "precisao_passes", "chutes_total", "chutes_alvo",
                    "pressoes", "duelos_ganhos", "posse_pct"]}
        ),
        on="match_id", how="left"
    ).merge(
        stats_fora[["match_id", "passes_total", "passes_completos",
                    "precisao_passes", "chutes_total", "chutes_alvo",
                    "pressoes", "duelos_ganhos", "posse_pct"]].rename(
            columns={c: f"{c}_fora" for c in ["passes_total", "passes_completos",
                    "precisao_passes", "chutes_total", "chutes_alvo",
                    "pressoes", "duelos_ganhos", "posse_pct"]}
        ),
        on="match_id", how="left"
    )

    # Calcula posse estimada se não disponível
    mask_sem_posse = fct["posse_pct_casa"].isna()
    total_passes   = fct["passes_total_casa"] + fct["passes_total_fora"]
    fct.loc[mask_sem_posse, "posse_pct_casa"] = (
        fct.loc[mask_sem_posse, "passes_total_casa"] /
        total_passes.loc[mask_sem_posse] * 100
    ).round(1)
    fct.loc[mask_sem_posse, "posse_pct_fora"] = (
        fct.loc[mask_sem_posse, "passes_total_fora"] /
        total_passes.loc[mask_sem_posse] * 100
    ).round(1)

    # Preenche nulos com 0
    cols_num = [c for c in fct.columns if any(
        c.endswith(s) for s in ["_casa", "_fora"]
    ) and c not in ["time_casa", "time_fora"]]
    fct[cols_num] = fct[cols_num].fillna(0)

    print(f"   ✅ {len(fct):,} partidas | {len(fct.columns)} colunas")
    return fct


# ─────────────────────────────────────────────────────────────
# 3. AGG_SELECAO_COPA
# ─────────────────────────────────────────────────────────────

def criar_agg_selecao_copa(matches_df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    """Métricas agregadas por seleção nas Copas do Mundo."""
    print("\n📊 Criando agg_selecao_copa...")

    # Filtra só Copas
    matches_copa = matches_df[matches_df["fonte"] == "statsbomb"]
    stats_copa   = stats_df[stats_df["fonte"] == "statsbomb"]

    rows = []
    for ano in [2018, 2022]:
        matches_ano = matches_copa[matches_copa["ano"] == ano]
        stats_ano   = stats_copa[stats_copa["ano"] == ano]

        for time in stats_ano["time_nome"].unique():
            s = stats_ano[stats_ano["time_nome"] == time]

            jogos    = len(s)
            vitorias = len(s[s["resultado"] == "V"])
            empates  = len(s[s["resultado"] == "E"])
            derrotas = len(s[s["resultado"] == "D"])
            pontos   = vitorias * 3 + empates
            gols_pro = int(s["gols_marcados"].sum())
            gols_con = int(s["gols_sofridos"].sum())

            rows.append({
                "ano":                   ano,
                "time_nome":             time,
                "jogos":                 jogos,
                "vitorias":              vitorias,
                "empates":               empates,
                "derrotas":              derrotas,
                "pontos":                pontos,
                "gols_pro":              gols_pro,
                "gols_contra":           gols_con,
                "saldo_gols":            gols_pro - gols_con,
                "aproveitamento":        round(pontos / (jogos * 3) * 100, 1),
                "passes_media":          round(pd.to_numeric(s["passes_total"], errors="coerce").mean(), 1),
                "precisao_passes_media": round(pd.to_numeric(s["precisao_passes"], errors="coerce").mean(), 1),
                "chutes_media":          round(pd.to_numeric(s["chutes_total"], errors="coerce").mean(), 1),
                "chutes_alvo_media":     round(pd.to_numeric(s["chutes_alvo"], errors="coerce").mean(), 1),
                "pressoes_media":        round(pd.to_numeric(s["pressoes"], errors="coerce").mean(), 1),
                "duelos_ganhos_media":   round(pd.to_numeric(s["duelos_ganhos"], errors="coerce").mean(), 1),
            })

    df = pd.DataFrame(rows)
    print(f"   ✅ {len(df)} registros | {df['time_nome'].nunique()} seleções")
    return df


# ─────────────────────────────────────────────────────────────
# 4. AGG_SELECAO_QUALI
# ─────────────────────────────────────────────────────────────

def criar_agg_selecao_quali(qualifiers_df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    """Métricas agregadas por seleção nas eliminatórias."""
    print("\n📊 Criando agg_selecao_quali...")

    if qualifiers_df.empty:
        print("   ⚠️  Sem dados de eliminatórias")
        return pd.DataFrame()

    stats_quali = stats_df[stats_df["fonte"].isin(["kaggle", "api_football"])]

    rows = []
    times = pd.concat([
        qualifiers_df["time_casa"],
        qualifiers_df["time_fora"]
    ]).unique()

    for time in times:
        if pd.isna(time) or time == "nan":
            continue

        s = stats_quali[stats_quali["time_nome"] == time]

        if s.empty:
            # Calcula só com resultados se não tiver stats detalhadas
            partidas_casa = qualifiers_df[qualifiers_df["time_casa"] == time]
            partidas_fora = qualifiers_df[qualifiers_df["time_fora"] == time]

            jogos    = len(partidas_casa) + len(partidas_fora)
            vitorias = (
                len(partidas_casa[partidas_casa["resultado_casa"] == "V"]) +
                len(partidas_fora[partidas_fora["resultado_casa"] == "D"])
            )
            empates  = (
                len(partidas_casa[partidas_casa["resultado_casa"] == "E"]) +
                len(partidas_fora[partidas_fora["resultado_casa"] == "E"])
            )
            derrotas = jogos - vitorias - empates
            gols_pro = (
                int(partidas_casa["gols_casa"].sum()) +
                int(partidas_fora["gols_fora"].sum())
            )
            gols_con = (
                int(partidas_casa["gols_fora"].sum()) +
                int(partidas_fora["gols_casa"].sum())
            )
        else:
            jogos    = len(s)
            vitorias = len(s[s["resultado"] == "V"])
            empates  = len(s[s["resultado"] == "E"])
            derrotas = len(s[s["resultado"] == "D"])
            gols_pro = int(s["gols_marcados"].sum())
            gols_con = int(s["gols_sofridos"].sum())

        pontos = vitorias * 3 + empates

        rows.append({
            "time_nome":             time,
            "jogos_quali":           jogos,
            "vitorias_quali":        vitorias,
            "empates_quali":         empates,
            "derrotas_quali":        derrotas,
            "pontos_quali":          pontos,
            "gols_pro_quali":        gols_pro,
            "gols_contra_quali":     gols_con,
            "saldo_gols_quali":      gols_pro - gols_con,
            "aproveitamento_quali":  round(pontos / (jogos * 3) * 100, 1) if jogos > 0 else 0,
            "passes_media_quali":    round(s["passes_total"].mean(), 1) if not s.empty else 0,
            "precisao_passes_quali": round(s["precisao_passes"].mean(), 1) if not s.empty else 0,
            "chutes_media_quali":    round(s["chutes_total"].mean(), 1) if not s.empty else 0,
            "posse_media_quali":     round(pd.to_numeric(s["posse_pct"], errors="coerce").mean(), 1) if not s.empty else 0,
        })

    df = pd.DataFrame(rows)
    print(f"   ✅ {len(df)} seleções com dados de eliminatórias")
    return df


# ─────────────────────────────────────────────────────────────
# 5. FEAT_XG
# ─────────────────────────────────────────────────────────────

def criar_feat_xg(shots_df: pd.DataFrame) -> pd.DataFrame:
    """Features para treinar o modelo de xG."""
    print("\n🤖 Criando feat_xg...")

    df = shots_df.copy()

    # Features numéricas
    features_num = ["distancia_gol", "angulo_gol", "minuto", "periodo"]
    for col in features_num:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # One-hot encoding
    df = pd.get_dummies(df, columns=["parte_corpo", "tipo_jogada"],
                        prefix=["corpo", "jogada"])

    # Converte boolean
    df["sob_pressao"] = pd.to_numeric(df["sob_pressao"], errors="coerce").fillna(0).astype(int)

    # Seleciona features
    feat_cols = (
        features_num +
        ["sob_pressao"] +
        [c for c in df.columns if c.startswith("corpo_") or c.startswith("jogada_")]
    )

    feat_df = df[feat_cols + ["foi_gol", "xg_statsbomb", "match_id", "ano", "fase"]].copy()
    feat_df = feat_df.dropna(subset=features_num)
    feat_df["foi_gol"] = pd.to_numeric(feat_df["foi_gol"], errors="coerce").fillna(0).astype(int)

    print(f"   ✅ {len(feat_df):,} amostras | {len(feat_cols)} features")
    print(f"   📊 Balanceamento: {feat_df['foi_gol'].mean()*100:.1f}% gols")
    return feat_df


# ─────────────────────────────────────────────────────────────
# 6. FEAT_PREDICAO
# ─────────────────────────────────────────────────────────────

def criar_feat_predicao(
    matches_df: pd.DataFrame,
    agg_copa: pd.DataFrame,
    agg_quali: pd.DataFrame
) -> pd.DataFrame:
    """
    Features para o modelo de predição de resultado.
    Usa dados das Copas anteriores + eliminatórias como contexto.
    Aplica pesos por fonte conforme proximidade da competição.
    """
    print("\n🤖 Criando feat_predicao...")

    fases_cod = {
        "Fase de Grupos": 1, "Oitavas de Final": 2,
        "Quartas de Final": 3, "Semifinal": 4,
        "Disputa de 3º Lugar": 5, "Final": 6
    }
    resultado_cod = {"V": 1, "E": 0, "D": -1}

    rows = []

    for _, match in matches_df[matches_df["fonte"] == "statsbomb"].iterrows():
        casa = match["time_casa"]
        fora = match["time_fora"]
        ano  = match["ano"]

        # Busca dados de Copa
        sc_copa = agg_copa[(agg_copa["time_nome"] == casa) & (agg_copa["ano"] == ano)]
        sf_copa = agg_copa[(agg_copa["time_nome"] == fora) & (agg_copa["ano"] == ano)]

        # Busca dados de eliminatórias
        sc_quali = agg_quali[agg_quali["time_nome"] == casa]
        sf_quali = agg_quali[agg_quali["time_nome"] == fora]

        if sc_copa.empty or sf_copa.empty:
            continue

        sc = sc_copa.iloc[0]
        sf = sf_copa.iloc[0]

        # Aproveitamento nas eliminatórias (se disponível)
        aprov_quali_casa = sc_quali.iloc[0]["aproveitamento_quali"] if not sc_quali.empty else 50.0
        aprov_quali_fora = sf_quali.iloc[0]["aproveitamento_quali"] if not sf_quali.empty else 50.0

        rows.append({
            "match_id":                  match["match_id"],
            "ano":                       ano,
            "fase_cod":                  fases_cod.get(match["fase"], 1),

            # Features Copa (peso 0.8)
            "casa_aproveitamento_copa":  sc["aproveitamento"],
            "casa_gols_pro_media_copa":  sc["gols_pro"] / max(sc["jogos"], 1),
            "casa_gols_con_media_copa":  sc["gols_contra"] / max(sc["jogos"], 1),
            "casa_chutes_media_copa":    sc["chutes_media"],
            "casa_precisao_passes_copa": sc["precisao_passes_media"],
            "casa_pressoes_media_copa":  sc["pressoes_media"],

            "fora_aproveitamento_copa":  sf["aproveitamento"],
            "fora_gols_pro_media_copa":  sf["gols_pro"] / max(sf["jogos"], 1),
            "fora_gols_con_media_copa":  sf["gols_contra"] / max(sf["jogos"], 1),
            "fora_chutes_media_copa":    sf["chutes_media"],
            "fora_precisao_passes_copa": sf["precisao_passes_media"],
            "fora_pressoes_media_copa":  sf["pressoes_media"],

            # Features eliminatórias (peso 0.5)
            "casa_aproveitamento_quali": aprov_quali_casa,
            "fora_aproveitamento_quali": aprov_quali_fora,

            # Diferenças
            "diff_aproveitamento_copa":  sc["aproveitamento"] - sf["aproveitamento"],
            "diff_gols_pro_copa":        sc["gols_pro"] - sf["gols_pro"],
            "diff_chutes_copa":          sc["chutes_media"] - sf["chutes_media"],
            "diff_aproveitamento_quali": aprov_quali_casa - aprov_quali_fora,

            # Variável alvo
            "resultado": resultado_cod.get(match["resultado_casa"], 0),
        })

    df = pd.DataFrame(rows)
    print(f"   ✅ {len(df):,} partidas | {len(df.columns)-3} features")
    return df


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _fase_maxima(matches_df: pd.DataFrame, time: str) -> str:
    ordem = {
        "Fase de Grupos": 1, "Oitavas de Final": 2,
        "Quartas de Final": 3, "Semifinal": 4,
        "Disputa de 3º Lugar": 5, "Final": 6,
    }
    partidas = matches_df[
        (matches_df["time_casa"] == time) |
        (matches_df["time_fora"] == time)
    ]
    if partidas.empty:
        return "Fase de Grupos"
    fases = partidas["fase"].unique()
    return sorted(fases, key=lambda x: ordem.get(x, 0), reverse=True)[0]


def salvar_parquet(df: pd.DataFrame, nome: str):
    if df.empty:
        print(f"   ⚠️  {nome}.parquet — vazio, pulando")
        return
    path = GOLD_DIR / f"{nome}.parquet"
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).replace("nan", None)
    df.to_parquet(path, index=False, engine="pyarrow")
    size_kb = path.stat().st_size / 1024
    print(f"   💾 {path} ({size_kb:.1f} KB | {len(df):,} linhas | {len(df.columns)} colunas)")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run():
    print_header()

    # Lê camada silver
    print("📂 Lendo camada silver...")
    matches_df    = pd.read_parquet(SILVER_DIR / "matches.parquet")
    shots_df      = pd.read_parquet(SILVER_DIR / "shots.parquet")
    stats_df      = pd.read_parquet(SILVER_DIR / "stats.parquet")
    qualifiers_df = pd.read_parquet(SILVER_DIR / "qualifiers.parquet")
    print(f"   ✅ {len(matches_df):,} partidas | {len(shots_df):,} chutes | {len(stats_df):,} stats | {len(qualifiers_df):,} eliminatórias\n")

    # Cria tabelas gold
    dim_selecoes  = criar_dim_selecoes(matches_df, qualifiers_df)
    fct_partidas  = criar_fct_partidas(matches_df, stats_df)
    agg_copa      = criar_agg_selecao_copa(matches_df, stats_df)
    agg_quali     = criar_agg_selecao_quali(qualifiers_df, stats_df)
    feat_xg       = criar_feat_xg(shots_df)
    feat_predicao = criar_feat_predicao(matches_df, agg_copa, agg_quali)

    # Salva
    print("\n💾 Salvando camada gold...")
    salvar_parquet(dim_selecoes,  "dim_selecoes")
    salvar_parquet(fct_partidas,  "fct_partidas")
    salvar_parquet(agg_copa,      "agg_selecao_copa")
    salvar_parquet(agg_quali,     "agg_selecao_quali")
    salvar_parquet(feat_xg,       "feat_xg")
    salvar_parquet(feat_predicao, "feat_predicao")

    print("\n" + "="*55)
    print("  Gold concluído!")
    print("="*55)
    print(f"""
  Arquivos gerados:
  ─────────────────────────────────────────────
  data/gold/
  ├── dim_selecoes.parquet      → {len(dim_selecoes):>4} seleções
  ├── fct_partidas.parquet      → {len(fct_partidas):>4,} partidas
  ├── agg_selecao_copa.parquet  → {len(agg_copa):>4} registros (Copas)
  ├── agg_selecao_quali.parquet → {len(agg_quali):>4} registros (Eliminatórias)
  ├── feat_xg.parquet           → {len(feat_xg):>4,} amostras modelo xG
  └── feat_predicao.parquet     → {len(feat_predicao):>4,} amostras predição

  Próximo passo:
  ─────────────────────────────────────────────
  python src/models/xg_model.py
    """)


if __name__ == "__main__":
    run()
