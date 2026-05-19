"""
src/pipeline/bronze.py
───────────────────────────────────────────────────────────────
Camada Bronze — transforma os JSONs brutos em Parquet tipado.

Fontes processadas:
    1. StatsBomb (Copas 2018 e 2022) — partidas, chutes, stats
    2. Kaggle (Eliminatórias 2026)   — partidas e stats
    3. API-Football (UEFA e CAF)     — partidas e stats

Arquivos gerados:
    data/bronze/matches.parquet      → todas as partidas
    data/bronze/shots.parquet        → chutes (modelo xG)
    data/bronze/stats.parquet        → estatísticas por time/partida
    data/bronze/qualifiers.parquet   → eliminatórias consolidadas

Como rodar:
    python src/pipeline/bronze.py
"""

import json
import pandas as pd
from pathlib import Path

# ── Caminhos ──────────────────────────────────────────────────
RAW_DIR    = Path("data/raw")
BRONZE_DIR = Path("data/bronze")
BRONZE_DIR.mkdir(parents=True, exist_ok=True)


def print_header():
    print("\n" + "="*55)
    print("  Pipeline Bronze — todas as fontes → Parquet")
    print("="*55 + "\n")


# ─────────────────────────────────────────────────────────────
# PARTE 1 — STATSBOMB (Copas 2018 e 2022)
# ─────────────────────────────────────────────────────────────

def processar_matches_statsbomb() -> pd.DataFrame:
    """Processa partidas das Copas 2018 e 2022 da StatsBomb."""
    print("📋 [StatsBomb] Processando partidas...")

    arquivos = [
        (RAW_DIR / "statsbomb/matches_2022.json", 2022),
        (RAW_DIR / "statsbomb/matches_2018.json", 2018),
    ]

    rows = []
    for arquivo, ano in arquivos:
        if not arquivo.exists():
            print(f"   ⚠️  {arquivo} não encontrado")
            continue

        with open(arquivo, encoding="utf-8") as f:
            matches = json.load(f)

        for m in matches:
            rows.append({
                "match_id":       m["match_id"],
                "fonte":          "statsbomb",
                "competicao":     "Copa do Mundo",
                "ano":            ano,
                "data":           m["match_date"],
                "time_casa_id":   m["home_team"]["home_team_id"],
                "time_casa":      m["home_team"]["home_team_name"],
                "time_fora_id":   m["away_team"]["away_team_id"],
                "time_fora":      m["away_team"]["away_team_name"],
                "gols_casa":      int(m["home_score"]),
                "gols_fora":      int(m["away_score"]),
                "resultado_casa": _resultado(m["home_score"], m["away_score"]),
                "fase":           m["competition_stage"]["name"],
                "estadio":        m.get("stadium", {}).get("name"),
            })

        print(f"   ✅ Copa {ano}: {len(matches)} partidas")

    df = pd.DataFrame(rows)
    df["data"] = pd.to_datetime(df["data"])
    return df


def processar_shots_statsbomb(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Extrai chutes de cada partida da StatsBomb para o modelo xG."""
    print("\n⚽ [StatsBomb] Processando chutes...")

    events_dir = RAW_DIR / "statsbomb/events"
    rows       = []

    for _, match in matches_df[matches_df["fonte"] == "statsbomb"].iterrows():
        match_id   = match["match_id"]
        event_file = events_dir / f"{match_id}.json"

        if not event_file.exists():
            continue

        with open(event_file, encoding="utf-8") as f:
            events = json.load(f)

        for shot in [e for e in events if e["type"]["name"] == "Shot"]:
            loc      = shot.get("location", [None, None])
            shot_det = shot.get("shot", {})

            rows.append({
                "match_id":      match_id,
                "fonte":         "statsbomb",
                "ano":           match["ano"],
                "fase":          match["fase"],
                "time_casa":     match["time_casa"],
                "time_fora":     match["time_fora"],
                "time_nome":     shot.get("team", {}).get("name"),
                "jogador_nome":  shot.get("player", {}).get("name"),
                "loc_x":         float(loc[0]) if loc[0] else None,
                "loc_y":         float(loc[1]) if loc[1] else None,
                "parte_corpo":   shot_det.get("body_part", {}).get("name"),
                "tipo_jogada":   shot_det.get("type", {}).get("name"),
                "tecnica":       shot_det.get("technique", {}).get("name"),
                "sob_pressao":   bool(shot.get("under_pressure", False)),
                "resultado":     shot_det.get("outcome", {}).get("name"),
                "foi_gol":       1 if shot_det.get("outcome", {}).get("name") == "Goal" else 0,
                "xg_statsbomb":  float(shot_det["statsbomb_xg"]) if "statsbomb_xg" in shot_det else None,
                "minuto":        int(shot.get("minute", 0)),
                "periodo":       int(shot.get("period", 1)),
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        df["distancia_gol"] = ((df["loc_x"] - 120) ** 2 + (df["loc_y"] - 40) ** 2) ** 0.5
        df["angulo_gol"]    = (df["loc_y"] - 40).abs() / (120 - df["loc_x"] + 0.0001)
        df["foi_gol"]       = df["foi_gol"].astype(int)

    gols = df["foi_gol"].sum()
    print(f"   ✅ {len(df):,} chutes | {gols:,} gols | conversão: {gols/len(df)*100:.1f}%")
    return df


def processar_stats_statsbomb(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula estatísticas por time por partida da StatsBomb."""
    print("\n📊 [StatsBomb] Processando estatísticas...")

    events_dir = RAW_DIR / "statsbomb/events"
    rows       = []

    for _, match in matches_df[matches_df["fonte"] == "statsbomb"].iterrows():
        match_id   = match["match_id"]
        event_file = events_dir / f"{match_id}.json"

        if not event_file.exists():
            continue

        with open(event_file, encoding="utf-8") as f:
            events = json.load(f)

        times = {
            match["time_casa_id"]: match["time_casa"],
            match["time_fora_id"]: match["time_fora"],
        }

        for time_id, time_nome in times.items():
            ev = [e for e in events if e.get("team", {}).get("id") == time_id]

            passes      = [e for e in ev if e["type"]["name"] == "Pass"]
            passes_ok   = [p for p in passes if p.get("pass", {}).get("outcome") is None]
            chutes      = [e for e in ev if e["type"]["name"] == "Shot"]
            chutes_alvo = [c for c in chutes if c.get("shot", {}).get("outcome", {}).get("name") in ["Goal", "Saved"]]
            pressoes    = [e for e in ev if e["type"]["name"] == "Pressure"]
            duelos      = [e for e in ev if e["type"]["name"] == "Duel"]
            duelos_g    = [d for d in duelos if d.get("duel", {}).get("outcome", {}).get("name") in ["Won", "Success"]]

            eh_casa = time_nome == match["time_casa"]

            rows.append({
                "match_id":          match_id,
                "fonte":             "statsbomb",
                "competicao":        "Copa do Mundo",
                "ano":               match["ano"],
                "fase":              match["fase"],
                "time_id":           time_id,
                "time_nome":         time_nome,
                "eh_casa":           eh_casa,
                "adversario":        match["time_fora"] if eh_casa else match["time_casa"],
                "gols_marcados":     match["gols_casa"] if eh_casa else match["gols_fora"],
                "gols_sofridos":     match["gols_fora"] if eh_casa else match["gols_casa"],
                "resultado":         match["resultado_casa"] if eh_casa else _inverter(match["resultado_casa"]),
                "passes_total":      len(passes),
                "passes_completos":  len(passes_ok),
                "precisao_passes":   round(len(passes_ok) / len(passes) * 100, 1) if passes else 0.0,
                "chutes_total":      len(chutes),
                "chutes_alvo":       len(chutes_alvo),
                "pressoes":          len(pressoes),
                "duelos_total":      len(duelos),
                "duelos_ganhos":     len(duelos_g),
                # Campos não disponíveis na StatsBomb
                "posse_pct":         None,
                "escanteios":        None,
                "cartoes_amarelos":  None,
                "cartoes_vermelhos": None,
                "faltas":            None,
            })

    df = pd.DataFrame(rows)
    print(f"   ✅ {len(df):,} registros | média passes/jogo: {df['passes_total'].mean():.0f}")
    return df


# ─────────────────────────────────────────────────────────────
# PARTE 2 — KAGGLE (Eliminatórias 2026)
# ─────────────────────────────────────────────────────────────

def processar_qualifiers_kaggle() -> pd.DataFrame:
    """Processa eliminatórias Copa 2026 do dataset Kaggle."""
    print("\n📋 [Kaggle] Processando eliminatórias...")

    path = RAW_DIR / "qualifiers/all_qualifiers_2026.csv"
    if not path.exists():
        print("   ⚠️  all_qualifiers_2026.csv não encontrado")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)

    # Filtra só jogos finalizados
    df = df[df["result"].isin(["H", "A", "D"])].copy()

    # Mapeia resultado para V/E/D
    resultado_map = {"H": "V", "D": "E", "A": "D"}

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "match_id":          f"kaggle_{row['match_id']}",
            "fonte":             "kaggle",
            "competicao":        row["league_division"],
            "ano":               2026,
            "data":              row["date"],
            "time_casa_id":      None,
            "time_casa":         row["home_team"],
            "time_fora_id":      None,
            "time_fora":         row["away_team"],
            "gols_casa":         int(row["home_goals"]) if pd.notna(row["home_goals"]) else 0,
            "gols_fora":         int(row["away_goals"]) if pd.notna(row["away_goals"]) else 0,
            "resultado_casa":    resultado_map.get(row["result"], "E"),
            "fase":              row.get("round", "Qualificação"),
            "estadio":           row.get("stadium"),
        })

    matches_df = pd.DataFrame(rows)
    matches_df["data"] = pd.to_datetime(matches_df["data"], errors="coerce")

    print(f"   ✅ {len(matches_df):,} partidas | {matches_df['time_casa'].nunique()} seleções únicas")
    return matches_df


def processar_stats_kaggle(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Extrai estatísticas das eliminatórias do Kaggle."""
    print("\n📊 [Kaggle] Processando estatísticas...")

    path = RAW_DIR / "qualifiers/all_qualifiers_2026.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    df = df[df["result"].isin(["H", "A", "D"])].copy()

    resultado_map = {"H": "V", "D": "E", "A": "D"}
    rows = []

    for _, row in df.iterrows():
        match_id = f"kaggle_{row['match_id']}"
        fase     = row.get("round", "Qualificação")

        for lado in ["home", "away"]:
            eh_casa    = lado == "home"
            time_nome  = row[f"{lado}_team"]
            adversario = row["away_team"] if eh_casa else row["home_team"]
            resultado  = resultado_map.get(row["result"], "E")
            if not eh_casa:
                resultado = _inverter(resultado)

            rows.append({
                "match_id":          match_id,
                "fonte":             "kaggle",
                "competicao":        row["league_division"],
                "ano":               2026,
                "fase":              fase,
                "time_id":           None,
                "time_nome":         time_nome,
                "eh_casa":           eh_casa,
                "adversario":        adversario,
                "gols_marcados":     _safe_int(row[f"{lado}_goals"]),
                "gols_sofridos":     _safe_int(row["away_goals" if eh_casa else "home_goals"]),
                "resultado":         resultado,
                "passes_total":      _safe_int(row.get(f"{lado}_passes_total")),
                "passes_completos":  _safe_int(row.get(f"{lado}_passes_successful")),
                "precisao_passes":   _safe_float(row.get(f"{lado}_passes_pct")),
                "chutes_total":      _safe_int(row.get(f"{lado}_shots_total")),
                "chutes_alvo":       _safe_int(row.get(f"{lado}_shots_on_target")),
                "pressoes":          None,
                "duelos_total":      _safe_int(row.get(f"{lado}_duels_won")),
                "duelos_ganhos":     _safe_int(row.get(f"{lado}_duels_won")),
                "posse_pct":         _safe_float(row.get(f"{lado}_possession_pct")),
                "escanteios":        _safe_int(row.get(f"{lado}_corners")),
                "cartoes_amarelos":  _safe_int(row.get(f"{lado}_yellow_cards")),
                "cartoes_vermelhos": _safe_int(row.get(f"{lado}_red_cards")),
                "faltas":            _safe_int(row.get(f"{lado}_fouls_committed")),
            })

    df_stats = pd.DataFrame(rows)
    print(f"   ✅ {len(df_stats):,} registros | {df_stats['time_nome'].nunique()} seleções")
    return df_stats


# ─────────────────────────────────────────────────────────────
# PARTE 3 — API-FOOTBALL (UEFA e CAF)
# ─────────────────────────────────────────────────────────────

def processar_qualifiers_api() -> tuple:
    """Processa eliminatórias UEFA e CAF da API-Football."""
    print("\n📋 [API-Football] Processando UEFA e CAF...")

    confederacoes = [
        ("UEFA", RAW_DIR / "qualifiers/uefa.json", 2024),
        ("CAF",  RAW_DIR / "qualifiers/caf.json",  2023),
    ]

    all_matches = []
    all_stats   = []

    for nome, path, ano in confederacoes:
        if not path.exists():
            print(f"   ⚠️  {nome}: arquivo não encontrado")
            continue

        with open(path, encoding="utf-8") as f:
            fixtures = json.load(f)

        finalizados = [f for f in fixtures if f["fixture"]["status"]["short"] == "FT"]

        for fix in finalizados:
            match_id   = f"api_{fix['fixture']['id']}"
            home       = fix["teams"]["home"]["name"]
            away       = fix["teams"]["away"]["name"]
            gols_casa  = fix["goals"]["home"] or 0
            gols_fora  = fix["goals"]["away"] or 0
            resultado  = _resultado(gols_casa, gols_fora)

            all_matches.append({
                "match_id":       match_id,
                "fonte":          "api_football",
                "competicao":     f"Eliminatórias {nome}",
                "ano":            ano,
                "data":           fix["fixture"]["date"],
                "time_casa_id":   fix["teams"]["home"]["id"],
                "time_casa":      home,
                "time_fora_id":   fix["teams"]["away"]["id"],
                "time_fora":      away,
                "gols_casa":      int(gols_casa),
                "gols_fora":      int(gols_fora),
                "resultado_casa": resultado,
                "fase":           fix.get("league", {}).get("round", "Qualificação"),
                "estadio":        fix.get("fixture", {}).get("venue", {}).get("name"),
            })

            # Stats básicas disponíveis no fixture
            for lado, eh_casa in [("home", True), ("away", False)]:
                time_nome  = home if eh_casa else away
                adversario = away if eh_casa else home
                gols_m     = gols_casa if eh_casa else gols_fora
                gols_s     = gols_fora if eh_casa else gols_casa
                res        = resultado if eh_casa else _inverter(resultado)

                all_stats.append({
                    "match_id":          match_id,
                    "fonte":             "api_football",
                    "competicao":        f"Eliminatórias {nome}",
                    "ano":               ano,
                    "fase":              fix.get("league", {}).get("round", "Qualificação"),
                    "time_id":           fix["teams"][lado]["id"],
                    "time_nome":         time_nome,
                    "eh_casa":           eh_casa,
                    "adversario":        adversario,
                    "gols_marcados":     int(gols_m),
                    "gols_sofridos":     int(gols_s),
                    "resultado":         res,
                    "passes_total":      None,
                    "passes_completos":  None,
                    "precisao_passes":   None,
                    "chutes_total":      None,
                    "chutes_alvo":       None,
                    "pressoes":          None,
                    "duelos_total":      None,
                    "duelos_ganhos":     None,
                    "posse_pct":         None,
                    "escanteios":        None,
                    "cartoes_amarelos":  None,
                    "cartoes_vermelhos": None,
                    "faltas":            None,
                })

        print(f"   ✅ {nome}: {len(finalizados)} partidas processadas")

    matches_df = pd.DataFrame(all_matches)
    stats_df   = pd.DataFrame(all_stats)

    if not matches_df.empty:
        matches_df["data"] = pd.to_datetime(matches_df["data"], errors="coerce")

    return matches_df, stats_df


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _resultado(gols_casa, gols_fora) -> str:
    if gols_casa > gols_fora:   return "V"
    elif gols_casa == gols_fora: return "E"
    return "D"

def _inverter(resultado: str) -> str:
    return {"V": "D", "D": "V", "E": "E"}.get(resultado, "E")

def _safe_int(val) -> int:
    try:    return int(float(val)) if pd.notna(val) else 0
    except: return 0

def _safe_float(val) -> float:
    try:    return float(val) if pd.notna(val) else 0.0
    except: return 0.0


# ─────────────────────────────────────────────────────────────
# SALVAR
# ─────────────────────────────────────────────────────────────

def salvar_parquet(df: pd.DataFrame, nome: str):
    if df.empty:
        print(f"   ⚠️  {nome}.parquet — vazio, pulando")
        return
    
    # Converte todas as colunas object para string para evitar tipos mistos
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).replace("nan", None)
    
    path = BRONZE_DIR / f"{nome}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")
    size_kb = path.stat().st_size / 1024
    print(f"   💾 {path} ({size_kb:.1f} KB | {len(df):,} linhas | {len(df.columns)} colunas)")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run():
    print_header()

    # ── StatsBomb ──────────────────────────────────────────
    matches_sb  = processar_matches_statsbomb()
    shots_sb    = processar_shots_statsbomb(matches_sb)
    stats_sb    = processar_stats_statsbomb(matches_sb)

    # ── Kaggle ─────────────────────────────────────────────
    matches_kg  = processar_qualifiers_kaggle()
    stats_kg    = processar_stats_kaggle(matches_kg)

    # ── API-Football ───────────────────────────────────────
    matches_api, stats_api = processar_qualifiers_api()

    # ── Consolida ──────────────────────────────────────────
    print("\n🔗 Consolidando todas as fontes...")

    all_matches = pd.concat([matches_sb, matches_kg, matches_api], ignore_index=True)
    all_stats   = pd.concat([stats_sb,   stats_kg,   stats_api],   ignore_index=True)
    all_shots   = shots_sb  # chutes só da StatsBomb

    # Consolida eliminatórias separado
    qualifiers  = pd.concat([matches_kg, matches_api], ignore_index=True)

    print(f"   ✅ Partidas totais:       {len(all_matches):,}")
    print(f"      → Copas (StatsBomb):  {len(matches_sb):,}")
    print(f"      → Eliminatórias:      {len(qualifiers):,}")
    print(f"   ✅ Stats totais:          {len(all_stats):,}")
    print(f"   ✅ Chutes (modelo xG):    {len(all_shots):,}")
    # Converte match_id para string em todos os DataFrames
    all_matches["match_id"] = all_matches["match_id"].astype(str)
    all_stats["match_id"]   = all_stats["match_id"].astype(str)
    all_shots["match_id"]   = all_shots["match_id"].astype(str)
    qualifiers["match_id"]  = qualifiers["match_id"].astype(str)

    # ── Salva ──────────────────────────────────────────────
    print("\n💾 Salvando camada bronze...")
    salvar_parquet(all_matches, "matches")
    salvar_parquet(all_shots,   "shots")
    salvar_parquet(all_stats,   "stats")
    salvar_parquet(qualifiers,  "qualifiers")

    print("\n" + "="*55)
    print("  Bronze concluído!")
    print("="*55)
    print(f"""
  Arquivos gerados:
  ─────────────────────────────────────────────
  data/bronze/
  ├── matches.parquet     → {len(all_matches):>6,} partidas (todas as fontes)
  ├── shots.parquet       → {len(all_shots):>6,} chutes (modelo xG)
  ├── stats.parquet       → {len(all_stats):>6,} registros time×partida
  └── qualifiers.parquet  → {len(qualifiers):>6,} partidas eliminatórias

  Próximo passo:
  ─────────────────────────────────────────────
  python src/pipeline/silver.py
    """)


if __name__ == "__main__":
    run()