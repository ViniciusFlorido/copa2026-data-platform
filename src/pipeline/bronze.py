"""
src/pipeline/bronze.py
───────────────────────────────────────────────────────────────
Camada Bronze — transforma os JSONs brutos da StatsBomb em
arquivos Parquet tipados e estruturados.

O que faz:
    1. Lê os JSONs de partidas (matches_2022.json, matches_2018.json)
    2. Lê os JSONs de eventos de cada partida
    3. Extrai e tipifica os dados relevantes
    4. Salva em Parquet na pasta data/bronze/

Arquivos gerados:
    data/bronze/matches.parquet    → informações das partidas
    data/bronze/shots.parquet      → chutes (para modelo xG)
    data/bronze/stats.parquet      → estatísticas por time/partida

Como rodar:
    python src/pipeline/bronze.py
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Caminhos ──────────────────────────────────────────────────
RAW_DIR    = Path("data/raw/statsbomb")
BRONZE_DIR = Path("data/bronze")
BRONZE_DIR.mkdir(parents=True, exist_ok=True)


def print_header():
    print("\n" + "="*55)
    print("  Pipeline Bronze — StatsBomb → Parquet")
    print("="*55 + "\n")


# ─────────────────────────────────────────────────────────────
# 1. MATCHES — informações gerais de cada partida
# ─────────────────────────────────────────────────────────────

def processar_matches() -> pd.DataFrame:
    """
    Lê os JSONs de partidas das Copas 2018 e 2022 e transforma
    em um DataFrame flat com os campos relevantes.

    Returns:
        DataFrame com todas as partidas
    """
    print("📋 Processando partidas...")

    arquivos = [
        (RAW_DIR / "matches_2022.json", 2022),
        (RAW_DIR / "matches_2018.json", 2018),
    ]

    rows = []

    for arquivo, ano in arquivos:
        if not arquivo.exists():
            print(f"   ⚠️  {arquivo} não encontrado — rode setup_data.py primeiro")
            continue

        with open(arquivo, encoding="utf-8") as f:
            matches = json.load(f)

        for m in matches:
            rows.append({
                # Identificação
                "match_id":          m["match_id"],
                "ano":               ano,
                "data":              m["match_date"],

                # Times
                "time_casa_id":      m["home_team"]["home_team_id"],
                "time_casa":         m["home_team"]["home_team_name"],
                "time_fora_id":      m["away_team"]["away_team_id"],
                "time_fora":         m["away_team"]["away_team_name"],

                # Resultado
                "gols_casa":         int(m["home_score"]),
                "gols_fora":         int(m["away_score"]),
                "resultado_casa":    _resultado(m["home_score"], m["away_score"]),

                # Fase do torneio
                "fase":              m["competition_stage"]["name"],

                # Árbitro e estádio
                "estadio":           m.get("stadium", {}).get("name", None),
                "pais_estadio":      m.get("stadium", {}).get("country", {}).get("name", None),
            })

        print(f"   ✅ Copa {ano}: {len(matches)} partidas processadas")

    df = pd.DataFrame(rows)

    # Tipos corretos
    df["match_id"]     = df["match_id"].astype(int)
    df["ano"]          = df["ano"].astype(int)
    df["gols_casa"]    = df["gols_casa"].astype(int)
    df["gols_fora"]    = df["gols_fora"].astype(int)
    df["data"]         = pd.to_datetime(df["data"])

    print(f"   📊 Total: {len(df)} partidas | Colunas: {len(df.columns)}")
    return df


def _resultado(gols_casa: int, gols_fora: int) -> str:
    """Retorna V (vitória), E (empate) ou D (derrota) para o time da casa."""
    if gols_casa > gols_fora:
        return "V"
    elif gols_casa == gols_fora:
        return "E"
    return "D"


# ─────────────────────────────────────────────────────────────
# 2. SHOTS — chutes para treinar o modelo de xG
# ─────────────────────────────────────────────────────────────

def processar_shots(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai todos os eventos de chute de cada partida.
    Esses dados são usados para treinar o modelo de xG.

    Campos extraídos:
        - Localização (x, y) do chute
        - Distância e ângulo calculados
        - Parte do corpo (pé, cabeça)
        - Tipo de jogada (aberto, pênalti, falta, etc.)
        - Se foi gol ou não (variável alvo do modelo)

    Args:
        matches_df: DataFrame de partidas

    Returns:
        DataFrame com todos os chutes
    """
    print("\n⚽ Processando chutes para modelo xG...")

    events_dir = RAW_DIR / "events"
    rows       = []
    partidas_ok = 0
    partidas_erro = 0

    for _, match in matches_df.iterrows():
        match_id   = match["match_id"]
        event_file = events_dir / f"{match_id}.json"

        if not event_file.exists():
            partidas_erro += 1
            continue

        with open(event_file, encoding="utf-8") as f:
            events = json.load(f)

        shots = [e for e in events if e["type"]["name"] == "Shot"]
        partidas_ok += 1

        for shot in shots:
            loc      = shot.get("location", [None, None])
            shot_det = shot.get("shot", {})

            rows.append({
                # Identificação
                "match_id":        match_id,
                "ano":             match["ano"],
                "fase":            match["fase"],
                "time_casa":       match["time_casa"],
                "time_fora":       match["time_fora"],

                # Quem chutou
                "time_id":         shot.get("team", {}).get("id"),
                "time_nome":       shot.get("team", {}).get("name"),
                "jogador_id":      shot.get("player", {}).get("id"),
                "jogador_nome":    shot.get("player", {}).get("name"),

                # Localização do chute
                "loc_x":           float(loc[0]) if loc[0] else None,
                "loc_y":           float(loc[1]) if loc[1] else None,

                # Características do chute
                "parte_corpo":     shot_det.get("body_part", {}).get("name"),
                "tipo_jogada":     shot_det.get("type", {}).get("name"),
                "tecnica":         shot_det.get("technique", {}).get("name"),
                "sob_pressao":     bool(shot.get("under_pressure", False)),

                # Resultado
                "resultado":       shot_det.get("outcome", {}).get("name"),
                "foi_gol":         1 if shot_det.get("outcome", {}).get("name") == "Goal" else 0,

                # xG já calculado pela StatsBomb (para comparação)
                "xg_statsbomb":    float(shot_det["statsbomb_xg"]) if "statsbomb_xg" in shot_det else None,

                # Minuto do chute
                "minuto":          int(shot.get("minute", 0)),
                "periodo":         int(shot.get("period", 1)),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        print("   ❌ Nenhum chute encontrado!")
        return df

    # Calcula distância e ângulo do gol
    # Gol fica em x=120, y=40 no campo da StatsBomb (120x80)
    df["distancia_gol"] = (
        ((df["loc_x"] - 120) ** 2 + (df["loc_y"] - 40) ** 2) ** 0.5
    ).round(2)

    df["angulo_gol"] = (
        (df["loc_y"] - 40).abs() / (120 - df["loc_x"] + 0.0001)
    ).round(4)

    # Tipos corretos
    df["foi_gol"]   = df["foi_gol"].astype(int)
    df["minuto"]    = df["minuto"].astype(int)
    df["periodo"]   = df["periodo"].astype(int)

    gols = df["foi_gol"].sum()
    taxa = (gols / len(df) * 100).round(1)

    print(f"   ✅ {partidas_ok} partidas processadas | {partidas_erro} sem arquivo")
    print(f"   📊 {len(df):,} chutes | {gols:,} gols | taxa de conversão: {taxa}%")
    return df


# ─────────────────────────────────────────────────────────────
# 3. STATS — estatísticas por time por partida
# ─────────────────────────────────────────────────────────────

def processar_stats(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estatísticas agregadas por time por partida
    a partir dos eventos da StatsBomb.

    Métricas calculadas:
        - Total de passes e passes completos
        - Total de chutes e chutes no alvo
        - Pressão aplicada no adversário
        - Duelos ganhos

    Args:
        matches_df: DataFrame de partidas

    Returns:
        DataFrame com estatísticas por time/partida
    """
    print("\n📊 Processando estatísticas por time...")

    events_dir = RAW_DIR / "events"
    rows       = []

    for _, match in matches_df.iterrows():
        match_id   = match["match_id"]
        event_file = events_dir / f"{match_id}.json"

        if not event_file.exists():
            continue

        with open(event_file, encoding="utf-8") as f:
            events = json.load(f)

        # Agrupa por time
        times = {
            match["time_casa_id"]: match["time_casa"],
            match["time_fora_id"]: match["time_fora"],
        }

        for time_id, time_nome in times.items():
            ev_time = [e for e in events if e.get("team", {}).get("id") == time_id]

            passes        = [e for e in ev_time if e["type"]["name"] == "Pass"]
            passes_ok     = [p for p in passes if p.get("pass", {}).get("outcome") is None]
            chutes        = [e for e in ev_time if e["type"]["name"] == "Shot"]
            chutes_alvo   = [c for c in chutes if c.get("shot", {}).get("outcome", {}).get("name") in ["Goal", "Saved"]]
            pressoes      = [e for e in ev_time if e["type"]["name"] == "Pressure"]
            duelos        = [e for e in ev_time if e["type"]["name"] == "Duel"]
            duelos_ganhos = [d for d in duelos if d.get("duel", {}).get("outcome", {}).get("name") in ["Won", "Success"]]

            eh_casa = time_nome == match["time_casa"]

            rows.append({
                "match_id":          match_id,
                "ano":               match["ano"],
                "fase":              match["fase"],
                "time_id":           time_id,
                "time_nome":         time_nome,
                "eh_casa":           eh_casa,
                "adversario":        match["time_fora"] if eh_casa else match["time_casa"],

                # Resultado
                "gols_marcados":     match["gols_casa"] if eh_casa else match["gols_fora"],
                "gols_sofridos":     match["gols_fora"] if eh_casa else match["gols_casa"],
                "resultado":         match["resultado_casa"] if eh_casa else _inverter_resultado(match["resultado_casa"]),

                # Passes
                "passes_total":      len(passes),
                "passes_completos":  len(passes_ok),
                "precisao_passes":   round(len(passes_ok) / len(passes) * 100, 1) if passes else 0.0,

                # Chutes
                "chutes_total":      len(chutes),
                "chutes_alvo":       len(chutes_alvo),

                # Pressão e duelos
                "pressoes":          len(pressoes),
                "duelos_total":      len(duelos),
                "duelos_ganhos":     len(duelos_ganhos),
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        print(f"   ✅ {len(df):,} registros (time × partida)")
        print(f"   📊 Média de passes por time por jogo: {df['passes_total'].mean():.0f}")

    return df


def _inverter_resultado(resultado: str) -> str:
    """Inverte o resultado do ponto de vista do time visitante."""
    return {"V": "D", "D": "V", "E": "E"}.get(resultado, "E")


# ─────────────────────────────────────────────────────────────
# 4. SALVAR EM PARQUET
# ─────────────────────────────────────────────────────────────

def salvar_parquet(df: pd.DataFrame, nome: str):
    """
    Salva o DataFrame em formato Parquet na camada bronze.

    Args:
        df:   DataFrame a salvar
        nome: Nome do arquivo (sem extensão)
    """
    if df.empty:
        print(f"   ⚠️  {nome}.parquet — DataFrame vazio, pulando")
        return

    path = BRONZE_DIR / f"{nome}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")

    size_kb = path.stat().st_size / 1024
    print(f"   💾 Salvo: {path} ({size_kb:.1f} KB | {len(df):,} linhas | {len(df.columns)} colunas)")


# ─────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────

def run():
    print_header()

    # Processa cada tabela
    matches_df = processar_matches()
    shots_df   = processar_shots(matches_df)
    stats_df   = processar_stats(matches_df)

    # Salva em Parquet
    print("\n💾 Salvando camada bronze...")
    salvar_parquet(matches_df, "matches")
    salvar_parquet(shots_df,   "shots")
    salvar_parquet(stats_df,   "stats")

    # Resumo final
    print("\n" + "="*55)
    print("  Bronze concluído!")
    print("="*55)
    print(f"""
  Arquivos gerados:
  ─────────────────────────────────────────────
  data/bronze/
  ├── matches.parquet  → {len(matches_df):>5,} partidas
  ├── shots.parquet    → {len(shots_df):>5,} chutes
  └── stats.parquet    → {len(stats_df):>5,} registros time×partida

  Próximo passo:
  ─────────────────────────────────────────────
  python src/pipeline/silver.py
    """)


if __name__ == "__main__":
    run()
