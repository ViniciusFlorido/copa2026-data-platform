"""
setup_data.py
───────────────────────────────────────────────────────────────
Script de setup — rode UMA VEZ após clonar o repositório.

O que ele faz:
    1. Baixa dados históricos da StatsBomb (Copa 2018 e 2022)
    2. Baixa eliminatórias Copa 2026 via Kaggle
    3. Busca dados das eliminatórias via API-Football (UEFA e CAF)
    4. Verifica integridade de tudo

Como rodar:
    python setup_data.py

Requisitos:
    - pip install -r requirements.txt já executado
    - Arquivo .env configurado com API_FOOTBALL_KEY
    - kaggle.json em C:\\Users\\SEU_USUARIO\\.kaggle\\kaggle.json
"""

import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

STATSBOMB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
API_KEY        = os.getenv("API_FOOTBALL_KEY")
API_HEADERS    = {"x-rapidapi-key": API_KEY}
API_BASE       = "https://v3.football.api-sports.io"

COPAS_STATSBOMB = [
    {"season_id": 106, "ano": 2022, "arquivo": "matches_2022.json"},
    {"season_id": 3,   "ano": 2018, "arquivo": "matches_2018.json"},
]

CONFEDERACOES_API = [
    {"nome": "UEFA", "league_id": 32, "season": 2024},
    {"nome": "CAF",  "league_id": 29, "season": 2023},
]

KAGGLE_DATASET = "omarameen99/football-matches-data-from-soccerway"


def print_header():
    print("\n" + "="*55)
    print("  Copa do Mundo 2026 — Setup completo de dados")
    print("="*55)
    print("""
  Fontes:
    1. StatsBomb Open Data  (gratuito, sem chave)
    2. Kaggle Dataset       (gratuito, requer conta)
    3. API-Football         (gratuito, requer chave)
  """)


def criar_pastas():
    pastas = [
        Path("data/raw/statsbomb/events"),
        Path("data/raw/qualifiers"),
        Path("data/raw/kaggle"),
        Path("data/bronze"),
        Path("data/silver"),
        Path("data/gold"),
    ]
    for pasta in pastas:
        pasta.mkdir(parents=True, exist_ok=True)
    print("✅ Estrutura de pastas criada\n")


def baixar_statsbomb():
    print("─"*55)
    print("  PARTE 1 — StatsBomb (Copas 2018 e 2022)")
    print("─"*55)

    all_matches = []

    for copa in COPAS_STATSBOMB:
        print(f"\n📥 Baixando partidas Copa {copa['ano']}...")
        url = f"{STATSBOMB_BASE}/matches/43/{copa['season_id']}.json"
        r   = requests.get(url, timeout=30)
        r.raise_for_status()
        matches = r.json()
        path    = Path(f"data/raw/statsbomb/{copa['arquivo']}")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(matches, f, ensure_ascii=False, indent=2)

        print(f"   ✅ {len(matches)} partidas salvas")
        all_matches.extend(matches)

        print(f"📥 Baixando eventos Copa {copa['ano']} ({len(matches)} partidas)...")
        events_dir = Path("data/raw/statsbomb/events")
        baixados = pulados = 0

        for i, match in enumerate(matches):
            match_id = match["match_id"]
            output   = events_dir / f"{match_id}.json"

            if output.exists():
                pulados += 1
                continue

            url    = f"{STATSBOMB_BASE}/events/{match_id}.json"
            r      = requests.get(url, timeout=30)
            r.raise_for_status()
            events = r.json()
            shots  = [e for e in events if e["type"]["name"] == "Shot"]

            with open(output, "w", encoding="utf-8") as f:
                json.dump(events, f, ensure_ascii=False, indent=2)

            baixados += 1
            print(f"   [{i+1}/{len(matches)}] ✅ {match['home_team']['home_team_name']} x {match['away_team']['away_team_name']} — {len(shots)} chutes")
            time.sleep(0.2)

        print(f"   Baixados: {baixados} | Já existiam: {pulados}")

    total_shots = 0
    for f in Path("data/raw/statsbomb/events").glob("*.json"):
        with open(f, encoding="utf-8") as fp:
            events = json.load(fp)
        total_shots += sum(1 for e in events if e["type"]["name"] == "Shot")

    print(f"\n✅ StatsBomb concluído! {len(all_matches)} partidas | {total_shots:,} chutes")


def baixar_kaggle():
    print("\n" + "─"*55)
    print("  PARTE 2 — Kaggle (Eliminatórias Copa 2026)")
    print("─"*55)

    output = Path("data/raw/qualifiers/all_qualifiers_2026.csv")

    if output.exists():
        df = pd.read_csv(output)
        print(f"\n⏭️  Já existe — {len(df)} jogos encontrados, pulando download")
        return

    print(f"\n📥 Baixando dataset do Kaggle...")

    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path="data/raw/kaggle",
            unzip=True,
            quiet=False
        )
    except Exception as e:
        print(f"   ❌ Erro no Kaggle: {e}")
        print("   Verifique se o kaggle.json está configurado corretamente")
        return

    csv_path = Path("data/raw/kaggle/scraped_dataset.csv")
    if not csv_path.exists():
        print("   ❌ Arquivo CSV não encontrado após download")
        return

    df = pd.read_csv(csv_path, low_memory=False)
    mask = (
        df["league_division"].str.contains("World Cup - Qualification", case=False, na=False) &
        (df["season"] == "2026")
    )
    qualifiers = df[mask].copy()
    qualifiers.to_csv(output, index=False, encoding="utf-8")

    times = pd.concat([qualifiers["home_team"], qualifiers["away_team"]]).unique()
    print(f"\n✅ Kaggle concluído! {len(qualifiers)} jogos | {len(times)} seleções")


def baixar_api_football():
    print("\n" + "─"*55)
    print("  PARTE 3 — API-Football (UEFA e CAF)")
    print("─"*55)

    if not API_KEY:
        print("\n   ⚠️  API_FOOTBALL_KEY não encontrada no .env — pulando")
        return

    print()
    for conf in CONFEDERACOES_API:
        output = Path(f"data/raw/qualifiers/{conf['nome'].lower()}.json")

        if output.exists():
            with open(output, encoding="utf-8") as f:
                existing = json.load(f)
            print(f"⏭️  {conf['nome']:10} — já existe ({len(existing)} jogos), pulando")
            continue

        r = requests.get(
            f"{API_BASE}/fixtures",
            headers=API_HEADERS,
            params={"league": conf["league_id"], "season": conf["season"]},
            timeout=15
        )
        data     = r.json()
        fixtures = data.get("response", [])
        errors   = data.get("errors", {})

        if errors:
            print(f"   ❌ {conf['nome']:10} — erro: {errors}")
            continue

        with open(output, "w", encoding="utf-8") as f:
            json.dump(fixtures, f, ensure_ascii=False, indent=2)

        finalizados = [x for x in fixtures if x["fixture"]["status"]["short"] == "FT"]
        print(f"✅ {conf['nome']:10} → {len(fixtures):3} jogos | {len(finalizados):3} finalizados")

    print(f"\n✅ API-Football concluído!")


def verificar_integridade():
    print("\n" + "─"*55)
    print("  Verificação de integridade")
    print("─"*55 + "\n")

    for copa in COPAS_STATSBOMB:
        path = Path(f"data/raw/statsbomb/{copa['arquivo']}")
        if path.exists():
            with open(path, encoding="utf-8") as f:
                matches = json.load(f)
            print(f"✅ StatsBomb Copa {copa['ano']}: {len(matches)} partidas")
        else:
            print(f"❌ StatsBomb Copa {copa['ano']}: não encontrado!")

    events = list(Path("data/raw/statsbomb/events").glob("*.json"))
    print(f"✅ Eventos StatsBomb: {len(events)} arquivos")

    kaggle_path = Path("data/raw/qualifiers/all_qualifiers_2026.csv")
    if kaggle_path.exists():
        df = pd.read_csv(kaggle_path)
        times = pd.concat([df["home_team"], df["away_team"]]).nunique()
        print(f"✅ Eliminatórias Kaggle: {len(df)} jogos | {times} seleções")
    else:
        print(f"❌ Eliminatórias Kaggle: não encontrado!")

    for conf in CONFEDERACOES_API:
        path = Path(f"data/raw/qualifiers/{conf['nome'].lower()}.json")
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            print(f"✅ API-Football {conf['nome']:10}: {len(data)} jogos")
        else:
            print(f"⚠️  API-Football {conf['nome']:10}: não encontrado")


def resumo_final():
    print("\n" + "="*55)
    print("  Setup concluído!")
    print("="*55)
    print("""
  Próximos passos:
  ─────────────────────────────────────────────
  1. python src/pipeline/bronze.py
  2. python src/pipeline/silver.py
  3. python src/pipeline/gold.py
  4. python src/models/xg_model.py
  5. streamlit run src/dashboard/app.py
    """)


def run():
    print_header()
    criar_pastas()
    baixar_statsbomb()
    baixar_kaggle()
    baixar_api_football()
    verificar_integridade()
    resumo_final()


if __name__ == "__main__":
    run()
