import requests
import json
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")
HEADERS = {"x-rapidapi-key": API_KEY}
BASE    = "https://v3.football.api-sports.io"

CONFEDERACOES = [
    {"nome": "CONMEBOL", "league_id": 34, "season": 2024},
    {"nome": "UEFA",     "league_id": 32, "season": 2024},
    {"nome": "CONCACAF", "league_id": 31, "season": 2024},
    {"nome": "CAF",      "league_id": 29, "season": 2023},
    {"nome": "AFC",      "league_id": 30, "season": 2024},
    {"nome": "OFC",      "league_id": 33, "season": 2024},
]

Path("data/raw/qualifiers").mkdir(parents=True, exist_ok=True)

print("\n📡 Buscando eliminatórias Copa 2026...\n")

for conf in CONFEDERACOES:
    r = requests.get(
        f"{BASE}/fixtures",
        headers=HEADERS,
        params={"league": conf["league_id"], "season": conf["season"]}
    )
    data = r.json()
    fixtures = data.get("response", [])
    errors   = data.get("errors", {})

    if errors:
        print(f"   ❌ {conf['nome']:10} → erro: {errors}")
        continue

    path = f"data/raw/qualifiers/{conf['nome'].lower()}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fixtures, f, ensure_ascii=False, indent=2)

    finalizados = [x for x in fixtures if x["fixture"]["status"]["short"] == "FT"]
    print(f"   ✅ {conf['nome']:10} → {len(fixtures):3} jogos | {len(finalizados):3} finalizados")

print("\nSalvo em data/raw/qualifiers/")