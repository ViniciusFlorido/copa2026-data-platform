"""
src/ingestion/explore_raw.py
───────────────────────────────────────────────────────────────
Segundo script — após salvar o raw, use este para entender
exatamente quais campos a API retorna antes de construir
o pipeline de transformação.

Como rodar:
    python src/ingestion/explore_raw.py 2026-06-15
"""

import json
import sys
from pathlib import Path


def explore_json(data: dict, prefix: str = "", max_depth: int = 3, depth: int = 0):
    """
    Percorre o JSON recursivamente e imprime a estrutura de campos.
    Útil para entender o schema antes de escrever o pipeline.
    """
    if depth >= max_depth:
        return

    if isinstance(data, dict):
        for key, value in data.items():
            tipo = type(value).__name__
            if isinstance(value, (dict, list)):
                print(f"{'  ' * depth}📁 {prefix}{key}: [{tipo}]")
                explore_json(value, prefix="", max_depth=max_depth, depth=depth + 1)
            else:
                print(f"{'  ' * depth}   {prefix}{key}: {repr(value)[:60]} ({tipo})")

    elif isinstance(data, list):
        if data:
            print(f"{'  ' * depth}   → {len(data)} item(s), mostrando o primeiro:")
            explore_json(data[0], prefix="", max_depth=max_depth, depth=depth + 1)


def summarize_fixture(fixture: dict):
    """Imprime um resumo legível de uma partida."""
    f    = fixture.get("fixture", {})
    home = fixture.get("teams", {}).get("home", {})
    away = fixture.get("teams", {}).get("away", {})
    goals = fixture.get("goals", {})
    stats = fixture.get("statistics", [])

    print(f"\n{'─'*40}")
    print(f"  ID:     {f.get('id')}")
    print(f"  Data:   {f.get('date')}")
    print(f"  Status: {f.get('status', {}).get('long')}")
    print(f"  Partida: {home.get('name')} {goals.get('home')} x {goals.get('away')} {away.get('name')}")
    if stats:
        print(f"  Stats disponíveis: {len(stats)} times")


def run(target_date: str):
    raw_dir = Path("data/raw") / target_date

    if not raw_dir.exists():
        print(f"❌ Pasta não encontrada: {raw_dir}")
        print("   Rode primeiro: python src/ingestion/fetch_matches.py")
        return

    print(f"\n{'='*50}")
    print(f"  Explorando raw de {target_date}")
    print(f"{'='*50}")

    # Lista os arquivos salvos
    files = list(raw_dir.glob("*.json"))
    print(f"\n📂 {len(files)} arquivo(s) encontrado(s):")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"   {f.name} ({size_kb:.1f} KB)")

    # Explora o fixtures.json
    fixtures_path = raw_dir / "fixtures.json"
    if not fixtures_path.exists():
        print("\n❌ fixtures.json não encontrado")
        return

    with open(fixtures_path, encoding="utf-8") as f:
        data = json.load(f)

    fixtures = data.get("response", [])
    print(f"\n\n{'='*50}")
    print(f"  RESUMO DAS PARTIDAS ({len(fixtures)} encontradas)")
    print(f"{'='*50}")
    for fixture in fixtures:
        summarize_fixture(fixture)

    # Mostra a estrutura completa do primeiro fixture
    if fixtures:
        print(f"\n\n{'='*50}")
        print("  ESTRUTURA DO JSON (primeiro fixture)")
        print("  Use isso para mapear os campos no pipeline")
        print(f"{'='*50}\n")
        explore_json(fixtures[0], max_depth=4)

    # Explora um arquivo de estatísticas se existir
    stats_files = list(raw_dir.glob("stats_*.json"))
    if stats_files:
        print(f"\n\n{'='*50}")
        print(f"  ESTRUTURA DE ESTATÍSTICAS ({stats_files[0].name})")
        print(f"{'='*50}\n")
        with open(stats_files[0], encoding="utf-8") as f:
            stats_data = json.load(f)
        response = stats_data.get("response", [])
        if response:
            explore_json(response[0], max_depth=4)


if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if not date_arg:
        print("Uso: python src/ingestion/explore_raw.py YYYY-MM-DD")
    else:
        run(date_arg)
