"""
setup_data.py
───────────────────────────────────────────────────────────────
Script de setup — rode UMA VEZ após clonar o repositório.

O que ele faz:
    1. Baixa os dados históricos da StatsBomb (Copa 2018 e 2022)
    2. Salva tudo na estrutura correta de pastas
    3. Verifica a integridade dos dados baixados

Como rodar:
    python setup_data.py

Requisitos:
    - Internet
    - pip install -r requirements.txt já executado
    - NÃO precisa de chave de API (StatsBomb é público e gratuito)
"""

import requests
import json
import time
from pathlib import Path

# ── Configurações ─────────────────────────────────────────────
BASE_URL   = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COPAS      = [
    {"season_id": 106, "ano": 2022, "arquivo": "matches_2022.json"},
    {"season_id": 3,   "ano": 2018, "arquivo": "matches_2018.json"},
]
COMPETITION_ID = 43  # FIFA World Cup na StatsBomb


def print_header():
    print("\n" + "="*55)
    print("  Copa do Mundo 2026 — Setup de dados históricos")
    print("  Fonte: StatsBomb Open Data (gratuito e público)")
    print("="*55 + "\n")


def criar_pastas():
    """Cria a estrutura de pastas necessária."""
    pastas = [
        Path("data/raw/statsbomb/events"),
        Path("data/bronze"),
        Path("data/silver"),
        Path("data/gold"),
    ]
    for pasta in pastas:
        pasta.mkdir(parents=True, exist_ok=True)
    print("✅ Estrutura de pastas criada\n")


def baixar_partidas(season_id: int, ano: int, arquivo: str) -> list:
    """
    Baixa a lista de partidas de uma Copa específica.

    Args:
        season_id: ID da temporada na StatsBomb
        ano:       Ano da Copa
        arquivo:   Nome do arquivo para salvar

    Returns:
        Lista de partidas
    """
    print(f"📥 Baixando partidas da Copa {ano}...")
    url = f"{BASE_URL}/matches/{COMPETITION_ID}/{season_id}.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    matches = r.json()
    output_path = Path(f"data/raw/statsbomb/{arquivo}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)

    print(f"   ✅ {len(matches)} partidas salvas em {output_path}")
    return matches


def baixar_eventos(matches: list, ano: int):
    """
    Baixa os eventos táticos de cada partida.
    Inclui chutes, passes, pressão, coordenadas x/y.

    Args:
        matches: Lista de partidas
        ano:     Ano da Copa (para exibição)
    """
    print(f"\n📥 Baixando eventos da Copa {ano} ({len(matches)} partidas)...")
    print("   (isso pode demorar alguns minutos)\n")

    events_dir = Path("data/raw/statsbomb/events")
    baixados   = 0
    pulados    = 0

    for i, match in enumerate(matches):
        match_id = match["match_id"]
        home     = match["home_team"]["home_team_name"]
        away     = match["away_team"]["away_team_name"]
        output   = events_dir / f"{match_id}.json"

        # Pula se já foi baixado
        if output.exists():
            pulados += 1
            print(f"   [{i+1}/{len(matches)}] ⏭️  {home} x {away} — já existe")
            continue

        url = f"{BASE_URL}/events/{match_id}.json"
        r   = requests.get(url, timeout=30)
        r.raise_for_status()

        events = r.json()
        shots  = [e for e in events if e["type"]["name"] == "Shot"]

        with open(output, "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)

        baixados += 1
        print(f"   [{i+1}/{len(matches)}] ✅ {home} x {away} — {len(shots)} chutes")

        # Pequena pausa para não sobrecarregar o servidor
        time.sleep(0.2)

    print(f"\n   Baixados: {baixados} | Já existiam: {pulados}")


def verificar_integridade():
    """Verifica se todos os arquivos foram baixados corretamente."""
    print("\n🔍 Verificando integridade dos dados...")

    # Verifica partidas
    for copa in COPAS:
        path = Path(f"data/raw/statsbomb/{copa['arquivo']}")
        if path.exists():
            with open(path, encoding="utf-8") as f:
                matches = json.load(f)
            print(f"   ✅ Copa {copa['ano']}: {len(matches)} partidas")
        else:
            print(f"   ❌ Copa {copa['ano']}: arquivo não encontrado!")

    # Verifica eventos
    events_dir = Path("data/raw/statsbomb/events")
    event_files = list(events_dir.glob("*.json"))
    total_size  = sum(f.stat().st_size for f in event_files)

    print(f"   ✅ Eventos: {len(event_files)} arquivos ({total_size / 1024 / 1024:.1f} MB)")

    # Conta total de chutes disponíveis
    total_shots = 0
    for event_file in event_files:
        with open(event_file, encoding="utf-8") as f:
            events = json.load(f)
        total_shots += sum(1 for e in events if e["type"]["name"] == "Shot")

    print(f"   ✅ Total de chutes para treino do modelo xG: {total_shots:,}")


def resumo_final():
    """Exibe resumo do que foi baixado."""
    print("\n" + "="*55)
    print("  Setup concluído!")
    print("="*55)
    print("""
  O que você tem agora:
  ─────────────────────────────────────────────
  data/raw/statsbomb/
  ├── matches_2022.json   → 64 partidas Copa 2022
  ├── matches_2018.json   → 64 partidas Copa 2018
  └── events/             → 128 arquivos de eventos
      └── *.json            chutes, passes, pressão

  Próximo passo:
  ─────────────────────────────────────────────
  python src/pipeline/bronze.py
  (em breve)
    """)


def run():
    print_header()
    criar_pastas()

    all_matches = []

    # Baixa partidas e eventos de cada Copa
    for copa in COPAS:
        matches = baixar_partidas(
            season_id=copa["season_id"],
            ano=copa["ano"],
            arquivo=copa["arquivo"]
        )
        all_matches.extend(matches)
        baixar_eventos(matches, copa["ano"])

    verificar_integridade()
    resumo_final()


if __name__ == "__main__":
    run()
