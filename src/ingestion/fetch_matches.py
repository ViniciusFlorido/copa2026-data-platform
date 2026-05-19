"""
src/ingestion/fetch_matches.py
───────────────────────────────────────────────────────────────
Primeiro script do projeto — busca os resultados do dia na
API-Football e salva o JSON bruto na camada raw.

Como rodar:
    python src/ingestion/fetch_matches.py

O que ele faz:
    1. Lê as credenciais do .env
    2. Chama a API-Football buscando jogos da Copa 2026
    3. Salva o JSON original em data/raw/YYYY-MM-DD/
    4. Imprime um resumo do que foi encontrado
"""

import os
import json
import requests
from datetime import date, datetime
from pathlib import Path
from dotenv import load_dotenv

# ── Carrega variáveis do .env ─────────────────────────────────
load_dotenv()

API_KEY       = os.getenv("API_FOOTBALL_KEY")
LEAGUE_ID     = os.getenv("WORLD_CUP_LEAGUE_ID", "1")
SEASON        = os.getenv("WORLD_CUP_SEASON", "2026")
RAW_PATH      = Path(os.getenv("RAW_PATH", "data/raw"))

BASE_URL      = "https://v3.football.api-sports.io"
HEADERS       = {
    "x-rapidapi-host": "v3.football.api-sports.io",
    "x-rapidapi-key": API_KEY,
}


def get_fixtures_by_date(target_date: str) -> dict:
    """
    Busca todas as partidas de uma data específica na Copa 2026.

    Args:
        target_date: Data no formato YYYY-MM-DD

    Returns:
        JSON completo retornado pela API
    """
    print(f"\n📡 Buscando partidas do dia {target_date}...")

    response = requests.get(
        url=f"{BASE_URL}/fixtures",
        headers=HEADERS,
        params={
            "league": LEAGUE_ID,
            "season": SEASON,
            "date": target_date,
        },
        timeout=15,
    )

    response.raise_for_status()
    return response.json()


def get_fixture_statistics(fixture_id: int) -> dict:
    """
    Busca estatísticas detalhadas de uma partida específica.
    (posse, chutes, passes, cartões, escanteios, etc.)

    Args:
        fixture_id: ID da partida na API-Football

    Returns:
        JSON com estatísticas da partida
    """
    print(f"   📊 Buscando estatísticas da partida {fixture_id}...")

    response = requests.get(
        url=f"{BASE_URL}/fixtures/statistics",
        headers=HEADERS,
        params={"fixture": fixture_id},
        timeout=15,
    )

    response.raise_for_status()
    return response.json()


def get_fixture_events(fixture_id: int) -> dict:
    """
    Busca os eventos de uma partida (gols, cartões, substituições).

    Args:
        fixture_id: ID da partida na API-Football

    Returns:
        JSON com eventos da partida
    """
    print(f"   ⚡ Buscando eventos da partida {fixture_id}...")

    response = requests.get(
        url=f"{BASE_URL}/fixtures/events",
        headers=HEADERS,
        params={"fixture": fixture_id},
        timeout=15,
    )

    response.raise_for_status()
    return response.json()


def save_raw(data: dict, filename: str, subfolder: str = "") -> Path:
    """
    Salva o JSON bruto na camada raw, organizado por data.

    Estrutura:
        data/raw/YYYY-MM-DD/fixtures.json
        data/raw/YYYY-MM-DD/stats_1234567.json
        data/raw/YYYY-MM-DD/events_1234567.json

    Args:
        data:      Dicionário com o JSON da API
        filename:  Nome do arquivo (ex: "fixtures.json")
        subfolder: Subpasta adicional (ex: "2026-06-15")

    Returns:
        Path do arquivo salvo
    """
    output_dir = RAW_PATH / subfolder
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"   ✅ Salvo em: {output_path}")
    return output_path


def check_api_status() -> bool:
    """
    Verifica quantas requisições ainda restam no plano gratuito.
    Imprime um resumo do status da API.

    Returns:
        True se a API está acessível, False caso contrário
    """
    print("\n🔑 Verificando status da API...")

    response = requests.get(
        url=f"{BASE_URL}/status",
        headers=HEADERS,
        timeout=10,
    )

    if response.status_code != 200:
        print(f"❌ Erro ao verificar status: {response.status_code}")
        return False

    data = response.json()
    subscription = data.get("response", {}).get("subscription", {})
    requests_info = data.get("response", {}).get("requests", {})

    print(f"   Plano:           {subscription.get('plan', 'N/A')}")
    print(f"   Req. usadas hoje: {requests_info.get('current', 'N/A')}")
    print(f"   Limite diário:   {requests_info.get('limit_day', 'N/A')}")

    return True


def run(target_date: str = None):
    """
    Função principal — orquestra a ingestão do dia.

    Args:
        target_date: Data alvo no formato YYYY-MM-DD.
                     Se None, usa a data de hoje.
    """
    if not API_KEY:
        print("❌ API_FOOTBALL_KEY não encontrada no .env")
        print("   Crie o arquivo .env a partir do .env.example")
        return

    # Define a data alvo
    if target_date is None:
        target_date = date.today().isoformat()

    print(f"\n{'='*50}")
    print(f"  Copa do Mundo 2026 — Ingestão diária")
    print(f"  Data: {target_date}")
    print(f"{'='*50}")

    # 1. Verifica status da API e saldo de requisições
    if not check_api_status():
        return

    # 2. Busca partidas do dia
    fixtures_data = get_fixtures_by_date(target_date)
    fixtures = fixtures_data.get("response", [])

    if not fixtures:
        print(f"\n⚠️  Nenhuma partida encontrada para {target_date}")
        print("   (Pode ser dia sem jogos ou o torneio ainda não começou)")
        # Salva mesmo assim para registrar que o script rodou
        save_raw(fixtures_data, "fixtures.json", subfolder=target_date)
        return

    # 3. Salva a lista de partidas do dia
    save_raw(fixtures_data, "fixtures.json", subfolder=target_date)
    print(f"\n✅ {len(fixtures)} partida(s) encontrada(s)")

    # 4. Para cada partida, busca estatísticas e eventos
    for fixture in fixtures:
        fixture_id   = fixture["fixture"]["id"]
        home_team    = fixture["teams"]["home"]["name"]
        away_team    = fixture["teams"]["away"]["name"]
        status       = fixture["fixture"]["status"]["short"]
        score_home   = fixture["goals"]["home"]
        score_away   = fixture["goals"]["away"]

        print(f"\n   🏟️  {home_team} {score_home} x {score_away} {away_team} [{status}]")

        # Só busca stats de partidas já finalizadas (FT = Full Time)
        if status == "FT":
            stats_data  = get_fixture_statistics(fixture_id)
            events_data = get_fixture_events(fixture_id)

            save_raw(stats_data,  f"stats_{fixture_id}.json",  subfolder=target_date)
            save_raw(events_data, f"events_{fixture_id}.json", subfolder=target_date)
        else:
            print(f"   ⏳ Partida com status '{status}' — pulando estatísticas")

    print(f"\n{'='*50}")
    print(f"  Ingestão concluída — {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Arquivos em: data/raw/{target_date}/")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Para testar com uma data específica, passe como argumento:
    # python src/ingestion/fetch_matches.py 2026-06-15
    import sys
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run(target_date=date_arg)
