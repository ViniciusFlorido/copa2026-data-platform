import pandas as pd
from pathlib import Path

df = pd.read_csv("data/raw/kaggle/scraped_dataset.csv", low_memory=False)

# Filtra eliminatórias Copa 2026 — season como string
mask = (
    df["league_division"].str.contains("World Cup - Qualification", case=False, na=False) &
    (df["season"] == "2026")
)

qualifiers = df[mask].copy()

times = pd.concat([qualifiers["home_team"], qualifiers["away_team"]]).unique()
finalizados = qualifiers[qualifiers["result"].isin(["H", "A", "D"])]

print(f"Total de jogos: {len(qualifiers)}")
print(f"Jogos finalizados: {len(finalizados)}")
print(f"Seleções únicas: {len(times)}")
print(f"\nJogos por competição:")
print(qualifiers["league_division"].value_counts().to_string())

# Salva
Path("data/raw/qualifiers").mkdir(parents=True, exist_ok=True)
qualifiers.to_csv("data/raw/qualifiers/all_qualifiers_2026.csv", index=False, encoding="utf-8")
print(f"\n✅ Salvo em data/raw/qualifiers/all_qualifiers_2026.csv")