"""
src/models/clustering.py
───────────────────────────────────────────────────────────────
Clustering de seleções por estilo de jogo — K-Means + PCA

O que faz:
    1. Lê métricas agregadas das seleções (gold)
    2. Agrupa seleções por estilo de jogo com K-Means
    3. Reduz dimensionalidade com PCA para visualização
    4. Salva clusters para uso no dashboard

Estilos identificados:
    - Posse e controle (Espanha, Brasil)
    - Pressão alta e intensidade (Alemanha, França)
    - Defensivo e contra-ataque (Uruguai, Costa Rica)
    - Equilibrado (maioria das seleções)

Como rodar:
    python src/models/clustering.py

Saída:
    models/clusters.parquet          → seleções com cluster
    models/clustering_meta.json      → metadados
    models/clustering_evaluation.png → gráficos PCA
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Caminhos ──────────────────────────────────────────────────
GOLD_DIR   = Path("data/gold")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def print_header():
    print("\n" + "="*55)
    print("  Clustering de Seleções (K-Means + PCA)")
    print("="*55 + "\n")


# ─────────────────────────────────────────────────────────────
# 1. CARREGA E PREPARA DADOS
# ─────────────────────────────────────────────────────────────

def carregar_dados() -> tuple:
    """
    Combina métricas de Copas e eliminatórias para criar
    o perfil tático de cada seleção.
    """
    print("📂 Carregando dados...")

    # Métricas das Copas
    path_copa  = GOLD_DIR / "agg_selecao_copa.parquet"
    path_quali = GOLD_DIR / "agg_selecao_quali.parquet"

    if not path_copa.exists():
        raise FileNotFoundError("agg_selecao_copa.parquet não encontrado!")

    df_copa  = pd.read_parquet(path_copa)
    df_quali = pd.read_parquet(path_quali) if path_quali.exists() else pd.DataFrame()

    # Agrega métricas de Copa por seleção (média das duas Copas)
    agg_copa = df_copa.groupby("time_nome").agg(
        aproveitamento_copa    = ("aproveitamento",        "mean"),
        gols_pro_media_copa    = ("gols_pro",              "mean"),
        gols_con_media_copa    = ("gols_contra",           "mean"),
        passes_media_copa      = ("passes_media",          "mean"),
        precisao_passes_copa   = ("precisao_passes_media", "mean"),
        chutes_media_copa      = ("chutes_media",          "mean"),
        chutes_alvo_media_copa = ("chutes_alvo_media",     "mean"),
        pressoes_media_copa    = ("pressoes_media",        "mean"),
        duelos_ganhos_copa     = ("duelos_ganhos_media",   "mean"),
    ).reset_index()

    print(f"   ✅ Copa: {len(agg_copa)} seleções")

    # Junta com eliminatórias se disponível
    if not df_quali.empty:
        df = agg_copa.merge(
            df_quali[["time_nome", "aproveitamento_quali",
                      "passes_media_quali", "precisao_passes_quali",
                      "chutes_media_quali", "posse_media_quali"]],
            on="time_nome", how="left"
        )
        print(f"   ✅ Eliminatórias: {len(df_quali)} seleções adicionadas")
    else:
        df = agg_copa.copy()

    # Preenche nulos com médias
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].mean())

    print(f"   ✅ Total: {len(df)} seleções para clustering")
    return df


# ─────────────────────────────────────────────────────────────
# 2. ENCONTRA NÚMERO IDEAL DE CLUSTERS
# ─────────────────────────────────────────────────────────────

def encontrar_k_ideal(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 8) -> int:
    """
    Usa o método do cotovelo (Elbow) e Silhouette Score
    para encontrar o número ideal de clusters.
    """
    print("\n🔍 Encontrando número ideal de clusters...")

    inertias   = []
    silhouettes = []
    ks         = range(k_min, k_max + 1)

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, km.labels_)
        silhouettes.append(sil)
        print(f"   k={k} | Inertia: {km.inertia_:.1f} | Silhouette: {sil:.3f}")

    # Escolhe k com melhor Silhouette Score
    k_ideal = ks[np.argmax(silhouettes)]
    print(f"\n   ✅ k ideal: {k_ideal} (melhor Silhouette: {max(silhouettes):.3f})")
    return k_ideal, inertias, silhouettes, list(ks)


# ─────────────────────────────────────────────────────────────
# 3. TREINA O CLUSTERING
# ─────────────────────────────────────────────────────────────

def treinar_clustering(df: pd.DataFrame, k: int) -> tuple:
    """
    Treina K-Means e aplica PCA para visualização.
    """
    print(f"\n🤖 Treinando K-Means com k={k}...")

    # Features para clustering
    features = [
        "aproveitamento_copa",
        "gols_pro_media_copa",
        "gols_con_media_copa",
        "passes_media_copa",
        "precisao_passes_copa",
        "chutes_media_copa",
        "pressoes_media_copa",
    ]

    # Adiciona features de eliminatórias se disponíveis
    for feat in ["aproveitamento_quali", "precisao_passes_quali", "posse_media_quali"]:
        if feat in df.columns:
            features.append(feat)

    features = [f for f in features if f in df.columns]

    X = df[features].values

    # Normaliza
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)

    # PCA para visualização 2D
    pca        = PCA(n_components=2, random_state=42)
    X_pca      = pca.fit_transform(X_scaled)
    df["pca1"] = X_pca[:, 0]
    df["pca2"] = X_pca[:, 1]

    variancia = pca.explained_variance_ratio_
    print(f"   ✅ Clusters gerados!")
    print(f"   📊 PCA explica {variancia.sum()*100:.1f}% da variância")
    print(f"      PC1: {variancia[0]*100:.1f}% | PC2: {variancia[1]*100:.1f}%")

    return df, km, scaler, pca, features, X_scaled


# ─────────────────────────────────────────────────────────────
# 4. NOMEIA OS CLUSTERS
# ─────────────────────────────────────────────────────────────

def nomear_clusters(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Analisa as características de cada cluster e atribui um nome
    descritivo baseado nas métricas mais distintivas.
    """
    print("\n📊 Analisando perfis dos clusters...")

    nomes_clusters = {}

    for cluster_id in sorted(df["cluster"].unique()):
        grupo    = df[df["cluster"] == cluster_id]
        media_gp = df.groupby("cluster")[features].mean()

        # Características do cluster vs média geral
        media_geral = df[features].mean()
        perfil      = media_gp.loc[cluster_id] - media_geral

        # Determina nome baseado nas características mais marcantes
        aprov     = perfil.get("aproveitamento_copa", 0)
        passes    = perfil.get("precisao_passes_copa", 0)
        pressoes  = perfil.get("pressoes_media_copa", 0)
        gols_pro  = perfil.get("gols_pro_media_copa", 0)
        gols_con  = perfil.get("gols_con_media_copa", 0)

        if aprov > 10 and passes > 2:
            nome = "🏆 Elite — Alta posse e eficiência"
        elif pressoes > 5 and gols_pro > 0.3:
            nome = "⚡ Pressão alta e ataque intenso"
        elif gols_con < -0.2 and aprov > 0:
            nome = "🛡️  Defensivo e sólido"
        elif gols_pro > 0.2 and passes < -2:
            nome = "🎯 Direto e eficiente"
        elif aprov < -10:
            nome = "📈 Em desenvolvimento"
        else:
            nome = "⚖️  Equilibrado"

        nomes_clusters[cluster_id] = nome

        times = grupo["time_nome"].tolist()
        print(f"\n   Cluster {cluster_id} — {nome}")
        print(f"   Times: {', '.join(times[:8])}{'...' if len(times) > 8 else ''}")
        print(f"   Aproveitamento médio: {grupo['aproveitamento_copa'].mean():.1f}%")
        if "precisao_passes_copa" in grupo.columns:
            print(f"   Precisão passes:      {grupo['precisao_passes_copa'].mean():.1f}%")

    df["cluster_nome"] = df["cluster"].map(nomes_clusters)
    return df, nomes_clusters


# ─────────────────────────────────────────────────────────────
# 5. GRÁFICOS
# ─────────────────────────────────────────────────────────────

def gerar_graficos(df: pd.DataFrame, inertias: list, silhouettes: list, ks: list, k_ideal: int):
    print("\n📊 Gerando gráficos...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Clustering de Seleções — Copa do Mundo 2026", fontsize=14, fontweight="bold")

    cores = ["#378ADD", "#E05A2B", "#1D9E75", "#F0A500", "#9B59B6", "#E74C3C"]

    # 1. Elbow method
    ax1 = axes[0]
    ax1.plot(ks, inertias, "o-", color="#378ADD", linewidth=2)
    ax1.axvline(x=k_ideal, color="#E05A2B", linestyle="--", alpha=0.7, label=f"k={k_ideal} (ideal)")
    ax1.set_xlabel("Número de clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Método do Cotovelo")
    ax1.legend()

    # 2. Silhouette Score
    ax2 = axes[1]
    ax2.plot(ks, silhouettes, "s-", color="#1D9E75", linewidth=2)
    ax2.axvline(x=k_ideal, color="#E05A2B", linestyle="--", alpha=0.7, label=f"k={k_ideal} (ideal)")
    ax2.set_xlabel("Número de clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score por k")
    ax2.legend()

    # 3. PCA — scatter plot dos clusters
    ax3 = axes[2]
    for i, cluster_id in enumerate(sorted(df["cluster"].unique())):
        mask  = df["cluster"] == cluster_id
        nome  = df[mask]["cluster_nome"].iloc[0]
        cor   = cores[i % len(cores)]
        ax3.scatter(
            df[mask]["pca1"], df[mask]["pca2"],
            c=cor, label=f"C{cluster_id}: {nome[:20]}",
            alpha=0.7, s=80
        )

        # Anota os times mais conhecidos
        times_conhecidos = ["Brasil", "Argentina", "França", "Alemanha",
                           "Espanha", "Inglaterra", "Portugal", "Japão",
                           "Marrocos", "Coreia do Sul"]
        for _, row in df[mask].iterrows():
            if row["time_nome"] in times_conhecidos:
                ax3.annotate(
                    row["time_nome"],
                    (row["pca1"], row["pca2"]),
                    fontsize=7, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points"
                )

    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_title("Mapa de Estilos de Jogo (PCA)")
    ax3.legend(fontsize=7, loc="best")

    plt.tight_layout()
    path = MODELS_DIR / "clustering_evaluation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Gráficos salvos em {path}")


# ─────────────────────────────────────────────────────────────
# 6. SALVA
# ─────────────────────────────────────────────────────────────

def salvar_resultados(df: pd.DataFrame, nomes_clusters: dict, features: list, k: int):
    """Salva clusters para uso no dashboard."""

    # Parquet com clusters
    cols_salvar = ["time_nome", "cluster", "cluster_nome", "pca1", "pca2",
                   "aproveitamento_copa", "gols_pro_media_copa", "gols_con_media_copa",
                   "passes_media_copa", "precisao_passes_copa", "chutes_media_copa"]
    cols_salvar = [c for c in cols_salvar if c in df.columns]

    path_parquet = MODELS_DIR / "clusters.parquet"
    df[cols_salvar].to_parquet(path_parquet, index=False)
    print(f"   💾 Clusters salvos: {path_parquet}")

    # Metadados
    meta = {
        "k":              k,
        "features":       features,
        "nomes_clusters": {str(k): v for k, v in nomes_clusters.items()},
        "descricao":      "Clustering de seleções por estilo de jogo",
        "versao":         "1.0",
    }
    path_meta = MODELS_DIR / "clustering_meta.json"
    with open(path_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"   💾 Metadados salvos: {path_meta}")


# ─────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────

def run():
    print_header()

    # 1. Carrega dados
    df = carregar_dados()

    # 2. Prepara features
    features_num = [c for c in df.columns if df[c].dtype in [np.float64, np.int64]
                    and c not in ["cluster", "pca1", "pca2"]]

    X        = df[features_num].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Encontra k ideal
    k_ideal, inertias, silhouettes, ks = encontrar_k_ideal(X_scaled)

    # 4. Treina com k ideal
    df, km, scaler, pca, features, X_scaled = treinar_clustering(df, k_ideal)

    # 5. Nomeia clusters
    df, nomes_clusters = nomear_clusters(df, features)

    # 6. Gráficos
    gerar_graficos(df, inertias, silhouettes, ks, k_ideal)

    # 7. Salva
    print("\n💾 Salvando resultados...")
    salvar_resultados(df, nomes_clusters, features, k_ideal)

    print("\n" + "="*55)
    print("  Clustering concluído!")
    print("="*55)
    print(f"""
  Arquivos gerados:
  ─────────────────────────────────────────────
  models/
  ├── clusters.parquet          → seleções + clusters
  ├── clustering_meta.json      → metadados
  └── clustering_evaluation.png → gráficos PCA

  Próximo passo:
  ─────────────────────────────────────────────
  streamlit run src/dashboard/app.py
    """)


if __name__ == "__main__":
    run()
