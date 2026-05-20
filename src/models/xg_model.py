"""
src/models/xg_model.py
───────────────────────────────────────────────────────────────
Modelo de xG (Expected Goals) — XGBoost

O que faz:
    1. Lê as features da camada gold (feat_xg.parquet)
    2. Treina modelo XGBoost para prever probabilidade de gol
    3. Avalia performance com métricas padrão
    4. Salva o modelo treinado para uso no dashboard
    5. Registra experimento no MLflow

Como rodar:
    python src/models/xg_model.py

Saída:
    models/xg_model.json     → modelo treinado
    models/xg_metrics.json   → métricas de avaliação
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, brier_score_loss,
    classification_report, confusion_matrix,
    log_loss
)
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Caminhos ──────────────────────────────────────────────────
GOLD_DIR   = Path("data/gold")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def print_header():
    print("\n" + "="*55)
    print("  Modelo xG — Expected Goals (XGBoost)")
    print("="*55 + "\n")


# ─────────────────────────────────────────────────────────────
# 1. CARREGA E PREPARA DADOS
# ─────────────────────────────────────────────────────────────

def carregar_dados() -> tuple:
    """
    Lê as features do gold e prepara X e y para treino.

    Returns:
        X: DataFrame com features
        y: Series com variável alvo (foi_gol)
        df: DataFrame completo
    """
    print("📂 Carregando dados...")

    path = GOLD_DIR / "feat_xg.parquet"
    if not path.exists():
        raise FileNotFoundError("feat_xg.parquet não encontrado! Rode gold.py primeiro.")

    df = pd.read_parquet(path)

    # Features numéricas principais
    features = [
        "distancia_gol",
        "angulo_gol",
        "minuto",
        "periodo",
        "sob_pressao",
    ]

    # Adiciona one-hot features se existirem
    features += [c for c in df.columns if c.startswith("corpo_") or c.startswith("jogada_")]

    # Converte para numérico
    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["foi_gol"] = pd.to_numeric(df["foi_gol"], errors="coerce").fillna(0).astype(int)

    # Remove linhas com nulos nas features principais
    df = df.dropna(subset=["distancia_gol", "angulo_gol"])

    # Filtra features que existem no DataFrame
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df["foi_gol"]

    print(f"   ✅ {len(df):,} amostras | {len(features)} features")
    print(f"   📊 Gols: {y.sum():,} ({y.mean()*100:.1f}%) | Não-gols: {(1-y).sum():,}")
    print(f"   📋 Features: {features[:5]}{'...' if len(features) > 5 else ''}")

    return X, y, df, features


# ─────────────────────────────────────────────────────────────
# 2. TREINA O MODELO
# ─────────────────────────────────────────────────────────────

def treinar_modelo(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Treina o modelo XGBoost com cross-validation.

    Args:
        X: Features
        y: Variável alvo

    Returns:
        modelo treinado, X_test, y_test, y_pred_proba
    """
    print("\n🤖 Treinando modelo XGBoost...")

    # Split treino/teste estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"   Treino: {len(X_train):,} amostras | Teste: {len(X_test):,} amostras")

    # Parâmetros do modelo
    # scale_pos_weight compensa o desbalanceamento (90% não-gol vs 10% gol)
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    params = {
        "n_estimators":      300,
        "max_depth":         4,
        "learning_rate":     0.05,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "scale_pos_weight":  scale_pos_weight,
        "eval_metric":       "auc",
        "use_label_encoder": False,
        "random_state":      42,
        "n_jobs":            -1,
    }

    print(f"   scale_pos_weight: {scale_pos_weight:.1f} (corrige desbalanceamento)")

    # Treina
    modelo = xgb.XGBClassifier(**params)
    modelo.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    print("   ✅ Modelo treinado!")

    # Cross-validation para avaliar robustez
    print("\n📊 Cross-validation (5 folds)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(modelo, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"   AUC médio: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    y_pred_proba = modelo.predict_proba(X_test)[:, 1]

    return modelo, X_train, X_test, y_train, y_test, y_pred_proba, cv_scores, params


# ─────────────────────────────────────────────────────────────
# 3. AVALIA O MODELO
# ─────────────────────────────────────────────────────────────

def avaliar_modelo(
    modelo,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred_proba: np.ndarray,
    cv_scores: np.ndarray,
    features: list
) -> dict:
    """
    Calcula e exibe métricas de avaliação do modelo.

    Métricas:
        AUC-ROC:     área sob a curva ROC — quanto maior melhor (máx 1.0)
        Brier Score: erro quadrático médio das probabilidades — menor melhor
        Log Loss:    penaliza predições confiantes e erradas — menor melhor
    """
    print("\n📈 Avaliando modelo...")

    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc       = roc_auc_score(y_test, y_pred_proba)
    brier     = brier_score_loss(y_test, y_pred_proba)
    logloss   = log_loss(y_test, y_pred_proba)

    print(f"\n   Métricas no conjunto de teste:")
    print(f"   ─────────────────────────────────")
    print(f"   AUC-ROC:     {auc:.4f}  (baseline: 0.5 | bom: >0.75)")
    print(f"   Brier Score: {brier:.4f}  (baseline: 0.09 | menor = melhor)")
    print(f"   Log Loss:    {logloss:.4f}  (menor = melhor)")
    print(f"   CV AUC:      {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Interpretação simples
    if auc >= 0.80:
        interpretacao = "🟢 Excelente"
    elif auc >= 0.75:
        interpretacao = "🟡 Bom"
    elif auc >= 0.70:
        interpretacao = "🟠 Razoável"
    else:
        interpretacao = "🔴 Abaixo do esperado"

    print(f"\n   Avaliação geral: {interpretacao}")

    # Feature importance
    print(f"\n   Top 5 features mais importantes:")
    importances = pd.Series(
        modelo.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    for i, (feat, imp) in enumerate(importances.head(5).items()):
        print(f"   {i+1}. {feat}: {imp:.4f}")

    metrics = {
        "auc_roc":      round(float(auc), 4),
        "brier_score":  round(float(brier), 4),
        "log_loss":     round(float(logloss), 4),
        "cv_auc_mean":  round(float(cv_scores.mean()), 4),
        "cv_auc_std":   round(float(cv_scores.std()), 4),
        "n_test":       int(len(y_test)),
        "n_gols_test":  int(y_test.sum()),
    }

    return metrics, importances


# ─────────────────────────────────────────────────────────────
# 4. GRÁFICOS
# ─────────────────────────────────────────────────────────────

def gerar_graficos(
    modelo,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred_proba: np.ndarray,
    importances: pd.Series
):
    """Gera gráficos de avaliação do modelo."""
    print("\n📊 Gerando gráficos...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Avaliação do Modelo xG — Copa do Mundo", fontsize=14, fontweight="bold")

    # 1. Distribuição de xG por resultado
    ax1 = axes[0]
    gols     = y_pred_proba[y_test == 1]
    nao_gols = y_pred_proba[y_test == 0]
    ax1.hist(nao_gols, bins=30, alpha=0.6, color="#378ADD", label="Não gol")
    ax1.hist(gols,     bins=30, alpha=0.6, color="#E05A2B", label="Gol")
    ax1.set_xlabel("xG previsto")
    ax1.set_ylabel("Frequência")
    ax1.set_title("Distribuição de xG")
    ax1.legend()

    # 2. Feature importance
    ax2 = axes[1]
    top10 = importances.head(8)
    ax2.barh(top10.index[::-1], top10.values[::-1], color="#378ADD")
    ax2.set_xlabel("Importância")
    ax2.set_title("Top 8 Features")
    ax2.tick_params(axis="y", labelsize=9)

    # 3. Calibração do modelo
    ax3 = axes[2]
    fraction_pos, mean_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    ax3.plot(mean_pred, fraction_pos, "s-", color="#E05A2B", label="Modelo xG")
    ax3.plot([0, 1], [0, 1], "--", color="gray", label="Perfeito")
    ax3.set_xlabel("xG previsto")
    ax3.set_ylabel("Fração real de gols")
    ax3.set_title("Calibração do Modelo")
    ax3.legend()

    plt.tight_layout()
    path = MODELS_DIR / "xg_model_evaluation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Gráficos salvos em {path}")


# ─────────────────────────────────────────────────────────────
# 5. FUNÇÃO DE PREDIÇÃO
# ─────────────────────────────────────────────────────────────

def calcular_xg(
    modelo,
    distancia: float,
    angulo: float,
    parte_corpo: str = "Pé Direito",
    tipo_jogada: str = "Jogo Aberto",
    sob_pressao: bool = False,
    minuto: int = 45,
    periodo: int = 1,
    features: list = None
) -> float:
    """
    Calcula o xG de um chute específico.

    Args:
        distancia:   distância do gol em metros
        angulo:      ângulo do chute
        parte_corpo: parte do corpo usada
        tipo_jogada: tipo da jogada
        sob_pressao: se estava sob pressão
        minuto:      minuto do jogo
        periodo:     período do jogo
        features:    lista de features do modelo

    Returns:
        xG (float entre 0 e 1)
    """
    data = {
        "distancia_gol": distancia,
        "angulo_gol":    angulo,
        "minuto":        minuto,
        "periodo":       periodo,
        "sob_pressao":   int(sob_pressao),
    }

    # Adiciona one-hot features zeradas
    if features:
        for feat in features:
            if feat.startswith("corpo_") or feat.startswith("jogada_"):
                data[feat] = 0

        # Ativa a feature correta
        corpo_feat  = f"corpo_{parte_corpo}"
        jogada_feat = f"jogada_{tipo_jogada}"
        if corpo_feat in data:
            data[corpo_feat] = 1
        if jogada_feat in data:
            data[jogada_feat] = 1

    df = pd.DataFrame([data])

    # Garante mesmas colunas do treino
    if features:
        for col in features:
            if col not in df.columns:
                df[col] = 0
        df = df[features]

    return float(modelo.predict_proba(df)[0][1])


# ─────────────────────────────────────────────────────────────
# 6. SALVA O MODELO
# ─────────────────────────────────────────────────────────────

def salvar_modelo(modelo, metrics: dict, features: list, params: dict):
    """Salva o modelo e metadados para uso no dashboard."""

    # Modelo XGBoost
    model_path = MODELS_DIR / "xg_model.json"
    modelo.save_model(str(model_path))
    print(f"   💾 Modelo salvo: {model_path}")

    # Metadados
    meta = {
        "features":  features,
        "metrics":   metrics,
        "params":    {k: v for k, v in params.items() if k != "use_label_encoder"},
        "descricao": "Modelo xG treinado com dados StatsBomb (Copas 2018 e 2022)",
        "versao":    "1.0",
    }
    meta_path = MODELS_DIR / "xg_model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"   💾 Metadados salvos: {meta_path}")


# ─────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────

def run():
    print_header()

    # Configura MLflow
    mlflow.set_experiment("copa2026_xg_model")

    with mlflow.start_run(run_name="xgboost_v1"):

        # 1. Carrega dados
        X, y, df, features = carregar_dados()

        # 2. Treina
        modelo, X_train, X_test, y_train, y_test, y_pred_proba, cv_scores, params = treinar_modelo(X, y)

        # 3. Avalia
        metrics, importances = avaliar_modelo(modelo, X_test, y_test, y_pred_proba, cv_scores, features)

        # 4. Gráficos
        gerar_graficos(modelo, X_test, y_test, y_pred_proba, importances)

        # 5. Loga no MLflow
        mlflow.log_params({k: v for k, v in params.items() if k != "use_label_encoder"})
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(modelo, "xg_model")

        # 6. Salva
        print("\n💾 Salvando modelo...")
        salvar_modelo(modelo, metrics, features, params)

        # 7. Testa com exemplos reais
        print("\n🧪 Testando com exemplos reais:")
        print("   ─────────────────────────────────────────────")

        exemplos = [
            {"desc": "Chute dentro da área, pé direito, jogo aberto", "dist": 12.0, "ang": 0.3, "corpo": "Pé Direito", "jogada": "Jogo Aberto", "pressao": False},
            {"desc": "Chute de fora da área, pé esquerdo",             "dist": 25.0, "ang": 0.1, "corpo": "Pé Esquerdo","jogada": "Jogo Aberto", "pressao": False},
            {"desc": "Cabeçada após escanteio",                        "dist": 8.0,  "ang": 0.4, "corpo": "Cabeça",     "jogada": "Escanteio",   "pressao": True},
            {"desc": "Chute de dentro da área pequena",                "dist": 4.0,  "ang": 0.8, "corpo": "Pé Direito", "jogada": "Jogo Aberto", "pressao": False},
            {"desc": "Chute de longe, sob pressão",                    "dist": 35.0, "ang": 0.05,"corpo": "Pé Direito", "jogada": "Jogo Aberto", "pressao": True},
        ]

        for ex in exemplos:
            xg = calcular_xg(
                modelo,
                distancia=ex["dist"],
                angulo=ex["ang"],
                parte_corpo=ex["corpo"],
                tipo_jogada=ex["jogada"],
                sob_pressao=ex["pressao"],
                features=features
            )
            print(f"   {ex['desc']}")
            print(f"   → xG: {xg:.3f} ({xg*100:.1f}% de chance de gol)\n")

    print("=" * 55)
    print("  Modelo xG concluído!")
    print("=" * 55)
    print(f"""
  Arquivos gerados:
  ─────────────────────────────────────────────
  models/
  ├── xg_model.json            → modelo treinado
  ├── xg_model_meta.json       → features e métricas
  └── xg_model_evaluation.png  → gráficos de avaliação

  Para ver os experimentos MLflow:
  ─────────────────────────────────────────────
  mlflow ui
  → acesse http://localhost:5000

  Próximo passo:
  ─────────────────────────────────────────────
  python src/models/prediction_model.py
    """)


if __name__ == "__main__":
    run()
