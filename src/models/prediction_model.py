"""
src/models/prediction_model.py
───────────────────────────────────────────────────────────────
Modelo de predição de resultado — LightGBM

O que faz:
    1. Lê features da camada gold (feat_predicao.parquet)
    2. Inclui dados das eliminatórias com peso menor
    3. Treina LightGBM para prever vitória/empate/derrota
    4. Avalia com métricas de classificação multiclasse
    5. Salva modelo para uso no dashboard
    6. Registra no MLflow

Como rodar:
    python src/models/prediction_model.py

Saída:
    models/prediction_model.txt       → modelo treinado
    models/prediction_model_meta.json → métricas e features
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, log_loss
)
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Caminhos ──────────────────────────────────────────────────
GOLD_DIR   = Path("data/gold")
SILVER_DIR = Path("data/silver")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def print_header():
    print("\n" + "="*55)
    print("  Modelo de Predição de Resultado (LightGBM)")
    print("="*55 + "\n")


# ─────────────────────────────────────────────────────────────
# 1. CARREGA E PREPARA DADOS
# ─────────────────────────────────────────────────────────────

def carregar_dados() -> tuple:
    print("📂 Carregando dados...")

    path_copa = GOLD_DIR / "feat_predicao.parquet"
    if not path_copa.exists():
        raise FileNotFoundError("feat_predicao.parquet não encontrado!")

    df_copa = pd.read_parquet(path_copa)
    df_copa["peso"]  = 0.8
    df_copa["fonte"] = "copa"

    print(f"   ✅ Copa: {len(df_copa)} partidas")

    df_quali = _carregar_features_eliminatorias()

    df = pd.concat([df_copa, df_quali], ignore_index=True)

    features = [
        "fase_cod",
        "casa_aproveitamento_copa",
        "casa_gols_pro_media_copa",
        "casa_gols_con_media_copa",
        "casa_chutes_media_copa",
        "casa_precisao_passes_copa",
        "casa_pressoes_media_copa",
        "fora_aproveitamento_copa",
        "fora_gols_pro_media_copa",
        "fora_gols_con_media_copa",
        "fora_chutes_media_copa",
        "fora_precisao_passes_copa",
        "fora_pressoes_media_copa",
        "casa_aproveitamento_quali",
        "fora_aproveitamento_quali",
        "diff_aproveitamento_copa",
        "diff_gols_pro_copa",
        "diff_chutes_copa",
        "diff_aproveitamento_quali",
    ]

    features = [f for f in features if f in df.columns]

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    le = LabelEncoder()
    df["resultado_cod"] = le.fit_transform(df["resultado"].astype(str))

    X     = df[features]
    y     = df["resultado_cod"]
    pesos = df["peso"]

    print(f"   ✅ Total: {len(df)} partidas | {len(features)} features")
    print(f"   📊 Distribuição:")
    for classe, nome in zip(le.classes_, ["Derrota", "Empate", "Vitória"]):
        n = (df["resultado"] == int(classe)).sum()
        print(f"      {nome}: {n} ({n/len(df)*100:.1f}%)")

    return X, y, pesos, df, features, le


def _carregar_features_eliminatorias() -> pd.DataFrame:
    path_quali = SILVER_DIR / "qualifiers.parquet"
    path_agg_q = GOLD_DIR   / "agg_selecao_quali.parquet"

    if not path_quali.exists() or not path_agg_q.exists():
        print("   ⚠️  Dados de eliminatórias não encontrados — usando só Copas")
        return pd.DataFrame()

    quali  = pd.read_parquet(path_quali)
    agg_q  = pd.read_parquet(path_agg_q)

    rows = []
    for _, match in quali.iterrows():
        casa = match["time_casa"]
        fora = match["time_fora"]

        sc = agg_q[agg_q["time_nome"] == casa]
        sf = agg_q[agg_q["time_nome"] == fora]

        if sc.empty or sf.empty:
            continue

        sc = sc.iloc[0]
        sf = sf.iloc[0]

        aprov_c = float(sc.get("aproveitamento_quali", 50))
        aprov_f = float(sf.get("aproveitamento_quali", 50))
        gols_c  = float(sc.get("gols_pro_quali", 0)) / max(float(sc.get("jogos_quali", 1)), 1)
        gols_f  = float(sf.get("gols_pro_quali", 0)) / max(float(sf.get("jogos_quali", 1)), 1)
        gcon_c  = float(sc.get("gols_contra_quali", 0)) / max(float(sc.get("jogos_quali", 1)), 1)
        gcon_f  = float(sf.get("gols_contra_quali", 0)) / max(float(sf.get("jogos_quali", 1)), 1)

        res_raw = match.get("resultado_casa", "E")
        res_map = {"V": 1, "E": 0, "D": -1}
        resultado = res_map.get(str(res_raw), 0)

        rows.append({
            "match_id":                   match.get("match_id", ""),
            "ano":                        2026,
            "fase_cod":                   1,
            "fonte":                      "quali",
            "peso":                       0.5,
            "resultado":                  resultado,
            "casa_aproveitamento_copa":   aprov_c,
            "casa_gols_pro_media_copa":   gols_c,
            "casa_gols_con_media_copa":   gcon_c,
            "casa_chutes_media_copa":     float(sc.get("chutes_media_quali", 0)),
            "casa_precisao_passes_copa":  float(sc.get("precisao_passes_quali", 0)),
            "casa_pressoes_media_copa":   0.0,
            "fora_aproveitamento_copa":   aprov_f,
            "fora_gols_pro_media_copa":   gols_f,
            "fora_gols_con_media_copa":   gcon_f,
            "fora_chutes_media_copa":     float(sf.get("chutes_media_quali", 0)),
            "fora_precisao_passes_copa":  float(sf.get("precisao_passes_quali", 0)),
            "fora_pressoes_media_copa":   0.0,
            "casa_aproveitamento_quali":  aprov_c,
            "fora_aproveitamento_quali":  aprov_f,
            "diff_aproveitamento_copa":   aprov_c - aprov_f,
            "diff_gols_pro_copa":         gols_c - gols_f,
            "diff_chutes_copa":           float(sc.get("chutes_media_quali", 0)) - float(sf.get("chutes_media_quali", 0)),
            "diff_aproveitamento_quali":  aprov_c - aprov_f,
        })

    df = pd.DataFrame(rows)
    print(f"   ✅ Eliminatórias: {len(df)} partidas")
    return df


# ─────────────────────────────────────────────────────────────
# 2. TREINA O MODELO
# ─────────────────────────────────────────────────────────────

def treinar_modelo(X: pd.DataFrame, y: pd.Series, pesos: pd.Series) -> tuple:
    print("\n🤖 Treinando modelo LightGBM...")

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, pesos,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"   Treino: {len(X_train):,} | Teste: {len(X_test):,}")

    params = {
        "objective":        "multiclass",
        "num_class":        3,
        "metric":           "multi_logloss",
        "n_estimators":     500,
        "max_depth":        4,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_samples":5,
        "random_state":     42,
        "n_jobs":           -1,
        "verbose":          -1,
    }

    modelo = lgb.LGBMClassifier(**params)
    modelo.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)]
    )

    print(f"   ✅ Modelo treinado! ({modelo.best_iteration_} iterações)")

    print("\n📊 Cross-validation (5 folds)...")
    modelo_cv = lgb.LGBMClassifier(**params)
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(modelo_cv, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"   Accuracy médio: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    y_pred       = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)

    return modelo, X_test, y_test, y_pred, y_pred_proba, cv_scores, params


# ─────────────────────────────────────────────────────────────
# 3. AVALIA O MODELO
# ─────────────────────────────────────────────────────────────

def avaliar_modelo(modelo, X_test, y_test, y_pred, y_pred_proba, cv_scores, features, le) -> dict:
    print("\n📈 Avaliando modelo...")

    acc     = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)

    print(f"\n   Métricas no conjunto de teste:")
    print(f"   ─────────────────────────────────────")
    print(f"   Accuracy:    {acc:.4f}  (baseline: 0.33 | bom: >0.50)")
    print(f"   Log Loss:    {logloss:.4f}  (menor = melhor)")
    print(f"   CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    if acc >= 0.60:
        interpretacao = "🟢 Excelente"
    elif acc >= 0.50:
        interpretacao = "🟡 Bom"
    elif acc >= 0.40:
        interpretacao = "🟠 Razoável"
    else:
        interpretacao = "🔴 Abaixo do esperado"

    print(f"\n   Avaliação geral: {interpretacao}")

    nomes  = ["Derrota", "Empate", "Vitória"]
    report = classification_report(y_test, y_pred, target_names=nomes, output_dict=True)
    print(f"\n   Relatório por resultado:")
    for nome in nomes:
        r = report[nome]
        print(f"   {nome:10} → Precisão: {r['precision']:.2f} | Recall: {r['recall']:.2f} | F1: {r['f1-score']:.2f}")

    print(f"\n   Top 5 features mais importantes:")
    importances = pd.Series(
        modelo.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    for i, (feat, imp) in enumerate(importances.head(5).items()):
        print(f"   {i+1}. {feat}: {imp}")

    metrics = {
        "accuracy":     round(float(acc), 4),
        "log_loss":     round(float(logloss), 4),
        "cv_acc_mean":  round(float(cv_scores.mean()), 4),
        "cv_acc_std":   round(float(cv_scores.std()), 4),
        "n_test":       int(len(y_test)),
    }

    return metrics, importances


# ─────────────────────────────────────────────────────────────
# 4. GRÁFICOS
# ─────────────────────────────────────────────────────────────

def gerar_graficos(y_test, y_pred, y_pred_proba, importances):
    print("\n📊 Gerando gráficos...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Avaliação do Modelo de Predição — Copa do Mundo", fontsize=14, fontweight="bold")

    nomes = ["Derrota", "Empate", "Vitória"]
    cores = ["#E05A2B", "#F0A500", "#1D9E75"]

    ax1 = axes[0]
    cm  = confusion_matrix(y_test, y_pred)
    ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks([0, 1, 2])
    ax1.set_yticks([0, 1, 2])
    ax1.set_xticklabels(nomes, rotation=45)
    ax1.set_yticklabels(nomes)
    ax1.set_xlabel("Previsto")
    ax1.set_ylabel("Real")
    ax1.set_title("Matriz de Confusão")
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")

    ax2 = axes[1]
    for i, nome in enumerate(nomes):
        mask = y_test == i
        if mask.sum() > 0:
            medias = y_pred_proba[mask].mean(axis=0)
            ax2.bar(
                [x + i*0.25 for x in range(3)],
                medias, width=0.25,
                label=f"Real: {nome}",
                color=cores[i], alpha=0.8
            )
    ax2.set_xticks([0.25, 1.25, 2.25])
    ax2.set_xticklabels(nomes)
    ax2.set_ylabel("Probabilidade média prevista")
    ax2.set_title("Calibração por Resultado")
    ax2.legend(fontsize=8)

    ax3 = axes[2]
    top8 = importances.head(8)
    ax3.barh(top8.index[::-1], top8.values[::-1], color="#378ADD")
    ax3.set_xlabel("Importância")
    ax3.set_title("Top 8 Features")
    ax3.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    path = MODELS_DIR / "prediction_model_evaluation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Gráficos salvos em {path}")


# ─────────────────────────────────────────────────────────────
# 5. FUNÇÃO DE PREDIÇÃO
# ─────────────────────────────────────────────────────────────

def prever_resultado(
    modelo,
    features: list,
    le: LabelEncoder,
    time_casa: str,
    time_fora: str,
    casa_aprov: float,
    fora_aprov: float,
    casa_gols_pro: float,
    fora_gols_pro: float,
    casa_gols_con: float,
    fora_gols_con: float,
    casa_chutes: float = 10.0,
    fora_chutes: float = 10.0,
    casa_passes_pct: float = 75.0,
    fora_passes_pct: float = 75.0,
    casa_pressoes: float = 20.0,
    fora_pressoes: float = 20.0,
    casa_aprov_quali: float = 50.0,
    fora_aprov_quali: float = 50.0,
    fase_cod: int = 1,
) -> dict:
    """
    Prevê o resultado de uma partida.

    Returns:
        dict com probabilidades associadas ao nome de cada seleção.
        Exemplo: {"Brasil": 70.0, "Empate": 10.0, "Sérvia": 20.0}
    """
    data = {
        "fase_cod":                   fase_cod,
        "casa_aproveitamento_copa":   casa_aprov,
        "casa_gols_pro_media_copa":   casa_gols_pro,
        "casa_gols_con_media_copa":   casa_gols_con,
        "casa_chutes_media_copa":     casa_chutes,
        "casa_precisao_passes_copa":  casa_passes_pct,
        "casa_pressoes_media_copa":   casa_pressoes,
        "fora_aproveitamento_copa":   fora_aprov,
        "fora_gols_pro_media_copa":   fora_gols_pro,
        "fora_gols_con_media_copa":   fora_gols_con,
        "fora_chutes_media_copa":     fora_chutes,
        "fora_precisao_passes_copa":  fora_passes_pct,
        "fora_pressoes_media_copa":   fora_pressoes,
        "casa_aproveitamento_quali":  casa_aprov_quali,
        "fora_aproveitamento_quali":  fora_aprov_quali,
        "diff_aproveitamento_copa":   casa_aprov - fora_aprov,
        "diff_gols_pro_copa":         casa_gols_pro - fora_gols_pro,
        "diff_chutes_copa":           casa_chutes - fora_chutes,
        "diff_aproveitamento_quali":  casa_aprov_quali - fora_aprov_quali,
    }

    df = pd.DataFrame([data])
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]

    proba   = modelo.predict_proba(df)[0]
    classes = le.classes_

    # Mapeia probabilidades para nomes das seleções
    resultado = {}
    for i, classe in enumerate(classes):
        if str(classe) == "-1":
            resultado[time_fora] = round(float(proba[i]) * 100, 1)
        elif str(classe) == "0":
            resultado["Empate"] = round(float(proba[i]) * 100, 1)
        elif str(classe) == "1":
            resultado[time_casa] = round(float(proba[i]) * 100, 1)

    return resultado


# ─────────────────────────────────────────────────────────────
# 6. SALVA O MODELO
# ─────────────────────────────────────────────────────────────

def salvar_modelo(modelo, metrics: dict, features: list, params: dict, le: LabelEncoder):
    model_path = MODELS_DIR / "prediction_model.txt"
    modelo.booster_.save_model(str(model_path))
    print(f"   💾 Modelo salvo: {model_path}")

    meta = {
        "features":      features,
        "metrics":       metrics,
        "classes":       list(le.classes_.astype(str)),
        "mapa_classes":  {"-1": "derrota", "0": "empate", "1": "vitoria"},
        "descricao":     "Modelo de predição treinado com Copas 2018/2022 + eliminatórias 2026",
        "versao":        "1.0",
    }
    meta_path = MODELS_DIR / "prediction_model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"   💾 Metadados salvos: {meta_path}")


# ─────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────

def run():
    print_header()

    mlflow.set_experiment("copa2026_prediction_model")

    with mlflow.start_run(run_name="lightgbm_v1"):

        X, y, pesos, df, features, le = carregar_dados()
        modelo, X_test, y_test, y_pred, y_pred_proba, cv_scores, params = treinar_modelo(X, y, pesos)
        metrics, importances = avaliar_modelo(modelo, X_test, y_test, y_pred, y_pred_proba, cv_scores, features, le)
        gerar_graficos(y_test, y_pred, y_pred_proba, importances)

        mlflow.log_params({k: v for k, v in params.items()})
        mlflow.log_metrics(metrics)

        print("\n💾 Salvando modelo...")
        salvar_modelo(modelo, metrics, features, params, le)

        # 7. Testa predições com nome das seleções
        print("\n🧪 Testando predições:")
        print("   ─────────────────────────────────────────────")

        exemplos = [
            {
                "casa": "Brasil",    "fora": "Sérvia",
                "c_aprov": 78.0,     "f_aprov": 45.0,
                "c_gp": 2.3,         "f_gp": 1.2,
                "c_gc": 0.8,         "f_gc": 1.8,
            },
            {
                "casa": "França",    "fora": "Argentina",
                "c_aprov": 72.0,     "f_aprov": 70.0,
                "c_gp": 2.0,         "f_gp": 2.1,
                "c_gc": 1.0,         "f_gc": 1.1,
            },
            {
                "casa": "Japão",     "fora": "Alemanha",
                "c_aprov": 55.0,     "f_aprov": 75.0,
                "c_gp": 1.5,         "f_gp": 2.2,
                "c_gc": 1.2,         "f_gc": 0.9,
            },
        ]

        for ex in exemplos:
            pred = prever_resultado(
                modelo, features, le,
                time_casa=ex["casa"],
                time_fora=ex["fora"],
                casa_aprov=ex["c_aprov"],
                fora_aprov=ex["f_aprov"],
                casa_gols_pro=ex["c_gp"],
                fora_gols_pro=ex["f_gp"],
                casa_gols_con=ex["c_gc"],
                fora_gols_con=ex["f_gc"],
            )
            print(f"   {ex['casa']} vs {ex['fora']}")
            for nome, prob in pred.items():
                print(f"   → {nome}: {prob}%")
            print()

    print("=" * 55)
    print("  Modelo de Predição concluído!")
    print("=" * 55)
    print(f"""
  Arquivos gerados:
  ─────────────────────────────────────────────
  models/
  ├── prediction_model.txt          → modelo treinado
  ├── prediction_model_meta.json    → features e métricas
  └── prediction_model_evaluation.png → gráficos

  Próximo passo:
  ─────────────────────────────────────────────
  python src/models/clustering.py
    """)


if __name__ == "__main__":
    run()