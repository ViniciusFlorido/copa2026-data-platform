# Documentação — Modelo de xG (Expected Goals)

## Visão Geral

O modelo de **xG (Expected Goals)** calcula a probabilidade de um chute virar gol com base nas características da jogada. É a métrica mais importante do projeto e alimenta os demais modelos.

---

## O que é xG?

Expected Goals (xG) é uma métrica que responde à pergunta:

> *"Dadas as condições desse chute, qual a probabilidade de ele virar gol?"*

Um chute recebe um valor entre **0 e 1**:

| xG | Interpretação |
|---|---|
| 0.90 | Chance quase certa de gol |
| 0.50 | 50% de probabilidade |
| 0.10 | Chance baixa |
| 0.02 | Chance muito baixa |

**Exemplos reais do nosso modelo:**

| Situação | xG |
|---|---|
| Chute na área pequena (4m) | 0.821 |
| Chute dentro da área (12m) | 0.442 |
| Cabeçada após escanteio | 0.248 |
| Chute de fora da área (25m) | 0.326 |
| Chute de longe sob pressão (35m) | 0.048 |

---

## Dados de Treinamento

| Item | Detalhe |
|---|---|
| Fonte | StatsBomb Open Data (gratuito) |
| Competições | Copa do Mundo 2018 e 2022 |
| Total de chutes | 3.068 |
| Gols | 287 (9.4%) |
| Não-gols | 2.781 (90.6%) |
| Pênaltis | Removidos (xG fixo ~0.76, distorcem o modelo) |

---

## Features (variáveis de entrada)

O modelo recebe 12 variáveis para calcular o xG:

### Numéricas

| Feature | Descrição | Impacto |
|---|---|---|
| `distancia_gol` | Distância do chute até o centro do gol (metros) | Alto — quanto mais longe, menor o xG |
| `angulo_gol` | Ângulo do chute em relação ao gol | Alto — ângulos fechados reduzem xG |
| `minuto` | Minuto do jogo em que o chute ocorreu | Baixo |
| `periodo` | Período do jogo (1º ou 2º tempo) | Baixo |
| `sob_pressao` | Se o jogador estava sob pressão defensiva | Médio |

### Categóricas (one-hot encoding)

| Feature | Valores possíveis |
|---|---|
| `parte_corpo` | Pé Direito, Pé Esquerdo, Cabeça |
| `tipo_jogada` | Jogo Aberto, Escanteio, Falta, Chute Inicial |

### Importância das features

```
distancia_gol      ████████████████  15.1%  (mais importante)
jogada_Jogo Aberto ████████████       10.5%
corpo_Cabeça       ████████████       10.5%
angulo_gol         ███████████         9.3%
jogada_Falta       ███████████         9.0%
sob_pressao        ████████            7.2%
corpo_Pé Direito   ███████             6.8%
distancia_gol      ██████              5.9%
```

---

## Algoritmo — XGBoost

**Por que XGBoost?**

XGBoost (Extreme Gradient Boosting) é um algoritmo de ensemble que constrói várias árvores de decisão em sequência, onde cada árvore corrige os erros da anterior. É o algoritmo mais usado em analytics esportivo por:

- Alta performance em dados tabulares
- Lida bem com desbalanceamento de classes
- Feature importance interpretável
- Velocidade de treino

**Hiperparâmetros utilizados:**

| Parâmetro | Valor | Motivo |
|---|---|---|
| `n_estimators` | 300 | Número de árvores |
| `max_depth` | 4 | Profundidade máxima — evita overfitting |
| `learning_rate` | 0.05 | Taxa de aprendizado conservadora |
| `subsample` | 0.8 | 80% dos dados por árvore — regularização |
| `colsample_bytree` | 0.8 | 80% das features por árvore |
| `scale_pos_weight` | 9.7 | Corrige desbalanceamento (90% não-gol vs 10% gol) |

---

## Métricas de Avaliação

| Métrica | Valor | Interpretação |
|---|---|---|
| **AUC-ROC** | **0.7617** | Em 76% das vezes o modelo ordena corretamente gol vs não-gol |
| **CV AUC** | **0.7226 ± 0.0108** | Estável em cross-validation — sem overfitting |
| **Brier Score** | **0.1337** | Erro quadrático médio das probabilidades |
| **Log Loss** | **0.4160** | Penaliza predições confiantes e erradas |

### O que significa AUC de 0.76?

```
AUC = 0.5   → Modelo aleatório (inútil)
AUC = 0.76  → Nosso modelo
AUC = 0.85  → Modelos profissionais (Opta, StatsBomb)
AUC = 1.0   → Modelo perfeito (impossível)
```

Com apenas 2 Copas do Mundo (3.068 chutes), AUC de 0.76 é um resultado sólido. Modelos comerciais usam milhões de chutes de múltiplas ligas.

### Cross-Validation (5 folds)

O modelo foi avaliado com 5-fold stratified cross-validation para garantir que o resultado não é fruto de um split de dados favorável:

```
Fold 1: AUC = 0.7118
Fold 2: AUC = 0.7334
Fold 3: AUC = 0.7226
Fold 4: AUC = 0.7312
Fold 5: AUC = 0.7140
─────────────────────
Média:  0.7226 ± 0.0108
```

O desvio padrão baixo (0.01) indica que o modelo é estável.

---

## Como o Modelo é Usado no Projeto

### 1. Análise de partidas históricas

Para cada partida das Copas 2018 e 2022, calculamos:

```
xG da seleção = soma dos xG de cada chute na partida
xGA da seleção = soma dos xG cedidos ao adversário
```

**Exemplo — Brasil x Sérvia (Copa 2022):**
```
Brasil:  8 chutes → xG total = 2.3
Sérvia:  5 chutes → xG total = 0.7
Placar:  Brasil 2 x 0 Sérvia

Interpretação: Brasil dominou as oportunidades (xG 2.3 vs 0.7)
e o placar refletiu bem a qualidade das chances criadas.
```

### 2. Feature para o modelo de predição

O xG acumulado de cada seleção nas partidas anteriores vira uma feature do modelo de predição de resultado:

```
feat_predicao["casa_xg_media"] = média do xG por jogo da seleção da casa
feat_predicao["fora_xga_media"] = média do xGA por jogo da seleção visitante
```

### 3. Visualização no dashboard

O dashboard exibe o mapa de campo com cada chute colorido pelo xG calculado:

```
🔴 xG > 0.3  → alta probabilidade (vermelho)
🟡 xG 0.1-0.3 → probabilidade média (amarelo)
🔵 xG < 0.1  → baixa probabilidade (azul)
```

---

## Limitações

| Limitação | Impacto | Mitigação |
|---|---|---|
| Poucos dados (3.068 chutes) | AUC abaixo do ideal | Adicionar mais ligas da StatsBomb |
| Sem dados de posição do goleiro | Perde informação importante | Limitação da fonte de dados |
| Sem dados de posição dos defensores | Subestima dificuldade de alguns chutes | Limitação da fonte de dados |
| Treinado só em Copas | Pode não generalizar bem para eliminatórias | Aceitável para o escopo do projeto |

---

## Como Melhorar o Modelo (trabalhos futuros)

1. **Mais dados** — Adicionar Premier League, La Liga, Champions League da StatsBomb (disponíveis gratuitamente)
2. **Novas features** — Posição do goleiro, número de defensores entre o chutador e o gol
3. **Redes neurais** — Com mais dados, uma MLP pode superar o XGBoost
4. **Ensemble** — Combinar XGBoost + LightGBM + Regressão Logística

---

## Arquivos Gerados

```
models/
├── xg_model.json            → modelo treinado (XGBoost)
├── xg_model_meta.json       → features, métricas e parâmetros
└── xg_model_evaluation.png  → gráficos de avaliação
```

---

## Como Usar o Modelo

```python
import xgboost as xgb
import json
import pandas as pd

# Carrega o modelo
modelo = xgb.XGBClassifier()
modelo.load_model("models/xg_model.json")

# Carrega metadados
with open("models/xg_model_meta.json") as f:
    meta = json.load(f)

features = meta["features"]

# Calcula xG de um chute
chute = pd.DataFrame([{
    "distancia_gol": 15.0,
    "angulo_gol":    0.3,
    "minuto":        67,
    "periodo":       2,
    "sob_pressao":   0,
    "corpo_Pé Direito": 1,
    "corpo_Cabeça":     0,
    "corpo_Pé Esquerdo":0,
    "jogada_Jogo Aberto": 1,
    "jogada_Escanteio":   0,
    "jogada_Falta":       0,
}])

# Garante mesmas colunas do treino
for col in features:
    if col not in chute.columns:
        chute[col] = 0
chute = chute[features]

xg = modelo.predict_proba(chute)[0][1]
print(f"xG: {xg:.3f} ({xg*100:.1f}% de chance de gol)")
```

---

## Referências

- [StatsBomb xG Model](https://statsbomb.com/articles/soccer/statsbomb-release-expected-goals/) — referência de como modelos profissionais são construídos
- [XGBoost Documentation](https://xgboost.readthedocs.io/) — documentação oficial do algoritmo
- [Friends of Tracking](https://www.youtube.com/channel/UCUBFJYcag8j2rm_9HkrrA7w) — canal do YouTube com tutoriais de analytics esportivo

---

*Projeto: Copa do Mundo 2026 Data Platform — Data Science*
