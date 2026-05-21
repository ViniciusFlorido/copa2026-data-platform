# Documentação — Clustering de Seleções por Estilo de Jogo

## Visão Geral

O modelo de **Clustering** agrupa as seleções da Copa do Mundo por similaridade de estilo de jogo, sem usar rótulos predefinidos. O modelo descobre sozinho quais times jogam de forma parecida.

---

## O que é Clustering?

Diferente dos modelos de xG e predição — que são **supervisionados** (aprendem com respostas certas) — o clustering é **não supervisionado**: ele não sabe de antemão quais grupos existem, apenas encontra padrões nos dados.

```
Supervisionado (xG, Predição):
  Input → modelo → resposta conhecida (gol/não gol, V/E/D)

Não supervisionado (Clustering):
  Input → modelo → grupos descobertos automaticamente
```

---

## Dados Utilizados

| Fonte | Seleções | Descrição |
|---|---|---|
| StatsBomb — Copas 2018 e 2022 | 40 | Métricas táticas detalhadas |
| Kaggle — Eliminatórias 2026 | 198 | Contexto recente |
| **Total processado** | **40** | Seleções com histórico de Copa |

O clustering foi feito com as 40 seleções que participaram das Copas 2018 e 2022, pois são as únicas com dados táticos completos da StatsBomb.

---

## Features Utilizadas

| Feature | Descrição |
|---|---|
| `aproveitamento_copa` | % de pontos conquistados nas Copas |
| `gols_pro_media_copa` | Média de gols marcados por jogo |
| `gols_con_media_copa` | Média de gols sofridos por jogo |
| `passes_media_copa` | Média de passes por jogo |
| `precisao_passes_copa` | % de precisão de passes |
| `chutes_media_copa` | Média de chutes por jogo |
| `pressoes_media_copa` | Média de ações de pressão por jogo |

---

## Algoritmo — K-Means + PCA

### K-Means

K-Means agrupa os dados em **k** grupos minimizando a distância de cada ponto ao centro do seu grupo. O algoritmo:

1. Escolhe k centros aleatórios
2. Atribui cada seleção ao centro mais próximo
3. Recalcula os centros
4. Repete até convergir

### Encontrando o k ideal

Testamos de k=2 a k=8 usando duas métricas:

| k | Inertia | Silhouette Score |
|---|---|---|
| 2 | 422.7 | 0.203 |
| **3** | **341.1** | **0.232** ← escolhido |
| 4 | 294.3 | 0.209 |
| 5 | 257.7 | 0.219 |
| 6 | 231.5 | 0.230 |
| 7 | 205.2 | 0.191 |
| 8 | 186.7 | 0.209 |

**k=3 foi escolhido** por ter o melhor Silhouette Score (0.232).

**O que é Silhouette Score?**

```
-1.0 → seleções estão no cluster errado
 0.0 → seleções na fronteira entre clusters
+1.0 → seleções bem agrupadas no cluster certo

0.232 → agrupamento razoável dado o número
         pequeno de seleções (40)
```

### PCA — Visualização 2D

Como temos 7 features, usamos PCA (Principal Component Analysis) para reduzir para 2 dimensões e plotar no gráfico:

```
PC1 (33.7%) → captura diferenças de qualidade geral
PC2 (19.4%) → captura diferenças de estilo de jogo
─────────────────────────────────────────────────
Total: 53.1% da variância explicada
```

53.1% é razoável para visualização — significa que o gráfico 2D captura mais da metade da informação original.

---

## Resultados — Os 3 Clusters

### 🏆 Cluster 2 — Elite: Alta posse e eficiência

**Aproveitamento médio: 57.6% | Precisão passes: 83.9%**

```
Alemanha, Argentina, Brasil, Bélgica, Croácia,
Espanha, França, Holanda, Inglaterra, Portugal...
```

São as seleções com histórico sólido em Copas — alto aproveitamento, boa precisão de passes e volume ofensivo. Dominam tecnicamente o jogo e chegam consistentemente nas fases finais. São as favoritas naturais para a Copa 2026.

---

### ⚖️ Cluster 0 — Equilibrado (nível médio-alto)

**Aproveitamento médio: 33.7% | Precisão passes: 79.1%**

```
Arábia Saudita, Austrália, Canadá, Catar,
Colômbia, Coreia do Sul, Dinamarca, Equador...
```

Seleções competitivas que chegam à Copa com capacidade de surpreender — como a Coreia do Sul que chegou às semifinais em 2002 e a Dinamarca que é sempre difícil de bater. Precisão de passes acima da média mas aproveitamento ainda abaixo das elites.

---

### ⚖️ Cluster 1 — Equilibrado (nível médio-baixo)

**Aproveitamento médio: 31.3% | Precisão passes: 74.0%**

```
Camarões, Costa Rica, Egito, Gana,
Irã, Islândia, Marrocos, Nigéria...
```

Seleções que representam bem suas confederações mas historicamente têm mais dificuldade nas Copas. Destaque para Marrocos, que na Copa 2022 chegou às semifinais — provando que esse cluster tem potencial de surpresa. Menor precisão de passes indica um jogo mais direto e físico.

---

## Interpretação Tática

O clustering revelou uma divisão clara entre três perfis:

```
Qualidade técnica
      ↑
  83.9% ─ 🏆 Elite (Cluster 2)
           Brasil, Arg, Fra, Ale...
           
  79.1% ─ ⚖️ Médio-alto (Cluster 0)
           Dinamarca, Coreia do Sul...
           
  74.0% ─ ⚖️ Médio-baixo (Cluster 1)
           Marrocos, Gana, Irã...
      ↓
      33%    34%    58%   → Aproveitamento
```

A principal dimensão de separação é a **qualidade geral** — aproveitamento e precisão de passes. Uma segunda dimensão capta diferenças de estilo (posse vs jogo direto), mas com apenas 40 seleções o sinal é mais fraco.

---

## Limitações

| Limitação | Impacto | Mitigação |
|---|---|---|
| Só 40 seleções (Copas 2018/2022) | Silhouette Score baixo (0.232) | Adicionar mais Copas históricas |
| Seleções novas na Copa 2026 | Não aparecem no clustering | Usar eliminatórias para estimar cluster |
| PCA captura só 53.1% | Visualização perde informação | Aceitar como aproximação visual |
| Dois clusters com mesmo nome | Difícil distinguir visualmente | Nomes refinados conforme Copa 2026 avança |

---

## Como é Usado no Dashboard

O clustering alimenta duas visualizações no dashboard:

**1. Mapa de estilos (scatter PCA)**
Cada seleção aparece como um ponto no gráfico — seleções próximas jogam de forma similar. Cores diferentes por cluster.

**2. Análise de confronto**
Quando o usuário seleciona duas seleções para ver a predição de resultado, o dashboard mostra se são do mesmo cluster (estilos parecidos) ou de clusters diferentes (estilos contrastantes).

```python
# Exemplo de uso no dashboard
import pandas as pd

clusters = pd.read_parquet("models/clusters.parquet")

# Busca cluster de uma seleção
def get_cluster(selecao: str) -> dict:
    row = clusters[clusters["time_nome"] == selecao]
    if row.empty:
        return None
    return {
        "cluster":      row.iloc[0]["cluster"],
        "nome":         row.iloc[0]["cluster_nome"],
        "aproveitamento": row.iloc[0]["aproveitamento_copa"],
    }

brasil   = get_cluster("Brasil")
marrocos = get_cluster("Marrocos")

print(f"Brasil:   Cluster {brasil['cluster']} — {brasil['nome']}")
print(f"Marrocos: Cluster {marrocos['cluster']} — {marrocos['nome']}")
```

---

## Diferença entre os Três Modelos

| | xG | Predição | Clustering |
|---|---|---|---|
| **Tipo** | Supervisionado | Supervisionado | Não supervisionado |
| **O que calcula** | Prob. de gol por chute | Prob. de resultado | Grupos por estilo |
| **Input** | Características do chute | Métricas das seleções | Métricas das seleções |
| **Output** | 0 a 1 (xG) | V% / E% / D% | Cluster 0, 1 ou 2 |
| **Algoritmo** | XGBoost | LightGBM | K-Means + PCA |
| **Uso no projeto** | Análise tática | Predição de partidas | Mapa de estilos |

---

## Arquivos Gerados

```
models/
├── clusters.parquet          → seleções com cluster e coordenadas PCA
├── clustering_meta.json      → k, features e nomes dos clusters
└── clustering_evaluation.png → gráficos elbow, silhouette e PCA
```

---

## Referências

- [K-Means Clustering — scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) — documentação oficial
- [PCA — scikit-learn](https://scikit-learn.org/stable/modules/decomposition.html#pca) — documentação oficial
- [Clustering Football Teams](https://www.kaggle.com/code/davidcariboo/player-scores) — benchmark de clustering em futebol

---

*Projeto: Copa do Mundo 2026 Data Platform — FIAP Data Science*
