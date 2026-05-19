# ⚽ Copa do Mundo 2026 — Data Platform

Dashboard analítico completo da Copa do Mundo 2026 com pipeline de dados automatizado, modelos de Machine Learning (xG, predição de resultados, clustering de seleções) e visualizações táticas interativas.

---

## 🏗️ Arquitetura

```
API-Football / StatsBomb
        ↓
   [ RAW ]  → JSON bruto salvo localmente
        ↓
  [ BRONZE ] → Parquet tipado (Pandas + PyArrow)
        ↓
  [ SILVER ] → Dados limpos e normalizados (dbt + DuckDB)
        ↓
   [ GOLD ]  → Métricas analíticas + features ML (dbt)
        ↓
  Streamlit Dashboard + Modelos ML
```

---

## 🚀 Setup inicial (primeiro passo)

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/copa2026-data-platform.git
cd copa2026-data-platform
```

### 2. Crie o ambiente virtual
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Configure as credenciais
```bash
cp .env.example .env
# Edite o .env e adicione sua chave da API-Football
```

> **Como obter a chave gratuita:**
> 1. Acesse https://www.api-football.com
> 2. Clique em "Get your free API key"
> 3. Crie sua conta (gratuita)
> 4. Copie a chave e cole no `.env`

### 5. Teste a conexão com a API
```bash
python src/ingestion/fetch_matches.py
```

### 6. Explore o que chegou
```bash
python src/ingestion/explore_raw.py 2026-06-15
```

---

## 📁 Estrutura do projeto

```
copa2026/
├── data/
│   ├── raw/          # JSONs brutos da API (não versionado)
│   ├── bronze/       # Parquet tipado
│   ├── silver/       # Dados limpos (DuckDB)
│   └── gold/         # Tabelas analíticas (DuckDB)
│
├── src/
│   ├── ingestion/    # Scripts de coleta da API
│   ├── pipeline/     # Transformações bronze → gold
│   ├── models/       # Modelos de ML (xG, predição, clustering)
│   └── dashboard/    # Aplicação Streamlit
│
├── notebooks/        # Análises exploratórias
├── tests/            # Testes automatizados
├── .github/
│   └── workflows/    # Pipeline automático diário
│
├── .env.example
├── .gitignore
└── requirements.txt
```

---

## 📊 Funcionalidades

| Módulo | Status | Descrição |
|--------|--------|-----------|
| Ingestão diária | ✅ | Busca resultados do dia via API |
| Pipeline dbt | 🔜 | Transformações bronze → gold |
| Modelo xG | 🔜 | Expected Goals por chute |
| Predição de resultado | 🔜 | Probabilidade por partida |
| Clustering | 🔜 | Estilos de jogo por seleção |
| Dashboard Streamlit | 🔜 | Visualização interativa pública |

---

## 🔄 Atualização automática

O pipeline roda automaticamente todo dia às **23h (horário de Brasília)** via GitHub Actions. Para rodar manualmente:

```bash
# Busca dados de hoje
python src/ingestion/fetch_matches.py

# Busca dados de uma data específica
python src/ingestion/fetch_matches.py 2026-06-15
```

---

## 🛠️ Stack tecnológica

| Camada | Ferramenta | Custo |
|--------|-----------|-------|
| Ingestão | API-Football | Gratuito (100 req/dia) |
| Storage | Pasta local + Parquet | Gratuito |
| Processamento | dbt Core + DuckDB | Gratuito |
| Qualidade | Great Expectations | Gratuito |
| ML | scikit-learn, XGBoost | Gratuito |
| Tracking ML | MLflow | Gratuito |
| Dashboard | Streamlit | Gratuito |
| Hospedagem | Streamlit Community Cloud | Gratuito |
| Orquestração | GitHub Actions | Gratuito |

**Custo total: R$ 0,00** 🎉

---

## 📚 Próximos passos

1. [ ] Configurar `.env` com a chave da API
2. [ ] Testar `fetch_matches.py` e explorar o JSON retornado
3. [ ] Criar pipeline bronze com Pandas + PyArrow
4. [ ] Configurar dbt com DuckDB
5. [ ] Criar modelos silver (normalização e limpeza)
6. [ ] Criar modelos gold (métricas agregadas)
7. [ ] Desenvolver modelo de xG
8. [ ] Construir dashboard Streamlit
9. [ ] Publicar no Streamlit Community Cloud

---

## 👨‍💻 Autor
Vinicius Florido Leite.

Desenvolvido como projeto de portfólio — Data Science
