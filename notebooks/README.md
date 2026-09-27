# Notebooks (wokibi-eval)

Exploração offline de **avaliação**: recommender (UMAP, MAP/recall) e pipeline clínico (`detalhe` vs `detalhe_{model}`).

## Ambiente

1. Na raiz de **wokibi-eval**: `poetry install` (instala `wokibi_eval` no venv).
2. Jupyter: `poetry install --with dev`.
3. Notebook clínico: `poetry install --with clinical` e `poetry run python -m spacy download pt_core_news_lg`.
4. Copie [`.env.example`](../.env.example) → `.env` na raiz (`WOKIBI_RUNTIME`, Mongo, `GOOGLE_API_KEY`; juiz alternativo comentado no example).
5. Inicie Jupyter com o **interpretador do venv**:

   ```bash
   poetry run jupyter lab
   ```

No VS Code/Cursor, selecione o Python de `wokibi-eval/.venv`.

Imports: use `wokibi_eval.*` (sem `sys.path`). Pacotes publicados: `wokibi_ai`, `wokibi_data`, `cliid`.

## Índice

| Notebook | Domínio | Entradas | Saídas / artefatos |
|----------|---------|----------|-------------------|
| [recommendation/1.0-embeddings-view.ipynb](recommendation/1.0-embeddings-view.ipynb) | Recommender | Mongo (embeddings individuals + catalogs) | Gráfico UMAP 2D |
| [recommendation/2.0-wokirec-test-user-item.ipynb](recommendation/2.0-wokirec-test-user-item.ipynb) | Recommender | Parquet local, Mongo, `Recommender` | MAP@k, recall@k, `eval_recommender` |
| [clinical/1.0-events-detalhe-eval.ipynb](clinical/1.0-events-detalhe-eval.ipynb) | Clínico | Mongo (`detalhe`, `detalhe_{model}`) | `README.md` na pasta + artefatos por modelo |

## `recommendation/1.0-embeddings-view.ipynb`

**Objetivo:** visualizar embeddings de **indivíduos** e **itens de catálogo** no mesmo parceiro, em 2D (UMAP), para inspecionar separação partner vs product.

**Imports principais:** `wokibi_eval.evaluation.embedding` (`get_umap`, `plot_umap`), `wokibi_data.service.mongo.MongoInterface`.

**Passo a passo:**

1. Definir `PARTNER` e os nomes das bases Mongo (`DATABASE_INDIVIDUALS`, `DATABASE_CATALOGS`) — alinhar sufixo `v*` com `data_version` do parceiro em `wokibi_data.config.partner_parameters`.
2. Conectar duas coleções via `MongoInterface` + `set_collection(PARTNER)`.
3. Carregar arrays com `get_embeddings_array`: campo `embeddings_avg` (individuals) e `embedding_gpt` (catalogs), label `nome`.
4. Concatenar embeddings; relabelar categorias `partner` / `product` para colorir no plot.
5. `get_umap(embeddings_array, labels)` → DataFrame com `x`, `y`; `plot_umap(...)` com título.
6. Explorar: `df.shape`, `sample`, filtros espaciais (`df.query(...)`).

**Ajustes típicos:** trocar parceiro e versões das bases Mongo quando mudar cohort.

---

## `recommendation/2.0-wokirec-test-user-item.ipynb`

**Objetivo:** medir se o **ranking** do recommender (embedding + reorder opcional via LLM) antecipa procedimentos que o paciente realizou **depois** de uma data de corte.

**Racional:** treino/histórico implícito até `reference_date`; eventos com `EVENT_DATE > reference_date` são o “ground truth” futuro.

**Imports principais:** `wokibi_eval.evaluation.ranking` (`mean_average_precision_at_k`, `recall_at_k`), `wokibi_eval.recommender.evaluation.eval_recommender`, `wokibi_ai.models.recommender.recommender.Recommender`.

**Passo a passo:**

1. Configurar `PARTNER`, `reference_date`, `embedding_user` / `embedding_item`, `individual_id` de teste.
2. Instanciar `Recommender(PARTNER, k=50)`.
3. Carregar `events.parquet` via DuckDB (`LOCAL_PATH` + query no mart `Events`).
4. Carregar clientes do Mongo (`IndividualBean` → DataFrame).
5. Filtrar eventos dos clientes com data após `reference_date` → conjunto de teste temporal.
6. `recommender.itens_by_user(individual_id)` → histórico e lista de candidatos.
7. `test_items` = `catalog_id` únicos do indivíduo no período futuro.
8. Baseline: `predict_items` = top-k por embedding (excluindo itens já no histórico) → `recall_at_k` / `mean_average_precision_at_k`.
9. LLM reorder: `get_data_by_ids` + `reorder_ranking` → parse JSON (`clean_text_to_json`) → MAP/recall após reorder.
10. Batch: `eval_recommender(memory, recommender, df_events, PARTNER, "id")` → agregar `np.mean(result_memory["users_recalls"])`, `users_maps`.

**Dependências:** parquet em `LOCAL_PATH`, Mongo, API LLM para `reorder_ranking`.

---

## `clinical/1.0-events-detalhe-eval.ipynb`

**Objetivo:** tutorial do job [`pipelines/clinical/events_detalhe_eval.py`](../pipelines/clinical/events_detalhe_eval.py) — smoke `ClinicalEvaluator`, batch `EventsDetalheEvaluator`, leitura do CSV.

**Imports principais:** `wokibi_eval.clinical.EventsDetalheEvaluator`, `wokibi_eval.evaluation.clinical_evaluator.ClinicalEvaluator`, `wokibi_eval.paths.clinical_detalhe_eval_dir`.

**Passo a passo:**

1. Configurar `DATASET`, `MODEL_NAME`, `TIPO_EVENTO`, `ENABLE_JUDGE`, `JUDGE_MODEL`; conferir caminho do CSV via `clinical_detalhe_eval_dir`.
2. Smoke em par sintético com `ClinicalEvaluator.evaluate` (sem Mongo).
3. `EventsDetalheEvaluator(...).run(break_on=True, max_events=1)` — um evento.
4. Amostra com `break_on=False` e `max_events` (paginação em lotes de 100 no backend).
5. `pd.read_csv` + agregados das métricas (compressão, NER, BioBERT, juiz).
6. Filtro de outliers; idempotência por `event_id` no CSV; `README.md` único na pasta; cohort JSONL (PHI — repo privado).

**Pré-requisitos extras:** grupo Poetry `clinical`, modelo spaCy `pt_core_news_lg`, Mongo com `detalhe` e `detalhe_{model}` preenchidos.

**CLI equivalente:** ver célula “Anexo” no notebook ou [README raiz](../README.md#job-batch-detalhe).
