# Notebooks (wokibi-eval)

Exploração offline de **avaliação** de recommender (UMAP, MAP/recall).

## Ambiente

1. Na raiz de **wokibi-eval**: `poetry install` (instala `wokibi_eval` no venv).
2. Opcional: `poetry install --with dev` para Jupyter.
3. Copie [`.env.example`](../.env.example) → `.env` na raiz do repo (ou filtre chaves do `wokibi-app/.env`).
4. Inicie Jupyter com o **interpretador do venv** do projeto:

   ```bash
   poetry run jupyter lab
   ```

No VS Code/Cursor, selecione o Python de `wokibi-eval/.venv`.

## Imports

| Notebook | Imports típicos |
|----------|-----------------|
| `recommendation/1.0-embeddings-view.ipynb` | `from wokibi_eval.evaluation.embedding import get_umap, plot_umap` |
| `recommendation/2.0-wokirec-test-user-item.ipynb` | `from wokibi_eval.evaluation.ranking import …`; `from wokibi_eval.recommender.evaluation import eval_recommender` |

Pacotes publicados: `wokibi_ai`, `wokibi_data`, `cliid`.
