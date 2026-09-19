
import numpy as np
from typing import List, Any, Set

# ==============================================================================
# 1. FUNÇÃO AUXILIAR (sem alterações)
# ==============================================================================


def _preparar_dados_para_metrica(data: List[Any]) -> List[Set[Any]]:
    """
    Converte uma lista de entrada para o formato padrão: uma lista de conjuntos.
    """
    # Lida com o caso de a entrada ser um array numpy
    if isinstance(data, np.ndarray):
        # Transforma o array em uma lista antes de processar
        data = data.tolist()

    if not isinstance(data, list) or len(data) == 0:
        return []

    is_nested_list = any(isinstance(i, (list, set, np.ndarray)) for i in data)
    if not is_nested_list:
        data = [data]

    return [set(row) for row in data]

# ==============================================================================
# 2. FUNÇÕES DE MÉTRICA ATUALIZADAS
# ==============================================================================


# <-- k padrão agora é 0
def recall_at_k(y_true: List[Any], y_pred: List[Any], k: int = 0) -> float:
    """
    Calcula o Macro-Average Recall @ k.
    Se k=0, usa o comprimento total da lista de predições para cada usuário.
    """
    y_true_sets = _preparar_dados_para_metrica(y_true)

    # Garante que y_pred seja uma lista de listas para consistência
    is_nested_pred = any(isinstance(i, (list, np.ndarray)) for i in y_pred)
    if not is_nested_pred:
        y_pred = [y_pred]

    recalls = []

    for i in range(len(y_true_sets)):
        true_set = y_true_sets[i]

        if not true_set:
            recalls.append(0.0)
            continue

        if i >= len(y_pred):
            recalls.append(0.0)
            continue

        pred_list_full = y_pred[i]

        # --- MUDANÇA PRINCIPAL AQUI ---
        # Se k for 0, o limite é o tamanho total da lista de previsões.
        # Caso contrário, usa o k especificado.
        limit = k if k > 0 else len(pred_list_full)

        pred_list_sliced = pred_list_full[:limit]

        hits = len(true_set.intersection(set(pred_list_sliced)))
        recall = hits / len(true_set)
        recalls.append(recall)

    if not recalls:
        return 0.0

    return sum(recalls) / len(recalls)


# <-- k padrão agora é 0
def mean_average_precision_at_k(y_true: List[Any], y_pred: List[Any], k: int = 0) -> float:
    """
    Calcula o Mean Average Precision (MAP) @ k.
    Se k=0, usa o comprimento total da lista de predições para cada usuário.
    """
    y_true_sets = _preparar_dados_para_metrica(y_true)

    is_nested_pred = any(isinstance(i, (list, np.ndarray)) for i in y_pred)
    if not is_nested_pred:
        y_pred = [y_pred]

    average_precisions = []

    for i in range(len(y_true_sets)):
        true_set = y_true_sets[i]

        if not true_set:
            average_precisions.append(0.0)
            continue

        if i >= len(y_pred):
            average_precisions.append(0.0)
            continue

        pred_list_full = y_pred[i]

        # --- MUDANÇA PRINCIPAL AQUI ---
        limit = k if k > 0 else len(pred_list_full)
        pred_list_sliced = pred_list_full[:limit]

        hits = 0
        precisions = []
        for j, item in enumerate(pred_list_sliced):
            if item in true_set:
                hits += 1
                precision_at_j = hits / (j + 1)
                precisions.append(precision_at_j)

        if not precisions:
            average_precisions.append(0.0)
        else:
            avg_precision = sum(precisions) / len(true_set)
            average_precisions.append(avg_precision)

    if not average_precisions:
        return 0.0

    return sum(average_precisions) / len(average_precisions)
