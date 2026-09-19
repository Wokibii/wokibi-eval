
import numpy as np
from tqdm import tqdm

from wokibi_eval.evaluation.ranking import mean_average_precision_at_k, recall_at_k

from wokibi_data.beans.mongo.individual import IndividualBean


def eval_recommender(memory, recommender, df_events, partner, client_id_column):

    y_true_geral = []
    y_pred_geral = []
    users_maps = []
    users_recalls = []

    count = 0
    count_without_events = 0
    
    individual_bean = IndividualBean(partner)
    individuals = individual_bean.getIndividualsWithEmbeddings(partner)
    
    for individual in tqdm(individuals):
        individual = individual.to_dict() # Use dict for compatibility with existing code
        individual_id = individual["id"]

        # Check if id already exists in memory
        if individual_id in memory:
            print(
                f"Individual {individual_id} already exists in memory, skipping...")
            continue

        train_items = individual['items']

        df = df_events[df_events[client_id_column]
                          == individual["id"]]
        if len(df) == 0:
            count_without_events += 1
            continue

        test_items = df["catalog_id"].unique()

        user, ids = recommender.itens_by_user(individual["id"])
        predict_items = np.array(
            [p for p in ids if p not in train_items])

        print("Individual: ", individual_id, ": train ", len(
            train_items), " test ", len(test_items), " similar ", len(predict_items))

        # Calculate metrics
        map_score = mean_average_precision_at_k(
            test_items, predict_items)
        recall_score = recall_at_k(test_items, predict_items)

        # Store results in memory dictionary
        memory[individual_id] = {
            'y_true': test_items,
            'y_pred': predict_items,
            'map_score': map_score,
            'recall_score': recall_score,
            'train_items': train_items
        }

        # Also append to global lists for backward compatibility
        y_true_geral.append(test_items)
        y_pred_geral.append(predict_items)
        users_maps.append(map_score)
        users_recalls.append(recall_score)

        count += 1

    print("Total individuals: ", count)
    print("Total individuals without events: ", count_without_events)

    count_with_events = count - count_without_events
    print("Total individuals with events: ", count_with_events)

    return {
        'memory': memory,
        'y_true_geral': y_true_geral,
        'y_pred_geral': y_pred_geral,
        'users_maps': users_maps,
        'users_recalls': users_recalls,
        'count': count,
        'count_without_events': count_without_events,
        'count_with_events': count_with_events
    }
