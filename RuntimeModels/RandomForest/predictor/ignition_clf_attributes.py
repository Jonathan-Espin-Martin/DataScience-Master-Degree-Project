import joblib
import json
import math
import sys
import os

def main():

    model_dir = sys.argv[1]

    #load the trained RandomForest model
    model_path = os.path.join(model_dir, "predictor", "trained_random_forest_pipeline_model.pkl")
    best_model = joblib.load(model_path)

    #load the random search results
    random_search_results_path = os.path.join(model_dir, "predictor", "random_search_results.pkl")
    random_search_results = joblib.load(random_search_results_path)

    #extract best_params and best_score from the random search results
    best_params = random_search_results.get('best_params', {})
    best_score = random_search_results.get('best_score', None)

    clf = best_model.named_steps.get('clf')
    if clf:
        all_attributes = dir(clf)

        #callable attributes
        model_attributes = {}
        for attr in all_attributes:
            if not attr.startswith('__') and not callable(getattr(clf, attr, None)):
                try:
                    model_attributes[attr] = getattr(clf, attr)
                except AttributeError:
                    model_attributes[attr] = 'N/A'
    else:
        model_attributes = {}

    #JSON attributes
    results = {
        "best_params": best_params,
        "best_score": best_score,
        "model_description": model_attributes
    }

    def serialize(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            return obj
        if isinstance(obj, dict):
            return {key: serialize(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [serialize(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(serialize(item) for item in obj)
        return str(obj)

    serialized_results = serialize(results)
    print(serialized_results)

    results_path = os.path.join(model_dir, "attributes_data.json")
    with open(results_path, 'w') as json_results:
        json.dump(serialized_results, json_results, indent=4)

if __name__ == "__main__":
    main()