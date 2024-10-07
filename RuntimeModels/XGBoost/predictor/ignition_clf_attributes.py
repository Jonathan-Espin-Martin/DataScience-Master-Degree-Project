from sklearn.metrics import fbeta_score
import joblib
import json
import math
import sys
import os


def f2_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    y_pred_label = (y_pred > 0.5).astype(int)
    return 'f2', fbeta_score(y_true, y_pred_label, beta=2)

def main():

    model_dir = sys.argv[1]

    #load XGBoost model
    model_path = os.path.join(model_dir, "predictor", "trained_xgboost_pipeline_model.pkl")
    best_model = joblib.load(model_path)

    #load features from training
    feature_names_filename = os.path.join(model_dir, "predictor", "feature_names.pkl")
    feature_names = joblib.load(feature_names_filename)

    #load random search results
    random_search_results_path = os.path.join(model_dir, "predictor", "random_search_results.pkl")
    random_search_results = joblib.load(random_search_results_path)

    #extract best_params and best_score from the random search results
    best_params = random_search_results.get('best_params', {})
    best_score = random_search_results.get('best_score', None)

    #get classifier
    clf = best_model.named_steps['clf']
    clf.get_booster().set_param({'device': 'cuda:0'})

    #get all attributes of the XGBClassifier
    all_attributes = dir(clf)
    model_attributes = {}
    for attr in all_attributes:
        if not attr.startswith('__') and not callable(getattr(clf, attr, None)):
            try:
                model_attributes[attr] = getattr(clf, attr)
            except AttributeError:
                model_attributes[attr] = 'N/A'

    #create a dictionary of the selected attributes and their corresponding values
    model_description = {attr: getattr(clf, attr, 'N/A') for attr in model_attributes}

    #results as a JSON object
    results = {
        "best_params": best_params,
        "best_score": best_score,
        "model_description": model_description
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

    #JSON-formatted results
    serialized_results = serialize(results)
    print(serialized_results)

    results_path = os.path.join(model_dir, "attributes_data.json")
    with open(results_path, 'w') as json_results:
        json.dump(serialized_results, json_results, indent=4)

if __name__ == "__main__":
    main()