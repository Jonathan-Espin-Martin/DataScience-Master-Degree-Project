import pandas as pd
import joblib
import json
import math
import sys
import os

def main():

    model_dir = sys.argv[1]
    file_data_name = sys.argv[2]

    #load the trained RandomForest model
    model_path = os.path.join(model_dir, "predictor", "trained_random_forest_pipeline_model.pkl")
    best_model = joblib.load(model_path)

    #load the feature names
    feature_names_filename = os.path.join(model_dir, "predictor", "feature_names.pkl")
    feature_names = joblib.load(feature_names_filename)

    #load the prediction data
    data_path = os.path.join(model_dir, file_data_name)
    with open(data_path, 'r') as openfile:
        prediction_data = json.load(openfile)

    #if prediction_data end
    if not prediction_data:
        print("{}")
        sys.exit(0)

    #predict data json to pd
    predict_df = pd.DataFrame.from_dict(prediction_data)
    predict_df = predict_df.dropna()
    predict_df = predict_df.drop_duplicates()

    if not predict_df.empty:
        #feaatures model
        features = predict_df[feature_names]
        predictions = best_model.predict(features)

        #predictions (threshold = 0.5)
        failure_probabilities = predictions
        binary_predictions = (failure_probabilities >= 0.5).astype(int)
        binary_predictions = binary_predictions.tolist()
        failure_probabilities = (failure_probabilities * 100).tolist()
    else:
        binary_predictions = [0]
        failure_probabilities = [0]

    #JSON results
    results = {
        "binary_predictions": max(binary_predictions),
        "failure_probabilities": max(failure_probabilities)
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

    results_path = os.path.join(model_dir, "results_data.json")
    with open(results_path, 'w') as json_results:
        json.dump(serialized_results, json_results, indent=4)

if __name__ == "__main__":
    main()