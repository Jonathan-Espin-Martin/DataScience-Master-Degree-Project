from pytorch_classifier import PyTorchClassifier, ANNModel
import pandas as pd
import joblib
import torch
import json
import math
import sys
import os

def main():

    #set torch device to cuda:0
    torch.cuda.set_device(0)

    #model dir
    model_dir = sys.argv[1]
    file_data_name = sys.argv[2]

    #load the feature names
    feature_names_filename = os.path.join(model_dir, "predictor", "feature_names.pkl")
    feature_names = joblib.load(feature_names_filename)

    #load the classifier and put on cuda device base 0
    classifier_path = os.path.join(model_dir, "predictor", 'pytorch_classifier.pth')
    classifier = torch.load(classifier_path, map_location=torch.device('cuda:0'))
    classifier.model.to('cuda:0')
    classifier.model.eval()

    #load the prediction data
    data_path = os.path.join(model_dir, file_data_name)
    with open(data_path, 'r') as openfile:
        prediction_data = json.load(openfile)
    predict_df = pd.DataFrame.from_dict(prediction_data)
    predict_df = predict_df.dropna()
    predict_df = predict_df.drop_duplicates()

    if not predict_df.empty:
        
        #feature as tensor on device
        features = predict_df[feature_names]
        features_tensor = torch.tensor(features.values, dtype=torch.float32).to('cuda:0')

        #make ann predictions
        with torch.no_grad():
            predictions = classifier.model(features_tensor).squeeze()

        #convert the predictions to probabilities and calculate failure likelihood
        failure_probabilities = torch.sigmoid(predictions).cpu().numpy()

        #binary predictions (threshold = 0.5)
        binary_predictions = (failure_probabilities >= 0.5).astype(int).tolist()
        failure_probabilities = (failure_probabilities * 100).tolist()
        binary_predictions = [binary_predictions] if not isinstance(binary_predictions, list) else binary_predictions
        failure_probabilities = [failure_probabilities] if not isinstance(failure_probabilities, list) else failure_probabilities
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

    #JSON-formatted results
    serialized_results = serialize(results)
    print(serialized_results)

    #save results to a JSON file
    results_path = os.path.join(model_dir, "results_data.json")
    with open(results_path, 'w') as json_results:
        json.dump(serialized_results, json_results, indent=4)

if __name__ == "__main__":
    main()