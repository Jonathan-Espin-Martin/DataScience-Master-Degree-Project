from pytorch_classifier import PyTorchClassifier, ANNModel
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

    #load the classifier and put on cuda device base 0
    classifier_path = os.path.join(model_dir, "predictor", 'pytorch_classifier.pth')
    classifier = torch.load(classifier_path, map_location=torch.device('cuda:0'))
    classifier.model.to('cuda:0')
    classifier.model.eval()

    #load the random search results
    random_search_results_path = os.path.join(model_dir, "predictor", "random_search_results.pkl")
    random_search_results = joblib.load(random_search_results_path)

    #extract best_params and best_score from the random search results
    best_params = random_search_results.get('best_params', {})
    best_score = random_search_results.get('best_score', None)

    #model attributes
    all_attributes = dir(classifier)
    model_attributes = [
        attr for attr in all_attributes 
        if not attr.startswith('__') and not callable(getattr(classifier, attr))
    ]
    model_description = {attr: getattr(classifier, attr) for attr in model_attributes}

    #JSON results
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

    #save results to a JSON file
    results_path = os.path.join(model_dir, "attributes_data.json")
    with open(results_path, 'w') as json_results:
        json.dump(serialized_results, json_results, indent=4)

if __name__ == "__main__":
    main()