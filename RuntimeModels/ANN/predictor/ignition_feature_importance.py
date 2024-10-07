from pytorch_classifier import PyTorchClassifier, ANNModel
import pandas as pd
import joblib
import torch
import sys
import os

def main():
    model_dir = sys.argv[1]

    #load model ann
    model_path = os.path.join(model_dir, "predictor", "pytorch_classifier.pth")
    best_model = torch.load(model_path, map_location=torch.device('cuda:0'))

    #load the feature names
    feature_names_filename = os.path.join(model_dir,'predictor', 'feature_names.pkl')
    feature_names = joblib.load(feature_names_filename)

    #get feature importances
    result_feature_importance = joblib.load(os.path.join(model_dir,'predictor', 'result_feature_importance.pkl'))
    feature_importance = pd.DataFrame({
            'Feature': result_feature_importance['features'],
            'Importance': result_feature_importance['importances']
        })
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    feature_importance_list = [{'tag': row['Feature'], 'importance': row['Importance']} 
                               for _, row in feature_importance.iterrows()]

    print(feature_importance_list)

if __name__ == "__main__":
    main()