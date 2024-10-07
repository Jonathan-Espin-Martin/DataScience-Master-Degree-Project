import joblib
import sys
import os

def main():

    model_dir = sys.argv[1]

    model_path = os.path.join(model_dir, "predictor", "trained_random_forest_pipeline_model.pkl")
    pipeline = joblib.load(model_path)

    estimator = pipeline.named_steps['clf'].estimators_[0]
    selected_features_mask = pipeline.named_steps['selector'].get_support()

    loaded_feature_names = joblib.load(os.path.join(model_dir, "predictor", 'feature_names.pkl'))
    selected_features = [feature for feature, mask in zip(loaded_feature_names,selected_features_mask) if mask]

    feature_importances = estimator.feature_importances_
    feature_importance_list = [{'tag': feature, 'importance': importance} 
                            for feature, importance in zip(selected_features, feature_importances)]
    feature_importance_list = sorted(feature_importance_list, key=lambda x: x['importance'], reverse=True)

    print(feature_importance_list)

if __name__ == "__main__":
    main()