from sklearn.metrics import fbeta_score
import joblib
import sys
import os

def f2_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    y_pred_label = (y_pred > 0.5).astype(int)
    return 'f2', fbeta_score(y_true, y_pred_label, beta=2)

def main():

    model_dir = sys.argv[1]

    model_path = os.path.join(model_dir, "predictor", "trained_xgboost_pipeline_model.pkl")
    pipeline = joblib.load(model_path)

    #load feature names
    loaded_feature_names = joblib.load(os.path.join(model_dir, "predictor", 'feature_names.pkl'))

    #get mask for selector features are selected
    if 'selector' in pipeline.named_steps:
        selected_features_mask = pipeline.named_steps['selector'].get_support()
        selected_features = [feature for feature, mask in zip(loaded_feature_names, selected_features_mask) if mask]
    else:
        selected_features = loaded_feature_names

    #get feature importance from estimator
    estimator = pipeline.named_steps['clf']
    feature_importances = estimator.feature_importances_

    feature_importance_dict = {feature: 0 for feature in loaded_feature_names}
    for feature, importance in zip(selected_features, feature_importances):
        feature_importance_dict[feature] = importance

    #sort features importance
    feature_importance_list = [{'tag': tag, 'importance': importance} 
                            for tag, importance in sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)]
    print(feature_importance_list)

if __name__ == "__main__":
    main()