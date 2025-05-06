from google.colab import drive
import pandas as pd

drive.mount('/content/drive', force_remount=True)

file_path = '/content/drive/MyDrive/project2/AI_agent_train_sepsis.csv'

data = pd.read_csv(file_path)

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score, precision_recall_curve
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import re
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import pickle
import json
import re
import matplotlib.pyplot as plt
import shap
import google.generativeai as genai


def load_data_and_model(data_path, model_path):
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return data, model


def predict_mortality_for_patient(patient_id, data, model):

    patient_data = data[data['icustayid'] == patient_id]

    if len(patient_data) == 0:
        print(f"Error: Patient ID {patient_id} not found in the dataset")
        return None, None


    numerical_cols = [col for col in patient_data.columns if col not in ['mortality_90d', 'icustayid', 'charttime', 'bloc']]
    agg_dict = {}

    if 'mortality_90d' in patient_data.columns:
        agg_dict['mortality_90d'] = 'first'

    for col in numerical_cols:
        if col == 'age':
            agg_dict[col] = 'first'
        else:
            agg_dict[col] = ['mean', 'min', 'max', 'std', 'last']

    if 'charttime' in patient_data.columns:
        agg_dict['charttime'] = 'count'

    aggregated_data = patient_data.groupby('icustayid').agg(agg_dict)

    aggregated_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in aggregated_data.columns]

    clean_df = aggregated_data.copy()
    clean_columns = {}
    for col in clean_df.columns:
        new_col = re.sub(r'[^\w\s_]', '', col)
        new_col = re.sub(r'\s+', '_', new_col)
        clean_columns[col] = new_col

    cleaned_data = clean_df.rename(columns=clean_columns)

    if 'mortality_90d_first' in cleaned_data.columns:
        X = cleaned_data.drop('mortality_90d_first', axis=1)
    else:
        X = cleaned_data

    if hasattr(model, 'feature_names_in_'):
        required_features = model.feature_names_in_
    elif hasattr(model, '_feature_name'):
        required_features = model._feature_name
    else:
        required_features = []

    if len(required_features) > 0:
        missing_features = set(required_features) - set(X.columns)
        print(f"Missing {len(missing_features)} features")

        for feature in missing_features:
            X[feature] = 0

        extra_features = set(X.columns) - set(required_features)
        if extra_features:
            X = X.drop(columns=list(extra_features))

        X = X[required_features]


    try:
        mortality_prob = model.predict_proba(X)[0][1]
        mortality_label = 1 if mortality_prob >= 0.2112 else 0
        print(f"Patient {patient_id}: Mortality prediction = {mortality_label} (Probability: {mortality_prob:.2%})")
        return mortality_prob, mortality_label, X
    except Exception as e:
        print(f"Error predicting for patient {patient_id}: {e}")
        return None, None, None

def calculate_patient_specific_shap(patient_id, data, model):
    print(f"Calculating SHAP values specifically for patient {patient_id}...")

    mortality_prob, mortality_label, X = predict_mortality_for_patient(patient_id, data, model)

    if X is None:
        print(f"Error: Failed to process data for patient {patient_id}")
        return None

    feature_names = X.columns.tolist()

    try:
        if hasattr(model, 'set_params'):
            print("Setting predict_disable_shape_check=True")
            model.set_params(predict_disable_shape_check=True)


        print("Creating SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(model)

        print(f"Calculating SHAP values for patient {patient_id}...")
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            if len(shap_values) > 1:
                print("Using positive class SHAP values (index 1)")
                shap_values = shap_values[1]
            else:
                print("Using available SHAP values (index 0)")
                shap_values = shap_values[0]

        key_features = [
            'age_first', 'BUN_last', 'output_4hourly_std', 'SOFA_mean',
            'Weight_kg_mean', 'input_4hourly_last', 'mechvent_std',
            'GCS_mean', 'SIRS_mean', 'Creatinine_min'
        ]

        patient_info = {}

        for feature in key_features:

            matched_feature = find_matching_feature(feature, feature_names)

            if matched_feature:
                try:
                    feature_idx = feature_names.index(matched_feature)

                    feature_value = X[matched_feature].values[0]

                    shap_value = shap_values[0, feature_idx]

                    patient_info[feature] = {
                        'value': float(feature_value),
                        'shap': float(shap_value),
                        'impact': 'positive' if shap_value > 0 else 'negative'
                    }
                except Exception as e:
                    print(f"Error extracting SHAP data for feature {feature}: {e}")

                    print(f"Feature {feature} not found or cannot calculate SHAP value")
            else:
                print(f"Warning: No matching feature found for {feature}")


        try:
            base_value = explainer.expected_value
            if isinstance(base_value, list):
                base_value = base_value[1]
            patient_info['base_value'] = float(base_value)
        except Exception as e:
            print(f"Error extracting base_value: {e}")
            patient_info['base_value'] = 0.5

        try:
            total_shap = np.sum(shap_values[0])
            patient_info['total_shap'] = float(total_shap)
        except Exception as e:
            print(f"Error calculating total SHAP: {e}")
            patient_info['total_shap'] = 0.0

        print(f"Successfully calculated SHAP values for patient {patient_id}")
        return patient_info

    except Exception as e:
        print(f"Error calculating SHAP values for patient {patient_id}: {e}")
        return None


def find_matching_feature(target_feature, feature_names):

    if target_feature in feature_names:
        return target_feature



    return None

def explain_mortality_with_gemini(patient_id, mortality_prob, patient_shap_data, api_key):

    genai.configure(api_key=api_key)

    feature_descriptions = {
        'age_first': 'Patient\'s age (first measurement)',
        'BUN_last': 'Blood Urea Nitrogen (BUN) level (last measurement)',
        'output_4hourly_std': 'Standard deviation of urine output measured every 4 hours',
        'SOFA_mean': 'Mean value of Sequential Organ Failure Assessment (SOFA) score',
        'Weight_kg_mean': 'Mean value of patient\'s weight (kg)',
        'input_4hourly_last': 'Fluid intake measured every 4 hours (last measurement)',
        'mechvent_std': 'Standard deviation of mechanical ventilation usage',
        'GCS_mean': 'Mean value of Glasgow Coma Scale (GCS) score',
        'SIRS_mean': 'Mean value of Systemic Inflammatory Response Syndrome (SIRS) score',
        'Creatinine_min': 'Minimum value of creatinine level',
    }

    prompt = f"""
    You are an AI medical assistant that analyzes ICU patient medical data to predict and explain the probability of death within 90 days.
    Patient ID {patient_id} has a mortality probability of {mortality_prob:.1f}%.
    This prediction comes from a machine learning model based on SHAP (SHapley Additive exPlanations) values.
    SHAP values indicate whether each feature (patient's vital signs, clinical data, etc.) increases (positive) or decreases (negative) the risk of death.
    Below are the main feature values for this patient and their corresponding SHAP values:
    {json.dumps(patient_shap_data, indent=2)}
    Feature descriptions:
    {json.dumps(feature_descriptions, indent=2)}
    Based on the above data, please answer the following questions:

    What are the 3 most important factors increasing this patient's risk of death?
    What are the 3 most important factors decreasing this patient's risk of death?
    Explain medically why each factor has such an impact.
    Summarize this patient's overall health status and risk factors.
    Suggest what interventions or treatments might be effective in improving this patient's chances of survival.

    Please provide a professional response that doctors and nurses can understand, while also explaining clearly so that the patient's family can understand
    """

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating explanation with Gemini: {e}")
        return f"""
        환자 ID {patient_id}의 사망 확률은 {mortality_prob:.1f}%로 예측됩니다.

        Gemini API에서 오류가 발생하여 자세한 설명을 생성할 수 없습니다.
        오류 메시지: {str(e)}

        주요 특성과 SHAP 값을 참고하여 의료진과 상담하세요.
        """

def mortality_explanation_agent(patient_id, data_path, model_path, gemini_api_key):
    try:

        data, model = load_data_and_model(data_path, model_path)

        mortality_prob, mortality_label, _ = predict_mortality_for_patient(patient_id, data, model)
        if mortality_prob is None:
            return f"환자 ID {patient_id}의 사망률을 예측할 수 없습니다."

        print("Calculating SHAP values specifically for this patient...")
        patient_shap_data = calculate_patient_specific_shap(patient_id, data, model)

        if patient_shap_data is None:
            return f"환자 ID {patient_id}의 SHAP 값을 계산할 수 없습니다."

        explanation = explain_mortality_with_gemini(patient_id, mortality_prob * 100, patient_shap_data, gemini_api_key)

        return explanation

    except Exception as e:
        print(f"Critical error in mortality_explanation_agent: {e}")
        return f"""
        환자 ID {patient_id}에 대한 사망률 예측 과정에서 오류가 발생했습니다.
        오류 메시지: {str(e)}

        오류가 반복될 경우 시스템 관리자에게 문의하세요.
        """

if __name__ == "__main__":
    """
    CHANGE HERE
    """
    data_path = '/content/drive/MyDrive/project2/AI_agent_test_sepsis_features.csv'
    model_path = '/content/Final_70_lightgbm_model.pkl'
    gemini_api_key = 'AIzaSyDn1F2k4894S4bvT_SsPnSbH6cjt9Bu06U'

    patient_id = int(input("Enter patient ID to analyze: "))

    explanation = mortality_explanation_agent(
        patient_id=patient_id,
        data_path=data_path,
        model_path=model_path,
        gemini_api_key=gemini_api_key
    )

    print("\n========== Mortality Explanation ==========")
    print(explanation)
    print("===========================================")