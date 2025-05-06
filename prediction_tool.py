import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import json
import pickle
import matplotlib.pyplot as plt
import io
import base64
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report


class PredictionTool:
    """
    Tool for predicting mortality risk in sepsis patients.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the tool with the path to the dataset.
        
        Args:
            data_path: Path to the CSV file containing the dataset
        """
        # Load training data
        self.df = pd.read_csv(data_path)
        
        # Remove the index column if it exists (first unnamed column)
        if self.df.columns[0] == 'Unnamed: 0':
            self.df = self.df.drop(columns=self.df.columns[0])
        
        # Initialize model and other components
        self.model = None
        self.feature_names = None
        self.features_to_use = None
        self.preprocessor = None
        self.feature_importance = {}
        self.threshold = 0.5  # Default threshold for binary classification
        
        # Identify which features are non-imputable (e.g., IDs, timestamps, target)
        self.non_imputable = ['bloc', 'icustayid', 'charttime', 'gender', 'mortality_90d']
        
        # Preprocess the data
        self._preprocess_data()
    
    def _preprocess_data(self):
        """
        Preprocess the dataset for model training, defining features and target.
        """
        # Convert charttime to datetime if it exists
        if 'charttime' in self.df.columns:
            try:
                self.df['charttime'] = pd.to_datetime(self.df['charttime'])
                
                # Extract useful datetime features if needed
                # self.df['hour_of_day'] = self.df['charttime'].dt.hour
                # self.df['day_of_week'] = self.df['charttime'].dt.dayofweek
            except:
                print("Could not convert charttime to datetime, will use as-is")
        
        # Convert age from days to years if exists
        if 'age' in self.df.columns:
            self.df['age'] = self.df['age'] / 365.25
        
        # Identify which features to use for prediction
        # Excluding non-predictive features or IDs
        self.features_to_use = [col for col in self.df.columns 
                              if col not in self.non_imputable
                              and pd.api.types.is_numeric_dtype(self.df[col])]
        
        # Store feature names for later use
        self.feature_names = self.features_to_use
        
        print(f"Preprocessing complete. Using {len(self.features_to_use)} features for prediction.")
    
    def get_aggregated_patient_data(self) -> pd.DataFrame:
        """
        Aggregate the time series data to create one record per patient.
        
        Returns:
            DataFrame with one row per patient, aggregated features
        """
        # Skip if no time series (e.g., already one record per patient)
        if 'icustayid' not in self.df.columns:
            return self.df.copy()
        
        # Group by patient ID and calculate aggregations
        agg_dict = {}
        
        # For each numeric feature, calculate various statistics
        for feature in self.features_to_use:
            agg_dict[feature] = ['mean', 'min', 'max', 'std', 'last']
        
        # For the target variable if it exists
        if 'mortality_90d' in self.df.columns:
            agg_dict['mortality_90d'] = 'first'  # Each patient has one mortality outcome
        
        # For non-numeric features we want to keep
        for feature in ['gender']:
            if feature in self.df.columns:
                agg_dict[feature] = 'first'
        
        # Perform the aggregation
        patient_df = self.df.groupby('icustayid').agg(agg_dict)
        
        # Flatten the column names
        patient_df.columns = ['_'.join(col).strip() for col in patient_df.columns.values]
        
        # Reset the index to make icustayid a column again
        patient_df = patient_df.reset_index()
        
        # Handle any NaN values from the aggregation
        patient_df = patient_df.fillna(method='ffill')
        
        return patient_df
    
    def train_model(self, test_size: float = 0.2, random_state: int = 42):
        """
        Train a CatBoost model for mortality prediction.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        print("Training mortality prediction model...")
        
        # Get aggregated patient data
        patient_df = self.get_aggregated_patient_data()
        
        # Check if mortality_90d exists
        if 'mortality_90d_first' in patient_df.columns:
            target_col = 'mortality_90d_first'
        elif 'mortality_90d' in patient_df.columns:
            target_col = 'mortality_90d'
        else:
            raise ValueError("Target column 'mortality_90d' not found in dataset")
        
        # Define features to use (excluding the target and icustayid)
        feature_cols = [col for col in patient_df.columns 
                      if col != target_col and col != 'icustayid'
                      and pd.api.types.is_numeric_dtype(patient_df[col])]
        
        # Split the data into training and testing sets
        X = patient_df[feature_cols]
        y = patient_df[target_col].astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Create a preprocessing pipeline
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Fit the preprocessor on the training data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Initialize and train the CatBoost model
        catboost_model = cb.CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function='Logloss',
            eval_metric='F1',
            random_seed=random_state,
            verbose=50
        )
        
        # Train the model
        catboost_model.fit(
            X_train_processed, y_train,
            eval_set=(X_test_processed, y_test),
            early_stopping_rounds=50
        )
        
        # Create a calibrated model to get better probability estimates
        calibrated_model = CalibratedClassifierCV(
            catboost_model, 
            method='sigmoid', 
            cv='prefit'
        )
        calibrated_model.fit(X_test_processed, y_test)
        
        # Evaluate the model
        y_pred_proba = calibrated_model.predict_proba(X_test_processed)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        accuracy = (y_pred == y_test).mean()
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Model evaluation results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Get feature importance
        feature_importance = catboost_model.get_feature_importance(type='PredictionValuesChange')
        self.feature_importance = {feature_cols[i]: float(importance) 
                                 for i, importance in enumerate(feature_importance)}
        
        # Store the trained model and preprocessor
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('model', calibrated_model)
        ])
        
        self.preprocessor = preprocessor
        self.feature_names = feature_cols
        
        print("Model training complete.")
    
    def save_model(self, model_path: str):
        """
        Save the trained model to a file.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No trained model to save")
        
        # Create a dictionary with all components to save
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'threshold': self.threshold
        }
        
        # Save to file
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """
        Load a trained model from a file.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            
            # Handle potentially missing keys with defaults
            if 'feature_names' in model_data:
                self.feature_names = model_data['feature_names']
            elif hasattr(self.model, 'feature_names_in_'):
                # Try to get feature names from model if available
                self.feature_names = self.model.feature_names_in_
            elif hasattr(self.model, 'named_steps') and 'model' in self.model.named_steps:
                if hasattr(self.model.named_steps['model'], 'feature_names_in_'):
                    self.feature_names = self.model.named_steps['model'].feature_names_in_
                # For CatBoost models
                elif hasattr(self.model.named_steps['model'], 'feature_names_'):
                    self.feature_names = self.model.named_steps['model'].feature_names_
            
            # If we still don't have feature names, use a placeholder
            if self.feature_names is None:
                print("Warning: Could not find feature names in the model. Using placeholder names.")
                if hasattr(self.model, 'n_features_in_'):
                    self.feature_names = [f'feature_{i}' for i in range(self.model.n_features_in_)]
                else:
                    # Default to common sepsis features
                    self.feature_names = ['GCS', 'HR', 'SysBP', 'MeanBP', 'RR', 'SpO2', 
                                         'Temp_C', 'SOFA', 'Arterial_lactate', 'age']
            
            # Handle feature importance
            if 'feature_importance' in model_data:
                self.feature_importance = model_data['feature_importance']
            else:
                # Create default feature importance if missing
                print("Warning: Feature importance not found. Using equal importance.")
                self.feature_importance = {f: 1.0/len(self.feature_names) for f in self.feature_names}
            
            # Get threshold
            self.threshold = model_data.get('threshold', 0.5)
            
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # Instead of raising, initialize with defaults
            print("Initializing with default values due to model loading error")
            self.model = None
            self.feature_names = None
            self.feature_importance = {}
            self.threshold = 0.5
    
    def predict_mortality(self, patient_data: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict mortality for a patient.
        
        Args:
            patient_data: DataFrame or dictionary with patient data
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded")
        
        # Convert dictionary to DataFrame if needed
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data.copy()
        
        # For time series data, aggregate to get one record per patient
        if 'icustayid' in patient_df.columns and len(patient_df) > 1:
            patient_id = patient_df['icustayid'].iloc[0]
            print(f"Aggregating time series data for patient {patient_id}")
            
            # Create aggregations for each feature
            agg_dict = {}
            for feature in self.features_to_use:
                if feature in patient_df.columns:
                    agg_dict[feature] = ['mean', 'min', 'max', 'std', 'last']
            
            # Perform aggregation
            agg_df = patient_df.groupby('icustayid').agg(agg_dict)
            
            # Flatten the column names
            agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
            
            # Reset the index to make icustayid a column again
            agg_df = agg_df.reset_index()
            
            # Use the aggregated data for prediction
            patient_df = agg_df
        
        # Prepare features for prediction
        # Find overlapping features between the model and patient data
        if len(patient_df.columns.intersection(self.feature_names)) == 0:
            # No direct feature match, try checking for aggregate features
            # (e.g., feature_mean, feature_max, etc.)
            feature_cols = []
            for feat in self.feature_names:
                if feat in patient_df.columns:
                    feature_cols.append(feat)
                else:
                    # If the base feature is not in the data but aggregations are
                    base_feat = feat.split('_')[0]
                    for col in patient_df.columns:
                        if col.startswith(base_feat + '_'):
                            feature_cols.append(col)
                            break
        else:
            # Use the intersection of features
            feature_cols = list(set(self.feature_names).intersection(patient_df.columns))
        
        # If still no matching features, raise an error
        if not feature_cols:
            raise ValueError("No matching features found between the model and patient data")
        
        # Align the feature order with the model's expected features
        # and fill missing features with NaN
        X = pd.DataFrame(index=patient_df.index)
        for feature in self.feature_names:
            if feature in patient_df.columns:
                X[feature] = patient_df[feature]
            else:
                X[feature] = np.nan
        
        # Make prediction
        if hasattr(self.model, 'predict_proba'):
            mortality_prob = self.model.predict_proba(X)[:, 1][0]
        else:
            # For pipeline models
            mortality_prob = self.model.predict_proba(X)[:, 1][0]
        
        # Determine binary prediction based on threshold
        mortality_class = int(mortality_prob >= self.threshold)
        
        # Calculate confidence (simplistic approach - distance from threshold)
        # A more sophisticated approach would use calibration or conformal prediction
        confidence = 1.0 - 2.0 * abs(mortality_prob - 0.5)
        
        # Determine risk level
        if mortality_prob < 0.25:
            risk_level = "Low"
        elif mortality_prob < 0.5:
            risk_level = "Moderate"
        elif mortality_prob < 0.75:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        # Get patient ID if available
        patient_id = None
        if 'icustayid' in patient_df.columns:
            patient_id = int(patient_df['icustayid'].iloc[0])
        
        # Get feature contributions
        contributing_features = self._get_feature_contributions(X)
        
        # Return prediction results
        return {
            "patient_id": patient_id,
            "mortality_probability": float(mortality_prob),
            "mortality_class": mortality_class,
            "risk_level": risk_level,
            "confidence": float(confidence),
            "contributing_features": contributing_features
        }
    
    def _get_feature_contributions(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Calculate feature contributions to the prediction.
        
        Args:
            X: Feature DataFrame for a patient
            
        Returns:
            List of dictionaries with feature contribution information
        """
        # Get the top 10 most important features from the model
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:10]
        
        contributions = []
        
        for feature_name, importance in sorted_features:
            # Skip if feature doesn't exist in the input
            if feature_name not in X.columns:
                continue
                
            # Get the feature value
            value = X[feature_name].iloc[0]
            
            # Skip if value is missing
            if pd.isna(value):
                continue
                
            # Determine direction based on feature value and importance
            # This is a simplistic approach - in a real system, you'd use SHAP or similar
            # For now, we're using the feature importance and comparing to global average
            
            # Get average value from preprocessor if available
            if hasattr(self.model, 'named_steps') and 'preprocessor' in self.model.named_steps:
                avg_value = self.model.named_steps['preprocessor'].steps[0][1].statistics_[
                    list(X.columns).index(feature_name)
                ]
            else:
                # Fallback to simple comparison with 0
                avg_value = 0.0
            
            # Determine direction
            if importance > 0:
                # Positive contribution to risk
                direction = "high" if value > avg_value else "low"
            else:
                # Negative contribution to risk
                direction = "low" if value > avg_value else "high"
                
            # Check if the feature was imputed (a more sophisticated system would track this)
            imputed = False  # Placeholder - in a real system, track imputations
                
            contributions.append({
                "feature": feature_name,
                "importance": abs(importance) / max(abs(imp) for _, imp in sorted_features),  # Normalize
                "value": float(value),
                "avg_value": float(avg_value),
                "direction": direction,
                "imputed": imputed
            })
        
        return contributions
    
    def generate_patient_report(self, patient_df: pd.DataFrame) -> str:
        """
        Generate a comprehensive report for a patient.
        
        Args:
            patient_df: DataFrame with patient data
            
        Returns:
            String containing the report
        """
        # Make prediction
        prediction = self.predict_mortality(patient_df)
        
        # Extract key information
        patient_id = prediction.get("patient_id", "Unknown")
        mortality_prob = prediction["mortality_probability"]
        risk_level = prediction["risk_level"]
        confidence = prediction["confidence"]
        contributing_features = prediction["contributing_features"]
        
        # Build report
        report = []
        report.append(f"# Mortality Risk Assessment Report for Patient {patient_id}")
        report.append("")
        
        # Summary section
        report.append("Summary")
        report.append(f"\n**Risk Level**: {risk_level}\n")
        report.append(f"\n**90-day Mortality Risk**: {mortality_prob:.1%}")
        # report.append(f"- **Prediction Confidence**: {confidence:.1%}")
        report.append("")
        
        # Key risk factors
        report.append("## Key Risk Factors")
        
        # Sort contributing features by importance
        sorted_features = sorted(contributing_features, key=lambda x: x['importance'], reverse=True)
        
        for idx, feature in enumerate(sorted_features[:5]):  # Show top 5 features
            feature_name = feature['feature'].replace('_', ' ').title()
            value = feature['value']
            avg_value = feature['avg_value']
            direction = feature['direction']
            importance = feature['importance']
            
            # Format the feature name nicely
            for suffix in ['_mean', '_max', '_min', '_last', '_std']:
                if feature_name.lower().endswith(suffix):
                    metric_type = suffix.strip('_').title()
                    base_name = feature_name[:-len(suffix)].strip()
                    feature_name = f"{base_name} ({metric_type})"
            
            # Create description of feature impact
            if direction == "high":
                impact = "higher"
                comparison = "above"
            else:
                impact = "lower"
                comparison = "below"
            
            # Format importance as percentage
            importance_pct = importance * 100
            
            report.append(f"### {idx+1}. {feature_name}")
            report.append(f"- **Value**: {value:.2f} ({comparison} average of {avg_value:.2f})")
            report.append(f"- **Impact**: Contributes to {impact} risk of mortality")
            report.append(f"- **Importance**: {importance_pct:.1f}% contribution to prediction")
            report.append("")
        
        # Missing data section if relevant
        missing_features = [f for f in self.feature_names if f not in patient_df.columns 
                          or patient_df[f].isna().all()]
        
        if missing_features:
            report.append("## Missing Data")
            report.append("The following important data points were missing and had to be estimated:")
            
            # Find important missing features
            important_missing = []
            for feature in missing_features:
                if feature in self.feature_importance:
                    importance = abs(self.feature_importance[feature])
                    important_missing.append((feature, importance))
            
            # Sort by importance and show top 5
            sorted_missing = sorted(important_missing, key=lambda x: x[1], reverse=True)[:5]
            
            for feature, importance in sorted_missing:
                feature_name = feature.replace('_', ' ').title()
                report.append(f"- {feature_name}")
            
            # Add impact assessment
            if len(sorted_missing) > 2:
                report.append("\nThe missing data may significantly affect the reliability of this prediction.")
            else:
                report.append("\nThe missing data is unlikely to significantly affect the prediction.")
        
        # Recommendations section
        report.append("## Clinical Recommendations")
        
        if risk_level == "Low":
            report.append("- Regular monitoring according to standard protocols")
            report.append("- Continue current treatment plan")
            report.append("- Reassess if clinical status changes")
        elif risk_level == "Moderate":
            report.append("- Increased monitoring frequency")
            report.append("- Consider additional diagnostic tests")
            report.append("- Review medication and treatment efficacy")
        elif risk_level == "High":
            report.append("- Frequent monitoring of vital signs and lab values")
            report.append("- Consider ICU admission if not already there")
            report.append("- Aggressive treatment of underlying conditions")
            report.append("- Early intervention for any new symptoms")
        else:  # Very High
            report.append("- Intensive monitoring")
            report.append("- ICU level care")
            report.append("- Aggressive intervention for organ support")
            report.append("- Consider discussion about goals of care")
        
        # Disclaimer
        report.append("\n## Disclaimer")
        report.append("This prediction is based on statistical analysis and should be used as a decision support tool only. "
                     "Clinical judgment should always take precedence. The model does not account for all possible "
                     "clinical factors and may have limitations in certain patient populations.")
        
        return "\n".join(report)
    
    def get_model_card(self) -> str:
        """
        Generate a model card describing the prediction model.
        
        Returns:
            String containing the model description
        """
        if self.model is None:
            return "No model has been trained or loaded yet."
        
        model_card = []
        model_card.append("# Sepsis Mortality Prediction Model Card")
        model_card.append("")
        
        # Model overview
        model_card.append("## Model Overview")
        
        # Determine the base classifier type
        if hasattr(self.model, 'named_steps') and 'model' in self.model.named_steps:
            if hasattr(self.model.named_steps['model'], 'estimator'):
                base_model = self.model.named_steps['model'].estimator
                model_type = str(type(base_model)).split("'")[1].split(".")[-1]
            else:
                model_type = str(type(self.model.named_steps['model'])).split("'")[1].split(".")[-1]
        else:
            model_type = str(type(self.model)).split("'")[1].split(".")[-1]
        
        model_card.append(f"- **Model Type**: {model_type}")
        model_card.append(f"- **Features Used**: {len(self.feature_names)} clinical variables")
        model_card.append(f"- **Target**: 90-day mortality after ICU admission")
        model_card.append(f"- **Classification Threshold**: {self.threshold}")
        model_card.append("")
        
        # Top features
        model_card.append("## Top Predictive Features")
        
        # Sort features by importance
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:10]  # Top 10 features
        
        for feature_name, importance in sorted_features:
            # Format feature name nicely
            display_name = feature_name.replace('_', ' ').title()
            for suffix in ['_mean', '_max', '_min', '_last', '_std']:
                if display_name.lower().endswith(suffix):
                    metric_type = suffix.strip('_').title()
                    base_name = display_name[:-len(suffix)].strip()
                    display_name = f"{base_name} ({metric_type})"
            
            # Direction of influence
            direction = "increases" if importance > 0 else "decreases"
            
            # Normalize importance for display
            norm_importance = abs(importance) / max(abs(imp) for _, imp in sorted_features)
            stars = "★" * int(round(norm_importance * 5))
            
            model_card.append(f"- {display_name}: {stars} ({direction} risk)")
        
        model_card.append("")
        
        # Intended use
        model_card.append("## Intended Use")
        model_card.append("This model is designed to provide decision support for clinical teams treating patients with "
                         "sepsis in the ICU. It predicts the risk of mortality within 90 days based on clinical "
                         "variables including vital signs, laboratory values, and treatment parameters.")
        model_card.append("")
        model_card.append("The model should be used as one of several tools to inform clinical decision-making, "
                         "not as the sole basis for treatment decisions.")
        model_card.append("")
        
        # Limitations
        model_card.append("## Limitations")
        model_card.append("- The model was trained on historical data and may not reflect changes in treatment protocols.")
        model_card.append("- Predictions are based on available data and may be less reliable when important measurements are missing.")
        model_card.append("- The model does not account for all possible clinical factors that might influence mortality.")
        model_card.append("- Performance may vary across different patient populations or institutions.")
        model_card.append("")
        
        # Usage guidelines
        model_card.append("## Usage Guidelines")
        model_card.append("- Always interpret predictions in the context of the patient's full clinical picture.")
        model_card.append("- Consider the confidence level when evaluating predictions.")
        model_card.append("- Use the model to identify patients who may benefit from increased monitoring or intervention.")
        model_card.append("- Regularly validate model performance with new data.")
        model_card.append("")
        
        return "\n".join(model_card)
    
    def plot_feature_importance(self) -> str:
        """
        Generate a feature importance plot and return it as a base64 encoded image.
        
        Returns:
            Base64 encoded image
        """
        if not self.feature_importance:
            return None
        
        # Sort features by importance
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:15]  # Top 15 features
        
        # Unpack for plotting
        feature_names = []
        importance_values = []
        
        for feature_name, importance in sorted_features:
            # Format feature name nicely
            display_name = feature_name.replace('_', ' ').title()
            for suffix in ['_mean', '_max', '_min', '_last', '_std']:
                if display_name.lower().endswith(suffix):
                    metric_type = suffix.strip('_').title()
                    base_name = display_name[:-len(suffix)].strip()
                    display_name = f"{base_name} ({metric_type})"
            
            feature_names.append(display_name)
            importance_values.append(importance)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        colors = ['#1f77b4' if v > 0 else '#d62728' for v in importance_values]
        y_pos = range(len(feature_names))
        
        plt.barh(y_pos, [abs(v) for v in importance_values], color=colors)
        plt.yticks(y_pos, feature_names)
        plt.xlabel('Feature Importance')
        plt.title('Top Features for Mortality Prediction')
        
        # Add a legend explaining colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', label='Increases Risk'),
            Patch(facecolor='#d62728', label='Decreases Risk')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Encode as base64
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_base64
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model information to a dictionary.
        
        Returns:
            Dictionary with model information
        """
        model_info = {
            "model_type": str(type(self.model)).split("'")[1].split(".")[-1] if self.model else None,
            "num_features": len(self.feature_names) if self.feature_names else 0,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance,
            "threshold": self.threshold
        }
        
        return model_info


# Example usage (would be wrapped in API endpoint)
if __name__ == "__main__":
    # Replace with your actual data path
    data_path = "./data/AI_agent_train_sepsis.csv"
    
    # Initialize the tool
    prediction_tool = PredictionTool(data_path)
    
    # Train the model
    prediction_tool.train_model()
    
    # Save the model
    prediction_tool.save_model("./model/sepsis_mortality_model.pkl")
    
    # Load the model
    # prediction_tool.load_model("./model/sepsis_mortality_model.pkl")
    
    # Get all patient IDs
    agg_df = prediction_tool.get_aggregated_patient_data()
    
    if 'icustayid' in agg_df.columns:
        patient_ids = agg_df['icustayid'].unique().tolist()
        
        if patient_ids:
            # Predict for the first patient
            first_patient_id = patient_ids[0]
            patient_df = agg_df[agg_df['icustayid'] == first_patient_id]
            
            # Get prediction
            prediction = prediction_tool.predict_mortality(patient_df)
            print(f"\nPrediction for patient {first_patient_id}:")
            print(json.dumps(prediction, indent=2))
            
            # Generate report
            report = prediction_tool.generate_patient_report(patient_df)
            print(f"\nReport for patient {first_patient_id}:")
            print(report)
            
            # Plot feature importance
            importance_plot = prediction_tool.plot_feature_importance()
            if importance_plot:
                print("\nFeature importance plot generated (base64 encoded)")
            
            # Get model card
            model_card = prediction_tool.get_model_card()
            print("\nModel Card:")
            print(model_card)
    
    # Example of how to use the prediction tool with new patient data
    print("\nExample of using the prediction tool with new patient data:")
    
    # Create a sample patient record (this would come from your data source)
    sample_patient = {
        'icustayid': 12345,
        'age_mean': 65.2,
        'GCS_mean': 12.5,
        'GCS_min': 10.0,
        'HR_mean': 95.3,
        'HR_max': 110.2,
        'SysBP_mean': 115.8,
        'SysBP_min': 95.0,
        'SOFA_max': 6.0,
        'SOFA_mean': 4.5,
        'Arterial_lactate_max': 2.1,
        'Arterial_lactate_mean': 1.8
    }
    
    # Convert to DataFrame
    sample_df = pd.DataFrame([sample_patient])
    
    # Get prediction
    try:
        prediction = prediction_tool.predict_mortality(sample_df)
        print("\nPrediction for sample patient:")
        print(json.dumps(prediction, indent=2))
        
        # Generate report
        report = prediction_tool.generate_patient_report(sample_df)
        print("\nReport for sample patient:")
        print(report)
    except Exception as e:
        print(f"Error predicting for sample patient: {str(e)}")