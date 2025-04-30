import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class PredictionTool:
    """
    Tool for predicting 90-day mortality in sepsis patients.
    """
    
    def __init__(self, train_data_path: str, model_type: str = 'random_forest'):
        """
        Initialize the tool with the path to the dataset and model type.
        
        Args:
            train_data_path: Path to the training CSV file
            model_type: Type of model to use ('random_forest', 'gradient_boosting', or 'logistic_regression')
        """
        # Load training data
        self.train_df = pd.read_csv(train_data_path)
        # Remove the index column if it exists (first unnamed column)
        if self.train_df.columns[0] == 'Unnamed: 0':
            self.train_df = self.train_df.drop(columns=self.train_df.columns[0])
        
        # Check if mortality_90d is in the training data
        if 'mortality_90d' not in self.train_df.columns:
            raise ValueError("mortality_90d not found in training data")
        
        # Define columns that should not be used as features
        self.non_features = ['bloc', 'icustayid', 'charttime', 'mortality_90d']
        
        # Store information about zero values that likely represent missing data
        self.zero_as_missing = self._detect_zero_as_missing()
        
        # Initialize model
        self.model_type = model_type
        self.model = None
        self.feature_importance = {}
        self.model_performance = {}
        self.scaler = None
        self.imputer = None
    
    def _detect_zero_as_missing(self) -> Dict[str, bool]:
        """
        Detect which variables have zeros that likely represent missing values.
        
        Returns:
            Dictionary mapping column names to boolean indicating if zeros should be treated as missing
        """
        zero_as_missing = {}
        
        # Skip non-numeric columns and columns that shouldn't be used as features
        for col in self.train_df.columns:
            if col in self.non_features or not pd.api.types.is_numeric_dtype(self.train_df[col]):
                continue
                
            # Get non-null values
            values = self.train_df[col].dropna()
            
            # Skip if no values or no zeros
            if len(values) == 0 or (values == 0).sum() == 0:
                zero_as_missing[col] = False
                continue
            
            # Calculate statistics
            zero_count = (values == 0).sum()
            zero_pct = zero_count / len(values)
            
            # Variables where zero is likely to be a missing value indicator
            likely_missing_vars = [
                'Potassium', 'Sodium', 'Chloride', 'Albumin', 'Hb', 'WBC_count', 
                'Platelets_count', 'paO2', 'paCO2', 'Arterial_pH', 'HCO3'
            ]
            
            # Variables where zero is likely to be a valid measurement
            valid_zero_vars = [
                'mechvent', 'input_4hourly', 'output_4hourly', 'median_dose_vaso', 
                'max_dose_vaso', 'Total_bili', 'SGOT', 'SGPT'
            ]
            
            # Determine if zeros should be treated as missing
            if col in likely_missing_vars:
                # For these variables, physiologically impossible to be zero
                zero_as_missing[col] = True
            elif col in valid_zero_vars:
                # For these variables, zero is a valid value
                zero_as_missing[col] = False
            elif zero_pct > 0.5:
                # If more than 50% of non-null values are zero, they're likely valid
                zero_as_missing[col] = False
            elif values[values > 0].min() > 10 * values[values > 0].std():
                # If the minimum non-zero value is much larger than the std, zeros likely represent missing values
                zero_as_missing[col] = True
            else:
                # Default: treat zeros as valid values
                zero_as_missing[col] = False
        
        return zero_as_missing
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for model training or prediction.
        
        Args:
            df: DataFrame to preprocess
            
        Returns:
            Preprocessed DataFrame
        """
        # Create a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Convert zeros to NaN where they likely represent missing values
        for col, treat_as_missing in self.zero_as_missing.items():
            if col in processed_df.columns and treat_as_missing:
                processed_df.loc[processed_df[col] == 0, col] = np.nan
        
        # Group by patient ID and aggregate time series data
        if 'icustayid' in processed_df.columns:
            # Define aggregation functions for different types of features
            agg_functions = {}
            
            for col in processed_df.columns:
                if col in self.non_features:
                    if col == 'mortality_90d':
                        # If this is the target variable, use the first value (should be the same for all records)
                        agg_functions[col] = 'first'
                    continue
                
                if not pd.api.types.is_numeric_dtype(processed_df[col]):
                    # For non-numeric columns, use the most common value
                    agg_functions[col] = pd.Series.mode
                else:
                    # For numeric features, calculate various statistics
                    agg_functions[col] = ['mean', 'min', 'max', 'std', 'last']
            
            # Apply aggregation
            grouped_df = processed_df.groupby('icustayid').agg(agg_functions)
            
            # Flatten multi-index columns
            if isinstance(grouped_df.columns, pd.MultiIndex):
                grouped_df.columns = ['_'.join(col).strip() for col in grouped_df.columns.values]
            
            # Reset index to make icustayid a column again
            grouped_df = grouped_df.reset_index()
            
            # Calculate the count of non-null values for each feature
            for col in processed_df.columns:
                if col in self.non_features or not pd.api.types.is_numeric_dtype(processed_df[col]):
                    continue
                
                count_col = f"{col}_count"
                grouped_df[count_col] = processed_df.groupby('icustayid')[col].count().values
                
                # Calculate percentage of missing values
                total_records = processed_df.groupby('icustayid').size().values
                grouped_df[f"{col}_missing_pct"] = 100 * (1 - grouped_df[count_col] / total_records)
            
            return grouped_df
        
        return processed_df
    
    def train_model(self, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train the mortality prediction model.
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training results
        """
        # Preprocess data
        processed_df = self.preprocess_data(self.train_df)
        
        # Check if mortality_90d is in the preprocessed data
        if 'mortality_90d' not in processed_df.columns and 'mortality_90d_first' not in processed_df.columns:
            raise ValueError("Target variable not found in preprocessed data")
        
        # Use mortality_90d or mortality_90d_first as the target
        target_col = 'mortality_90d' if 'mortality_90d' in processed_df.columns else 'mortality_90d_first'
        
        # Get feature columns (numeric columns not in non_features list)
        feature_cols = [col for col in processed_df.columns 
                       if col != target_col 
                       and col != 'icustayid'  # Exclude patient ID
                       and not any(nf in col for nf in self.non_features)  # Exclude derived columns from non_features
                       and pd.api.types.is_numeric_dtype(processed_df[col])]  # Only numeric columns
        
        # Split data into features and target
        X = processed_df[feature_cols]
        y = processed_df[target_col].astype(int)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        # Create preprocessing pipeline with imputation and scaling
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, feature_cols)
            ])
        
        # Create model
        if self.model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, 
                                          random_state=random_state, 
                                          class_weight='balanced')
        elif self.model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(n_estimators=100, 
                                               random_state=random_state)
        elif self.model_type == 'logistic_regression':
            model = LogisticRegression(random_state=random_state, 
                                      class_weight='balanced', 
                                      max_iter=1000)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Store model and preprocessing components
        self.model = pipeline
        self.imputer = pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['imputer']
        self.scaler = pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler']
        
        # Make predictions on test set
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate performance metrics
        performance = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'auc': float(roc_auc_score(y_test, y_pred_proba))
        }
        
        # Store performance metrics
        self.model_performance = performance
        
        # Calculate feature importance
        if self.model_type in ['random_forest', 'gradient_boosting']:
            # For tree-based models, use feature_importances_
            importances = pipeline.named_steps['model'].feature_importances_
            feature_importance = dict(zip(feature_cols, importances))
            
            # Sort by importance
            feature_importance = {k: float(v) for k, v in sorted(feature_importance.items(), 
                                                               key=lambda x: x[1], 
                                                               reverse=True)}
        
        elif self.model_type == 'logistic_regression':
            # For logistic regression, use coefficients
            coefficients = pipeline.named_steps['model'].coef_[0]
            feature_importance = dict(zip(feature_cols, abs(coefficients)))
            
            # Sort by absolute importance
            feature_importance = {k: float(v) for k, v in sorted(feature_importance.items(), 
                                                               key=lambda x: x[1], 
                                                               reverse=True)}
        
        # Store feature importance
        self.feature_importance = feature_importance
        
        return {
            'model_type': self.model_type,
            'performance': performance,
            'feature_importance': feature_importance,
            'num_features': len(feature_cols),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'class_balance': {
                'train': dict(pd.Series(y_train).value_counts(normalize=True)),
                'test': dict(pd.Series(y_test).value_counts(normalize=True))
            }
        }
    
    def predict_mortality(self, patient_data: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict mortality for a patient.
        
        Args:
            patient_data: DataFrame or dictionary with patient data
            
        Returns:
            Dictionary with prediction results
        """
        # Check if model is trained
        if self.model is None:
            self.train_model()
        
        # Convert dictionary to DataFrame if needed
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data.copy()
        
        # Preprocess data
        processed_df = self.preprocess_data(patient_df)
        
        # Get feature columns used by the model
        feature_cols = list(self.feature_importance.keys())
        
        # Check if all required features are present
        missing_features = [col for col in feature_cols if col not in processed_df.columns]
        if missing_features:
            # If features are missing, add them with NaN values
            for col in missing_features:
                processed_df[col] = np.nan
        
        # Select only the features used by the model
        X = processed_df[feature_cols]
        
        # Make prediction
        mortality_proba = self.model.predict_proba(X)[0, 1]
        mortality_class = 1 if mortality_proba >= 0.5 else 0
        
        # Get patient ID if available
        patient_id = None
        if 'icustayid' in processed_df.columns:
            patient_id = processed_df['icustayid'].iloc[0]
        
        # Get top contributing features
        contributing_features = self.get_contributing_features(X, top_n=5)
        
        return {
            'patient_id': patient_id,
            'mortality_probability': float(mortality_proba),
            'mortality_class': int(mortality_class),
            'confidence': float(max(mortality_proba, 1 - mortality_proba)),
            'contributing_features': contributing_features
        }
    
    def get_contributing_features(self, X: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the top contributing features for a prediction.
        
        Args:
            X: Feature DataFrame for a single patient
            top_n: Number of top features to return
            
        Returns:
            List of dictionaries with feature contribution information
        """
        # Check if model is trained
        if self.model is None or not self.feature_importance:
            return []
        
        # Get feature values
        feature_values = X.iloc[0].to_dict()
        
        if self.model_type == 'logistic_regression':
            # For logistic regression, get raw coefficients (not absolute values)
            coefficients = self.model.named_steps['model'].coef_[0]
            feature_importance = dict(zip(X.columns, coefficients))
            
            # Calculate contribution (coefficient * feature value)
            contributions = {}
            for feature, coef in feature_importance.items():
                value = feature_values[feature]
                if pd.isna(value):
                    # For missing values, use the imputed value
                    value = self.imputer.statistics_[list(X.columns).index(feature)]
                
                # Scale the value
                scaled_value = (value - self.scaler.mean_[list(X.columns).index(feature)]) / self.scaler.scale_[list(X.columns).index(feature)]
                
                contributions[feature] = float(coef * scaled_value)
            
            # Sort by absolute contribution
            sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Get top contributing features
            top_features = []
            for feature, contribution in sorted_contributions[:top_n]:
                direction = 'positive' if contribution > 0 else 'negative'
                top_features.append({
                    'feature': feature,
                    'contribution': float(contribution),
                    'direction': direction,
                    'value': float(feature_values[feature]) if not pd.isna(feature_values[feature]) else None,
                    'imputed': pd.isna(feature_values[feature])
                })
            
            return top_features
        
        else:
            # For tree-based models, use feature importance
            # Sort features by importance
            sorted_importance = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Get top important features
            top_features = []
            for feature, importance in sorted_importance[:top_n]:
                # Determine if the feature value is abnormal
                value = feature_values[feature]
                imputed = pd.isna(value)
                
                if imputed:
                    # For missing values, use the imputed value
                    value = self.imputer.statistics_[list(X.columns).index(feature)]
                
                # Determine direction of impact based on feature value compared to average
                avg_value = self.train_df[feature].mean() if feature in self.train_df else None
                
                if avg_value is not None and not pd.isna(avg_value):
                    direction = 'high' if value > avg_value else 'low'
                else:
                    direction = 'unknown'
                
                top_features.append({
                    'feature': feature,
                    'importance': float(importance),
                    'value': float(value),
                    'avg_value': float(avg_value) if avg_value is not None and not pd.isna(avg_value) else None,
                    'direction': direction,
                    'imputed': imputed
                })
            
            return top_features
    
    def get_model_card(self) -> str:
        """
        Generate a human-readable model card with information about the trained model.
        
        Returns:
            Text model card
        """
        # Check if model is trained
        if self.model is None:
            return "Model has not been trained yet."
        
        # Build model card
        card = []
        card.append("# Sepsis Mortality Prediction Model Card\n")
        
        # Model details
        card.append("## Model Details")
        card.append(f"- Model type: {self.model_type}")
        card.append(f"- Number of features: {len(self.feature_importance)}")
        card.append(f"- Task: Binary classification (90-day mortality prediction)")
        card.append(f"- Framework: scikit-learn")
        
        # Model performance
        card.append("\n## Model Performance")
        for metric, value in self.model_performance.items():
            card.append(f"- {metric.capitalize()}: {value:.4f}")
        
        # Feature importance
        card.append("\n## Important Features")
        
        # Get top 10 features
        top_features = list(self.feature_importance.items())[:10]
        
        for feature, importance in top_features:
            if '_mean' in feature:
                base_feature = feature.replace('_mean', '')
                card.append(f"- {base_feature} (mean): {importance:.4f}")
            elif '_max' in feature:
                base_feature = feature.replace('_max', '')
                card.append(f"- {base_feature} (maximum): {importance:.4f}")
            elif '_min' in feature:
                base_feature = feature.replace('_min', '')
                card.append(f"- {base_feature} (minimum): {importance:.4f}")
            else:
                card.append(f"- {feature}: {importance:.4f}")
        
        # Limitations and ethical considerations
        card.append("\n## Limitations")
        card.append("- The model is trained on a specific dataset and may not generalize to patients from different populations or hospitals.")
        card.append("- Missing data is imputed, which may introduce bias or uncertainty in predictions.")
        card.append("- The model provides probabilities but does not account for all possible clinical factors.")
        card.append("- The model should be used as a decision support tool, not as a replacement for clinical judgment.")
        
        # Usage recommendations
        card.append("\n## Recommended Use")
        card.append("- This model is intended to support clinical decision-making by providing an objective assessment of mortality risk.")
        card.append("- The model should be used in conjunction with clinical expertise, not as a standalone diagnostic tool.")
        card.append("- Predictions should be interpreted in the context of the patient's full clinical picture.")
        card.append("- The contributing factors identified by the model can help guide clinical focus to the most critical aspects of a patient's condition.")
        
        return "\n".join(card)
    
    def generate_patient_report(self, patient_data: Union[pd.DataFrame, Dict[str, Any]]) -> str:
        """
        Generate a human-readable report for a patient's mortality prediction.
        
        Args:
            patient_data: DataFrame or dictionary with patient data
            
        Returns:
            Text report
        """
        # Make prediction
        prediction = self.predict_mortality(patient_data)
        
        # Build report
        report = []
        report.append("# Sepsis Mortality Risk Assessment\n")
        
        # Prediction result
        mortality_prob = prediction['mortality_probability']
        mortality_class = prediction['mortality_class']
        
        # Determine risk level
        if mortality_prob < 0.25:
            risk_level = "Low"
        elif mortality_prob < 0.5:
            risk_level = "Moderate"
        elif mortality_prob < 0.75:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        # Summary
        report.append("## Summary")
        report.append(f"- **Risk Level**: {risk_level}")
        report.append(f"- **90-day Mortality Risk**: {mortality_prob:.1%}")
        report.append(f"- **Prediction**: {'Higher risk of mortality' if mortality_class == 1 else 'Lower risk of mortality'}")
        report.append(f"- **Confidence**: {prediction['confidence']:.1%}")
        
        # Key factors
        report.append("\n## Key Risk Factors")
        
        for factor in prediction['contributing_features']:
            feature = factor['feature']
            
            # Clean up feature name
            if '_mean' in feature:
                base_feature = feature.replace('_mean', '')
                feature_type = "average"
            elif '_max' in feature:
                base_feature = feature.replace('_max', '')
                feature_type = "maximum"
            elif '_min' in feature:
                base_feature = feature.replace('_min', '')
                feature_type = "minimum"
            elif '_last' in feature:
                base_feature = feature.replace('_last', '')
                feature_type = "most recent"
            # else:
            #     base

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where the model should be saved
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'feature_importance': self.feature_importance,
                'model_performance': self.model_performance,
                'zero_as_missing': self.zero_as_missing
            }, f)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.model_type = data['model_type']
        self.feature_importance = data['feature_importance']
        self.model_performance = data['model_performance']
        self.zero_as_missing = data['zero_as_missing']
        
        # Extract preprocessing components
        if hasattr(self.model, 'named_steps'):
            if 'preprocessor' in self.model.named_steps:
                preprocessor = self.model.named_steps['preprocessor']
                if hasattr(preprocessor, 'named_transformers_') and 'num' in preprocessor.named_transformers_:
                    numeric_transformer = preprocessor.named_transformers_['num']
                    if hasattr(numeric_transformer, 'named_steps'):
                        if 'imputer' in numeric_transformer.named_steps:
                            self.imputer = numeric_transformer.named_steps['imputer']
                        if 'scaler' in numeric_transformer.named_steps:
                            self.scaler = numeric_transformer.named_steps['scaler']