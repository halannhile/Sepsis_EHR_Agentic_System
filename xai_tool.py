import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import json
import matplotlib.pyplot as plt
import io
import base64
import shap
import warnings
warnings.filterwarnings('ignore')

class XAITool:
    """
    Tool for explaining model predictions using XAI techniques.
    """
    
    def __init__(self):
        """
        Initialize the XAI tool.
        """
        self.model = None
        self.feature_names = None
        self.explainer = None
        self.imputer = None
        
    def set_model(self, model: Any, feature_names: List[str]):
        """
        Set the model to be explained.
        
        Args:
            model: Trained model to explain
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        
        try:
            # Create a SHAP explainer for the model
            self.explainer = shap.TreeExplainer(model)
            print("SHAP TreeExplainer initialized successfully")
        except Exception as e:
            print(f"Error initializing SHAP explainer: {str(e)}")
            # Fallback approach
            try:
                self.explainer = shap.Explainer(model)
                print("Fallback SHAP Explainer initialized")
            except Exception as e2:
                print(f"Error initializing fallback SHAP explainer: {str(e2)}")
                self.explainer = None
                
    def set_imputer(self, imputer: Any):
        """
        Set the imputer used for handling missing values.
        
        Args:
            imputer: Trained imputer
        """
        self.imputer = imputer
    
    def generate_explanation(self, patient_data: pd.DataFrame, prediction: Dict[str, Any], skip_shap_plots: bool = False) -> Dict[str, Any]:
        """
        Generate a comprehensive explanation for a prediction.
        
        Args:
            patient_data: DataFrame with patient data
            prediction: Dictionary with prediction results
            skip_shap_plots: Whether to skip SHAP plots (for faster results)
            
        Returns:
            Dictionary with explanation components
        """
        if self.model is None:
            return {
                "error": "Model not set. Call set_model() first.",
                "summary": "Cannot generate explanation: Model not available."
            }
        
        # Extract patient ID if available
        patient_id = None
        if 'icustayid' in patient_data.columns:
            patient_id = patient_data['icustayid'].iloc[0]
            
        # 1. Calculate feature importances from the model
        model_importances = self._extract_model_importances()
        
        # 2. Calculate SHAP values for this patient
        shap_values, shap_base_value = self._calculate_shap_values(patient_data)
        
        # 3. Find missing values in the patient data
        missing_features = self._identify_missing_features(patient_data)
        
        # 4. Calculate prediction with and without imputation
        prediction_analysis = self._analyze_imputation_impact(patient_data, prediction)
        
        # 5. Generate SHAP plots if requested
        shap_plots = {}
        if not skip_shap_plots:
            shap_plots = self._generate_shap_plots(patient_data, shap_values)
        
        # 6. Generate summary text
        summary = self._generate_summary_text(
            patient_id=patient_id,
            prediction=prediction,
            model_importances=model_importances,
            shap_values=shap_values,
            shap_base_value=shap_base_value,
            missing_features=missing_features,
            prediction_analysis=prediction_analysis
        )
        
        # 7. Return the complete explanation
        explanation = {
            "patient_id": patient_id,
            "summary": summary,
            "model_importances": model_importances,
            "shap_values": self._format_shap_values(shap_values, patient_data),
            "shap_base_value": float(shap_base_value) if shap_base_value is not None else None,
            "missing_features": missing_features,
            "prediction_analysis": prediction_analysis,
            "plots": shap_plots
        }
        
        return explanation
    
    def generate_simplified_explanation(self, explanation: Dict[str, Any]) -> str:
        """
        Generate a simplified, human-readable explanation of the prediction.
        
        Args:
            explanation: Full explanation dictionary from generate_explanation()
            
        Returns:
            Simplified explanation text
        """
        # Check if the explanation has all necessary components
        if "summary" in explanation:
            return explanation["summary"]
        
        # If not, generate a new summary
        patient_id = explanation.get("patient_id", "Unknown")
        prediction = explanation.get("prediction", {})
        model_importances = explanation.get("model_importances", {})
        missing_features = explanation.get("missing_features", {})
        prediction_analysis = explanation.get("prediction_analysis", {})
        
        # Generate a simplified summary with available information
        return self._generate_summary_text(
            patient_id=patient_id,
            prediction=prediction,
            model_importances=model_importances,
            shap_values=None,
            shap_base_value=None,
            missing_features=missing_features,
            prediction_analysis=prediction_analysis
        )
    
    def _extract_model_importances(self) -> Dict[str, float]:
        """
        Extract feature importances from the model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None or self.feature_names is None:
            return {}
        
        try:
            # Try to get feature importance directly from the model
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                return {self.feature_names[i]: float(importances[i]) 
                       for i in range(len(self.feature_names))}
            
            # For CatBoost models
            elif hasattr(self.model, 'get_feature_importance'):
                importances = self.model.get_feature_importance()
                return {self.feature_names[i]: float(importances[i]) 
                       for i in range(len(self.feature_names))}
                
            # Fallback to permutation importance
            else:
                print("Model does not have direct feature importances. Using equal weights.")
                return {feat: 1.0/len(self.feature_names) for feat in self.feature_names}
                
        except Exception as e:
            print(f"Error extracting model importances: {str(e)}")
            return {}
    
    def _calculate_shap_values(self, patient_data: pd.DataFrame) -> tuple:
        """
        Calculate SHAP values for a patient.
        
        Args:
            patient_data: DataFrame with patient data
            
        Returns:
            Tuple of (shap_values, base_value)
        """
        if self.explainer is None:
            return None, None
        
        try:
            # Prepare features for SHAP calculation
            # Get features that match the model's feature names
            available_features = list(set(self.feature_names).intersection(patient_data.columns))
            
            # If no direct matches, try matching with suffixes
            if not available_features:
                available_features = []
                for model_feat in self.feature_names:
                    for col in patient_data.columns:
                        if model_feat in col or col in model_feat:
                            available_features.append(col)
                            break
            
            # If still no matches, use all numeric features
            if not available_features:
                available_features = [col for col in patient_data.columns 
                                    if col not in ['icustayid', 'bloc', 'charttime', 'gender', 'mortality_90d'] 
                                    and pd.api.types.is_numeric_dtype(patient_data[col])]
            
            # Create a DataFrame with the selected features
            X = patient_data[available_features].copy()
            
            # Handle NaN values
            X = X.fillna(0)
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)
            base_value = self.explainer.expected_value
            
            # Handle different formats of SHAP values
            if isinstance(shap_values, list):
                # For classifier models that return a list of arrays (one per class)
                if len(shap_values) > 1:
                    # Use values for positive class (mortality)
                    shap_values = shap_values[1]
                    if isinstance(base_value, (list, np.ndarray)):
                        base_value = base_value[1]
                else:
                    shap_values = shap_values[0]
                    if isinstance(base_value, (list, np.ndarray)):
                        base_value = base_value[0]
            
            return shap_values, base_value
        
        except Exception as e:
            print(f"Error calculating SHAP values: {str(e)}")
            return None, None
    
    def _identify_missing_features(self, patient_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify missing features in patient data.
        
        Args:
            patient_data: DataFrame with patient data
            
        Returns:
            Dictionary with missing feature information
        """
        # Count missing values for each column
        missing_counts = patient_data.isnull().sum().to_dict()
        
        # Calculate percentage of missing values
        total_records = len(patient_data)
        missing_pct = {col: count/total_records*100 for col, count in missing_counts.items() if count > 0}
        
        # Identify features with high missingness (>30%)
        high_missing = {col: pct for col, pct in missing_pct.items() if pct > 30}
        
        # Check for zeros that might represent missing values
        zero_counts = {col: ((patient_data[col] == 0).sum() / total_records * 100)
                     for col in patient_data.columns 
                     if pd.api.types.is_numeric_dtype(patient_data[col])}
        
        # Features that likely have zeros as missing values
        zero_as_missing = []
        for col in zero_counts:
            # Physiological parameters that can't be zero
            if col in ['Potassium', 'Sodium', 'Chloride', 'Albumin', 'Hb', 'WBC_count', 
                      'Platelets_count', 'paO2', 'paCO2', 'Arterial_pH', 'Arterial_lactate']:
                if zero_counts[col] > 0:
                    zero_as_missing.append(col)
        
        return {
            "missing_counts": missing_counts,
            "missing_percentages": missing_pct,
            "high_missing_features": high_missing,
            "zero_counts": zero_counts,
            "zero_as_missing": zero_as_missing
        }
    
    def _analyze_imputation_impact(self, patient_data: pd.DataFrame, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the impact of imputation on the prediction by comparing predictions
        with and without imputation.
        
        Args:
            patient_data: DataFrame with patient data
            prediction: Dictionary with prediction results
            
        Returns:
            Dictionary with imputation impact analysis
        """
        # Get the current prediction probability with imputation
        current_prob = prediction.get("mortality_probability", None)
        
        # Check if there are missing values to impute
        missing_any = patient_data.isnull().any().any()
        
        if not missing_any:
            return {
                "has_missing_values": False,
                "imputation_impact": "No missing values to impute",
                "pre_imputation_probability": current_prob,
                "post_imputation_probability": current_prob,
                "probability_difference": 0.0
            }
        
        # Count how many features have missing values
        missing_features = patient_data.columns[patient_data.isnull().any()].tolist()
        missing_count = len(missing_features)
        
        # Try to make a prediction with non-imputed data
        # This is done by replacing all NaN values with a sentinel value (-999)
        # that will likely be outside the normal range of values
        try:
            # Create a copy with missing values replaced by sentinel
            non_imputed_df = patient_data.copy()
            non_imputed_df = non_imputed_df.fillna(-999)
            
            # Use the model to predict on non-imputed data
            # Note: This requires access to the model, which should be available via self.model
            if self.model is not None:
                try:
                    if hasattr(self.model, 'predict_proba'):
                        non_imputed_prob = self.model.predict_proba(non_imputed_df)[:, 1][0]
                    else:
                        # For pipeline models
                        non_imputed_prob = 0.5  # Default if we can't predict
                    
                    # Calculate the difference
                    prob_diff = current_prob - non_imputed_prob
                    
                    # Determine impact level
                    impact_level = "Minimal"
                    if abs(prob_diff) > 0.2:
                        impact_level = "Substantial"
                    elif abs(prob_diff) > 0.05:
                        impact_level = "Moderate"
                    
                except Exception as e:
                    print(f"Error predicting on non-imputed data: {str(e)}")
                    non_imputed_prob = None
                    prob_diff = None
                    impact_level = "Unknown"
            else:
                non_imputed_prob = None
                prob_diff = None
                impact_level = "Unknown (model not available)"
        except Exception as e:
            print(f"Error in imputation impact analysis: {str(e)}")
            non_imputed_prob = None
            prob_diff = None
            impact_level = "Error in analysis"
        
        # Estimate potential uncertainty due to imputation
        uncertainty_level = "Low"
        if missing_count > 5:
            uncertainty_level = "High"
        elif missing_count > 2:
            uncertainty_level = "Moderate"
            
        # Identify which important features are missing
        important_missing = []
        if hasattr(self.model, 'feature_importances_'):
            # Get top 10 most important features
            importances = self.model.feature_importances_
            top_features = [self.feature_names[i] for i in importances.argsort()[-10:]]
            
            # Check which top features are missing
            for feat in top_features:
                if feat in missing_features or any(feat in f for f in missing_features):
                    important_missing.append(feat)
        
        return {
            "has_missing_values": True,
            "missing_count": missing_count,
            "missing_features": missing_features,
            "important_missing": important_missing,
            "uncertainty_level": uncertainty_level,
            "pre_imputation_probability": non_imputed_prob,
            "post_imputation_probability": current_prob,
            "probability_difference": prob_diff,
            "imputation_impact_level": impact_level,
            "current_probability": current_prob
        }
    
    def _generate_shap_plots(self, patient_data: pd.DataFrame, shap_values) -> Dict[str, str]:
        """
        Generate SHAP plots for visualization.
        
        Args:
            patient_data: DataFrame with patient data
            shap_values: SHAP values calculated for the patient
            
        Returns:
            Dictionary with base64-encoded plot images
        """
        if shap_values is None:
            return {}
            
        plots = {}
        
        try:
            # 1. SHAP Force Plot
            plt.figure(figsize=(10, 3))
            force_plot = shap.force_plot(
                self.explainer.expected_value,
                shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                patient_data.values[0] if len(patient_data) > 0 else None,
                feature_names=list(patient_data.columns),
                matplotlib=True,
                show=False
            )
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            plots["force_plot"] = base64.b64encode(buf.read()).decode('utf-8')
            
            # 2. SHAP Waterfall Plot
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                    base_values=self.explainer.expected_value,
                    data=patient_data.values[0] if len(patient_data) > 0 else None,
                    feature_names=list(patient_data.columns)
                ),
                max_display=10,
                show=False
            )
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            plots["waterfall_plot"] = base64.b64encode(buf.read()).decode('utf-8')
            
        except Exception as e:
            print(f"Error generating SHAP plots: {str(e)}")
            
        return plots
    
    def _format_shap_values(self, shap_values, patient_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Format SHAP values for API response.
        
        Args:
            shap_values: Raw SHAP values
            patient_data: DataFrame with patient data
            
        Returns:
            List of dictionaries with feature name, value, and SHAP value
        """
        if shap_values is None:
            return []
            
        try:
            # Get feature names
            feature_names = list(patient_data.columns)
            
            # Format SHAP values
            formatted_values = []
            
            if len(shap_values.shape) > 1:
                # Multiple patients
                for i, feat in enumerate(feature_names):
                    formatted_values.append({
                        "feature": feat,
                        "value": float(patient_data[feat].iloc[0]),
                        "shap_value": float(shap_values[0, i]),
                        "impact": "positive" if shap_values[0, i] > 0 else "negative"
                    })
            else:
                # Single patient
                for i, feat in enumerate(feature_names):
                    formatted_values.append({
                        "feature": feat,
                        "value": float(patient_data[feat].iloc[0]),
                        "shap_value": float(shap_values[i]),
                        "impact": "positive" if shap_values[i] > 0 else "negative"
                    })
            
            # Sort by absolute SHAP value
            return sorted(formatted_values, key=lambda x: abs(x["shap_value"]), reverse=True)
            
        except Exception as e:
            print(f"Error formatting SHAP values: {str(e)}")
            return []
        
    def _generate_summary_text(self, patient_id, prediction, model_importances, 
                            shap_values, shap_base_value, missing_features, 
                            prediction_analysis) -> str:
        """
        Generate a comprehensive summary text explanation.
        
        Args:
            Various explanation components
            
        Returns:
            Formatted summary text
        """
        # This will be the input for the LLM to generate the final explanation
        sections = []
        
        # 1. Introduction and prediction result
        mortality_prob = prediction.get("mortality_probability", None)
        risk_level = prediction.get("risk_level", None)
        
        intro = f"# Mortality Prediction Explanation for Patient {patient_id}\n\n"
        
        if mortality_prob is not None and risk_level is not None:
            intro += f"## Mortality Risk\n"
            intro += f"The machine learning model predicts a {'high' if mortality_prob > 0.5 else 'relatively low' if mortality_prob < 0.2 else 'moderate'} probability of death within 90 days "
            intro += f"for Patient ID {patient_id}: {mortality_prob:.1%}.\n"
        
        sections.append(intro)
        
        # 2. Key factors influencing prediction
        factors_section = "## Key Factors Influencing Risk (and their Impact)\n\n"
        factors_section += "The model uses SHAP values to determine how each factor increases or decreases the predicted probability of death. "
        factors_section += "A negative SHAP value means that a high value for that feature decreases the risk of death, and vice-versa.\n\n"
        
        # Use SHAP values if available
        if shap_values is not None:
            # Format SHAP values
            formatted_shap = self._format_shap_values(shap_values, pd.DataFrame())
            
            if formatted_shap:
                # Positive factors (increasing mortality risk)
                positive_factors = [f for f in formatted_shap if f["shap_value"] > 0][:3]
                if positive_factors:
                    factors_section += "### Top 3 Factors Increasing Risk of Death:\n"
                    for i, factor in enumerate(positive_factors):
                        feat_name = factor["feature"].replace('_', ' ').title()
                        feat_value = factor.get("value", "Unknown")
                        shap_val = factor["shap_value"]
                        
                        factors_section += f"{i+1}. '{feat_name}': (SHAP Value: {shap_val:.4f}, Positive Impact) "
                        if isinstance(feat_value, (int, float)):
                            factors_section += f"A value of {feat_value:.2f} is associated with increased risk.\n"
                        else:
                            factors_section += f"Associated with increased risk.\n"
                    factors_section += "\n"
                
                # Negative factors (decreasing mortality risk)
                negative_factors = [f for f in formatted_shap if f["shap_value"] < 0][:3]
                if negative_factors:
                    factors_section += "### Top 3 Factors Decreasing Risk of Death:\n"
                    for i, factor in enumerate(negative_factors):
                        feat_name = factor["feature"].replace('_', ' ').title()
                        feat_value = factor.get("value", "Unknown")
                        shap_val = factor["shap_value"]
                        
                        factors_section += f"{i+1}. '{feat_name}': (SHAP Value: {shap_val:.4f}, Negative Impact) "
                        if isinstance(feat_value, (int, float)):
                            factors_section += f"The {'last ' if 'last' in factor['feature'] else ''}value of {feat_value:.2f} is associated with decreased risk.\n"
                        else:
                            factors_section += f"Associated with decreased risk.\n"
                    factors_section += "\n"
        
        sections.append(factors_section)
        
        # 3. Missing Data and Imputation Analysis
        imputation_section = "## Impact of Missing Data and Imputation\n\n"
        
        # Check if there are missing values to analyze
        has_missing = prediction_analysis.get("has_missing_values", False)
        
        if has_missing:
            missing_count = prediction_analysis.get("missing_count", 0)
            important_missing = prediction_analysis.get("important_missing", [])
            uncertainty_level = prediction_analysis.get("uncertainty_level", "Unknown")
            
            # Get pre and post imputation probabilities
            pre_imp_prob = prediction_analysis.get("pre_imputation_probability", None)
            post_imp_prob = prediction_analysis.get("post_imputation_probability", None)
            prob_diff = prediction_analysis.get("probability_difference", None)
            impact_level = prediction_analysis.get("imputation_impact_level", "Unknown")
            
            imputation_section += f"This patient has {missing_count} features with missing values. "
            
            if important_missing:
                important_names = [f.replace('_', ' ').title() for f in important_missing]
                imputation_section += f"These include important predictors: {', '.join(important_names)}.\n\n"
            else:
                imputation_section += "None of the missing features are among the most important predictors.\n\n"
            
            if pre_imp_prob is not None and post_imp_prob is not None and prob_diff is not None:
                imputation_section += f"### Imputation Impact on Prediction\n"
                imputation_section += f"- Mortality probability before imputation: {pre_imp_prob:.1%}\n"
                imputation_section += f"- Mortality probability after imputation: {post_imp_prob:.1%}\n"
                imputation_section += f"- Difference due to imputation: {prob_diff:.1%} ({'increase' if prob_diff > 0 else 'decrease'})\n\n"
                imputation_section += f"The impact of imputation on this prediction is considered **{impact_level}**.\n\n"
            
            imputation_section += f"The overall uncertainty introduced by missing data and imputation is estimated to be **{uncertainty_level}**.\n\n"
            
        else:
            imputation_section += "This patient does not have significant missing data that would affect the prediction. No imputation was necessary.\n\n"
        
        sections.append(imputation_section)
        
        # Combine all sections to create the prompt for the LLM
        return "\n".join(sections)
    
    def generate_html_report(self, explanation: Dict[str, Any]) -> str:
        """
        Generate an HTML report for the explanation.
        
        Args:
            explanation: Dictionary with explanation components
            
        Returns:
            HTML string with the report
        """
        from html_templates import get_explanation_html_template
        
        # Extract components from explanation
        patient_id = explanation.get("patient_id", "Unknown")
        summary = explanation.get("summary", "")
        shap_values = explanation.get("shap_values", [])
        plots = explanation.get("plots", {})
        prediction_analysis = explanation.get("prediction_analysis", {})
        
        # Convert the full explanation to HTML
        explanation_html = summary.replace("# ", "<h1>").replace("## ", "<h2>").replace("### ", "<h3>")
        explanation_html = explanation_html.replace("\n\n", "<br><br>").replace("\n", "<br>")
        
        # Format imputation information
        imputation_html = "<p>No imputation was needed for this patient.</p>"
        if prediction_analysis.get("has_missing_values", False):
            pre_imp = prediction_analysis.get("pre_imputation_probability")
            post_imp = prediction_analysis.get("post_imputation_probability")
            diff = prediction_analysis.get("probability_difference")
            
            if pre_imp is not None and post_imp is not None and diff is not None:
                imputation_html = f"""
                <p><strong>Pre-imputation mortality probability:</strong> {pre_imp:.1%}</p>
                <p><strong>Post-imputation mortality probability:</strong> {post_imp:.1%}</p>
                <p><strong>Impact of imputation:</strong> {abs(diff):.1%} ({'increase' if diff > 0 else 'decrease'} in risk)</p>
                """
            
            missing_features = prediction_analysis.get("missing_features", [])
            if missing_features:
                missing_list = ", ".join([f.replace("_", " ").title() for f in missing_features[:10]])
                imputation_html += f"<p><strong>Missing features:</strong> {missing_list}</p>"
        
        # Generate feature table rows
        feature_rows = ""
        for feature in shap_values[:15]:  # Show top 15 features
            feature_name = feature.get("feature", "").replace("_", " ").title()
            feature_value = feature.get("value", 0)
            shap_value = feature.get("shap_value", 0)
            impact = feature.get("impact", "")
            
            # Format values
            if isinstance(feature_value, float):
                feature_value = f"{feature_value:.2f}"
            
            # Determine direction class
            direction_class = "positive" if impact == "positive" else "negative"
            direction_text = "Increases Risk" if impact == "positive" else "Decreases Risk"
            
            feature_rows += f"""
                <tr>
                    <td>{feature_name}</td>
                    <td>{feature_value}</td>
                    <td>{abs(shap_value):.4f}</td>
                    <td class="{direction_class}">{direction_text}</td>
                </tr>
            """
        
        # Get waterfall plot
        waterfall_plot_html = ""
        if "waterfall_plot" in plots:
            waterfall_plot_html = f"""
                <div class="plot-container">
                    <h3>Feature Impact Visualization (Waterfall Plot)</h3>
                    <img src="data:image/png;base64,{plots['waterfall_plot']}" alt="SHAP Waterfall Plot">
                </div>
            """
        
        # Get force plot
        force_plot_html = ""
        if "force_plot" in plots:
            force_plot_html = f"""
                <div class="plot-container">
                    <h3>Detailed Feature Impact (Force Plot)</h3>
                    <img src="data:image/png;base64,{plots['force_plot']}" alt="SHAP Force Plot">
                </div>
            """
        
        # Get the HTML template and fill in the values
        html = get_explanation_html_template().format(
            patient_id=patient_id,
            explanation_html=explanation_html,
            waterfall_plot=waterfall_plot_html,
            force_plot=force_plot_html,
            feature_rows=feature_rows,
            imputation_html=imputation_html
        )
        
        return html