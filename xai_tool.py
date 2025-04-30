import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Union
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import shap
from textstat import flesch_kincaid_grade, flesch_reading_ease
import re

class XAITool:
    """
    Tool for generating explanations for sepsis mortality predictions.
    """
    
    def __init__(self, model=None, preprocessor=None, feature_names=None):
        """
        Initialize the tool with a trained model and preprocessor.
        
        Args:
            model: Trained model (sklearn classifier)
            preprocessor: Preprocessor for the model
            feature_names: List of feature names
        """
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.explainer = None
        self.explanation_templates = self._load_explanation_templates()
    
    def set_model(self, model, preprocessor=None, feature_names=None):
        """
        Set the model and preprocessor.
        
        Args:
            model: Trained model (sklearn classifier)
            preprocessor: Preprocessor for the model
            feature_names: List of feature names
        """
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.explainer = None
    
    def _load_explanation_templates(self) -> Dict[str, str]:
        """
        Load templates for different types of explanations.
        
        Returns:
            Dictionary with explanation templates
        """
        templates = {
            # Templates for different feature types
            'GCS': {
                'high': "The Glasgow Coma Scale (GCS) score of {value} indicates better neurological function than average (average is {avg_value}), suggesting a {direction} risk of mortality.",
                'low': "The Glasgow Coma Scale (GCS) score of {value} is lower than average (average is {avg_value}), indicating reduced neurological function and contributing to a {direction} risk of mortality.",
                'definition': "GCS measures level of consciousness, with scores ranging from 3 (deep coma) to 15 (fully alert)."
            },
            'SOFA': {
                'high': "The Sequential Organ Failure Assessment (SOFA) score of {value} is higher than average (average is {avg_value}), indicating more severe organ dysfunction and contributing to a {direction} risk of mortality.",
                'low': "The Sequential Organ Failure Assessment (SOFA) score of {value} is lower than average (average is {avg_value}), indicating less severe organ dysfunction and contributing to a {direction} risk of mortality.",
                'definition': "SOFA scores organ function across six systems from 0 (normal) to 24 (severe dysfunction)."
            },
            'HR': {
                'high': "The heart rate of {value} beats per minute is higher than average (average is {avg_value}), which may indicate cardiovascular stress and contributes to a {direction} risk of mortality.",
                'low': "The heart rate of {value} beats per minute is lower than average (average is {avg_value}), which may indicate better cardiovascular stability and contributes to a {direction} risk of mortality.",
                'definition': "Heart rate is the number of times the heart beats per minute."
            },
            'SysBP': {
                'high': "The systolic blood pressure of {value} mmHg is higher than average (average is {avg_value}), which may indicate better cardiovascular function and contributes to a {direction} risk of mortality.",
                'low': "The systolic blood pressure of {value} mmHg is lower than average (average is {avg_value}), which may indicate cardiovascular compromise and contributes to a {direction} risk of mortality.",
                'definition': "Systolic blood pressure is the pressure in the arteries when the heart contracts."
            },
            'MeanBP': {
                'high': "The mean arterial pressure of {value} mmHg is higher than average (average is {avg_value}), which may indicate better tissue perfusion and contributes to a {direction} risk of mortality.",
                'low': "The mean arterial pressure of {value} mmHg is lower than average (average is {avg_value}), which may indicate compromised tissue perfusion and contributes to a {direction} risk of mortality.",
                'definition': "Mean arterial pressure is the average pressure in the arteries during one cardiac cycle."
            },
            'Arterial_lactate': {
                'high': "The arterial lactate level of {value} mmol/L is higher than average (average is {avg_value}), which can indicate tissue hypoxia and metabolic stress, contributing to a {direction} risk of mortality.",
                'low': "The arterial lactate level of {value} mmol/L is lower than average (average is {avg_value}), which can indicate better tissue oxygenation and contributes to a {direction} risk of mortality.",
                'definition': "Arterial lactate is a marker of anaerobic metabolism and tissue hypoxia."
            },
            'SpO2': {
                'high': "The oxygen saturation of {value}% is higher than average (average is {avg_value}%), which indicates better oxygenation and contributes to a {direction} risk of mortality.",
                'low': "The oxygen saturation of {value}% is lower than average (average is {avg_value}%), which indicates reduced oxygenation and contributes to a {direction} risk of mortality.",
                'definition': "Oxygen saturation (SpO2) measures the percentage of hemoglobin binding sites occupied by oxygen."
            },
            'Creatinine': {
                'high': "The creatinine level of {value} mg/dL is higher than average (average is {avg_value}), which can indicate kidney dysfunction and contributes to a {direction} risk of mortality.",
                'low': "The creatinine level of {value} mg/dL is lower than average (average is {avg_value}), which can indicate better kidney function and contributes to a {direction} risk of mortality.",
                'definition': "Creatinine is a waste product filtered by the kidneys and elevated levels indicate kidney dysfunction."
            },
            'WBC_count': {
                'high': "The white blood cell count of {value} ×10^9/L is higher than average (average is {avg_value}), which can indicate infection or inflammation and contributes to a {direction} risk of mortality.",
                'low': "The white blood cell count of {value} ×10^9/L is lower than average (average is {avg_value}), which can indicate immune suppression and contributes to a {direction} risk of mortality.",
                'definition': "White blood cell count measures immune cells circulating in the blood."
            },
            'PaO2_FiO2': {
                'high': "The PaO2/FiO2 ratio of {value} mmHg is higher than average (average is {avg_value}), which indicates better lung function and contributes to a {direction} risk of mortality.",
                'low': "The PaO2/FiO2 ratio of {value} mmHg is lower than average (average is {avg_value}), which indicates worse lung function and contributes to a {direction} risk of mortality.",
                'definition': "PaO2/FiO2 ratio measures lung function by comparing oxygen in the blood to the concentration of oxygen being delivered."
            },
            'age': {
                'high': "The age of {value:.1f} years is higher than average (average is {avg_value:.1f}), which is associated with increased frailty and contributes to a {direction} risk of mortality.",
                'low': "The age of {value:.1f} years is lower than average (average is {avg_value:.1f}), which is associated with greater physiological reserve and contributes to a {direction} risk of mortality.",
                'definition': "Age is a key factor in determining overall health and resilience to illness."
            },
            'elixhauser': {
                'high': "The Elixhauser comorbidity score of {value} is higher than average (average is {avg_value}), indicating more chronic health conditions and contributing to a {direction} risk of mortality.",
                'low': "The Elixhauser comorbidity score of {value} is lower than average (average is {avg_value}), indicating fewer chronic health conditions and contributing to a {direction} risk of mortality.",
                'definition': "Elixhauser score summarizes the burden of chronic diseases."
            },
            
            # Generic templates for other variables
            'generic': {
                'high': "The {feature_name} of {value:.2f} is higher than average (average is {avg_value:.2f}), which contributes to a {direction} risk of mortality.",
                'low': "The {feature_name} of {value:.2f} is lower than average (average is {avg_value:.2f}), which contributes to a {direction} risk of mortality.",
                'definition': "{feature_name} is a clinical parameter used in patient assessment."
            },
            
            # Templates for missing values
            'missing': {
                'high_importance': "The {feature_name} value was missing and had to be estimated. This feature is important for prediction, so this could affect the reliability of the prediction.",
                'medium_importance': "The {feature_name} value was missing and had to be estimated. This might slightly affect the prediction reliability.",
                'low_importance': "The {feature_name} value was missing and had to be estimated, but this likely has minimal impact on the prediction."
            },
            
            # Summary templates
            'summary': {
                'high_risk': "Based on the analysis, this patient has a high risk of mortality within 90 days. The key contributing factors are {factor_list}.",
                'medium_risk': "Based on the analysis, this patient has a moderate risk of mortality within 90 days. The key contributing factors are {factor_list}.",
                'low_risk': "Based on the analysis, this patient has a low risk of mortality within 90 days. The key contributing factors are {factor_list}."
            },
            
            # Confidence templates
            'confidence': {
                'high': "The prediction has high confidence ({confidence:.1f}%), meaning the model is fairly certain about this assessment.",
                'medium': "The prediction has moderate confidence ({confidence:.1f}%), indicating some uncertainty in this assessment.",
                'low': "The prediction has low confidence ({confidence:.1f}%), indicating significant uncertainty in this assessment."
            },
            
            # Missing data impact templates
            'missing_impact': {
                'high': "Several important clinical measurements were missing, which significantly reduces the reliability of this prediction.",
                'medium': "Some clinical measurements were missing, which moderately affects the reliability of this prediction.",
                'low': "Few or no important clinical measurements were missing, so this prediction should be reliable.",
                'none': "All important clinical measurements were available, making this a highly reliable prediction."
            }
        }
        
        return templates
    
    def _initialize_shap_explainer(self):
        """
        Initialize the SHAP explainer for the model.
        """
        if self.model is None:
            raise ValueError("Model has not been set")
        
        # Determine model type
        if isinstance(self.model, RandomForestClassifier):
            self.explainer = shap.TreeExplainer(self.model)
        elif isinstance(self.model, GradientBoostingClassifier):
            self.explainer = shap.TreeExplainer(self.model)
        elif isinstance(self.model, LogisticRegression):
            self.explainer = shap.LinearExplainer(self.model, self.preprocessor)
        else:
            # For other model types, use KernelExplainer
            self.explainer = shap.KernelExplainer(self.model.predict_proba, self.preprocessor)
    
    def get_shap_values(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate SHAP values for a patient.
        
        Args:
            X: Feature DataFrame for a patient
            
        Returns:
            Tuple of (SHAP values, expected value)
        """
        if self.explainer is None:
            self._initialize_shap_explainer()
        
        # Process DataFrame to handle Timestamp objects
        X_processed = X.copy()
        
        # Convert any datetime columns to numeric (timestamp in seconds)
        for col in X_processed.columns:
            if pd.api.types.is_datetime64_any_dtype(X_processed[col]):
                X_processed[col] = pd.to_numeric(X_processed[col].astype(np.int64) // 10**9, errors='coerce')
        
        # Calculate SHAP values
        try:
            shap_values = self.explainer.shap_values(X_processed)
            
            # For classification models, shap_values is a list of arrays (one per class)
            if isinstance(shap_values, list):
                # Use the positive class (index 1)
                shap_values = shap_values[1]
            
            # Get expected value
            if hasattr(self.explainer, 'expected_value'):
                expected_value = self.explainer.expected_value
                if isinstance(expected_value, list):
                    expected_value = expected_value[1]
            else:
                expected_value = 0.0
            
            return shap_values, expected_value
        except Exception as e:
            print(f"Error calculating SHAP values: {str(e)}")
            # Return empty arrays as fallback
            dummy_values = np.zeros((X.shape[0], X.shape[1]))
            return dummy_values, 0.0
    
    def plot_shap_summary(self, X: pd.DataFrame) -> str:
        """
        Generate a SHAP summary plot for a patient and return it as a base64 encoded image.
        
        Args:
            X: Feature DataFrame for a patient
            
        Returns:
            Base64 encoded image
        """
        if self.explainer is None:
            self._initialize_shap_explainer()
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # For classification models, shap_values is a list of arrays (one per class)
        if isinstance(shap_values, list):
            # Use the positive class (index 1)
            plot_shap_values = shap_values[1]
        else:
            plot_shap_values = shap_values
        
        # Create plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(plot_shap_values, X, show=False)
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        
        # Encode as base64
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return image_base64
    
    def plot_shap_force(self, X: pd.DataFrame) -> str:
        """
        Generate a SHAP force plot for a patient and return it as a base64 encoded image.
        
        Args:
            X: Feature DataFrame for a patient
            
        Returns:
            Base64 encoded image
        """
        if self.explainer is None:
            self._initialize_shap_explainer()
        
        # Calculate SHAP values
        shap_values, expected_value = self.get_shap_values(X)
        
        # Create plot
        plt.figure(figsize=(12, 3))
        shap.force_plot(expected_value, shap_values[0], X.iloc[0], matplotlib=True, show=False)
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        
        # Encode as base64
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return image_base64
    
    def plot_shap_waterfall(self, X: pd.DataFrame) -> str:
        """
        Generate a SHAP waterfall plot for a patient and return it as a base64 encoded image.
        
        Args:
            X: Feature DataFrame for a patient
            
        Returns:
            Base64 encoded image
        """
        if self.explainer is None:
            self._initialize_shap_explainer()
        
        # Calculate SHAP values
        shap_values, expected_value = self.get_shap_values(X)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                           base_values=expected_value, 
                                           data=X.iloc[0].values, 
                                           feature_names=X.columns), 
                          show=False)
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        
        # Encode as base64
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return image_base64
    
    def create_feature_explanations(self, top_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create human-readable explanations for top features.
        
        Args:
            top_features: List of dictionaries with feature contribution information
            
        Returns:
            List of dictionaries with human-readable explanations
        """
        explanations = []
        
        for feature in top_features:
            feature_name = feature['feature']
            value = feature.get('value')
            avg_value = feature.get('avg_value')
            direction = feature.get('direction')
            imputed = feature.get('imputed', False)
            importance = feature.get('importance', 0.5)
            
            # Create template context
            context = {
                'feature_name': feature_name.replace('_', ' ').title(),
                'value': value,
                'avg_value': avg_value,
                'direction': 'higher' if direction in ['high', 'positive'] else 'lower',
                'importance': importance
            }
            
            # Get base feature name (without _mean, _max, etc.)
            base_feature = feature_name
            for suffix in ['_mean', '_max', '_min', '_last', '_std']:
                if base_feature.endswith(suffix):
                    base_feature = base_feature[:-len(suffix)]
                    break
            
            # Find the appropriate template
            template_key = base_feature if base_feature in self.explanation_templates else 'generic'
            templates = self.explanation_templates.get(template_key, self.explanation_templates['generic'])
            
            # Generate explanation
            explanation = {}
            if imputed:
                # Use missing value template
                if importance > 0.7:
                    importance_level = 'high_importance'
                elif importance > 0.3:
                    importance_level = 'medium_importance'
                else:
                    importance_level = 'low_importance'
                
                explanation['text'] = self.explanation_templates['missing'][importance_level].format(**context)
                explanation['category'] = 'missing'
            else:
                # Use regular template
                if direction in ['high', 'positive']:
                    explanation['text'] = templates['high'].format(**context)
                else:
                    explanation['text'] = templates['low'].format(**context)
                
                explanation['category'] = 'feature'
            
            # Add technical definition
            explanation['definition'] = templates.get('definition', '').format(**context)
            
            # Add metadata
            explanation['feature'] = feature_name
            explanation['importance'] = importance
            
            explanations.append(explanation)
        
        return explanations
    
    def create_summary_explanation(self, prediction: Dict[str, Any], feature_explanations: List[Dict[str, Any]]) -> str:
        """
        Create a summary explanation for a prediction.
        
        Args:
            prediction: Dictionary with prediction results
            feature_explanations: List of dictionaries with feature explanations
            
        Returns:
            Summary explanation text
        """
        # Get prediction probability
        probability = prediction['mortality_probability']
        confidence = prediction['confidence'] * 100
        
        # Determine risk level
        if probability < 0.25:
            risk_level = 'low_risk'
        elif probability < 0.75:
            risk_level = 'medium_risk'
        else:
            risk_level = 'high_risk'
        
        # Determine confidence level
        if confidence > 85:
            confidence_level = 'high'
        elif confidence > 65:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        # Get top features
        top_features = []
        for explanation in feature_explanations[:3]:  # Use top 3 features
            feature_name = explanation['feature'].replace('_', ' ').title()
            if '_mean' in explanation['feature']:
                feature_name = feature_name.replace('Mean', '(avg)')
            elif '_max' in explanation['feature']:
                feature_name = feature_name.replace('Max', '(max)')
            elif '_min' in explanation['feature']:
                feature_name = feature_name.replace('Min', '(min)')
            elif '_last' in explanation['feature']:
                feature_name = feature_name.replace('Last', '(last)')
            
            top_features.append(feature_name)
        
        # Create factor list string
        factor_list = ', '.join(top_features)
        
        # Create summary using template
        summary = self.explanation_templates['summary'][risk_level].format(factor_list=factor_list)
        
        # Add confidence statement
        confidence_statement = self.explanation_templates['confidence'][confidence_level].format(confidence=confidence)
        
        # Count imputed features
        imputed_count = sum(1 for f in prediction.get('contributing_features', []) if f.get('imputed', False))
        
        # Determine missing data impact
        if imputed_count >= 3:
            missing_impact = 'high'
        elif imputed_count > 0:
            missing_impact = 'medium'
        elif imputed_count == 0:
            missing_impact = 'none'
        else:
            missing_impact = 'low'
        
        # Add missing data impact statement
        missing_statement = self.explanation_templates['missing_impact'][missing_impact]
        
        # Combine all parts
        full_explanation = f"{summary}\n\n{confidence_statement}\n\n{missing_statement}"
        
        return full_explanation
    
    def generate_explanation(self, patient_data: Union[pd.DataFrame, Dict[str, Any]], prediction: Dict[str, Any], skip_shap_plots: bool = False) -> Dict[str, Any]:
        """
        Generate a comprehensive explanation for a patient's mortality prediction.
        
        Args:
            patient_data: DataFrame or dictionary with patient data
            prediction: Dictionary with prediction results
            skip_shap_plots: Whether to skip generating SHAP visualizations
            
        Returns:
            Dictionary with explanation components
        """
        # Convert dictionary to DataFrame if needed
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data.copy()
        
        # Get feature names if not provided
        if self.feature_names is None and self.model is not None:
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = self.model.feature_names_in_
            elif 'contributing_features' in prediction:
                self.feature_names = [f['feature'] for f in prediction['contributing_features']]
        
        # Generate SHAP values if model is available and SHAP plots are not skipped
        shap_values = None
        expected_value = None
        shap_plots = {}
        
        if self.model is not None and not skip_shap_plots:
            try:
                # Calculate SHAP values - try with a timeout to avoid long processing
                shap_values, expected_value = self.get_shap_values(patient_df)
                
                # Generate plots with a simplified approach
                try:
                    # Only generate the simplest plot (waterfall) to reduce complexity
                    shap_plots['waterfall'] = self.plot_shap_waterfall(patient_df)
                except Exception as e:
                    print(f"Error generating SHAP waterfall plot: {str(e)}")
            except Exception as e:
                print(f"Error generating SHAP explanations: {str(e)}")
                # Continue without SHAP values
        
        # Create explanations for contributing features
        feature_explanations = []
        if 'contributing_features' in prediction:
            feature_explanations = self.create_feature_explanations(prediction['contributing_features'])
        
        # Create summary explanation
        summary_explanation = self.create_summary_explanation(prediction, feature_explanations)
        
        # Calculate readability scores
        readability = self.calculate_readability(summary_explanation, feature_explanations)
        
        # Return complete explanation
        return {
            'summary': summary_explanation,
            'feature_explanations': feature_explanations,
            'shap_values': shap_values.tolist() if shap_values is not None else None,
            'expected_value': float(expected_value) if expected_value is not None else None,
            'shap_plots': shap_plots,
            'readability': readability,
            'prediction': prediction
        }
    
    def calculate_readability(self, summary: str, feature_explanations: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate readability scores for the explanations.
        
        Args:
            summary: Summary explanation text
            feature_explanations: List of dictionaries with feature explanations
            
        Returns:
            Dictionary with readability scores
        """
        # Combine all explanation texts
        all_text = summary + "\n\n"
        for explanation in feature_explanations:
            all_text += explanation['text'] + "\n"
        
        # Calculate Flesch-Kincaid Grade Level (lower is more readable)
        fk_grade = flesch_kincaid_grade(all_text)
        
        # Calculate Flesch Reading Ease (higher is more readable)
        fk_ease = flesch_reading_ease(all_text)
        
        return {
            'flesch_kincaid_grade': float(fk_grade),
            'flesch_reading_ease': float(fk_ease),
            'interpretation': self._interpret_readability(fk_grade, fk_ease)
        }
    
    def _interpret_readability(self, fk_grade: float, fk_ease: float) -> str:
        """
        Interpret readability scores.
        
        Args:
            fk_grade: Flesch-Kincaid Grade Level
            fk_ease: Flesch Reading Ease
            
        Returns:
            Interpretation of readability
        """
        # Interpret Flesch-Kincaid Grade Level
        if fk_grade <= 6:
            grade_interp = "very easy to understand (elementary school level)"
        elif fk_grade <= 8:
            grade_interp = "easy to understand (middle school level)"
        elif fk_grade <= 12:
            grade_interp = "fairly easy to understand (high school level)"
        elif fk_grade <= 16:
            grade_interp = "moderately difficult (college level)"
        else:
            grade_interp = "difficult (graduate level)"
        
        # Interpret Flesch Reading Ease
        if fk_ease >= 90:
            ease_interp = "very easy to read"
        elif fk_ease >= 80:
            ease_interp = "easy to read"
        elif fk_ease >= 70:
            ease_interp = "fairly easy to read"
        elif fk_ease >= 60:
            ease_interp = "standard difficulty"
        elif fk_ease >= 50:
            ease_interp = "moderately difficult"
        else:
            ease_interp = "difficult"
        
        return f"The explanation text is {grade_interp} and {ease_interp}."
    
    def generate_simplified_explanation(self, explanation: Dict[str, Any], target_grade_level: float = 8.0) -> str:
        """
        Generate a simplified version of the explanation for improved readability.
        
        Args:
            explanation: Dictionary with explanation components
            target_grade_level: Target Flesch-Kincaid Grade Level
            
        Returns:
            Simplified explanation text
        """
        current_grade = explanation['readability']['flesch_kincaid_grade']
        
        # Skip if already at or below target grade level
        if current_grade <= target_grade_level:
            return explanation['summary']
        
        # Simplification strategies
        simplified_text = explanation['summary']
        
        # Strategy 1: Shorten sentences
        simplified_text = self._shorten_sentences(simplified_text)
        
        # Strategy 2: Replace complex words
        simplified_text = self._simplify_vocabulary(simplified_text)
        
        # Strategy 3: Add explanations for medical terms
        simplified_text = self._add_term_explanations(simplified_text)
        
        # Check new grade level
        new_grade = flesch_kincaid_grade(simplified_text)
        
        # If still above target, apply more aggressive simplification
        if new_grade > target_grade_level:
            # Strategy 4: Further simplify structure
            simplified_text = self._further_simplify(simplified_text)
        
        return simplified_text
    
    def _shorten_sentences(self, text: str) -> str:
        """
        Shorten long sentences in the text.
        
        Args:
            text: Original text
            
        Returns:
            Text with shortened sentences
        """
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Process each sentence
        simplified_sentences = []
        for sentence in sentences:
            # Skip short sentences
            if len(sentence.split()) < 15:
                simplified_sentences.append(sentence)
                continue
            
            # Split long sentences at conjunctions
            parts = re.split(r',\s+|\s+and\s+|\s+but\s+|\s+or\s+|\s+because\s+|\s+which\s+|\s+while\s+', sentence)
            
            # Recombine parts into shorter sentences
            current_parts = []
            for part in parts:
                current_parts.append(part)
                
                # If we have enough words for a sentence, add it
                if sum(len(p.split()) for p in current_parts) >= 10:
                    combined = ' '.join(current_parts)
                    if not combined.endswith(('.', '!', '?')):
                        combined += '.'
                    simplified_sentences.append(combined)
                    current_parts = []
            
            # Add any remaining parts
            if current_parts:
                combined = ' '.join(current_parts)
                if not combined.endswith(('.', '!', '?')):
                    combined += '.'
                simplified_sentences.append(combined)
        
        return ' '.join(simplified_sentences)
    
    def _simplify_vocabulary(self, text: str) -> str:
        """
        Replace complex words with simpler alternatives.
        
        Args:
            text: Original text
            
        Returns:
            Text with simpler vocabulary
        """
        # Define complex words and their simpler alternatives
        word_replacements = {
            'mortality': 'death risk',
            'contributing': 'adding to',
            'assessment': 'review',
            'significantly': 'greatly',
            'moderately': 'somewhat',
            'indicates': 'shows',
            'dysfunction': 'problems',
            'reliability': 'trustworthiness',
            'imputed': 'estimated',
            'physiological': 'bodily',
            'elevated': 'high',
            'compromised': 'weakened',
            'monitoring': 'watching',
            'interventions': 'treatments',
            'cardiovascular': 'heart',
            'neurological': 'brain',
            'metabolic': 'body chemistry',
            'oxygenation': 'oxygen levels',
            'perfusion': 'blood flow',
            'hypoxia': 'low oxygen',
            'anaerobic': 'without oxygen'
        }
        
        # Replace complex words
        simplified_text = text
        for complex_word, simple_word in word_replacements.items():
            simplified_text = re.sub(r'\b' + complex_word + r'\b', simple_word, simplified_text, flags=re.IGNORECASE)
        
        return simplified_text
    
    def _add_term_explanations(self, text: str) -> str:
        """
        Add explanations for medical terms.
        
        Args:
            text: Original text
            
        Returns:
            Text with added explanations
        """
        # Define medical terms and their explanations
        term_explanations = {
            'SOFA': 'SOFA (which measures how well organs are working)',
            'GCS': 'GCS (which measures brain function)',
            'lactate': 'lactate (a substance that increases when tissues don\'t get enough oxygen)',
            'creatinine': 'creatinine (which shows kidney function)',
            'Elixhauser': 'Elixhauser score (which counts chronic health problems)',
            'PaO2/FiO2': 'PaO2/FiO2 ratio (which shows how well lungs are working)',
            'SpO2': 'SpO2 (blood oxygen level)',
            'systolic': 'systolic (upper blood pressure number)',
            'diastolic': 'diastolic (lower blood pressure number)'
        }
        
        # Add explanations for first occurrence of each term
        simplified_text = text
        for term, explanation in term_explanations.items():
            # Only replace the first occurrence
            pattern = r'\b' + term + r'\b'
            match = re.search(pattern, simplified_text, re.IGNORECASE)
            if match:
                pos = match.start()
                simplified_text = simplified_text[:pos] + explanation + simplified_text[pos + len(term):]
        
        return simplified_text
    
    def _further_simplify(self, text: str) -> str:
        """
        Apply more aggressive simplification strategies.
        
        Args:
            text: Original text
            
        Returns:
            Further simplified text
        """
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        
        # Process each paragraph
        simplified_paragraphs = []
        for paragraph in paragraphs:
            # Skip short paragraphs
            if len(paragraph.split()) < 25:
                simplified_paragraphs.append(paragraph)
                continue
            
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            
            # Simplify sentences
            simplified_sentences = []
            for sentence in sentences:
                # Skip very short sentences
                if len(sentence.split()) < 8:
                    simplified_sentences.append(sentence)
                    continue
                
                # Simplify sentence structure
                simplified = sentence
                
                # Remove parenthetical expressions
                simplified = re.sub(r'\([^)]*\)', '', simplified)
                
                # Remove qualifying phrases
                simplified = re.sub(r',\s*which[^,]*,', ',', simplified)
                simplified = re.sub(r',\s*including[^,]*,', ',', simplified)
                
                # Add the simplified sentence
                if simplified.strip():
                    simplified_sentences.append(simplified)
            
            # Combine sentences into paragraph
            if simplified_sentences:
                simplified_paragraphs.append(' '.join(simplified_sentences))
        
        return '\n\n'.join(simplified_paragraphs)
    
    def generate_html_report(self, explanation: Dict[str, Any]) -> str:
        """
        Generate an HTML report for the explanation.
        
        Args:
            explanation: Dictionary with explanation components
            
        Returns:
            HTML report
        """
        # Extract components
        summary = explanation['summary']
        feature_explanations = explanation['feature_explanations']
        prediction = explanation['prediction']
        readability = explanation['readability']
        shap_plots = explanation.get('shap_plots', {})
        
        # Determine risk level for styling
        mortality_prob = prediction['mortality_probability']
        if mortality_prob < 0.25:
            risk_color = "#4CAF50"  # Green
            risk_level = "Low"
        elif mortality_prob < 0.5:
            risk_color = "#FFC107"  # Amber
            risk_level = "Moderate"
        elif mortality_prob < 0.75:
            risk_color = "#FF9800"  # Orange
            risk_level = "High"
        else:
            risk_color = "#F44336"  # Red
            risk_level = "Very High"
        
        # Build HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sepsis Mortality Prediction Explanation</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1000px; margin: 0 auto; padding: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .risk-badge {{ display: inline-block; padding: 5px 10px; border-radius: 4px; color: white; background-color: {risk_color}; margin-left: 10px; }}
                .probability {{ font-size: 24px; font-weight: bold; margin: 20px 0; }}
                .probability-bar {{ background-color: #eee; height: 30px; border-radius: 4px; margin-bottom: 20px; }}
                .probability-fill {{ height: 100%; border-radius: 4px; width: {mortality_prob * 100}%; background-color: {risk_color}; }}
                .explanation {{ background-color: #f9f9f9; border-left: 4px solid #2c3e50; padding: 15px; margin: 15px 0; }}
                .feature {{ margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px solid #eee; }}
                .feature-header {{ font-weight: bold; }}
                .feature-explanation {{ margin-top: 5px; }}
                .feature-definition {{ font-style: italic; color: #666; margin-top: 5px; }}
                .missing {{ background-color: #fff3e0; }}
                .shap-plots {{ margin: 20px 0; }}
                .shap-plot {{ margin-bottom: 20px; }}
                .readability {{ background-color: #e3f2fd; padding: 10px; border-radius: 4px; margin-top: 20px; }}
                .disclaimer {{ font-style: italic; color: #666; margin-top: 20px; padding-top: 20px; border-top: 1px solid #eee; }}
            </style>
        </head>
        <body>
            <h1>Sepsis Mortality Risk Assessment <span class="risk-badge">{risk_level} Risk</span></h1>
            
            <div class="probability">
                90-day Mortality Probability: {mortality_prob:.1%}
            </div>
            
            <div class="probability-bar">
                <div class="probability-fill"></div>
            </div>
            
            <div class="explanation">
                <p>{summary}</p>
            </div>
            
            <h2>Key Risk Factors</h2>
        """
        
        # Add SHAP plots if available
        if shap_plots:
            html += '<h2>Visual Explanations</h2><div class="shap-plots">'
            
            if 'waterfall' in shap_plots:
                html += f"""
                <div class="shap-plot">
                    <h3>Feature Impact on Prediction</h3>
                    <img src="data:image/png;base64,{shap_plots['waterfall']}" alt="SHAP Waterfall Plot" width="100%">
                    <p>This chart shows how each feature pushed the prediction higher (red) or lower (blue).</p>
                </div>
                """
            
            if 'summary' in shap_plots:
                html += f"""
                <div class="shap-plot">
                    <h3>Feature Importance Summary</h3>
                    <img src="data:image/png;base64,{shap_plots['summary']}" alt="SHAP Summary Plot" width="100%">
                    <p>This plot shows the importance of each feature. Features are ranked by importance, with the most important at the top.</p>
                </div>
                """
            
            html += '</div>'
        
        # Add readability information
        html += f"""
            <div class="readability">
                <h3>Explanation Readability</h3>
                <p>Flesch-Kincaid Grade Level: {readability['flesch_kincaid_grade']:.1f}</p>
                <p>Flesch Reading Ease: {readability['flesch_reading_ease']:.1f}</p>
                <p>{readability['interpretation']}</p>
            </div>
            
            <div class="disclaimer">
                <p>This prediction is based on statistical analysis and should be used as a decision support tool only. 
                Clinical judgment should always take precedence. The model does not account for all possible clinical 
                factors and may have limitations in certain patient populations.</p>
            </div>
        </body>
        </html>
        """
        
        return html


# Example usage (would be wrapped in API endpoint)
if __name__ == "__main__":
    # This is a placeholder - in a real implementation, you would use the actual model and data
    
    # Create a mock prediction
    mock_prediction = {
        'patient_id': 12345,
        'mortality_probability': 0.75,
        'mortality_class': 1,
        'confidence': 0.8,
        'contributing_features': [
            {
                'feature': 'SOFA_max',
                'importance': 0.32,
                'value': 10.0,
                'avg_value': 6.2,
                'direction': 'high',
                'imputed': False
            },
            {
                'feature': 'Arterial_lactate_mean',
                'importance': 0.28,
                'value': 4.2,
                'avg_value': 2.1,
                'direction': 'high',
                'imputed': False
            },
            {
                'feature': 'age',
                'importance': 0.18,
                'value': 76.5,
                'avg_value': 65.2,
                'direction': 'high',
                'imputed': False
            },
            {
                'feature': 'GCS_min',
                'importance': 0.15,
                'value': 8.0,
                'avg_value': 12.5,
                'direction': 'low',
                'imputed': True
            },
            {
                'feature': 'SysBP_min',
                'importance': 0.12,
                'value': 85.0,
                'avg_value': 110.3,
                'direction': 'low',
                'imputed': False
            }
        ]
    }
    
    # Initialize the XAI tool
    xai_tool = XAITool()
    
    # Create feature explanations
    feature_explanations = xai_tool.create_feature_explanations(mock_prediction['contributing_features'])
    print("Feature explanations:")
    for explanation in feature_explanations:
        print(f"- {explanation['feature']}: {explanation['text']}")
        print(f"  Definition: {explanation['definition']}\n")
    
    # Create summary explanation
    summary = xai_tool.create_summary_explanation(mock_prediction, feature_explanations)
    print("\nSummary explanation:")
    print(summary)
    
    # Calculate readability
    readability = xai_tool.calculate_readability(summary, feature_explanations)
    print("\nReadability assessment:")
    print(f"Flesch-Kincaid Grade Level: {readability['flesch_kincaid_grade']:.1f}")
    print(f"Flesch Reading Ease: {readability['flesch_reading_ease']:.1f}")
    print(readability['interpretation'])
    
    # Generate simplified explanation
    mock_explanation = {
        'summary': summary,
        'feature_explanations': feature_explanations,
        'readability': readability,
        'prediction': mock_prediction
    }
    simplified = xai_tool.generate_simplified_explanation(mock_explanation, target_grade_level=6.0)
    print("\nSimplified explanation:")
    print(simplified)
    
    # Generate HTML report
    # Note: In a real implementation, you would include SHAP plots
    mock_explanation['shap_plots'] = {}
    html_report = xai_tool.generate_html_report(mock_explanation)
    print("\nHTML report generated (first 500 characters):")
    print(html_report[:500] + "...")