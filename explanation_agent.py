import os
import json
from typing import Dict, Any, List, Optional

# Import OpenAI if using their API for the reasoning agent
import os
from dotenv import load_dotenv

# Load environment variables 
load_dotenv()

# Check if OPENAI_API_KEY is set
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class ExplanationAgent:
    """
    Agent for reasoning about model explanations and generating human-readable summaries.
    """
    
    def __init__(self):
        """
        Initialize the explanation agent.
        """
        self.llm = None
        self.initialize_llm()
    
    def initialize_llm(self):
        """
        Initialize the LLM for reasoning.
        """
        if OPENAI_API_KEY:
            try:
                # Try newer LangChain version with OpenAI
                try:
                    from langchain_openai import ChatOpenAI
                    self.llm = ChatOpenAI(
                        api_key=OPENAI_API_KEY,
                        temperature=0,
                        model="gpt-3.5-turbo"
                    )
                    print("Initialized OpenAI ChatGPT with newer LangChain version")
                except ImportError:
                    # Try older LangChain version
                    from langchain.chat_models import ChatOpenAI
                    self.llm = ChatOpenAI(
                        openai_api_key=OPENAI_API_KEY,
                        temperature=0,
                        model_name="gpt-3.5-turbo"
                    )
                    print("Initialized OpenAI ChatGPT with older LangChain version")
            except Exception as e:
                print(f"Error initializing OpenAI: {str(e)}")
                self.llm = None
        else:
            print("OPENAI_API_KEY not found. Using fallback explanation method")
            self.llm = None
    
    def analyze_explanation(self, explanation_data: Dict[str, Any]) -> str:
        """
        Analyze the explanation data and generate a human-readable summary.
        
        Args:
            explanation_data: Dictionary with explanation components
            
        Returns:
            Human-readable summary text
        """
        # Try to use the LLM-based explanation first
        try:
            from llm_explanation_agent import LLMExplanationAgent
            llm_agent = LLMExplanationAgent()
            llm_explanation = llm_agent.generate_explanation(explanation_data)
            
            # If we got a good explanation from the LLM, use it
            if llm_explanation and len(llm_explanation) > 200:  # Simple check to see if we got something substantial
                return llm_explanation
        except Exception as e:
            print(f"Error using LLM explanation agent: {str(e)}")
        
        # If LLM explanation failed or is not available, fall back to template-based approach
        if self.llm:
            try:
                # Call the LLM with the explanation data
                return self._generate_with_llm(explanation_data)
            except Exception as e:
                print(f"Error generating explanation with LLM: {str(e)}")
                # Fall back to template-based explanation
                return self._generate_template_explanation(explanation_data)
        else:
            # Use template-based explanation
            return self._generate_template_explanation(explanation_data)
    
    def _generate_with_llm(self, explanation_data: Dict[str, Any]) -> str:
        """
        Generate explanation using LLM integration.
        
        Args:
            explanation_data: Dictionary with explanation components
            
        Returns:
            Explanation text from LLM
        """
        # Extract key components for the prompt
        patient_id = explanation_data.get("patient_id", "Unknown")
        mortality_prob = explanation_data.get("prediction", {}).get("mortality_probability", 0)
        risk_level = explanation_data.get("prediction", {}).get("risk_level", "Unknown")
        
        # Get top SHAP values or model importances
        feature_impacts = []
        if "shap_values" in explanation_data and explanation_data["shap_values"]:
            # Use SHAP values
            for feature in explanation_data["shap_values"][:10]:  # Top 10 features
                feature_impacts.append({
                    "feature": feature["feature"],
                    "value": feature["value"],
                    "impact": feature["impact"],
                    "shap_value": feature["shap_value"]
                })
        elif "model_importances" in explanation_data:
            # Use model importances
            importances = explanation_data["model_importances"]
            for feat, imp in sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
                feature_impacts.append({
                    "feature": feat,
                    "importance": imp
                })
        
        # Get missing data information
        missing_info = explanation_data.get("missing_features", {})
        missing_counts = missing_info.get("missing_counts", {})
        high_missing = missing_info.get("high_missing_features", {})
        zero_as_missing = missing_info.get("zero_as_missing", [])
        
        # Get imputation impact
        imputation_impact = explanation_data.get("prediction_analysis", {})
        has_missing = imputation_impact.get("has_missing_values", False)
        missing_count = imputation_impact.get("missing_count", 0)
        important_missing = imputation_impact.get("important_missing", [])
        uncertainty_level = imputation_impact.get("uncertainty_level", "Unknown")
        
        # Construct the prompt template
        prompt = f"""
        You are an AI medical assistant that analyzes ICU patient data to explain mortality risk predictions.
        
        Patient ID {patient_id} has a mortality probability of {mortality_prob:.1%} with a risk level of {risk_level}.
        
        Based on the analysis data below, please provide a detailed explanation of:
        
        1. Feature importance: Identify the top positive features that increase mortality risk and their impact magnitude, and the top negative features that decrease risk and their impact magnitude.
        
        2. Impact of Missing Data: Explain whether missing features influenced the prediction and how crucial the missing data was to the model's decision-making process.
        
        3. Impact of Imputation: Evaluate whether imputing missing data affects prediction uncertainty, including how imputation might affect the final outcome.
        
        ANALYSIS DATA:
        - Feature impacts: {json.dumps(feature_impacts, indent=2)}
        - Missing feature counts: {json.dumps(dict(list(missing_counts.items())[:10]), indent=2)}
        - High missing features: {json.dumps(high_missing, indent=2)}
        - Features with zeros treated as missing: {json.dumps(zero_as_missing, indent=2)}
        - Imputation impact: {json.dumps(imputation_impact, indent=2)}
        
        Please provide a comprehensive, human-readable explanation suitable for medical professionals to understand the prediction. Use medical terminology appropriately but ensure the explanation is understandable to patients and families as well.
        """
        
        # Get response from LLM
        try:
            from langchain.schema import HumanMessage, SystemMessage
            # Using langchain messages style for compatibility with multiple LLM versions
            messages = [
                SystemMessage(content="You are a helpful AI assistant that explains medical predictions."),
                HumanMessage(content=prompt)
            ]
            response = self.llm.predict_messages(messages)
            return response.content
        except Exception as e:
            print(f"Error in LLM prediction: {str(e)}")
            try:
                # Alternative direct call method for newer LLM versions
                response = self.llm.invoke(prompt)
                if hasattr(response, 'content'):
                    return response.content
                return str(response)
            except:
                # As a last resort, just return the prompt
                print("Failed to get LLM response, using template explanation")
                return self._generate_template_explanation(explanation_data)
    
    def _generate_template_explanation(self, explanation_data: Dict[str, Any]) -> str:
        """
        Generate explanation using a template-based approach without LLM.
        
        Args:
            explanation_data: Dictionary with explanation components
            
        Returns:
            Template-based explanation text
        """
        # Extract key components
        patient_id = explanation_data.get("patient_id", "Unknown")
        
        # Get prediction information
        prediction = explanation_data.get("prediction", {})
        mortality_prob = prediction.get("mortality_probability", 0)
        risk_level = prediction.get("risk_level", "Unknown")
        
        # Get feature impacts
        feature_impacts = []
        positive_impacts = []
        negative_impacts = []
        
        if "shap_values" in explanation_data and explanation_data["shap_values"]:
            # Use SHAP values
            shap_values = explanation_data["shap_values"]
            
            # Separate positive and negative impacts
            for feature in shap_values:
                if feature["impact"] == "positive":
                    positive_impacts.append(feature)
                else:
                    negative_impacts.append(feature)
                    
            # Sort by magnitude
            positive_impacts = sorted(positive_impacts, key=lambda x: abs(x["shap_value"]), reverse=True)
            negative_impacts = sorted(negative_impacts, key=lambda x: abs(x["shap_value"]), reverse=True)
        
        # Get missing data information
        missing_info = explanation_data.get("missing_features", {})
        high_missing = missing_info.get("high_missing_features", {})
        zero_as_missing = missing_info.get("zero_as_missing", [])
        
        # Get imputation impact
        imputation_impact = explanation_data.get("prediction_analysis", {})
        has_missing = imputation_impact.get("has_missing_values", False)
        missing_count = imputation_impact.get("missing_count", 0)
        important_missing = imputation_impact.get("important_missing", [])
        uncertainty_level = imputation_impact.get("uncertainty_level", "Unknown")
        
        # Build the explanation
        sections = []
        
        # 1. Introduction
        intro = f"# Mortality Prediction Explanation for Patient {patient_id}\n\n"
        intro += f"## Prediction Summary\n"
        intro += f"- **Risk Level**: {risk_level}\n"
        intro += f"- **90-day Mortality Risk**: {mortality_prob:.1%}\n\n"
        sections.append(intro)
        
        # 2. Feature Importance
        importance_section = "## Feature Importance Analysis\n\n"
        
        # Positive factors (increasing mortality risk)
        if positive_impacts:
            importance_section += "### Factors Increasing Mortality Risk\n"
            for i, factor in enumerate(positive_impacts[:3]):  # Top 3
                feat_name = factor["feature"].replace('_', ' ').title()
                importance_section += f"{i+1}. **{feat_name}**: "
                importance_section += f"Contributes {abs(factor['shap_value']):.4f} to increased risk"
                importance_section += f" (value: {factor['value']:.2f})\n"
            importance_section += "\n"
        
        # Negative factors (decreasing mortality risk)
        if negative_impacts:
            importance_section += "### Factors Decreasing Mortality Risk\n"
            for i, factor in enumerate(negative_impacts[:3]):  # Top 3
                feat_name = factor["feature"].replace('_', ' ').title()
                importance_section += f"{i+1}. **{feat_name}**: "
                importance_section += f"Contributes {abs(factor['shap_value']):.4f} to decreased risk"
                importance_section += f" (value: {factor['value']:.2f})\n"
            importance_section += "\n"
        
        sections.append(importance_section)
        
        # 3. Missing Data Analysis
        missing_section = "## Missing Data Analysis\n\n"
        
        if has_missing:
            missing_section += f"This patient has {missing_count} features with missing values. "
            
            if important_missing:
                important_names = [f.replace('_', ' ').title() for f in important_missing]
                missing_section += f"Notably, these include important predictors: {', '.join(important_names)}. "
            
            missing_section += f"The impact of these missing values on prediction uncertainty is estimated to be **{uncertainty_level}**.\n\n"
            
            # Add more details about zeros treated as missing
            if zero_as_missing:
                zero_names = [f.replace('_', ' ').title() for f in zero_as_missing]
                missing_section += f"Additionally, zero values in these features likely represent missing data: {', '.join(zero_names)}.\n\n"
        else:
            missing_section += "This patient does not have significant missing data that would affect the prediction.\n\n"
        
        sections.append(missing_section)
        
        # 4. Imputation Impact
        imputation_section = "## Impact of Data Imputation\n\n"
        
        if has_missing:
            imputation_section += f"Missing data was imputed to make this prediction. "
            imputation_section += f"The imputation process introduces a {uncertainty_level.lower()} level of uncertainty.\n\n"
            
            if uncertainty_level == "High":
                imputation_section += "**Recommendation**: Consider collecting the missing data for more reliable prediction.\n\n"
            elif uncertainty_level == "Moderate":
                imputation_section += "**Recommendation**: The prediction is usable but interpret with caution given the missing data.\n\n"
            else:
                imputation_section += "**Recommendation**: The missing data likely has minimal impact on prediction reliability.\n\n"
        else:
            imputation_section += "No significant imputation was needed for this prediction.\n\n"
        
        sections.append(imputation_section)
        
        # 5. Conclusion
        conclusion = "## Conclusion\n\n"
        
        if mortality_prob > 0.7:
            conclusion += "This patient shows a **high risk of mortality** within 90 days. Close monitoring and aggressive intervention may be warranted.\n\n"
        elif mortality_prob > 0.3:
            conclusion += "This patient shows a **moderate risk of mortality** within 90 days. Regular monitoring and appropriate interventions should be considered.\n\n"
        else:
            conclusion += "This patient shows a **low risk of mortality** within 90 days. Standard care protocols may be appropriate.\n\n"
        
        if has_missing and uncertainty_level == "High":
            conclusion += "**Note**: Due to significant missing data, this prediction should be interpreted with caution.\n"
        
        sections.append(conclusion)
        
        # Combine all sections
        return "\n".join(sections)