import os
import json
from typing import Dict, Any, Optional

class LLMExplanationAgent:
    """
    Agent for generating human-readable explanations of model predictions using LLMs.
    """
    
    def __init__(self):
        """Initialize the explanation agent."""
        self.llm = None
        self.initialize_llm()
    
    def initialize_llm(self):
        """Initialize the LLM for generating explanations."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            try:
                # Try newer LangChain version with OpenAI
                try:
                    from langchain_openai import ChatOpenAI
                    self.llm = ChatOpenAI(
                        api_key=api_key,
                        temperature=0,
                        model="gpt-3.5-turbo"
                    )
                    print("Initialized OpenAI ChatGPT with newer LangChain version")
                except ImportError:
                    # Try older LangChain version
                    from langchain.chat_models import ChatOpenAI
                    self.llm = ChatOpenAI(
                        openai_api_key=api_key,
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
    
    def generate_explanation(self, explanation_data: Dict[str, Any]) -> str:
        """
        Generate a comprehensive explanation using LLM.
        
        Args:
            explanation_data: Dictionary with explanation components
            
        Returns:
            Human-readable explanation
        """
        if self.llm is None:
            return explanation_data.get("summary", "LLM explanation not available.")
        
        try:
            # Extract key components
            patient_id = explanation_data.get("patient_id", "Unknown")
            mortality_prob = explanation_data.get("prediction", {}).get("mortality_probability", 0)
            risk_level = explanation_data.get("prediction", {}).get("risk_level", "Unknown")
            shap_values = explanation_data.get("shap_values", [])
            missing_features = explanation_data.get("missing_features", {})
            prediction_analysis = explanation_data.get("prediction_analysis", {})
            
            # Get pre and post imputation probabilities
            pre_imp_prob = prediction_analysis.get("pre_imputation_probability", None)
            post_imp_prob = prediction_analysis.get("post_imputation_probability", None)
            prob_diff = prediction_analysis.get("probability_difference", None)
            
            # Sort SHAP values to get top positive and negative factors
            positive_factors = [s for s in shap_values if s.get("shap_value", 0) > 0]
            negative_factors = [s for s in shap_values if s.get("shap_value", 0) < 0]
            
            # Sort by absolute SHAP value
            positive_factors = sorted(positive_factors, key=lambda x: abs(x.get("shap_value", 0)), reverse=True)
            negative_factors = sorted(negative_factors, key=lambda x: abs(x.get("shap_value", 0)), reverse=True)
            
            # Select top 3 of each (or fewer if not available)
            top_pos = positive_factors[:3]
            top_neg = negative_factors[:3]
            
            # Build the prompt for the LLM with a more structured format
            prompt = f"""
            You are a medical AI assistant that explains sepsis mortality predictions to medical professionals.
            You need to explain a prediction for Patient ID {patient_id}, who has a mortality probability of {mortality_prob:.1%} and risk level of {risk_level}.
            
            The model uses SHAP values to determine how each factor increases or decreases the predicted probability of death.
            
            MUST INCLUDE EXACTLY THE FOLLOWING SECTIONS WITH EXACTLY 3 FACTORS FOR EACH CATEGORY (or fewer only if less than 3 are available):
            
            TOP 3 POSITIVE FACTORS (INCREASING RISK):
            {json.dumps([{"feature": f.get("feature", ""), "value": f.get("value", ""), "shap_value": f.get("shap_value", 0)} for f in top_pos], indent=2)}
            
            TOP 3 NEGATIVE FACTORS (DECREASING RISK):
            {json.dumps([{"feature": f.get("feature", ""), "value": f.get("value", ""), "shap_value": f.get("shap_value", 0)} for f in top_neg], indent=2)}
            
            MISSING DATA:
            {json.dumps(missing_features, indent=2)}
            
            IMPUTATION IMPACT:
            Pre-imputation probability: {pre_imp_prob if pre_imp_prob is not None else 'Not available'}
            Post-imputation probability: {post_imp_prob if post_imp_prob is not None else 'Not available'}
            Probability difference: {prob_diff if prob_diff is not None else 'Not available'}
            
            Structure your response using this exact format:
            
            # Mortality Risk:
            [Overview of prediction in 2-3 sentences]

            # Model's Predictions Before vs. After Imputation
            [Summarize model's predictions (probability, risk level) before and after imputation]

            # Key Factors Influencing Risk (and their Impact):
            
            ## Top 3 Factors Increasing Risk of Death:
            **[Feature Name 1]**: (SHAP Value: [value], Positive Impact) [Brief description]
            - Medical Explanation: [Detailed clinical reasoning]
            - Layperson Explanation: [Simple explanation]
            
            **[Feature Name 2]**: (SHAP Value: [value], Positive Impact) [Brief description]
            - Medical Explanation: [Detailed clinical reasoning]
            - Layperson Explanation: [Simple explanation]
            
            **[Feature Name 3]**: (SHAP Value: [value], Positive Impact) [Brief description]
            - Medical Explanation: [Detailed clinical reasoning]
            - Layperson Explanation: [Simple explanation]
            
            ## Top 3 Factors Decreasing Risk of Death:
            [Feature Name 1]: (SHAP Value: [value], Negative Impact) [Brief description]
            - Medical Explanation: [Detailed clinical reasoning]
            - Layperson Explanation: [Simple explanation]
            
            [Feature Name 2]: (SHAP Value: [value], Negative Impact) [Brief description]
            - Medical Explanation: [Detailed clinical reasoning]
            - Layperson Explanation: [Simple explanation]
            
            [Feature Name 3]: (SHAP Value: [value], Negative Impact) [Brief description]
            - Medical Explanation: [Detailed clinical reasoning]
            - Layperson Explanation: [Simple explanation]
            
            # Impact of Missing Data:
            [Analysis of how missing data affected the prediction]

            # Impact of Imputation:
            [Analysis of how imputation affected the prediction. Compare and contrast model's predictions before and after imputation]
            
            Include all features listed above, and make sure to include all 3 factors for both increasing and decreasing risk (or as many as are available if fewer than 3). Do not skip any factors from the lists provided.
            
            Your explanation should be medically accurate, clear, and insightful, with specific clinical reasoning for why each factor affects mortality risk.
            """
            
            # Get response from LLM
            from langchain.schema import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content="You are a helpful AI assistant that explains medical predictions."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.predict_messages(messages)
            return response.content
            
        except Exception as e:
            print(f"Error generating explanation with LLM: {str(e)}")
            # Return the summary from explanation_data as fallback
            return explanation_data.get("summary", f"Failed to generate explanation: {str(e)}")