import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pickle
import re
import time
from enum import Enum

# Import Langchain components
from langchain.agents import Tool, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish

# Import our tools
from data_summary_tool import DataSummaryTool
from patient_retrieval_tool import PatientRetrievalTool
from imputation_tool import ImputationTool
from prediction_tool import PredictionTool
from xai_tool import XAITool


import os
from dotenv import load_dotenv

# Define path to data
TRAIN_DATA_PATH = "./data/AI_agent_train_sepsis.csv"
TEST_DATA_PATH = "./data/AI_agent_test_sepsis_features.csv"
MODEL_PATH = "./model/sepsis_mortality_model.pkl"


# Load .env file
load_dotenv()

# Access the API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY environment variable not set. The agent will not work properly.")
    print("Set it using: export OPENAI_API_KEY='your-api-key'")
    # For development purposes, you could set a default key here
    # OPENAI_API_KEY = "your_key_here"

class ToolType(str, Enum):
    """Enum for available tool types"""
    DATA_SUMMARY = "data_summary"
    PATIENT_RETRIEVAL = "patient_retrieval"
    IMPUTATION = "imputation"
    PREDICTION = "prediction"
    XAI = "xai"
    NONE = "none"

class AgentRequest(BaseModel):
    """Request for the agent"""
    instruction: str
    patient_id: Optional[int] = None
    context: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    """Response from the agent"""
    message: str
    report: Optional[str] = None
    explanation: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Custom prompt template for the LangChain agent
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        intermediate_steps = kwargs.pop("intermediate_steps", [])
        
        # Format agent scratchpad
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action.tool}\nAction Input: {action.tool_input}\nObservation: {observation}\n"
        
        # Set the agent scratchpad
        kwargs["agent_scratchpad"] = thoughts
        
        # Create a list of tool names and descriptions
        tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tools"] = tool_descriptions
        
        # Create a list of tool names
        tool_names = ", ".join([tool.name for tool in self.tools])
        kwargs["tool_names"] = tool_names
        
        return self.template.format(**kwargs)
    
# Custom output parser for the LangChain agent
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if the agent wants to finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Parse the LLM output to extract the tool name and arguments
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        
        action = match.group(1).strip()
        action_input = match.group(2).strip()
        
        # Return the agent action
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)

class SepsisAgent:
    """
    Agent for sepsis EHR analysis using LangChain.
    """
    
    def __init__(self):
        """Initialize the agent and load all tools"""
        print("Initializing Sepsis Agent with LangChain...")
        
        # Initialize all tools
        try:
            self.data_summary_tool = DataSummaryTool(TRAIN_DATA_PATH)
            print("Data summary tool loaded.")
        except Exception as e:
            print(f"Error loading data summary tool: {str(e)}")
            self.data_summary_tool = None
        
        try:
            self.patient_retrieval_tool = PatientRetrievalTool(TRAIN_DATA_PATH, TEST_DATA_PATH)
            print("Patient retrieval tool loaded.")
        except Exception as e:
            print(f"Error loading patient retrieval tool: {str(e)}")
            self.patient_retrieval_tool = None
        
        try:
            self.imputation_tool = ImputationTool(TRAIN_DATA_PATH, TEST_DATA_PATH)
            print("Imputation tool loaded.")
        except Exception as e:
            print(f"Error loading imputation tool: {str(e)}")
            self.imputation_tool = None
        
        try:
            self.prediction_tool = PredictionTool(TRAIN_DATA_PATH)
            
            # Check if model file exists and load it
            if os.path.exists(MODEL_PATH):
                self.prediction_tool.load_model(MODEL_PATH)
                print(f"Prediction model loaded from {MODEL_PATH}")
            else:
                # Train and save the model
                print("Training prediction model...")
                self.prediction_tool.train_model()
                self.prediction_tool.save_model(MODEL_PATH)
                print(f"Prediction model trained and saved to {MODEL_PATH}")
        except Exception as e:
            print(f"Error with prediction tool: {str(e)}")
            self.prediction_tool = None
        
        try:
            self.xai_tool = XAITool()
            if self.prediction_tool is not None and self.prediction_tool.model is not None:
                self.xai_tool.set_model(
                    model=self.prediction_tool.model.named_steps['model'],
                    feature_names=list(self.prediction_tool.feature_importance.keys())
                )
            print("XAI tool loaded.")
        except Exception as e:
            print(f"Error loading XAI tool: {str(e)}")
            self.xai_tool = None
        
        # Keep track of the current patient being analyzed
        self.current_patient_id = None
        self.current_patient_data = None
        self.current_prediction = None
        self.current_explanation = None
        
        # Initialize LangChain components
        self._setup_langchain_agent()
        
        print("Agent initialization complete.")
    
    def _setup_langchain_agent(self):
        """Set up the LangChain agent with tools and LLM."""
        print("Setting up LangChain agent...")
        
        # Define tools
        tools = []
        
        # Data Summary Tool
        if self.data_summary_tool:
            tools.append(
                Tool(
                    name="DataSummaryTool",
                    func=self._execute_data_summary,
                    description="Use this tool to get statistics and summaries about the sepsis dataset. You can ask for basic statistics, variable distributions, or correlations with mortality."
                )
            )
            
        # Patient Retrieval Tool
        if self.patient_retrieval_tool:
            tools.append(
                Tool(
                    name="PatientRetrievalTool",
                    func=self._execute_patient_retrieval,
                    description="Use this tool to get information about specific patients. You can request full patient data, a summary, or time series data. The input should include a patient_id number."
                )
            )
            
        # Imputation Tool
        if self.imputation_tool:
            tools.append(
                Tool(
                    name="ImputationTool",
                    func=self._execute_imputation,
                    description="Use this tool to detect and impute missing values for patients. You can specify a patient_id to impute data for a specific patient."
                )
            )
            
        # Prediction Tool
        if self.prediction_tool:
            tools.append(
                Tool(
                    name="PredictionTool",
                    func=self._execute_prediction,
                    description="Use this tool to predict mortality risk for patients. You should specify a patient_id for prediction."
                )
            )
            
        # XAI Tool
        if self.xai_tool:
            tools.append(
                Tool(
                    name="ExplanationTool",
                    func=self._execute_explanation,
                    description="Use this tool to explain predictions. You should specify a patient_id to get an explanation for their prediction."
                )
            )
        
        # Skip trying to use OpenAI entirely and use rule-based agent instead
        print("Using rule-based agent to avoid OpenAI dependency issues.")
        self.agent_executor = self._create_rule_based_agent(tools)
        print("Rule-based agent setup complete.")


    def _create_rule_based_agent(self, tools):
        """
        Create a simple rule-based agent that doesn't rely on LangChain's agent framework.
        This is a fallback mechanism when OpenAI integration fails.
        """
        class RuleBasedExecutor:
            def __init__(self, tools, owner):
                self.tools = tools
                self.owner = owner
                self.tool_map = {tool.name: tool for tool in tools}
                
            def run(self, input=None):
                if not input:
                    return "Please provide a valid instruction."
                
                input_lower = input.lower()
                result = None
                
                # Try to match keywords to tools
                if any(word in input_lower for word in ["summary", "statistics", "dataset", "distribution"]):
                    if "DataSummaryTool" in self.tool_map:
                        result = self.tool_map["DataSummaryTool"].func(input)
                
                # Check for patient info request
                if any(word in input_lower for word in ["patient", "record", "data"]):
                    # Extract patient ID if present
                    patient_id_match = re.search(r"patient (?:id |#)?(\d+)", input_lower)
                    if patient_id_match and "PatientRetrievalTool" in self.tool_map:
                        result = self.tool_map["PatientRetrievalTool"].func(input)
                
                # Check for imputation request
                if any(word in input_lower for word in ["impute", "missing", "fill"]):
                    if "ImputationTool" in self.tool_map:
                        result = self.tool_map["ImputationTool"].func(input)
                
                # Check for prediction request
                if any(word in input_lower for word in ["predict", "mortality", "risk", "chance", "likelihood"]):
                    if "PredictionTool" in self.tool_map:
                        result = self.tool_map["PredictionTool"].func(input)
                
                # Check for explanation request
                if any(word in input_lower for word in ["explain", "explanation", "why", "how", "reason"]):
                    if "ExplanationTool" in self.tool_map:
                        result = self.tool_map["ExplanationTool"].func(input)
                
                # If no tool matched or execution failed, return a helpful message
                if result is None:
                    tools_available = ", ".join([f"{tool.name}" for tool in self.tools])
                    return (f"I'm not sure how to process that instruction. Available tools are: {tools_available}. "
                            f"Please try being more specific about what you're looking for.")
                
                return result
        
        return RuleBasedExecutor(tools, self)
    
    # Tool execution methods
    def _execute_data_summary(self, input_str: str) -> str:
        """Execute data summary tool based on input string."""
        try:
            # Parse input to determine what type of summary is needed
            input_lower = input_str.lower()
            
            # Prepare parameters
            params = {}
            
            # Check if a specific summary type is requested
            if "basic stats" in input_lower or "basic statistics" in input_lower:
                params["summary_type"] = "basic_stats"
            elif "distribution" in input_lower:
                params["summary_type"] = "variable_distributions"
                # Extract column names if specified
                columns_match = re.search(r"columns:?\s*\[(.*?)\]", input_lower)
                if columns_match:
                    columns_str = columns_match.group(1)
                    params["columns"] = [col.strip() for col in columns_str.split(",")]
            elif "correlation" in input_lower:
                params["summary_type"] = "correlation_with_mortality"
            
            # Check if a specific patient ID is mentioned
            patient_id_match = re.search(r"patient (?:id |#)?(\d+)", input_lower)
            if patient_id_match:
                patient_id = int(patient_id_match.group(1))
                # Filter the summary for this patient
                # Note: We'll need to modify the DataSummaryTool to handle patient-specific summaries
                params["patient_id"] = patient_id
                
                # If we have a patient ID, we might want to get a summary from PatientRetrievalTool instead
                if self.patient_retrieval_tool:
                    return self._execute_patient_retrieval(f"patient_id: {patient_id}, format: summary")
            
            # Execute the data summary tool
            if params.get("summary_type"):
                if params["summary_type"] == "basic_stats":
                    result = self.data_summary_tool.get_basic_stats()
                elif params["summary_type"] == "variable_distributions":
                    columns = params.get("columns", None)
                    result = self.data_summary_tool.get_variable_distributions(columns)
                elif params["summary_type"] == "correlation_with_mortality":
                    result = self.data_summary_tool.get_correlation_with_mortality()
                return json.dumps(result, default=str)
            else:
                # Generate comprehensive summary report
                report = self.data_summary_tool.get_summary_report()
                return report
        
        except Exception as e:
            return f"Error executing data summary tool: {str(e)}"
    
    def _execute_patient_retrieval(self, input_str: str) -> str:
        """Execute patient retrieval tool based on input string."""
        try:
            # Parse input to extract parameters
            patient_id_match = re.search(r"patient(?:_| )id:?\s*(\d+)", input_str)
            
            if not patient_id_match:
                # Try another pattern
                patient_id_match = re.search(r"patient (?:id |#)?(\d+)", input_str)
            
            if patient_id_match:
                patient_id = int(patient_id_match.group(1))
                
                # Update current patient
                self.current_patient_id = patient_id
                
                # Determine format type
                format_type = "summary"  # Default
                if "full" in input_str.lower():
                    format_type = "full"
                elif "time series" in input_str.lower() or "timeseries" in input_str.lower():
                    format_type = "time_series"
                
                # Get variables for time series if specified
                variables = ["GCS", "HR", "SysBP", "SOFA"]  # Default
                variables_match = re.search(r"variables:?\s*\[(.*?)\]", input_str.lower())
                if variables_match:
                    variables_str = variables_match.group(1)
                    variables = [var.strip() for var in variables_str.split(",")]
                
                # Execute the patient retrieval
                if format_type == "full":
                    result = self.patient_retrieval_tool.get_patient_data(patient_id)
                    return json.dumps(result, default=str)
                elif format_type == "summary":
                    summary = self.patient_retrieval_tool.get_patient_summary(patient_id)
                    return summary
                elif format_type == "time_series":
                    result = self.patient_retrieval_tool.get_patient_time_series(patient_id, variables)
                    return json.dumps(result, default=str)
            
            # Check if similar patients are requested
            elif "similar" in input_str.lower():
                ref_patient_match = re.search(r"reference(?:_| )patient(?:_| )id:?\s*(\d+)", input_str.lower())
                if ref_patient_match:
                    ref_patient_id = int(ref_patient_match.group(1))
                    n_match = re.search(r"n:?\s*(\d+)", input_str.lower())
                    n = int(n_match.group(1)) if n_match else 5
                    
                    result = self.patient_retrieval_tool.find_similar_patients(ref_patient_id, n)
                    return json.dumps(result, default=str)
            
            # If no specific parameters, return list of patient IDs
            else:
                patient_ids = self.patient_retrieval_tool.get_patient_ids()
                return f"Available patient IDs: {patient_ids[:20]}..."
        
        except Exception as e:
            return f"Error executing patient retrieval tool: {str(e)}"
    
    def _execute_imputation(self, input_str: str) -> str:
        """Execute imputation tool based on input string."""
        try:
            # Parse input to extract parameters
            patient_id_match = re.search(r"patient(?:_| )id:?\s*(\d+)", input_str)
            
            if not patient_id_match:
                # Try another pattern
                patient_id_match = re.search(r"patient (?:id |#)?(\d+)", input_str)
            
            if patient_id_match:
                patient_id = int(patient_id_match.group(1))
                
                # Determine imputation method
                method = "mice"  # Default
                if "mean" in input_str.lower():
                    method = "mean"
                elif "median" in input_str.lower():
                    method = "median"
                elif "knn" in input_str.lower():
                    method = "knn"
                
                # Execute imputation
                imputation_result = self.imputation_tool.impute_patient_data(patient_id, method)
                
                # Generate report
                if "report" in input_str.lower() or "generate_report" in input_str.lower():
                    report = self.imputation_tool.generate_imputation_report(patient_id)
                    return report
                
                return json.dumps(imputation_result, default=str)
            
            # If no patient ID specified, generate general imputation report
            else:
                report = self.imputation_tool.generate_imputation_report()
                return report
        
        except Exception as e:
            return f"Error executing imputation tool: {str(e)}"
    
    def _execute_prediction(self, input_str: str) -> str:
        """Execute prediction tool based on input string."""
        try:
            # Parse input to extract parameters
            patient_id_match = re.search(r"patient(?:_| )id:?\s*(\d+)", input_str)
            
            if not patient_id_match:
                # Try another pattern
                patient_id_match = re.search(r"patient (?:id |#)?(\d+)", input_str)
            
            if patient_id_match:
                patient_id = int(patient_id_match.group(1))
                
                # Update current patient
                self.current_patient_id = patient_id
                
                # Get patient data from PatientRetrievalTool
                if self.patient_retrieval_tool:
                    # Check if patient exists in training data
                    train_patient_exists = 'icustayid' in self.patient_retrieval_tool.train_df.columns and patient_id in self.patient_retrieval_tool.train_df['icustayid'].values
                    
                    # Check if patient exists in test data
                    test_patient_exists = (self.patient_retrieval_tool.test_df is not None and 
                                        'icustayid' in self.patient_retrieval_tool.test_df.columns and 
                                        patient_id in self.patient_retrieval_tool.test_df['icustayid'].values)
                    
                    if not train_patient_exists and not test_patient_exists:
                        return f"Patient ID {patient_id} not found in the dataset."
                    
                    # Get patient data
                    if train_patient_exists:
                        patient_df = self.patient_retrieval_tool.train_df[self.patient_retrieval_tool.train_df['icustayid'] == patient_id].copy()
                    else:
                        patient_df = self.patient_retrieval_tool.test_df[self.patient_retrieval_tool.test_df['icustayid'] == patient_id].copy()
                    
                    self.current_patient_data = patient_df
                    
                    # Make prediction
                    prediction = self.prediction_tool.predict_mortality(patient_df)
                    self.current_prediction = prediction
                    
                    # Generate report
                    if "report" in input_str.lower():
                        report = self.prediction_tool.generate_patient_report(patient_df)
                        return report
                    
                    # Check if explanation is requested
                    if "explain" in input_str.lower() and self.xai_tool is not None:
                        explanation = self.xai_tool.generate_explanation(patient_df, prediction)
                        self.current_explanation = explanation
                        return json.dumps({"prediction": prediction, "explanation": explanation["summary"]}, default=str)
                    
                    # Format output
                    mortality_prob = prediction["mortality_probability"]
                    confidence = prediction["confidence"]
                    
                    # Determine risk level
                    if mortality_prob < 0.25:
                        risk_level = "low"
                    elif mortality_prob < 0.5:
                        risk_level = "moderate"
                    elif mortality_prob < 0.75:
                        risk_level = "high"
                    else:
                        risk_level = "very high"
                    
                    return f"The patient {patient_id} has a {risk_level} risk of 90-day mortality with a probability of {mortality_prob:.1%}."
            
            # If no patient ID specified, return model information
            else:
                model_card = self.prediction_tool.get_model_card()
                return model_card
        
        except Exception as e:
            return f"Error executing prediction tool: {str(e)}"
    
    def _execute_explanation(self, input_str: str) -> str:
        """Execute XAI tool based on input string."""
        try:
            # Parse input to extract parameters
            patient_id_match = re.search(r"patient(?:_| )id:?\s*(\d+)", input_str)
            
            if not patient_id_match:
                # Try another pattern
                patient_id_match = re.search(r"patient (?:id |#)?(\d+)", input_str)
            
            if patient_id_match:
                patient_id = int(patient_id_match.group(1))
                
                # Update current patient
                self.current_patient_id = patient_id
                
                # Check if simplified explanation is requested
                simplified_requested = "simple" in input_str.lower() or "simplified" in input_str.lower()
                
                # Check if plots should be skipped (to avoid timeout issues)
                skip_plots = "no plots" in input_str.lower() or "text only" in input_str.lower()
                
                # Check if we already have a prediction for this patient
                if self.current_patient_id == patient_id and self.current_prediction is not None and self.current_patient_data is not None:
                    # Generate explanation
                    explanation = self.xai_tool.generate_explanation(
                        self.current_patient_data, 
                        self.current_prediction,
                        skip_shap_plots=skip_plots
                    )
                    self.current_explanation = explanation
                    
                    # Check if simplified explanation is requested
                    if simplified_requested:
                        simplified_explanation = self.xai_tool.generate_simplified_explanation(explanation)
                        return simplified_explanation
                    
                    return explanation["summary"]
                
                # Otherwise, get patient data and make prediction first
                else:
                    # Get patient data from PatientRetrievalTool
                    if self.patient_retrieval_tool:
                        # Check if patient exists in training data
                        train_patient_exists = 'icustayid' in self.patient_retrieval_tool.train_df.columns and patient_id in self.patient_retrieval_tool.train_df['icustayid'].values
                        
                        # Check if patient exists in test data
                        test_patient_exists = (self.patient_retrieval_tool.test_df is not None and 
                                            'icustayid' in self.patient_retrieval_tool.test_df.columns and 
                                            patient_id in self.patient_retrieval_tool.test_df['icustayid'].values)
                        
                        if not train_patient_exists and not test_patient_exists:
                            return f"Patient ID {patient_id} not found in the dataset."
                        
                        # Get patient data
                        if train_patient_exists:
                            patient_df = self.patient_retrieval_tool.train_df[self.patient_retrieval_tool.train_df['icustayid'] == patient_id].copy()
                        else:
                            patient_df = self.patient_retrieval_tool.test_df[self.patient_retrieval_tool.test_df['icustayid'] == patient_id].copy()
                        
                        self.current_patient_data = patient_df
                        
                        # Make prediction
                        prediction = self.prediction_tool.predict_mortality(patient_df)
                        self.current_prediction = prediction
                        
                        # Generate explanation
                        explanation = self.xai_tool.generate_explanation(
                            patient_df, 
                            prediction,
                            skip_shap_plots=skip_plots
                        )
                        self.current_explanation = explanation
                        
                        # Check if simplified explanation is requested
                        if simplified_requested:
                            simplified_explanation = self.xai_tool.generate_simplified_explanation(explanation)
                            return simplified_explanation
                        
                        return explanation["summary"]
            
            # If no specific parameters, generate general explanation
            else:
                if self.current_patient_id is not None and self.current_prediction is not None and self.current_patient_data is not None:
                    # Generate explanation for current patient
                    explanation = self.xai_tool.generate_explanation(
                        self.current_patient_data, 
                        self.current_prediction,
                        skip_shap_plots=True  # Skip plots for general explanation
                    )
                    self.current_explanation = explanation
                    return explanation["summary"]
                else:
                    return "No patient is currently selected. Please specify a patient ID to generate an explanation."
        
        except Exception as e:
            return f"Error executing explanation tool: {str(e)}"
    
    def _get_patient_data(self, patient_id: int) -> Optional[pd.DataFrame]:
        """
        Get data for a specific patient.
        
        Args:
            patient_id: Patient ID
            
        Returns:
            DataFrame with patient data or None if not found
        """
        # Make sure patient retrieval tool is available
        if self.patient_retrieval_tool is None:
            return None
        
        # Check if patient exists
        if not self.patient_retrieval_tool.get_patient_ids() or patient_id not in self.patient_retrieval_tool.get_patient_ids():
            return None
        
        # Get patient data
        patient_df = self.patient_retrieval_tool.train_df[self.patient_retrieval_tool.train_df['icustayid'] == patient_id].copy()
        
        # If not found in training data, try test data if available
        if len(patient_df) == 0 and hasattr(self.patient_retrieval_tool, 'test_df') and self.patient_retrieval_tool.test_df is not None:
            patient_df = self.patient_retrieval_tool.test_df[self.patient_retrieval_tool.test_df['icustayid'] == patient_id].copy()
        
        return patient_df if len(patient_df) > 0 else None
    
    def process_instruction(self, instruction: str, patient_id: Optional[int] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user instruction"""
        try:
            # Update current patient if provided
            if patient_id is not None:
                self.current_patient_id = patient_id
                self.current_patient_data = self._get_patient_data(patient_id)
                
                # Add patient ID to instruction if not present
                if not re.search(r"patient (?:id |#)?(\d+)", instruction.lower()):
                    instruction = f"{instruction} for patient {patient_id}"
            
            # Use the agent executor with robust error handling
            try:
                # Direct execution without relying on run method
                if hasattr(self.agent_executor, 'run'):
                    result = self.agent_executor.run(input=instruction)
                else:
                    # For our rule-based fallback agent
                    result = self.agent_executor(instruction)
            except Exception as e:
                print(f"Error executing agent: {str(e)}")
                # Fallback to using the rule-based approach directly
                if hasattr(self, '_create_rule_based_agent'):
                    fallback_agent = self._create_rule_based_agent(self.agent_executor.tools if hasattr(self.agent_executor, 'tools') else [])
                    result = fallback_agent.run(input=instruction)
                else:
                    result = f"Error processing instruction: {str(e)}"
            
            # Parse the result
            response = self._parse_agent_result(result, instruction)
            
            return response
        except Exception as e:
            print(f"General error in process_instruction: {str(e)}")
            return {
                "message": f"Error processing instruction: {str(e)}",
                "error": str(e)
            }
    
    def _parse_agent_result(self, result: str, instruction: str) -> Dict[str, Any]:
        """
        Parse the agent result to create a structured response.
        
        Args:
            result: Result from agent execution
            instruction: Original instruction
            
        Returns:
            Structured response dictionary
        """
        response = {"message": result}
        
        # Check if result contains a report or explanation
        if "##" in result or "#" in result:
            # Likely a markdown report
            response["report"] = result
        
        # Check if the result contains prediction information
        if "risk of mortality" in result.lower():
            # Extract probability if present
            prob_match = re.search(r"probability of (\d+\.\d+%)", result)
            if prob_match:
                probability = float(prob_match.group(1).rstrip("%")) / 100
                risk_level = "low"
                if probability >= 0.25 and probability < 0.5:
                    risk_level = "moderate"
                elif probability >= 0.5 and probability < 0.75:
                    risk_level = "high"
                else:
                    risk_level = "very high"
                
                response["data"] = {
                    "prediction": {
                        "mortality_probability": probability,
                        "risk_level": risk_level
                    }
                }
        
        # Include current patient ID if available
        if self.current_patient_id is not None:
            if "data" not in response:
                response["data"] = {}
            response["data"]["patient_id"] = self.current_patient_id
        
        return response

# FastAPI app for serving the agent
app = FastAPI(title="Sepsis EHR AI Agent API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
agent = None

@app.on_event("startup")
async def startup_event():
    global agent
    agent = SepsisAgent()

@app.post("/process_instruction", response_model=AgentResponse)
async def process_instruction(request: AgentRequest):
    global agent
    
    if agent is None:
        agent = SepsisAgent()
    
    response = agent.process_instruction(
        instruction=request.instruction,
        patient_id=request.patient_id,
        context=request.context
    )
    
    return response

@app.get("/patient_ids", response_class=JSONResponse)
async def get_patient_ids():
    global agent
    
    if agent is None:
        agent = SepsisAgent()
    
    if agent.patient_retrieval_tool is None:
        return {"error": "Patient retrieval tool not available"}
    
    patient_ids = agent.patient_retrieval_tool.get_patient_ids()
    return {"patient_ids": patient_ids}

@app.get("/html_report/{patient_id}", response_class=HTMLResponse)
async def get_html_report(patient_id: int):
    global agent
    
    if agent is None:
        agent = SepsisAgent()
    
    if agent.xai_tool is None or agent.prediction_tool is None:
        return "<html><body><h1>Error</h1><p>XAI tool or prediction tool not available</p></body></html>"
    
    # Get patient data
    patient_df = agent._get_patient_data(patient_id)
    
    if patient_df is None:
        return "<html><body><h1>Error</h1><p>Patient not found</p></body></html>"
    
    # Make prediction
    prediction = agent.prediction_tool.predict_mortality(patient_df)
    
    # Generate explanation
    explanation = agent.xai_tool.generate_explanation(patient_df, prediction)
    
    # Generate HTML report
    html_report = agent.xai_tool.generate_html_report(explanation)
    
    return html_report

# Run app if main
if __name__ == "__main__":
    uvicorn.run("agent_langchain:app", host="0.0.0.0", port=8000) # reload=True