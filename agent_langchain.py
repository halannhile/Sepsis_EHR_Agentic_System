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
from langchain.agents import Tool, AgentOutputParser, LLMSingleActionAgent, AgentExecutor
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
        
        # Check if OpenAI API key is available
        if OPENAI_API_KEY:
            try:
                # Try newer LangChain version first
                try:
                    from langchain_openai import ChatOpenAI
                    from langchain.agents import AgentType, initialize_agent
                    from langchain.prompts import PromptTemplate
                    
                    # Define a custom prompt that instructs the agent to output tool results directly
                    custom_prefix = """You are an AI assistant for Sepsis EHR Analysis. Your job is to decide which tool to use based on the user's question.
                    
                    IMPORTANT: You must ONLY decide which tool to use. DO NOT summarize or rephrase the tool's output.
                    Simply execute the appropriate tool and return its exact output without modification.

                    IMPORTANT: when you are requested to provide a prediction, ONLY use the PredictionTool and return its results immediately. DO NOT call the ExplanationTool.                    
                    You have access to the following tools:
                    - DataSummaryTool: to generate a summary of the entire dataset 
                    - PatientRetrievalTool: to generate a summary of a retrieved patient
                    - ImputationTool: to impute missing values for a retrieved patient
                    - PredictionTool: to predict the 90-day mortality risk for a retrieved patient
                    - ExplanationTool: to generate explanations for the mortality prediction of the retrieved patient
                    """
                    
                    custom_format_instructions = """Use the following format:
                    
                    Question: the input question you must answer
                    Action: the action to take, should be one of [{tool_names}]
                    Action Input: the input to the action
                    Observation: the result of the action
                    Final Answer: the original, unmodified output from the tool
                    """
                    
                    # Initialize LLM
                    llm = ChatOpenAI(
                        api_key=OPENAI_API_KEY,
                        temperature=0,
                        model="gpt-3.5-turbo"
                    )
                    
                    # Create agent with custom template
                    agent_executor = initialize_agent(
                        tools,
                        llm,
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        verbose=True,
                        handle_parsing_errors=True,
                        max_iterations=1,  # Limit to one tool call
                        early_stopping_method="force",  # Force early stopping after one tool call
                        prefix=custom_prefix,
                        format_instructions=custom_format_instructions,
                        return_intermediate_steps=True  # This will help with debugging
                    )
                    
                    self.agent_executor = agent_executor
                    print("LangChain agent with OpenAI setup complete.")
                    return
                    
                except ImportError:
                    # Try older LangChain version
                    from langchain.chat_models import ChatOpenAI
                    from langchain.agents import AgentType, initialize_agent
                    
                    # Define a custom prompt that instructs the agent to output tool results directly
                    custom_prefix = """You are an AI assistant for Sepsis EHR Analysis. Your job is to decide which tool to use based on the user's question.
                    
                    IMPORTANT: You must ONLY decide which tool to use. DO NOT summarize or rephrase the tool's output.
                    Simply execute the appropriate tool and return its exact output without modification.
                    
                    IMPORTANT: when you are requested to provide a prediction, ONLY use the PredictionTool and return its results immediately. DO NOT call the ExplanationTool.                    

                    You have access to the following tools:
                    - DataSummaryTool: to generate a summary of the entire dataset 
                    - PatientRetrievalTool: to generate a summary of a retrieved patient
                    - ImputationTool: to impute missing values for a retrieved patient
                    - PredictionTool: to predict the 90-day mortality risk for a retrieved patient
                    - ExplanationTool: to generate explanations for the mortality prediction of the retrieved patient
                    """
                    
                    custom_format_instructions = """Use the following format:
                    
                    Question: the input question you must answer
                    Action: the action to take, should be one of [{tool_names}]
                    Action Input: the input to the action
                    Observation: the result of the action
                    Final Answer: the original, unmodified output from the tool
                    """
                    
                    # Initialize LLM
                    llm = ChatOpenAI(
                        openai_api_key=OPENAI_API_KEY,
                        temperature=0,
                        model_name="gpt-3.5-turbo"
                    )
                    
                    # Create agent with custom template
                    agent_kwargs = {
                        "prefix": custom_prefix,
                        "format_instructions": custom_format_instructions,
                        "handle_parsing_errors": True,
                        "early_stopping_method": "force",  # Forces the agent to stop after one tool use if possible
                    }

                    agent_executor = initialize_agent(
                        tools,
                        llm,
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        verbose=True,
                        agent_kwargs=agent_kwargs,
                        return_intermediate_steps=True
                    )
                    
                    self.agent_executor = agent_executor
                    print("LangChain agent with OpenAI setup complete (using older LangChain version).")
                    return
                    
            except Exception as e:
                print(f"Error setting up OpenAI agent: {str(e)}")
                print("Falling back to rule-based agent.")
        else:
            print("OpenAI API key not available. Using rule-based agent.")
        
        # If we get here, use rule-based agent
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
            # Check for parameter-style input (patient_id=NUMBER)
            param_match = re.search(r"patient_id=(\d+\.?\d*)", input_str)
            if param_match:
                patient_id = float(param_match.group(1))
                
                # Check if the patient exists
                patient_ids = self.patient_retrieval_tool.get_patient_ids()
                if patient_id not in patient_ids:
                    return f"Patient ID {patient_id} not found in the dataset. Available patient IDs: {patient_ids[:20]}..."
                
                # Update current patient
                self.current_patient_id = patient_id
                
                # Get patient summary
                summary = self.patient_retrieval_tool.get_patient_summary(patient_id)
                return summary
                
            # Check for simple patient ID first (pure number)
            if input_str.strip().isdigit() or input_str.strip().replace('.', '', 1).isdigit():
                patient_id = float(input_str.strip())
                
                # Check if the patient exists
                patient_ids = self.patient_retrieval_tool.get_patient_ids()
                if patient_id not in patient_ids:
                    return f"Patient ID {patient_id} not found in the dataset. Available patient IDs: {patient_ids[:20]}..."
                
                # Update current patient
                self.current_patient_id = patient_id
                
                # Get patient summary
                summary = self.patient_retrieval_tool.get_patient_summary(patient_id)
                return summary
            
            # Parse input to extract parameters
            patient_id_match = re.search(r"patient(?:_| )id:?\s*(\d+\.?\d*)", input_str)
            
            if not patient_id_match:
                # Try another pattern
                patient_id_match = re.search(r"patient (?:id |#)?(\d+\.?\d*)", input_str)
            
            if patient_id_match:
                patient_id = float(patient_id_match.group(1))
                
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
            
            # Check if summary is requested for the CURRENT patient (no ID specified)
            elif ("summary" in input_str.lower() or "get patient" in input_str.lower()) and self.current_patient_id is not None:
                patient_id = self.current_patient_id
                summary = self.patient_retrieval_tool.get_patient_summary(patient_id)
                return summary
            
            # Check if similar patients are requested
            elif "similar" in input_str.lower():
                ref_patient_match = re.search(r"reference(?:_| )patient(?:_| )id:?\s*(\d+\.?\d*)", input_str.lower())
                if ref_patient_match:
                    ref_patient_id = float(ref_patient_match.group(1))
                    n_match = re.search(r"n:?\s*(\d+)", input_str.lower())
                    n = int(n_match.group(1)) if n_match else 5
                    
                    result = self.patient_retrieval_tool.find_similar_patients(ref_patient_id, n)
                    return json.dumps(result, default=str)
            
            # If no specific parameters, return list of patient IDs
            else:
                patient_ids = self.patient_retrieval_tool.get_patient_ids()
                
                # If we have a current patient but no specific request, get summary
                if self.current_patient_id is not None:
                    summary = self.patient_retrieval_tool.get_patient_summary(self.current_patient_id)
                    return summary
                else:
                    return f"Available patient IDs: {patient_ids[:20]}..."
        
        except Exception as e:
            return f"Error executing patient retrieval tool: {str(e)}"
    
    def _execute_imputation(self, input_str: str) -> str:
        """Execute imputation tool based on input string."""
        try:
            # Parse input to extract patient ID
            patient_id_match = re.search(r"patient(?:_| )id:?\s*(\d+)", input_str)
            
            if not patient_id_match:
                # Try another pattern
                patient_id_match = re.search(r"patient (?:id |#)?(\d+)", input_str)
            
            # Also check if input is just a number
            if not patient_id_match and input_str.strip().isdigit():
                patient_id = int(input_str.strip())
            elif patient_id_match:
                patient_id = int(patient_id_match.group(1))
            else:
                return "Please provide a valid patient ID for imputation."
            
            # Update current patient
            self.current_patient_id = patient_id
            
            # Determine imputation method
            method = "mean"  # Use mean as default for faster processing
            if "mice" in input_str.lower():
                method = "mice"
            elif "median" in input_str.lower():
                method = "median"
            elif "knn" in input_str.lower():
                method = "knn"
                
            # Execute imputation
            print(f"Starting imputation for patient {patient_id} using {method} method...")
            imputation_result = self.imputation_tool.impute_patient_data(patient_id, method)
            print(f"Imputation completed for patient {patient_id}")
            
            # Check if the imputation was completed successfully
            if imputation_result.get("status") == "imputation_completed":
                # Format the results into a readable report
                report = []
                report.append(f"# Imputation Report for Patient {patient_id}")
                
                # Add imputation method
                report.append("## Imputation Method")
                report.append(f"Method used: {method.upper()} {self._get_method_description(method)}")
                
                # Add missing variables section - each on its own line
                missing_cols = imputation_result.get("missing_columns", {})
                
                report.append("## Missing Variables")
                if missing_cols:
                    for col, count in missing_cols.items():
                        report.append(f"- {col}: {count} missing values")
                else:
                    report.append("No missing values were detected for this patient.")
                
                # Imputation details - show values after imputation
                report.append("## Imputation Details")
                
                # Extract imputation results
                imputation_results = imputation_result.get("imputation_results", [])
                imputation_stats = imputation_result.get("imputation_stats", {})
                
                # Count total imputed values
                total_imputed = 0
                all_imputed_vars = set()
                
                for record in imputation_results:
                    imputed_values = record.get("imputed", {})
                    total_imputed += len(imputed_values)
                    for var_name in imputed_values.keys():
                        all_imputed_vars.add(var_name)
                
                # Organize imputed values by variable
                variable_imputed_values = {}
                for record_idx, record in enumerate(imputation_results):
                    imputed_values = record.get("imputed", {})
                    for var_name, value in imputed_values.items():
                        if var_name not in variable_imputed_values:
                            variable_imputed_values[var_name] = []
                        
                        # Get info about how it was calculated
                        var_stats = imputation_stats.get(var_name, {})
                        imp_method = var_stats.get("method", method)
                        imp_value = var_stats.get("value")
                        
                        # Create description based on method
                        if imp_method in ["mean", "median"] and imp_value is not None:
                            method_desc = f"using {imp_method} value: {imp_value:.2f}"
                        elif imp_method == "knn":
                            method_desc = "using K-Nearest Neighbors"
                        elif imp_method == "mice":
                            method_desc = "using MICE"
                        else:
                            method_desc = f"using {imp_method}"
                        
                        variable_imputed_values[var_name].append({
                            "record_idx": record_idx,
                            "value": value,
                            "method_desc": method_desc
                        })
                
                # List all variables with their imputed values
                if variable_imputed_values:
                    report.append(f"Total imputed values: {total_imputed}")
                    for var_name, values in sorted(variable_imputed_values.items()):
                        report.append(f"### {var_name}")
                        for value_info in values:
                            report.append(f"- Record {value_info['record_idx']+1}: {value_info['value']:.2f} (imputed {value_info['method_desc']})")
                else:
                    report.append("No values were imputed.")
                
                return "\n".join(report)
            
            elif imputation_result.get("status") == "no_imputation_needed":
                return f"No missing values to impute for patient {patient_id}."
            else:
                # If there was an error or unexpected result format
                if "error" in imputation_result:
                    return f"Error during imputation: {imputation_result['error']}"
                else:
                    return f"Imputation could not be completed. Result: {json.dumps(imputation_result, default=str)}"
        
        except Exception as e:
            print(f"Detailed error in imputation: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error executing imputation tool: {str(e)}"

    def _get_method_description(self, method: str) -> str:
        """Get a description of the imputation method."""
        if method == "mean":
            return "Mean imputation replaces missing values with the average value across all patients."
        elif method == "median":
            return "Median imputation replaces missing values with the median value across all patients."
        elif method == "knn":
            return "KNN imputation uses similar patients to predict missing values."
        elif method == "mice":
            return "MICE uses relationships between variables to predict missing values."
        else:
            return f"{method} imputation"
    
    def _execute_prediction(self, input_str: str) -> str:
        """Execute prediction tool based on input string."""
        try:
            # Parse input to extract patient ID
            patient_id_match = re.search(r"patient(?:_| )id:?\s*(\d+)", input_str)
            
            if not patient_id_match:
                # Try another pattern
                patient_id_match = re.search(r"patient (?:id |#)?(\d+)", input_str)
            
            # Also check if input is just a number
            if not patient_id_match and input_str.strip().isdigit():
                patient_id = int(input_str.strip())
            elif patient_id_match:
                patient_id = int(patient_id_match.group(1))
            else:
                return "Please provide a valid patient ID for prediction."
            
            # Update current patient
            self.current_patient_id = patient_id
            
            # Get patient data from PatientRetrievalTool
            if self.patient_retrieval_tool:
                # Check if patient exists in either dataset
                patient_df = None
                
                # Check training dataset
                if 'icustayid' in self.patient_retrieval_tool.train_df.columns:
                    train_patient_exists = patient_id in self.patient_retrieval_tool.train_df['icustayid'].values
                    if train_patient_exists:
                        patient_df = self.patient_retrieval_tool.train_df[
                            self.patient_retrieval_tool.train_df['icustayid'] == patient_id
                        ].copy()
                
                # Check test dataset if not found in training
                if patient_df is None and self.patient_retrieval_tool.test_df is not None:
                    if 'icustayid' in self.patient_retrieval_tool.test_df.columns:
                        test_patient_exists = patient_id in self.patient_retrieval_tool.test_df['icustayid'].values
                        if test_patient_exists:
                            patient_df = self.patient_retrieval_tool.test_df[
                                self.patient_retrieval_tool.test_df['icustayid'] == patient_id
                            ].copy()
                
                if patient_df is None:
                    return f"Patient ID {patient_id} not found in the dataset."
                
                self.current_patient_data = patient_df
                
                # IMPORTANT: Make actual prediction using the prediction tool
                if self.prediction_tool and self.prediction_tool.model is not None:
                    # Make prediction
                    prediction = self.prediction_tool.predict_mortality(patient_df)
                    self.current_prediction = prediction
                    
                    # Extract key information
                    mortality_prob = prediction["mortality_probability"]
                    risk_level = prediction["risk_level"]
                    confidence = prediction.get("confidence", 0.8)  # Default if not available
                    
                    # Generate report if requested
                    if "report" in input_str.lower() or "detailed" in input_str.lower():
                        report = self.prediction_tool.generate_patient_report(patient_df)
                        return report
                    
                    # Format a clear response with the prediction results
                    response = f"""# Mortality Prediction for Patient {patient_id}"""
                    response += "\n## Summary"
                    response += f"\n**Risk Level**: {risk_level}\n"             
                    response += f"\n**90-day Mortality Risk**: {mortality_prob:.1%}\n"

                    # # Add key contributing factors if available
                    # if "contributing_features" in prediction and prediction["contributing_features"]:
                    #     response += "\n## Key Contributing Factors\n"
                    #     for i, feature in enumerate(prediction["contributing_features"][:3]):
                    #         feature_name = feature["feature"].replace('_', ' ').title()
                    #         value = feature["value"]
                    #         response += f"{i+1}. {feature_name}: {value:.2f}\n"
                    
                    # Add recommendation for detailed report
                    response += "\nFor a detailed explanation, you can request a full report by asking for 'explain prediction for patient " + str(patient_id) + "'."
                    
                    return response
                else:
                    return "Prediction model is not available. Please ensure the model has been trained or loaded correctly."
            else:
                return "Patient retrieval tool is not available. Cannot get patient data for prediction."
        
        except Exception as e:
            return f"Error executing prediction tool: {str(e)}"
    
    def _execute_explanation(self, input_str: str) -> str:
        """Execute XAI tool based on input string."""
        try:
            # Try to directly parse the input as a patient ID if it's just a number
            if input_str.strip().isdigit():
                patient_id = int(input_str.strip())
                
                # Update current patient
                self.current_patient_id = patient_id
            else:
                # Original parsing logic for more complex queries
                patient_id_match = re.search(r"patient(?:_| )id:?\s*(\d+)", input_str)
                
                if not patient_id_match:
                    # Try another pattern
                    patient_id_match = re.search(r"patient (?:id |#)?(\d+)", input_str)
                
                if patient_id_match:
                    patient_id = int(patient_id_match.group(1))
                    
                    # Update current patient
                    self.current_patient_id = patient_id
                else:
                    # If no patient ID specified, use current patient if available
                    if self.current_patient_id is not None:
                        patient_id = self.current_patient_id
                    else:
                        return "Please specify a patient ID to explain the prediction."
            
            # Check if simplified explanation is requested
            simplified_requested = "simple" in input_str.lower() or "simplified" in input_str.lower()
            
            # Check if plots should be skipped (to avoid timeout issues)
            skip_plots = "no plots" in input_str.lower() or "text only" in input_str.lower()
            
            # Get patient data and make prediction if not already done
            if (self.current_patient_id != patient_id or 
                self.current_prediction is None or 
                self.current_patient_data is None):
                
                # Get patient data
                patient_df = self._get_patient_data(patient_id)
                
                if patient_df is None:
                    return f"Patient ID {patient_id} not found in the dataset."
                
                self.current_patient_data = patient_df
                
                # Make prediction
                prediction = self.prediction_tool.predict_mortality(patient_df)
                self.current_prediction = prediction
            else:
                # Use cached data and prediction
                patient_df = self.current_patient_data
                prediction = self.current_prediction
            
            # Generate explanation
            explanation = self.xai_tool.generate_explanation(
                patient_df, 
                prediction,
                skip_shap_plots=skip_plots
            )
            self.current_explanation = explanation
            
            # Use Explanation Agent to reason about the explanation
            from explanation_agent import ExplanationAgent
            explanation_agent = ExplanationAgent()
            
            # Add the prediction to the explanation data for better reasoning
            explanation_data = explanation.copy()
            explanation_data["prediction"] = prediction
            
            # Get the reasoned explanation
            reasoned_explanation = explanation_agent.analyze_explanation(explanation_data)
            
            # Check if simplified explanation is requested
            if simplified_requested:
                simplified_explanation = self.xai_tool.generate_simplified_explanation(explanation)
                return simplified_explanation
            
            return reasoned_explanation
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error executing explanation tool: {str(e)}"
    
    def _get_patient_data(self, patient_id: int) -> Optional[pd.DataFrame]:
        """
        Get data for a specific patient from either training or test dataset.
        
        Args:
            patient_id: Patient ID
            
        Returns:
            DataFrame with patient data or None if not found
        """
        # Check if patient retrieval tool is available
        if self.patient_retrieval_tool is None:
            return None
        
        # Check if patient exists in training data
        train_patient_exists = ('icustayid' in self.patient_retrieval_tool.train_df.columns and 
                            patient_id in self.patient_retrieval_tool.train_df['icustayid'].values)
        
        # Check if patient exists in test data
        test_patient_exists = (self.patient_retrieval_tool.test_df is not None and 
                            'icustayid' in self.patient_retrieval_tool.test_df.columns and 
                            patient_id in self.patient_retrieval_tool.test_df['icustayid'].values)
        
        if not train_patient_exists and not test_patient_exists:
            return None
        
        # Get patient data from the appropriate dataset
        if train_patient_exists:
            patient_df = self.patient_retrieval_tool.train_df[self.patient_retrieval_tool.train_df['icustayid'] == patient_id].copy()
        else:
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
                if hasattr(self.agent_executor, 'run'):
                    # For LangChain agent
                    if hasattr(self.agent_executor, 'return_intermediate_steps') and self.agent_executor.return_intermediate_steps:
                        # For agents that return intermediate steps
                        result_with_steps = self.agent_executor(instruction)
                        
                        # Extract the last observation from the intermediate steps
                        if 'intermediate_steps' in result_with_steps:
                            steps = result_with_steps['intermediate_steps']
                            if steps:
                                # Get the last observation (tool output)
                                last_action, last_observation = steps[-1]
                                result = last_observation
                            else:
                                result = result_with_steps.get('output', str(result_with_steps))
                        else:
                            result = result_with_steps.get('output', str(result_with_steps))
                    else:
                        # Standard run method for regular agents
                        result = self.agent_executor.run(instruction)
                else:
                    # Direct execution for rule-based agents or other non-standard agents
                    result = self.agent_executor(instruction)
            except Exception as e:
                print(f"Error executing agent: {str(e)}")
                # Fallback to using the rule-based approach directly
                fallback_agent = self._create_rule_based_agent([])
                rule_based_class = fallback_agent.__class__
                
                if not isinstance(self.agent_executor, rule_based_class):
                    # If not already using rule-based agent, create a new one with the available tools
                    tools = self.agent_executor.tools if hasattr(self.agent_executor, 'tools') else []
                    fallback_agent = self._create_rule_based_agent(tools)
                    result = fallback_agent.run(input=instruction)
                else:
                    # If already using rule-based agent, try direct execution with input parameter
                    result = self.agent_executor.run(input=instruction)
            
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
        # For prediction results, return them unchanged
        if "Mortality Prediction for Patient" in result or "Risk Level" in result:
            return {"message": result, "report": result}
        
        # Otherwise, use the normal parsing logic
        response = {"message": result}
        
        # Check if result contains a report or explanation
        if "##" in result or "#" in result:
            # Likely a markdown report
            response["report"] = result
        
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
    uvicorn.run("agent_langchain:app", host="0.0.0.0", port=8008, reload=True) # reload=True