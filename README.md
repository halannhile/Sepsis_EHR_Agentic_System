# Sepsis EHR Analysis AI Agent

An AI-powered agent for analyzing electronic health records (EHR) of sepsis patients, predicting mortality outcomes, and generating interpretable explanations.

## Developers

Nhi Le, Yurim Lee, Zepeng Hu (Brandeis University)

## Overview

This project implements an intelligent agent system that assists healthcare professionals in analyzing sepsis patient data. The agent can:

1. **Summarize Dataset Statistics**: Provide comprehensive statistics about the sepsis dataset
2. **Retrieve Patient Data**: Access and summarize individual patient records
3. **Detect and Impute Missing Values**: Identify missing features and intelligently impute them
4. **Predict Mortality Risk**: Estimate 90-day mortality risk for specific patients
5. **Explain Predictions**: Generate detailed, interpretable explanations for predictions

The system uses machine learning models to predict mortality and explainable AI techniques (SHAP) to generate interpretations of these predictions.

## Architecture

The system consists of several integrated components:

- **Agent Core** (`agent_langchain.py`): Main agent logic and API implementation
- **Data Summary Tool** (`data_summary_tool.py`): Dataset statistics and visualization
- **Patient Retrieval Tool** (`patient_retrieval_tool.py`): Patient data access and summarization
- **Imputation Tool** (`imputation_tool.py`): Missing value detection and imputation
- **Prediction Tool** (`prediction_tool.py`): Mortality prediction model
- **XAI Tool** (`xai_tool.py`): Explainable AI for interpreting predictions
- **Explanation Agents** (`explanation_agent.py`, `llm_explanation_agent.py`): LLM-based explanation generation
- **Frontend Interface** (`frontend_interface.py`): Web UI for interacting with the agent
- **HTML Templates** (`html_templates.py`): Templates for generating visual reports

## Requirements

- Python 3.8+
- Libraries listed in `requirements.txt`
- OpenAI API key for LLM-based explanations

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/sepsis-ehr-agent.git
   cd sepsis-ehr-agent
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the project root directory
   - Add your OpenAI API key: `OPENAI_API_KEY=your_openai_api_key`

5. Prepare data files:
   - Place `AI_agent_train_sepsis.csv` in the `data/` directory
   - Place `AI_agent_test_sepsis_features.csv` in the `data/` directory

## Execution

* Install packages: 
```
pip install -r requirements.txt
```

* Ensure you have an `OPENAI_API_KEY` in `.env` file 

* Run the app: 
```
python start_app.py 
```

## Usage

### Starting the Application

Run the application using:

```
python start_app.py
```

This will start both the backend API server and the frontend web interface.

To start only the backend:
```
python start_app.py --backend-only
```

To start only the frontend:
```
python start_app.py --frontend-only
```

### Accessing the Web Interface

Once the application is running, access the web interface at:
```
http://localhost:5005
```

### Using the Agent

The agent supports the following commands:

1. **Dataset Summary**: `Give me a summary of the dataset`
2. **Patient Information**: `Get patient summary for patient 12345`
3. **Missing Value Analysis**: `Impute missing values for patient 12345`
4. **Mortality Prediction**: `Predict mortality for patient 12345`
5. **Explanation**: `Explain the prediction for patient 12345`

## Explanation Features

The system generates detailed explanations for mortality predictions, including:

- **Mortality Risk Assessment**: Clear statement of risk level and probability
- **Key Contributing Factors**: Top factors that increase or decrease mortality risk
- **Medical Explanations**: Detailed clinical reasoning for each factor
- **Layperson Explanations**: Simplified explanations for non-medical professionals
- **Missing Data Impact**: Analysis of how missing data affects predictions
- **Imputation Analysis**: Comparison of predictions before and after imputation

## Model Information

The mortality prediction uses a machine learning model trained on sepsis EHR data. The model:

- Handles time series data for each patient
- Detects and imputes missing values
- Generates probability estimates for 90-day mortality
- Provides confidence levels for predictions
