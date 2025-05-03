import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import json
from sklearn.impute import SimpleImputer, KNNImputer
# from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class ImputationTool:
    """
    Tool for detecting and imputing missing values in the sepsis EHR dataset.
    """
    
    def __init__(self, train_data_path: str, test_data_path: Optional[str] = None):
        """
        Initialize the tool with the paths to the datasets.
        
        Args:
            train_data_path: Path to the training CSV file 
            test_data_path: Path to the test CSV file (optional)
        """
        # Load training data
        self.train_df = pd.read_csv(train_data_path)
        # Remove the index column if it exists (first unnamed column)
        if self.train_df.columns[0] == 'Unnamed: 0':
            self.train_df = self.train_df.drop(columns=self.train_df.columns[0])
        
        # Load test data if provided
        self.test_df = None
        if test_data_path:
            self.test_df = pd.read_csv(test_data_path)
            # Remove the index column if it exists
            if self.test_df.columns[0] == 'Unnamed: 0':
                self.test_df = self.test_df.drop(columns=self.test_df.columns[0])
        
        # Initialize imputation models
        self.imputers = {}
        
        # Define variables that should not be imputed
        self.non_imputable = ['bloc', 'icustayid', 'charttime', 'gender', 'mortality_90d']
        
        # Store information about zero values that likely represent missing data
        self.zero_as_missing = self._detect_zero_as_missing()
        
        # Store statistics for imputation strategies
        self.imputation_stats = {}
    
    def _detect_zero_as_missing(self) -> Dict[str, bool]:
        """
        Detect which variables have zeros that likely represent missing values.
        
        Returns:
            Dictionary mapping column names to boolean indicating if zeros should be treated as missing
        """
        zero_as_missing = {}
        
        # Skip non-numeric columns and columns that shouldn't be imputed
        for col in self.train_df.columns:
            if col in self.non_imputable or not pd.api.types.is_numeric_dtype(self.train_df[col]):
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
    
    def detect_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect missing values in a DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with missing value statistics
        """
        # Calculate missing values (NaN)
        na_counts = df.isna().sum()
        na_pcts = na_counts / len(df) * 100
        
        # Calculate zeros that should be treated as missing
        zero_counts = {}
        zero_pcts = {}
        for col, treat_as_missing in self.zero_as_missing.items():
            if col in df.columns and treat_as_missing:
                zero_count = (df[col] == 0).sum()
                zero_counts[col] = int(zero_count)
                zero_pcts[col] = float(zero_count / len(df) * 100)
        
        # Calculate total missing (NaN + zeros treated as missing)
        total_missing = {}
        total_missing_pcts = {}
        for col in df.columns:
            if col in self.non_imputable:
                continue
                
            na_count = na_counts[col]
            zero_count = zero_counts.get(col, 0)
            total = na_count + zero_count
            
            total_missing[col] = int(total)
            total_missing_pcts[col] = float(total / len(df) * 100)
        
        # Get columns with high missingness
        high_missing = {col: pct for col, pct in total_missing_pcts.items() if pct > 50}
        
        # Group by icustayid and calculate missingness per patient
        patient_missingness = {}
        if 'icustayid' in df.columns:
            for patient_id in df['icustayid'].unique():
                patient_df = df[df['icustayid'] == patient_id]
                
                # Calculate missing values per column for this patient
                missing = {}
                for col in patient_df.columns:
                    if col in self.non_imputable:
                        continue
                        
                    na_count = patient_df[col].isna().sum()
                    zero_count = 0
                    if col in self.zero_as_missing and self.zero_as_missing[col]:
                        zero_count = (patient_df[col] == 0).sum()
                    
                    total = na_count + zero_count
                    missing[col] = int(total)
                
                patient_missingness[str(patient_id)] = missing
        
        return {
            "na_counts": na_counts.to_dict(),
            "na_percentages": {k: round(v, 2) for k, v in na_pcts.to_dict().items()},
            "zero_as_missing": self.zero_as_missing,
            "zero_counts": zero_counts,
            "zero_percentages": {k: round(v, 2) for k, v in zero_pcts.items()},
            "total_missing": total_missing,
            "total_missing_percentages": {k: round(v, 2) for k, v in total_missing_pcts.items()},
            "high_missing_columns": {k: round(v, 2) for k, v in high_missing.items()},
            "patient_missingness": patient_missingness
        }
    
    def preprocess_for_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess DataFrame for imputation, converting zeros to NaN where appropriate.
        
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
        
        return processed_df
    
    def fit_imputation_models(self, method: str = 'mice') -> None:
        """
        Fit imputation models on the training data.
        
        Args:
            method: Imputation method ('mean', 'median', 'knn', or 'mice')
        """
        # Preprocess data
        processed_df = self.preprocess_for_imputation(self.train_df)
        
        # Get columns to impute (numeric columns not in non_imputable list)
        imputable_cols = [col for col in processed_df.columns 
                         if col not in self.non_imputable 
                         and pd.api.types.is_numeric_dtype(processed_df[col])]
        
        # Store imputation statistics for each column
        self.imputation_stats = {}
        
        if method == 'mean':
            # Fit a simple mean imputer
            imputer = SimpleImputer(strategy='mean')
            imputer.fit(processed_df[imputable_cols])
            self.imputers['mean'] = imputer
            
            # Store imputation values
            for i, col in enumerate(imputable_cols):
                self.imputation_stats[col] = {
                    'method': 'mean',
                    'value': float(imputer.statistics_[i])
                }
        
        elif method == 'median':
            # Fit a simple median imputer
            imputer = SimpleImputer(strategy='median')
            imputer.fit(processed_df[imputable_cols])
            self.imputers['median'] = imputer
            
            # Store imputation values
            for i, col in enumerate(imputable_cols):
                self.imputation_stats[col] = {
                    'method': 'median',
                    'value': float(imputer.statistics_[i])
                }
        
        elif method == 'knn':
            # Fit a KNN imputer
            imputer = KNNImputer(n_neighbors=5)
            imputer.fit(processed_df[imputable_cols])
            self.imputers['knn'] = imputer
            
            # No simple statistics to store for KNN
            for col in imputable_cols:
                self.imputation_stats[col] = {
                    'method': 'knn',
                    'value': None
                }
        
        elif method == 'mice':
            # Fit a MICE imputer (Multiple Imputation by Chained Equations)
            imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=10), 
                                      random_state=42, 
                                      max_iter=10)
            imputer.fit(processed_df[imputable_cols])
            self.imputers['mice'] = imputer
            
            # No simple statistics to store for MICE
            for col in imputable_cols:
                self.imputation_stats[col] = {
                    'method': 'mice',
                    'value': None
                }
        
        else:
            raise ValueError(f"Unknown imputation method: {method}")
    
    def impute_patient_data(self, patient_id: int, method: str = 'mice') -> Dict[str, Any]:
        """
        Impute missing values for a specific patient.
        
        Args:
            patient_id: Patient's unique identifier (icustayid)
            method: Imputation method ('mean', 'median', 'knn', or 'mice')
            
        Returns:
            Dictionary with imputation results
        """
        # Check if patient exists in training data
        train_patient_exists = 'icustayid' in self.train_df.columns and patient_id in self.train_df['icustayid'].values
        
        # Check if patient exists in test data
        test_patient_exists = (self.test_df is not None and 
                              'icustayid' in self.test_df.columns and 
                              patient_id in self.test_df['icustayid'].values)
        
        if not train_patient_exists and not test_patient_exists:
            return {"error": f"Patient ID {patient_id} not found in either dataset"}
        
        # Determine which dataset to use
        if train_patient_exists:
            df = self.train_df[self.train_df['icustayid'] == patient_id].copy()
        else:
            df = self.test_df[self.test_df['icustayid'] == patient_id].copy()
        
        # Check if imputation models are fitted
        if not self.imputers:
            self.fit_imputation_models(method=method)
        
        # Preprocess data
        processed_df = self.preprocess_for_imputation(df)
        
        # Get columns to impute (numeric columns not in non_imputable list)
        imputable_cols = [col for col in processed_df.columns 
                         if col not in self.non_imputable 
                         and pd.api.types.is_numeric_dtype(processed_df[col])]
        
        # Check which columns have missing values
        missing_cols = {}
        for col in imputable_cols:
            missing_count = processed_df[col].isna().sum()
            if missing_count > 0:
                missing_cols[col] = int(missing_count)
        
        # Don't proceed if no missing values
        if not missing_cols:
            return {
                "patient_id": patient_id,
                "status": "no_imputation_needed",
                "message": "No missing values to impute for this patient"
            }
        
        # Get the appropriate imputer
        imputer = self.imputers.get(method)
        if imputer is None:
            self.fit_imputation_models(method=method)
            imputer = self.imputers.get(method)
        
        # Create imputation results data structure
        imputation_results = []
        
        # Process each record separately to maintain time series structure
        for idx, row in processed_df.iterrows():
            record = {}
            
            # Copy non-imputable columns
            for col in self.non_imputable:
                if col in row:
                    record[col] = row[col]
            
            # Check which values need imputation in this record
            missing_in_record = {}
            for col in imputable_cols:
                if pd.isna(row[col]):
                    missing_in_record[col] = True
            
            # Skip record if no missing values
            if not missing_in_record:
                # Copy all original values
                for col in imputable_cols:
                    record[col] = row[col]
                imputation_results.append({"original": record, "imputed": {}})
                continue
            
            # Prepare data for imputation
            record_df = pd.DataFrame([row[imputable_cols]])
            
            # Perform imputation
            imputed_values = imputer.transform(record_df)[0]
            
            # Create record with imputed values
            imputed_record = {}
            for i, col in enumerate(imputable_cols):
                imputed_record[col] = float(imputed_values[i])
            
            # Add original and imputed data to results
            original_values = {}
            imputed_values_dict = {}
            for i, col in enumerate(imputable_cols):
                original_values[col] = float(row[col]) if not pd.isna(row[col]) else None
                if col in missing_in_record:
                    imputed_values_dict[col] = float(imputed_values[i])
            
            imputation_results.append({
                "original": original_values,
                "imputed": imputed_values_dict
            })
        
        return {
            "patient_id": patient_id,
            "status": "imputation_completed",
            "method": method,
            "missing_columns": missing_cols,
            "imputation_stats": {k: v for k, v in self.imputation_stats.items() if k in missing_cols},
            "imputation_results": imputation_results
        }
    
    def impute_dataset(self, dataset: str = 'train', method: str = 'mice') -> pd.DataFrame:
        """
        Impute missing values for an entire dataset.
        
        Args:
            dataset: Which dataset to impute ('train' or 'test')
            method: Imputation method ('mean', 'median', 'knn', or 'mice')
            
        Returns:
            DataFrame with imputed values
        """
        # Determine which dataset to use
        if dataset == 'train':
            df = self.train_df.copy()
        elif dataset == 'test':
            if self.test_df is None:
                raise ValueError("Test dataset not provided")
            df = self.test_df.copy()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Check if imputation models are fitted
        if not self.imputers:
            self.fit_imputation_models(method=method)
        
        # Preprocess data
        processed_df = self.preprocess_for_imputation(df)
        
        # Get columns to impute (numeric columns not in non_imputable list)
        imputable_cols = [col for col in processed_df.columns 
                         if col not in self.non_imputable 
                         and pd.api.types.is_numeric_dtype(processed_df[col])]
        
        # Get the appropriate imputer
        imputer = self.imputers.get(method)
        if imputer is None:
            self.fit_imputation_models(method=method)
            imputer = self.imputers.get(method)
        
        # Copy non-imputable columns to the result
        result_df = df.copy()
        
        # Perform imputation for imputable columns
        imputed_values = imputer.transform(processed_df[imputable_cols])
        
        # Replace values in the result
        result_df[imputable_cols] = imputed_values
        
        return result_df
    
    def generate_imputation_report(self, patient_id: Optional[int] = None) -> str:
        """
        Generate a human-readable report about imputation.
        
        Args:
            patient_id: Optional patient ID to generate a report for
            
        Returns:
            Text report about imputation
        """
        if patient_id is not None:
            # Generate report for a specific patient
            imputation_data = self.impute_patient_data(patient_id, method='mice')
            
            if "error" in imputation_data:
                return imputation_data["error"]
            
            if imputation_data["status"] == "no_imputation_needed":
                return f"No missing values to impute for patient {patient_id}."
            
            # Build report
            report = []
            report.append(f"# Imputation Report for Patient {patient_id}\n")
            
            # Missing data overview
            missing_cols = imputation_data["missing_columns"]
            report.append(f"## Missing Data Overview")
            report.append(f"The patient has missing values in {len(missing_cols)} variables:")
            for col, count in missing_cols.items():
                report.append(f"- {col}: {count} missing values")
            
            # Imputation method
            report.append(f"\n## Imputation Method")
            report.append(f"Method used: {imputation_data['method']}")
            
            # Imputation details
            report.append(f"\n## Imputation Details")
            
            # Get top 5 most important variables with missing values
            important_vars = ["SOFA", "GCS", "Arterial_lactate", "PaO2_FiO2", "SysBP", "WBC_count", "Creatinine"]
            important_missing = [v for v in important_vars if v in missing_cols]
            
            if not important_missing:
                # If none of the important vars are missing, use the ones that are missing
                important_missing = list(missing_cols.keys())
            
            for var in important_missing[:5]:
                if var in imputation_data["imputation_stats"]:
                    stats = imputation_data["imputation_stats"][var]
                    
                    report.append(f"### {var}")
                    if stats["method"] in ["mean", "median"]:
                        report.append(f"- Imputation method: {stats['method']}")
                        report.append(f"- Imputation value: {stats['value']:.2f}")
                    else:
                        report.append(f"- Imputation method: {stats['method']} (uses relationships between variables to predict missing values)")
            
            # Impact on analysis
            report.append(f"\n## Potential Impact on Analysis")
            report.append(f"Missing data and imputation can affect the reliability of predictions and analysis. Here's an assessment of the impact:")
            
            # Determine severity based on which variables are missing and how many
            critical_vars = set(["SOFA", "GCS", "Arterial_lactate", "HR", "SysBP", "MeanBP", "SpO2"])
            missing_critical = critical_vars.intersection(set(missing_cols.keys()))
            
            if len(missing_critical) >= 3:
                impact = "High"
                explanation = "Several critical clinical variables are missing, which could significantly affect the reliability of predictions."
            elif len(missing_critical) > 0:
                impact = "Moderate"
                explanation = "Some important clinical variables are missing, which could affect the accuracy of predictions."
            elif len(missing_cols) > 10:
                impact = "Moderate"
                explanation = "A large number of variables are missing, which may impact the overall reliability of predictions."
            else:
                impact = "Low"
                explanation = "The missing variables are not critical for clinical assessment, and imputation should provide reliable values."
            
            report.append(f"- Impact level: {impact}")
            report.append(f"- Explanation: {explanation}")
            
            return "\n".join(report)
        
        else:
            # Generate general report about imputation in the dataset
            # Detect missing values in training data
            missing_info = self.detect_missing_values(self.train_df)
            
            # Build report
            report = []
            report.append("# Imputation Strategy Report\n")
            
            # Overview of missing data
            report.append("## Missing Data Overview")
            high_missing = missing_info["high_missing_columns"]
            if high_missing:
                report.append(f"The dataset has {len(high_missing)} variables with high missingness (>50%):")
                for col, pct in sorted(high_missing.items(), key=lambda x: x[1], reverse=True)[:10]:
                    report.append(f"- {col}: {pct:.1f}% missing")
            else:
                report.append("No variables have more than 50% missing values.")
            
            # Zeros treated as missing
            zero_as_missing = {k: v for k, v in self.zero_as_missing.items() if v}
            if zero_as_missing:
                report.append(f"\nFor the following variables, zero values are treated as missing data:")
                for col in sorted(zero_as_missing.keys()):
                    pct = missing_info["zero_percentages"].get(col, 0)
                    report.append(f"- {col}: {pct:.1f}% zeros (treated as missing)")
            
            # Imputation strategy
            report.append(f"\n## Imputation Strategy")
            report.append("The system uses the following strategies for handling missing values:")
            report.append(f"1. **Identification**: Missing values are identified both as NaN and as zeros for certain physiological variables where zero is not a valid value.")
            report.append(f"2. **Method**: Multiple Imputation by Chained Equations (MICE) is used as the primary imputation method. This approach uses relationships between variables to predict missing values.")
            report.append(f"3. **Validation**: For any prediction, the system will assess how imputation uncertainty affects the prediction reliability.")
            
            # Impact on analysis
            report.append(f"\n## Impact on Analysis")
            report.append(f"Missing data and imputation can affect the reliability of predictions. The system handles this by:")
            report.append(f"- Transparently reporting which values were imputed for each patient")
            report.append(f"- Assessing the impact of missingness on prediction reliability")
            report.append(f"- Using an imputation method that preserves relationships between variables")
            report.append(f"- Providing confidence estimates that account for imputation uncertainty")
            
            return "\n".join(report)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert imputation data to a dictionary format.
        
        Returns:
            Dictionary representation of imputation data
        """
        # Detect missing values in training data
        missing_info = self.detect_missing_values(self.train_df)
        
        return {
            "missing_info": missing_info,
            "zero_as_missing": self.zero_as_missing,
            "imputation_stats": self.imputation_stats,
        }
    
    def to_json(self) -> str:
        """
        Convert imputation data to a JSON string.
        
        Returns:
            JSON string representation of imputation data
        """
        return json.dumps(self.to_dict(), default=str, indent=2)


# Example usage (would be wrapped in API endpoint)
if __name__ == "__main__":
    # Replace with your actual data paths
    train_data_path = "./data/AI_agent_train_sepsis.csv"
    test_data_path = "./data/AI_agent_test_sepsis_features.csv"  # Optional
    
    # Initialize the tool
    imputation_tool = ImputationTool(train_data_path, test_data_path)
    
    # Detect missing values
    missing_info = imputation_tool.detect_missing_values(imputation_tool.train_df)
    print("Missing value statistics:")
    print(json.dumps(missing_info, indent=2, default=str)[:1000] + "...")  # Truncate for readability
    
    # Fit imputation models
    imputation_tool.fit_imputation_models(method='mice')
    
    # Get all patient IDs
    if 'icustayid' in imputation_tool.train_df.columns:
        patient_ids = imputation_tool.train_df['icustayid'].unique().tolist()
        
        if patient_ids:
            # Impute data for the first patient
            first_patient_id = patient_ids[0]
            imputation_results = imputation_tool.impute_patient_data(first_patient_id, method='mice')
            print(f"\nImputation results for patient {first_patient_id}:")
            print(json.dumps(imputation_results, indent=2, default=str)[:1000] + "...")  # Truncate for readability
            
            # Generate imputation report
            report = imputation_tool.generate_imputation_report(first_patient_id)
            print(f"\nImputation report for patient {first_patient_id}:")
            print(report)
    
    # Generate general imputation report
    general_report = imputation_tool.generate_imputation_report()
    print("\nGeneral imputation report:")
    print(general_report)