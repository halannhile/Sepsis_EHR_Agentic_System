import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json

class PatientRetrievalTool:
    """
    Tool for retrieving specific patient records from the sepsis EHR dataset.
    """
    
    def __init__(self, data_path: str, test_data_path: Optional[str] = None):
        """
        Initialize the tool with the path to the dataset.
        
        Args:
            data_path: Path to the CSV file containing the dataset
            test_data_path: Optional path to test dataset
        """
        # Load training data
        self.train_df = pd.read_csv(data_path)
        
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
        
        # Convert charttime to datetime if it exists
        if 'charttime' in self.train_df.columns:
            try:
                self.train_df['charttime'] = pd.to_datetime(self.train_df['charttime'])
            except:
                # If conversion fails, keep as is
                pass
                
        if self.test_df is not None and 'charttime' in self.test_df.columns:
            try:
                self.test_df['charttime'] = pd.to_datetime(self.test_df['charttime'])
            except:
                # If conversion fails, keep as is
                pass
    
    def get_patient_ids(self) -> List[int]:
        """
        Get a list of all unique patient IDs in the dataset.
        
        Returns:
            List of all unique patient IDs
        """
        train_ids = self.train_df['icustayid'].unique().tolist() if 'icustayid' in self.train_df.columns else []
        
        if self.test_df is not None and 'icustayid' in self.test_df.columns:
            test_ids = self.test_df['icustayid'].unique().tolist()
            # Combine and deduplicate IDs
            all_ids = list(set(train_ids + test_ids))
            return sorted(all_ids)
        
        return sorted(train_ids)
    
    def get_patient_data(self, patient_id: int) -> Dict[str, Any]:
        """
        Get all data for a specific patient.
        
        Args:
            patient_id: Patient's unique identifier (icustayid)
            
        Returns:
            Dictionary containing patient data and statistics
        """
        # Check if patient exists
        if patient_id not in self.get_patient_ids():
            return {"error": f"Patient ID {patient_id} not found in the dataset"}
        
        # Get all records for this patient from training data
        patient_df = self.train_df[self.train_df['icustayid'] == patient_id].copy() if 'icustayid' in self.train_df.columns else pd.DataFrame()
        
        # If not found in training data or patient data is empty, try test data
        if len(patient_df) == 0 and self.test_df is not None:
            patient_df = self.test_df[self.test_df['icustayid'] == patient_id].copy() if 'icustayid' in self.test_df.columns else pd.DataFrame()
        
        # If still no data, return error
        if len(patient_df) == 0:
            return {"error": f"No data found for patient ID {patient_id}"}
        
        # Sort by charttime if available
        if 'charttime' in patient_df.columns:
            patient_df = patient_df.sort_values('charttime')
        
        # Get static information (should be the same across all records)
        static_info = {}
        for col in ['gender', 'age', 'elixhauser', 're_admission']:
            if col in patient_df.columns:
                # Use first non-null value
                values = patient_df[col].dropna()
                if len(values) > 0:
                    if col == 'age':
                        # Convert age from days to years
                        static_info[col] = float(values.iloc[0]) / 365.25
                    else:
                        static_info[col] = values.iloc[0]
        
        # Get mortality label if available
        mortality = None
        if 'mortality_90d' in patient_df.columns:
            mortality = bool(patient_df['mortality_90d'].iloc[0])
        
        # Get timeline of measurements
        timeline = []
        for _, row in patient_df.iterrows():
            record = {}
            if 'charttime' in row:
                record['timestamp'] = row['charttime']
            
            # Add vital signs
            vital_signs = {}
            for col in ['GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'SpO2', 'Temp_C']:
                if col in row and not pd.isna(row[col]):
                    vital_signs[col] = row[col]
            record['vital_signs'] = vital_signs
            
            # Add lab results
            lab_results = {}
            for col in ['Potassium', 'Sodium', 'Chloride', 'Glucose', 'BUN', 'Creatinine',
                       'Magnesium', 'Calcium', 'Ionised_Ca', 'CO2_mEqL', 'SGOT', 'SGPT',
                       'Total_bili', 'Albumin', 'Hb', 'WBC_count', 'Platelets_count',
                       'PTT', 'PT', 'INR', 'Arterial_pH', 'paO2', 'paCO2', 'Arterial_BE',
                       'HCO3', 'Arterial_lactate']:
                if col in row and not pd.isna(row[col]):
                    lab_results[col] = row[col]
            record['lab_results'] = lab_results
            
            # Add treatment information
            treatments = {}
            for col in ['FiO2_1', 'mechvent', 'median_dose_vaso', 'max_dose_vaso',
                       'input_total', 'input_4hourly', 'output_total', 'output_4hourly',
                       'cumulated_balance']:
                if col in row and not pd.isna(row[col]):
                    treatments[col] = row[col]
            record['treatments'] = treatments
            
            # Add clinical scores
            scores = {}
            for col in ['SOFA', 'SIRS', 'Shock_Index', 'PaO2_FiO2']:
                if col in row and not pd.isna(row[col]):
                    scores[col] = row[col]
            record['clinical_scores'] = scores
            
            timeline.append(record)
        
        # Calculate summary statistics for key variables over time
        summary_stats = {}
        numeric_cols = [col for col in patient_df.columns if 
                       pd.api.types.is_numeric_dtype(patient_df[col]) and
                       col not in ['bloc', 'icustayid', 'mortality_90d']]
        
        for col in numeric_cols:
            values = patient_df[col].dropna()
            if len(values) > 0:
                summary_stats[col] = {
                    'mean': float(values.mean()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'first': float(values.iloc[0]),
                    'last': float(values.iloc[-1]),
                    'trend': float(values.iloc[-1] - values.iloc[0]) if len(values) > 1 else 0,
                    'count': int(len(values)),
                    'missing': int(patient_df[col].isna().sum()),
                    'zero_count': int((values == 0).sum())
                }
        
        # Calculate alert flags based on abnormal values
        alerts = self._calculate_alerts(patient_df)
        
        # Return complete patient information
        return {
            "patient_id": patient_id,
            "static_info": static_info,
            "mortality_90d": mortality,
            "num_records": len(patient_df),
            "timeline": timeline,
            "summary_stats": summary_stats,
            "alerts": alerts,
            "raw_data": patient_df.to_dict(orient='records')
        }
    
    def _calculate_alerts(self, patient_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Calculate clinical alerts based on abnormal values.
        
        Args:
            patient_df: DataFrame containing patient records
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Define normal ranges for vital signs and lab values
        normal_ranges = {
            'HR': (60, 100),      # Heart rate (bpm)
            'SysBP': (90, 140),   # Systolic BP (mmHg)
            'MeanBP': (70, 105),  # Mean BP (mmHg)
            'DiaBP': (60, 90),    # Diastolic BP (mmHg)
            'RR': (12, 20),       # Respiratory rate (breaths/min)
            'SpO2': (95, 100),    # Oxygen saturation (%)
            'Temp_C': (36.5, 37.5),  # Temperature (°C)
            'GCS': (15, 15),      # Glasgow Coma Scale (15 is normal)
            'Glucose': (70, 180),  # Blood glucose (mg/dL)
            'WBC_count': (4, 11),  # White blood cell count (×10^9/L)
            'Arterial_lactate': (0, 2),  # Lactate (mmol/L)
            'SOFA': (0, 0),       # SOFA score (0 is normal)
            'SIRS': (0, 0),       # SIRS criteria (0 is normal)
            'Shock_Index': (0, 0.7),  # Shock Index (HR/SysBP)
            'Creatinine': (0.5, 1.2),  # Creatinine (mg/dL)
            'Platelets_count': (150, 450),  # Platelets (×10^9/L)
            'PaO2_FiO2': (300, 500),  # P/F ratio (mmHg)
        }
        
        # Check each variable against normal ranges
        for var, (lower, upper) in normal_ranges.items():
            if var in patient_df.columns:
                # Get non-null values
                values = patient_df[var].dropna()
                if len(values) > 0:
                    # Check for values outside normal range
                    abnormal_high = values[values > upper]
                    abnormal_low = values[values < lower]
                    
                    # Create alerts for high values
                    if len(abnormal_high) > 0:
                        max_val = float(abnormal_high.max())
                        timestamp = None
                        if 'charttime' in patient_df.columns:
                            idx = values.idxmax()
                            timestamp = patient_df.loc[idx, 'charttime']
                        
                        alerts.append({
                            "variable": var,
                            "value": max_val,
                            "condition": "high",
                            "normal_range": normal_ranges[var],
                            "timestamp": timestamp,
                            "severity": "critical" if max_val > upper * 1.5 else "warning"
                        })
                    
                    # Create alerts for low values
                    if len(abnormal_low) > 0:
                        min_val = float(abnormal_low.min())
                        timestamp = None
                        if 'charttime' in patient_df.columns:
                            idx = values.idxmin()
                            timestamp = patient_df.loc[idx, 'charttime']
                        
                        alerts.append({
                            "variable": var,
                            "value": min_val,
                            "condition": "low",
                            "normal_range": normal_ranges[var],
                            "timestamp": timestamp,
                            "severity": "critical" if min_val < lower * 0.5 and lower > 0 else "warning"
                        })
        
        # Sort alerts by severity (critical first)
        return sorted(alerts, key=lambda x: 0 if x['severity'] == 'critical' else 1)
    
    def get_patient_summary(self, patient_id: int) -> str:
        """
        Generate a human-readable text summary for a specific patient.
        
        Args:
            patient_id: Patient's unique identifier (icustayid)
            
        Returns:
            Text summary of patient data
        """
        data = self.get_patient_data(patient_id)
        if "error" in data:
            return data["error"]
        
        # Build summary text
        summary = []
        summary.append(f"# Patient Summary for ID: {patient_id}\n")
        
        # Patient demographics
        static_info = data["static_info"]
        summary.append("## Patient Information")
        if "gender" in static_info:
            summary.append(f"- Gender: {static_info['gender']}")
        if "age" in static_info:
            summary.append(f"- Age: {static_info['age']:.1f} years")
        if "elixhauser" in static_info:
            summary.append(f"- Elixhauser comorbidity score: {static_info['elixhauser']}")
        if "re_admission" in static_info and static_info["re_admission"] == 1:
            summary.append("- This is a readmission")
        
        # Mortality information if available
        if data["mortality_90d"] is not None:
            outcome = "Deceased" if data["mortality_90d"] else "Survived"
            summary.append(f"- 90-day outcome: {outcome}")
        
        summary.append(f"- Total records: {data['num_records']}")
        
        # Clinical alerts
        if data["alerts"]:
            summary.append("\n## Critical Alerts")
            critical_alerts = [a for a in data["alerts"] if a["severity"] == "critical"]
            if critical_alerts:
                for alert in critical_alerts[:5]:  # Show top 5 critical alerts
                    var = alert["variable"]
                    val = alert["value"]
                    cond = alert["condition"]
                    normal = alert["normal_range"]
                    summary.append(f"- {var}: {val:.1f} is {cond.upper()} (normal range: {normal[0]}-{normal[1]})")
            else:
                summary.append("- No critical alerts detected")
        
        # Key vital signs
        vital_stats = {}
        for var in ['HR', 'SysBP', 'MeanBP', 'RR', 'SpO2', 'Temp_C', 'GCS']:
            if var in data["summary_stats"]:
                vital_stats[var] = data["summary_stats"][var]
        
        if vital_stats:
            summary.append("\n## Vital Signs Summary")
            for var, stats in vital_stats.items():
                summary.append(f"- {var}: Latest {stats['last']:.1f}, Range {stats['min']:.1f}-{stats['max']:.1f}")
        
        # Key lab values
        lab_stats = {}
        for var in ['WBC_count', 'Platelets_count', 'Hb', 'Creatinine', 'Arterial_lactate', 'Glucose']:
            if var in data["summary_stats"]:
                lab_stats[var] = data["summary_stats"][var]
                
        if lab_stats:
            summary.append("\n## Laboratory Results Summary")
            for var, stats in lab_stats.items():
                summary.append(f"- {var}: Latest {stats['last']:.1f}, Range {stats['min']:.1f}-{stats['max']:.1f}")
        
        # Treatment information
        treatment_info = []
        for var in ['mechvent', 'median_dose_vaso', 'max_dose_vaso', 'input_total', 'output_total']:
            if var in data["summary_stats"]:
                stats = data["summary_stats"][var]
                
                if var == 'mechvent' and stats['mean'] > 0:
                    treatment_info.append(f"- Patient received mechanical ventilation")
                elif var in ['median_dose_vaso', 'max_dose_vaso'] and stats['max'] > 0:
                    treatment_info.append(f"- Patient received vasopressors (max dose: {stats['max']:.2f})")
                elif var == 'input_total':
                    treatment_info.append(f"- Total fluid input: {stats['last']:.1f}")
                elif var == 'output_total':
                    treatment_info.append(f"- Total fluid output: {stats['last']:.1f}")
        
        if treatment_info:
            summary.append("\n## Treatment Information")
            summary.extend(treatment_info)
        
        # Clinical scores
        score_info = []
        for var in ['SOFA', 'SIRS', 'Shock_Index', 'PaO2_FiO2']:
            if var in data["summary_stats"]:
                stats = data["summary_stats"][var]
                
                if var == 'SOFA':
                    score_info.append(f"- SOFA score: Latest {stats['last']:.1f}, Max {stats['max']:.1f}")
                elif var == 'SIRS':
                    score_info.append(f"- SIRS criteria: Latest {stats['last']:.1f}, Max {stats['max']:.1f}")
                elif var == 'Shock_Index':
                    score_info.append(f"- Shock Index: Latest {stats['last']:.2f}, Max {stats['max']:.2f}")
                elif var == 'PaO2_FiO2':
                    score_info.append(f"- P/F ratio: Latest {stats['last']:.1f}, Min {stats['min']:.1f}")
        
        if score_info:
            summary.append("\n## Clinical Scores")
            summary.extend(score_info)
        
        # Missing data report
        missing_data = []
        for var, stats in data["summary_stats"].items():
            if stats['missing'] > 0:
                missing_pct = stats['missing'] / (stats['count'] + stats['missing']) * 100
                if missing_pct > 50:  # Only report if more than 50% missing
                    missing_data.append(f"- {var}: {missing_pct:.1f}% missing")
        
        if missing_data:
            summary.append("\n## Missing Data Report")
            summary.extend(missing_data[:5])  # Show top 5 missing variables
            
        return "\n".join(summary)

    def get_patient_time_series(self, patient_id: int, variables: List[str]) -> Dict[str, Any]:
        """
        Get time series data for specific variables for a patient.
        
        Args:
            patient_id: Patient's unique identifier (icustayid)
            variables: List of variable names to retrieve
            
        Returns:
            Dictionary with time series data for requested variables
        """
        # Check if patient exists
        if patient_id not in self.get_patient_ids():
            return {"error": f"Patient ID {patient_id} not found in the dataset"}
        
        # Get patient data from training set
        patient_df = self.train_df[self.train_df['icustayid'] == patient_id].copy() if 'icustayid' in self.train_df.columns else pd.DataFrame()
        
        # If not found in training data, try test data
        if len(patient_df) == 0 and self.test_df is not None:
            patient_df = self.test_df[self.test_df['icustayid'] == patient_id].copy() if 'icustayid' in self.test_df.columns else pd.DataFrame()
            
        # If still no data found, return error
        if len(patient_df) == 0:
            return {"error": f"No data found for patient ID {patient_id}"}
        
        # Sort by charttime if available
        if 'charttime' in patient_df.columns:
            patient_df = patient_df.sort_values('charttime')
        
        # Filter variables to only include those in the dataset
        valid_vars = [var for var in variables if var in patient_df.columns]
        
        if not valid_vars:
            return {"error": f"None of the requested variables {variables} found in the dataset"}
        
        # Create time series data
        time_series = {}
        
        # Get timestamps if available
        timestamps = None
        if 'charttime' in patient_df.columns:
            timestamps = patient_df['charttime'].tolist()
        else:
            # Use bloc as a proxy for time if charttime is not available
            if 'bloc' in patient_df.columns:
                timestamps = patient_df['bloc'].tolist()
            else:
                # Use simple indices if no time information is available
                timestamps = list(range(len(patient_df)))
        
        # Extract time series for each variable
        for var in valid_vars:
            values = patient_df[var].tolist()
            time_series[var] = list(zip(timestamps, values))
        
        return {
            "patient_id": patient_id,
            "time_series": time_series,
            "num_records": len(patient_df)
        }
    
    def find_similar_patients(self, patient_id: int, n: int = 5) -> List[Dict[str, Any]]:
        """
        Find patients with similar clinical characteristics.
        
        Args:
            patient_id: Reference patient's unique identifier (icustayid)
            n: Number of similar patients to return
            
        Returns:
            List of dictionaries containing similar patient information
        """
        # Check if patient exists
        if patient_id not in self.get_patient_ids():
            return [{"error": f"Patient ID {patient_id} not found in the dataset"}]
        
        # Get unique patient IDs
        all_patient_ids = self.get_patient_ids()
        
        # Calculate feature averages for each patient
        patient_features = {}
        
        # Choose features for similarity comparison
        similarity_features = ['age', 'GCS', 'HR', 'SysBP', 'RR', 'SpO2', 'Temp_C',
                             'Creatinine', 'WBC_count', 'SOFA', 'SIRS']
        
        # Filter to features actually present in the dataset
        similarity_features = [f for f in similarity_features if f in self.train_df.columns]
        
        # Calculate average feature values for each patient
        for pid in all_patient_ids:
            # Get patient data from training set
            patient_df = self.train_df[self.train_df['icustayid'] == pid] if 'icustayid' in self.train_df.columns else pd.DataFrame()
            
            # If not found in training data, try test data
            if len(patient_df) == 0 and self.test_df is not None:
                patient_df = self.test_df[self.test_df['icustayid'] == pid] if 'icustayid' in self.test_df.columns else pd.DataFrame()
                
            # Skip if no data found
            if len(patient_df) == 0:
                continue
                
            # Calculate average values for each feature
            feature_avgs = {}
            for feature in similarity_features:
                if feature in patient_df.columns:
                    values = patient_df[feature].dropna()
                    if len(values) > 0:
                        feature_avgs[feature] = values.mean()
                    else:
                        feature_avgs[feature] = np.nan
                else:
                    feature_avgs[feature] = np.nan
            
            patient_features[pid] = feature_avgs
        
        # Get reference patient features
        ref_features = patient_features[patient_id]
        
        # Calculate similarity (Euclidean distance)
        similarities = []
        
        for pid in all_patient_ids:
            if pid == patient_id:
                continue  # Skip the reference patient
                
            # Get features for this patient
            features = patient_features.get(pid)
            if not features:
                continue
                
            # Calculate distance
            squared_diffs = []
            for feature in similarity_features:
                if feature in ref_features and feature in features:
                    # Skip if either value is missing
                    if np.isnan(ref_features[feature]) or np.isnan(features[feature]):
                        continue
                    
                    # Normalize the difference by the range of the feature in the dataset
                    feature_range = self.train_df[feature].max() - self.train_df[feature].min() if feature in self.train_df.columns else 1
                    if feature_range == 0:
                        feature_range = 1  # Avoid division by zero
                    
                    diff = (ref_features[feature] - features[feature]) / feature_range
                    squared_diffs.append(diff**2)
            
            if squared_diffs:
                distance = np.sqrt(sum(squared_diffs))
                similarities.append((pid, distance))
        
        # Sort by similarity (lowest distance first)
        similarities.sort(key=lambda x: x[1])
        
        # Get the top N similar patients
        similar_patients = []
        for pid, distance in similarities[:n]:
            # Get mortality status if available
            mortality = None
            if 'mortality_90d' in self.train_df.columns:
                patient_rows = self.train_df[self.train_df['icustayid'] == pid]
                if len(patient_rows) > 0:
                    mortality = bool(patient_rows['mortality_90d'].iloc[0])
            
            # Get static information
            info = {}
            for col in ['gender', 'age', 'elixhauser']:
                if col in self.train_df.columns:
                    patient_rows = self.train_df[self.train_df['icustayid'] == pid]
                    values = patient_rows[col].dropna() if len(patient_rows) > 0 else pd.Series()
                    if len(values) > 0:
                        if col == 'age':
                            # Convert age from days to years
                            info[col] = float(values.iloc[0]) / 365.25
                        else:
                            info[col] = values.iloc[0]
            
            # Calculate average values for key clinical features
            clinical_avgs = {}
            for feature in ['GCS', 'HR', 'SysBP', 'MeanBP', 'RR', 'SpO2', 'SOFA']:
                if feature in self.train_df.columns:
                    patient_rows = self.train_df[self.train_df['icustayid'] == pid]
                    values = patient_rows[feature].dropna() if len(patient_rows) > 0 else pd.Series()
                    if len(values) > 0:
                        clinical_avgs[feature] = float(values.mean())
            
            similar_patients.append({
                "patient_id": pid,
                "similarity_score": 1.0 / (1.0 + distance),  # Convert distance to similarity score
                "mortality_90d": mortality,
                "info": info,
                "clinical_averages": clinical_avgs
            })
        
        return similar_patients
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert patient data to a dictionary format.
        
        Returns:
            Dictionary representation of patient data
        """
        return {
            "total_patients": len(self.get_patient_ids()),
            "patient_ids": self.get_patient_ids()
        }
    
    def to_json(self) -> str:
        """
        Convert patient data to a JSON string.
        
        Returns:
            JSON string representation of patient data
        """
        return json.dumps(self.to_dict(), default=str)


# Example usage (would be wrapped in API endpoint)
if __name__ == "__main__":
    # Replace with your actual data path
    data_path = "AI_agent_train_sepsis.csv"
    test_data_path = "AI_agent_test_sepsis_features.csv"  # Optional
    
    # Initialize the tool
    patient_tool = PatientRetrievalTool(data_path, test_data_path)
    
    # Get all patient IDs
    patient_ids = patient_tool.get_patient_ids()
    print(f"Found {len(patient_ids)} unique patients")
    
    if patient_ids:
        # Get data for the first patient
        first_patient_id = patient_ids[0]
        patient_data = patient_tool.get_patient_data(first_patient_id)
        print(f"\nPatient {first_patient_id} data:")
        print(json.dumps(patient_data, indent=2, default=str)[:1000] + "...")  # Truncate for readability
        
        # Get summary for the first patient
        summary = patient_tool.get_patient_summary(first_patient_id)
        print(f"\nPatient {first_patient_id} summary:")
        print(summary)
        
        # Get time series data for some variables
        time_series = patient_tool.get_patient_time_series(first_patient_id, ["HR", "GCS", "SOFA"])
        print(f"\nPatient {first_patient_id} time series:")
        print(json.dumps(time_series, indent=2, default=str)[:500] + "...")  # Truncate for readability
        
        # Find similar patients
        similar = patient_tool.find_similar_patients(first_patient_id, n=3)
        print(f"\nPatients similar to {first_patient_id}:")
        print(json.dumps(similar, indent=2, default=str))