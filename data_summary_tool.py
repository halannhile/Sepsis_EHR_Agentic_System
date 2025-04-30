import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json

class DataSummaryTool:
    """
    Tool for generating statistical summaries of the sepsis EHR dataset.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the tool with the path to the dataset.
        
        Args:
            data_path: Path to the CSV file containing the dataset
        """
        self.df = pd.read_csv(data_path)
        # Remove the index column if it exists (first unnamed column)
        if self.df.columns[0] == 'Unnamed: 0':
            self.df = self.df.drop(columns=self.df.columns[0])
        
    def get_basic_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics about the dataset.
        
        Returns:
            Dictionary containing basic statistics about the dataset
        """
        # Get the shape of the dataset
        num_rows, num_cols = self.df.shape
        
        # Get the number of unique patients (icustayid)
        num_patients = self.df['icustayid'].nunique()
        
        # Get the number of unique chartimes per patient (average)
        avg_records_per_patient = self.df.groupby('icustayid').size().mean()
        
        # Get mortality ratio (if mortality_90d exists in the dataset)
        mortality_ratio = None
        if 'mortality_90d' in self.df.columns:
            mortality_ratio = self.df.groupby('icustayid')['mortality_90d'].first().mean()
        
        # Get the percentage of missing values for each column
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100).to_dict()
        
        # Get the percentage of zero values for each column (which might indicate missing data)
        zero_pct = {col: ((self.df[col] == 0).sum() / len(self.df) * 100) 
                   for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])}
        
        # Get the distribution of gender
        gender_dist = self.df['gender'].value_counts(normalize=True).to_dict() if 'gender' in self.df.columns else None
        
        # Get the age distribution
        age_stats = None
        if 'age' in self.df.columns:
            # Convert age from days to years
            age_years = self.df['age'] / 365.25
            age_stats = {
                'mean': age_years.mean(),
                'median': age_years.median(),
                'min': age_years.min(),
                'max': age_years.max(),
                'std': age_years.std()
            }
        
        # Return all statistics
        return {
            'num_rows': num_rows,
            'num_cols': num_cols,
            'num_patients': num_patients,
            'avg_records_per_patient': avg_records_per_patient,
            'mortality_ratio': mortality_ratio,
            'missing_percentage': {k: round(v, 2) for k, v in missing_pct.items()},
            'zero_percentage': {k: round(v, 2) for k, v in zero_pct.items()},
            'gender_distribution': gender_dist,
            'age_stats_years': age_stats
        }
    
    def get_variable_distributions(self, columns: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get distribution statistics for selected columns.
        
        Args:
            columns: List of column names to analyze. If None, uses all numeric columns.
            
        Returns:
            Dictionary with distribution statistics for each column
        """
        if columns is None:
            # Use all numeric columns except for the ID columns and timestamp
            exclude_cols = ['bloc', 'icustayid', 'charttime', 'gender']
            columns = [col for col in self.df.columns if col not in exclude_cols and 
                      pd.api.types.is_numeric_dtype(self.df[col])]
        
        distributions = {}
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                # Filter out NaN values for statistics
                values = self.df[col].dropna()
                
                # Calculate quartiles
                q1, median, q3 = np.percentile(values, [25, 50, 75])
                
                # Calculate statistics
                distributions[col] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'q1': q1,
                    'median': median,
                    'q3': q3,
                    'missing_pct': (self.df[col].isnull().sum() / len(self.df) * 100),
                    'zero_pct': ((values == 0).sum() / len(values) * 100)
                }
        
        return {k: {k2: round(v2, 2) if isinstance(v2, (float, np.float64)) else v2 
                   for k2, v2 in v.items()} 
                for k, v in distributions.items()}
    
    def get_correlation_with_mortality(self) -> Dict[str, float]:
        """
        Get correlation of each feature with mortality_90d.
        
        Returns:
            Dictionary with correlation coefficient for each feature
        """
        if 'mortality_90d' not in self.df.columns:
            return {"error": "mortality_90d column not found in dataset"}
        
        # Get all numeric columns
        numeric_cols = [col for col in self.df.columns if 
                       pd.api.types.is_numeric_dtype(self.df[col]) and 
                       col != 'mortality_90d' and
                       col not in ['bloc', 'icustayid']]
        
        # Calculate correlation with mortality
        correlations = {}
        for col in numeric_cols:
            # Group by patient and get the average value for each feature
            # and the mortality label (which should be the same for all records of a patient)
            patient_df = self.df.groupby('icustayid').agg({
                col: 'mean',
                'mortality_90d': 'first'
            }).dropna()
            
            if len(patient_df) > 0:
                corr = patient_df[col].corr(patient_df['mortality_90d'])
                correlations[col] = round(corr, 3)
        
        # Sort by absolute correlation value
        return {k: v for k, v in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)}
    
    def get_summary_report(self) -> str:
        """
        Generate a comprehensive text summary of the dataset.
        
        Returns:
            Text summary of the dataset
        """
        stats = self.get_basic_stats()
        
        # Start building the report
        report = []
        report.append(f"# Sepsis EHR Dataset Summary Report\n")
        
        # Basic statistics
        report.append(f"## Basic Statistics\n")
        report.append(f"- Dataset contains {stats['num_rows']} total records for {stats['num_patients']} unique patients")
        report.append(f"- Average of {stats['avg_records_per_patient']:.2f} records per patient")
        
        if stats['mortality_ratio'] is not None:
            report.append(f"- 90-day mortality rate: {stats['mortality_ratio']*100:.2f}%")
        
        if stats['gender_distribution']:
            gender_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in stats['gender_distribution'].items()])
            report.append(f"- Gender distribution: {gender_str}")
        
        if stats['age_stats_years']:
            report.append(f"- Age statistics (in years): Mean: {stats['age_stats_years']['mean']:.1f}, "
                         f"Median: {stats['age_stats_years']['median']:.1f}, "
                         f"Range: {stats['age_stats_years']['min']:.1f} to {stats['age_stats_years']['max']:.1f}")
        
        # Features with high missing values
        high_missing = {k: v for k, v in stats['missing_percentage'].items() if v > 20}
        if high_missing:
            report.append("\n## Features with High Missing Values (>20%)")
            for feat, pct in sorted(high_missing.items(), key=lambda x: x[1], reverse=True):
                report.append(f"- {feat}: {pct:.1f}% missing")
        
        # Features with high zero values (potential missing indicators)
        high_zero = {k: v for k, v in stats['zero_percentage'].items() 
                    if v > 30 and k not in ['mortality_90d', 'mechvent']}
        if high_zero:
            report.append("\n## Features with High Zero Values (>30%)")
            for feat, pct in sorted(high_zero.items(), key=lambda x: x[1], reverse=True):
                report.append(f"- {feat}: {pct:.1f}% zeros")
        
        # Correlations with mortality if available
        if 'mortality_90d' in self.df.columns:
            correlations = self.get_correlation_with_mortality()
            if correlations and not isinstance(correlations, dict) and 'error' in correlations:
                top_positive = {k: v for k, v in correlations.items() if v > 0.15}
                top_negative = {k: v for k, v in correlations.items() if v < -0.15}
                
                if top_positive:
                    report.append("\n## Features Most Positively Correlated with Mortality")
                    for feat, corr in sorted(top_positive.items(), key=lambda x: x[1], reverse=True)[:10]:
                        report.append(f"- {feat}: {corr:.3f}")
                
                if top_negative:
                    report.append("\n## Features Most Negatively Correlated with Mortality")
                    for feat, corr in sorted(top_negative.items(), key=lambda x: x[1])[:10]:
                        report.append(f"- {feat}: {corr:.3f}")
        
        return "\n".join(report)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert all summary statistics to a dictionary for API response.
        
        Returns:
            Dictionary containing all summary statistics
        """
        return {
            "basic_stats": self.get_basic_stats(),
            "variable_distributions": self.get_variable_distributions(),
            "correlation_with_mortality": self.get_correlation_with_mortality() if 'mortality_90d' in self.df.columns else None,
            "summary_report": self.get_summary_report()
        }

    def to_json(self) -> str:
        """
        Convert all summary statistics to a JSON string for API response.
        
        Returns:
            JSON string containing all summary statistics
        """
        return json.dumps(self.to_dict(), default=str, indent=2)


# Example usage (would be wrapped in API endpoint)
if __name__ == "__main__":
    # Replace with your actual data path
    data_path = "./data/AI_agent_train_sepsis.csv"
    
    # Initialize the tool
    summary_tool = DataSummaryTool(data_path)
    
    # Get basic statistics
    basic_stats = summary_tool.get_basic_stats()
    print("Basic Statistics:")
    print(json.dumps(basic_stats, indent=2, default=str))
    
    # Get variable distributions
    distributions = summary_tool.get_variable_distributions(["GCS", "HR", "SysBP", "SpO2", "Temp_C"])
    print("\nVariable Distributions:")
    print(json.dumps(distributions, indent=2))
    
    # Get correlation with mortality
    correlations = summary_tool.get_correlation_with_mortality()
    print("\nCorrelations with Mortality:")
    print(json.dumps(correlations, indent=2))
    
    # Generate summary report
    report = summary_tool.get_summary_report()
    print("\nSummary Report:")
    print(report)