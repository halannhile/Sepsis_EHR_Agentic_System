"""
HTML templates for the XAI tool and explanation components.
This file contains all HTML templates used to generate reports and visualizations.
Separating these templates avoids f-string syntax errors in the main code.
"""

def get_explanation_html_template() -> str:
    """
    Get the HTML template for explanation reports.
    
    Returns:
        HTML template string with placeholders for dynamic content
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Mortality Prediction Explanation for Patient {patient_id}</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .plot-container {
                margin: 20px 0;
            }
            .feature-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            .feature-table th, .feature-table td {
                padding: 8px 12px;
                border: 1px solid #ddd;
                text-align: left;
            }
            .feature-table th {
                background-color: #f2f2f2;
            }
            .positive {
                color: #e74c3c;
            }
            .negative {
                color: #2ecc71;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Mortality Prediction Explanation</h1>
            <h2>Patient ID: {patient_id}</h2>
            
            <!-- Summary Section -->
            <div class="summary-section">
                {summary}
            </div>
            
            <!-- SHAP Plots Section -->
            <h2>Visualization of Prediction Factors</h2>
            
            {waterfall_plot}
            
            {force_plot}
            
            <!-- Feature Table Section -->
            <h2>Detailed Feature Contributions</h2>
            <table class="feature-table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Value</th>
                        <th>Impact</th>
                        <th>Direction</th>
                    </tr>
                </thead>
                <tbody>
                    {feature_rows}
                </tbody>
            </table>
            
            <div class="disclaimer">
                <h3>Disclaimer</h3>
                <p>This prediction is based on statistical analysis and should be used as a decision support tool only. Clinical judgment should always take precedence.</p>
            </div>
        </div>
    </body>
    </html>
    """