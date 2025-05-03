from flask import Flask, render_template, request, jsonify
import requests
import json
import os
import re
import markdown

app = Flask(__name__)

# Configuration
API_URL = "http://localhost:8000"  # URL of the backend API

@app.route('/')
def index():
    """
    Render the main chat interface
    """
    return render_template('index.html')

@app.route('/api/patient_ids', methods=['GET'])
def get_patient_ids():
    """
    Get list of patient IDs from the backend
    """
    try:
        response = requests.get(f"{API_URL}/patient_ids")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/process_instruction', methods=['POST'])
def process_instruction():
    """
    Send user instruction to the backend for processing
    """
    try:
        data = request.json
        
        # Prepare request body
        request_body = {
            "instruction": data["instruction"]
        }
        
        # Add patient_id if provided
        if "patient_id" in data and data["patient_id"]:
            request_body["patient_id"] = int(data["patient_id"])
        
        # Add context if provided
        if "context" in data and data["context"]:
            request_body["context"] = data["context"]
        
        # Send request to backend
        response = requests.post(
            f"{API_URL}/process_instruction",
            json=request_body
        )
        
        result = response.json()
        
        # Convert any markdown in the response to HTML
        if "report" in result and result["report"]:
            result["report_html"] = markdown.markdown(result["report"])
        
        if "explanation" in result and result["explanation"]:
            result["explanation_html"] = markdown.markdown(result["explanation"])
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/html_report/<int:patient_id>', methods=['GET'])
def get_html_report(patient_id):
    """
    Get HTML report for a specific patient
    """
    try:
        response = requests.get(f"{API_URL}/html_report/{patient_id}")
        return response.text
    except Exception as e:
        return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"

# Templates directory
os.makedirs('templates', exist_ok=True)

# Create index.html template
index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sepsis EHR AI Agent</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 1200px;
            padding: 20px;
        }
        .chat-container {
            height: calc(100vh - 220px);
            overflow-y: auto;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            background-color: white;
            margin-bottom: 20px;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 8px;
            max-width: 80%;
        }
        .user-message {
            background-color: #007bff;
            color: white;
            margin-left: auto;
        }
        .agent-message {
            background-color: #f1f1f1;
            color: #333;
        }
        .input-group {
            margin-top: 10px;
        }
        .sidebar {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #dee2e6;
        }
        .patient-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .patient-item {
            cursor: pointer;
            padding: 8px 12px;
            border-radius: 4px;
        }
        .patient-item:hover {
            background-color: #f1f1f1;
        }
        .patient-item.active {
            background-color: #007bff;
            color: white;
        }
        .report-container {
            background-color: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }
        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 2s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row mt-3">
            <div class="col-12">
                <h1 class="text-center">Sepsis EHR AI Agent</h1>
                <p class="text-center">An AI agent for analyzing electronic health records and predicting sepsis outcomes</p>
            </div>
        </div>
        
        <div class="row mt-3">
            <!-- Sidebar with patient list -->
            <div class="col-md-3">
                <div class="sidebar">
                    <h4>Patient Selection</h4>
                    <div class="input-group mb-3">
                        <input type="text" id="patient-search" class="form-control" placeholder="Search patient ID">
                        <button class="btn btn-outline-secondary" type="button" id="patient-search-btn">Search</button>
                    </div>
                    <div id="patient-list" class="patient-list">
                        <div class="loader" id="patient-loader"></div>
                        <!-- Patient list will be populated here -->
                    </div>
                    <div class="mt-3">
                        <h5>Current Patient</h5>
                        <div id="current-patient">None selected</div>
                    </div>
                    <div class="mt-3">
                        <h5>Quick Commands</h5>
                        <button class="btn btn-sm btn-outline-primary mb-1 w-100" onclick="sendQuickCommand('Give me a summary of the dataset')">Dataset Summary</button>
                        <button class="btn btn-sm btn-outline-primary mb-1 w-100" onclick="sendQuickCommand('Get patient summary')">Patient Summary</button>
                        <button class="btn btn-sm btn-outline-primary mb-1 w-100" onclick="sendQuickCommand('Impute missing values')">Impute Missing Values</button>
                        <button class="btn btn-sm btn-outline-primary mb-1 w-100" onclick="sendQuickCommand('Predict mortality')">Predict Mortality</button>
                        <button class="btn btn-sm btn-outline-primary mb-1 w-100" onclick="sendQuickCommand('Explain the prediction')">Explain Prediction</button>
                        <button class="btn btn-sm btn-outline-primary mb-1 w-100" onclick="sendQuickCommand('Show detailed HTML report')">Detailed HTML Report</button>
                    </div>
                </div>
            </div>
            
            <!-- Main chat area -->
            <div class="col-md-9">
                <div class="chat-container" id="chat-container">
                    <!-- Initial welcome message -->
                    <div class="message agent-message">
                        <strong>AI Agent:</strong> Welcome to the Sepsis EHR Analysis Agent. I can help you analyze patient data, predict mortality risk, and explain those predictions. You can select a patient from the list, use the quick command buttons, or type your own questions in natural language below.
                    </div>
                </div>
                
                <!-- Input area -->
                <div class="input-group">
                    <input type="text" id="user-input" class="form-control" placeholder="Type your message here...">
                    <button class="btn btn-primary" type="button" id="send-btn">Send</button>
                </div>
                
                <!-- Report area -->
                <div class="report-container" id="report-container" style="display: none;">
                    <h4 id="report-title">Report</h4>
                    <div id="report-content"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Global variables
        let currentPatientId = null;
        let patientIds = [];
        let chatHistory = [];
        
        // DOM elements
        const chatContainer = document.getElementById('chat-container');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const patientList = document.getElementById('patient-list');
        const patientLoader = document.getElementById('patient-loader');
        const currentPatientElement = document.getElementById('current-patient');
        const patientSearchInput = document.getElementById('patient-search');
        const patientSearchBtn = document.getElementById('patient-search-btn');
        const reportContainer = document.getElementById('report-container');
        const reportTitle = document.getElementById('report-title');
        const reportContent = document.getElementById('report-content');
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            // Load patient IDs
            loadPatientIds();
            
            // Add event listeners
            sendBtn.addEventListener('click', sendMessage);
            userInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            patientSearchBtn.addEventListener('click', searchPatient);
            patientSearchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    searchPatient();
                }
            });
        });
        
        // Function to load patient IDs
        function loadPatientIds() {
            fetch('/api/patient_ids')
                .then(response => response.json())
                .then(data => {
                    patientLoader.style.display = 'none';
                    
                    if (data.error) {
                        patientList.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                        return;
                    }
                    
                    patientIds = data.patient_ids || [];
                    displayPatientIds(patientIds);
                })
                .catch(error => {
                    patientLoader.style.display = 'none';
                    patientList.innerHTML = `<div class="alert alert-danger">Error loading patient IDs: ${error}</div>`;
                });
        }
        
        // Function to display patient IDs
        function displayPatientIds(ids) {
            patientList.innerHTML = '';
            
            if (ids.length === 0) {
                patientList.innerHTML = '<div>No patients found</div>';
                return;
            }
            
            ids.forEach(id => {
                const element = document.createElement('div');
                element.className = 'patient-item';
                if (id === currentPatientId) {
                    element.className += ' active';
                }
                element.textContent = `Patient ${id}`;
                element.addEventListener('click', () => selectPatient(id));
                patientList.appendChild(element);
            });
        }
        
        // Function to search for a patient
        function searchPatient() {
            const searchTerm = patientSearchInput.value.trim();
            
            if (!searchTerm) {
                displayPatientIds(patientIds);
                return;
            }
            
            // If search term is a number, try to find exact match
            if (/^\d+$/.test(searchTerm)) {
                const id = parseInt(searchTerm);
                if (patientIds.includes(id)) {
                    selectPatient(id);
                    const filteredIds = [id];
                    displayPatientIds(filteredIds);
                    return;
                }
            }
            
            // Filter patient IDs that contain the search term
            const filteredIds = patientIds.filter(id => 
                id.toString().includes(searchTerm)
            );
            
            displayPatientIds(filteredIds);
        }
        
        // Function to select a patient
        function selectPatient(id) {
            currentPatientId = id;
            currentPatientElement.textContent = `Patient ${id}`;
            
            // Update patient list to show active patient
            const patientItems = document.querySelectorAll('.patient-item');
            patientItems.forEach(item => {
                if (item.textContent === `Patient ${id}`) {
                    item.className = 'patient-item active';
                } else {
                    item.className = 'patient-item';
                }
            });
            
            // Add system message about selected patient
            addMessage('system', `Patient ${id} selected. You can now ask questions about this patient.`);
            
            // Hide any open reports
            reportContainer.style.display = 'none';
        }
        
        // Function to send a message
        function sendMessage() {
            const message = userInput.value.trim();
            
            if (!message) {
                return;
            }
            
            // Add user message to chat
            addMessage('user', message);
            
            // Clear input
            userInput.value = '';
            
            // Show loader
            const loaderId = addLoader();
            
            // Prepare request data
            const requestData = {
                instruction: message
            };
            
            // Add patient ID if selected
            if (currentPatientId !== null) {
                requestData.patient_id = currentPatientId;
            }
            
            // Send request to backend
            fetch('/api/process_instruction', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            })
            .then(response => response.json())
            .then(data => {
                // Remove loader
                removeLoader(loaderId);
                
                // Handle error
                if (data.error) {
                    addMessage('agent', `Error: ${data.error}`);
                    return;
                }
                
                // Add agent message
                addMessage('agent', data.message);
                
                // Handle report if present
                if (data.report_html || data.report) {
                    reportTitle.textContent = 'Report';
                    reportContent.innerHTML = data.report_html || data.report;
                    reportContainer.style.display = 'block';
                } else if (data.explanation_html || data.explanation) {
                    reportTitle.textContent = 'Explanation';
                    reportContent.innerHTML = data.explanation_html || data.explanation;
                    reportContainer.style.display = 'block';
                } else {
                    reportContainer.style.display = 'none';
                }
                
                // Check if this is a command to show HTML report
                if (message.toLowerCase().includes('html report') && currentPatientId !== null) {
                    openHtmlReport(currentPatientId);
                }
                
                // Update current patient ID if one was returned in the response
                if (data.data && data.data.patient_id && data.data.patient_id !== currentPatientId) {
                    selectPatient(data.data.patient_id);
                }
            })
            .catch(error => {
                // Remove loader
                removeLoader(loaderId);
                
                // Add error message
                addMessage('agent', `Error: ${error.message}`);
            });
        }
        
        // Function to send a quick command
        function sendQuickCommand(command) {
            // For dataset summary specifically, we need to bypass the current patient ID
            if (command === "Give me a summary of the dataset" || command === "Dataset Summary") {
                // Create a direct request to get dataset summary without patient ID
                fetch('/api/process_instruction', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    // Important: Do NOT include patient_id here
                    body: JSON.stringify({
                        instruction: "Give me a summary of the entire dataset"
                    })
                })
                .then(response => response.json())
                .then(data => {
                    // Handle response as usual
                    if (data.error) {
                        addMessage('agent', `Error: ${data.error}`);
                        return;
                    }
                    
                    // Add agent message
                    addMessage('agent', data.message);
                    
                    // Handle report if present
                    if (data.report_html || data.report) {
                        reportTitle.textContent = 'Dataset Report';
                        reportContent.innerHTML = data.report_html || data.report;
                        reportContainer.style.display = 'block';
                    }
                })
                .catch(error => {
                    addMessage('agent', `Error: ${error.message}`);
                });
                
                // Add user message to chat
                addMessage('user', "Give me a summary of the dataset");
                
                // Clear input
                userInput.value = '';
            } else {
                // For all other commands, use the regular flow
                userInput.value = command;
                sendMessage();
            }
        }
                
        // Function to add a message to the chat
        function addMessage(sender, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            
            if (sender === 'user') {
                messageDiv.innerHTML = `<strong>You:</strong> ${content}`;
            } else if (sender === 'agent') {
                messageDiv.innerHTML = `<strong>AI Agent:</strong> ${content}`;
            } else if (sender === 'system') {
                messageDiv.innerHTML = `<strong>System:</strong> ${content}`;
                messageDiv.style.backgroundColor = '#d1ecf1';
                messageDiv.style.color = '#0c5460';
            }
            
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            // Add to history
            chatHistory.push({
                sender: sender,
                content: content,
                timestamp: new Date().toISOString()
            });
        }
        
        // Function to add a loader to the chat
        function addLoader() {
            const id = Date.now().toString();
            const loaderDiv = document.createElement('div');
            loaderDiv.id = `loader-${id}`;
            loaderDiv.className = 'loader';
            
            chatContainer.appendChild(loaderDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            return id;
        }
        
        // Function to remove a loader from the chat
        function removeLoader(id) {
            const loader = document.getElementById(`loader-${id}`);
            if (loader) {
                loader.remove();
            }
        }
        
        // Function to open HTML report in a new window
        function openHtmlReport(patientId) {
            window.open(`/api/html_report/${patientId}`, '_blank');
        }
    </script>
</body>
</html>
"""

with open('templates/index.html', 'w') as f:
    f.write(index_html)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)