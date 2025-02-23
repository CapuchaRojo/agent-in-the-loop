import json
import requests
from collections import defaultdict

class GoalBasedIncidentAgent:
    def __init__(self, knowledge_base, granite_api_key):
        self.knowledge_base = knowledge_base
        self.incident_logs = []
        self.categorized_incidents = defaultdict(list)
        self.granite_api_key = granite_api_key

    def analyze_with_granite(self, log):
        """Sends log data to IBM Granite NLP API for categorization."""
        url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.granite_api_key}"
        }
        payload = {
            "input": f"<|start_of_role|>system<|end_of_role|>Analyze the following IT log and categorize it based on severity level: {log}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 900,
                "min_new_tokens": 0,
                "stop_sequences": [],
                "repetition_penalty": 1
            },
            "model_id": "ibm/granite-3-8b-instruct",
            "project_id": "a9b46767-1ccb-4c23-b7c4-7f89f6fbe8e7"
        }

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result.get("generated_text", "Unknown")
        else:
            return "API Error"

    def parse_logs(self, log_data):
        """Parses IT logs using IBM Granite NLP for enhanced log understanding."""
        for log in log_data:
            category = self.analyze_with_granite(log)  # Get AI-generated category
            suggested_solution = self.knowledge_base.get(category, {}).get("solution", "No predefined solution.")
            self.incident_logs.append((log, category, suggested_solution))
            self.categorized_incidents[category].append(log)

    def diagnose_incidents(self):
        """Generates an incident report with categorized incidents and suggested solutions."""
        return [{"Log": log, "Category": category, "Suggested Solution": solution} 
                for log, category, solution in self.incident_logs]

    def recommend_action(self):
        """Suggests proactive steps based on incident severity and frequency."""
        recommendations = {}
        for category, logs in self.categorized_incidents.items():
            if len(logs) > 3:
                recommendations[category] = "Recurring issue detected. Consider a permanent fix."
        return recommendations

# Sample knowledge base
knowledge_base = {
    "Critical": {"solution": "Restart the server and check logs."},
    "High": {"solution": "Optimize queries or scale database resources."},
    "Medium": {"solution": "Analyze application memory usage and fix leaks."}
}

# Example log data
log_data = [
    "2025-02-22 10:00:01 - Server crashed due to overload.",
    "2025-02-22 10:05:43 - Database timeout on user request.",
    "2025-02-22 10:15:12 - Memory leak detected in process X."
]

# Initialize and run the agent
granite_api_key = "YOUR_IBM_GRANITE_API_KEY"
incident_agent = GoalBasedIncidentAgent(knowledge_base, granite_api_key)

incident_agent.parse_logs(log_data)
incident_report = incident_agent.diagnose_incidents()
action_recommendations = incident_agent.recommend_action()

print("Incident Report:")
print(json.dumps(incident_report, indent=4))

print("\nRecommended Actions:")
print(json.dumps(action_recommendations, indent=4))
