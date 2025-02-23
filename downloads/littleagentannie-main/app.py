import json
import os
from collections import defaultdict
from dotenv import load_dotenv
from ibm_watson_machine_learning.foundation_models import Model  # Correct IBM Granite API import

# Load environment variables
load_dotenv()
granite_api_key = os.getenv("IBM_GRANITE_API_KEY")

# Set up IBM Watson credentials
credentials = {
    "apikey": granite_api_key,
    "url": "https://us-south.ml.cloud.ibm.com"  # Update if using a different region
}

class GoalBasedIncidentAgent:
    def __init__(self, knowledge_base, credentials):
        self.knowledge_base = knowledge_base
        self.incident_logs = []
        self.categorized_incidents = defaultdict(list)
        
        # Initialize IBM Granite Model correctly
        self.granite_model = Model(auth_credentials=credentials)

    def parse_logs(self, log_data):
        """Parses IT logs using IBM Granite NLP to categorize logs and suggest solutions."""
        for log in log_data:
            try:
                response = self.granite_model.generate(
                    prompt=f"Classify this log: {log}", 
                    params={"decoding_method": "greedy", "max_new_tokens": 50}
                )
                response_text = response.get("results", [{}])[0].get("generated_text", "").strip()
                category = self.classify_log(response_text)

                suggested_solution = self.knowledge_base.get(category, {}).get("solution", "No predefined solution.")
                self.incident_logs.append((log, category, suggested_solution))
                self.categorized_incidents[category].append(log)

            except Exception as e:
                print(f"Error processing log: {log}\n{str(e)}")

    def classify_log(self, response_text):
        """Classifies logs based on AI-generated response."""
        response_text = response_text.lower()
        if "crash" in response_text or "failure" in response_text:
            return "Critical"
        elif "timeout" in response_text or "latency" in response_text:
            return "High"
        elif "memory leak" in response_text or "slow" in response_text:
            return "Medium"
        return "Unknown"

    def diagnose_incidents(self):
        """Generates an incident report with categorized logs and suggested solutions."""
        return [
            {"Log": log, "Category": category, "Suggested Solution": solution}
            for log, category, solution in self.incident_logs
        ]

    def recommend_action(self):
        """Suggests proactive steps based on incident frequency."""
        recommendations = {}
        for category, logs in self.categorized_incidents.items():
            if len(logs) > 3:
                recommendations[category] = f"Recurring {category} issues detected. Consider a permanent fix."
        return recommendations

# Sample knowledge base with predefined solutions
knowledge_base = {
    "Critical": {"solution": "Restart the server and check system logs."},
    "High": {"solution": "Optimize database queries and scale infrastructure."},
    "Medium": {"solution": "Analyze memory usage and fix memory leaks."}
}

# Example log data
log_data = [
    "2025-02-22 10:00:01 - Server crashed due to overload.",
    "2025-02-22 10:05:43 - Database timeout on user request.",
    "2025-02-22 10:15:12 - Memory leak detected in process X.",
    "2025-02-22 10:30:00 - Web server experienced high latency.",
    "2025-02-22 10:45:00 - Another database timeout detected."
]

# Initialize and run the agent
incident_agent = GoalBasedIncidentAgent(knowledge_base, credentials)
incident_agent.parse_logs(log_data)
incident_report = incident_agent.diagnose_incidents()
action_recommendations = incident_agent.recommend_action()

# Print results
print("Incident Report:")
print(json.dumps(incident_report, indent=4))
print("\nRecommended Actions:")
print(json.dumps(action_recommendations, indent=4))
