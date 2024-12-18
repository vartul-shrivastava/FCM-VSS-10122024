from flask import Flask, render_template, request, jsonify
import ollama
import subprocess
import re

app = Flask(__name__)
# Global variable to store selected model name
ollama_model = ''
node_stats = ''
kosko_results = ''
fixation_results = ''
last_kosko_row_sorted = ''
last_fixation_row = ''

import subprocess

def is_ollama_running():
    try:
        subprocess.run(
            ["ollama"],
            check=True,
            stdout=subprocess.DEVNULL,  # Suppress stdout
            stderr=subprocess.DEVNULL,  # Suppress stderr
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# Route to check AI readiness and available models
@app.route('/check_ai_readiness', methods=['GET'])
def check_ai_readiness():
    if not is_ollama_running():
        return jsonify({"ollama_ready": False, "models": [], "error": "Ollama is not running or not found in PATH."})

    try:
        # Fetch available models from Ollama
        model_data = str(ollama.list())  # Assume this returns the list of Model objects
        # Regular expression to match the model name
        
        pattern = r"model='(.*?)'"  # Captures content between model=' and '
        # Use re.findall to extract all matches
        models = re.findall(pattern, model_data)
        models = [name for name in models if name.strip()]  # Strip whitespace and filter out empty strings

        print(models)
        return jsonify({"ollama_ready": True, "models": models})
    except Exception as e:
        return jsonify({"ollama_ready": True, "models": [], "error": f"Error fetching Ollama models: {e}"})

fixed_nodes = []  # Initialize the global variable
fixed_node_values = []  # Initialize the global variable

@app.route('/push_fixed_nodes', methods=['POST'])
def push_fixed_nodes():
    global fixed_nodes, fixed_node_values  # Declare globals

    try:
        # Get the JSON data sent from JavaScript
        data = request.get_json()
        fixed_nodes = data.get('fixedNodes', [])  # Update global variables
        fixed_node_values = data.get('fixedNodeValues', [])

        # Log the data
        print('Fixed Nodes:', fixed_nodes)
        print('Fixed Node Values:', fixed_node_values)

        # Return a success response
        return jsonify({'status': 'success', 'message': 'Data received successfully!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/')
def home():
    return render_template('base.html')

import traceback  # For detailed error tracebacks

def process_last_fixation_row(last_fixation_row):
    # Threshold for effectively zero values
    zero_threshold = 1e-3

    # Filter and round non-zero values
    processed_row = {
        key: round(float(value), 3)
        for key, value in last_fixation_row.items()
        if key != 'iteration'  # Exclude the 'iteration' key
        and abs(float(value)) > zero_threshold  # Filter out effectively zero values
    }

    return processed_row

# Global variables to store the custom and default prompts
custom_prompt = None
default_prompt = """
Please provide a detailed analysis of the Fuzzy Cognitive Map (FCM) data in the following sections:

1. Driver Nodes:
Summarize the driver nodes, their roles, and contributions based on outdegree and centrality values. Key details include:
- Nodes: {driver_summary[nodes]}
- Outdegree (Total/Average): {driver_summary[outdegree_summary]}
- Centrality (Total/Average): {driver_summary[centrality_summary]}

2. Intermediate Nodes:
Provide insights into the intermediate nodes and their dual roles based on indegree, outdegree, and centrality values:
- Nodes: {intermediate_summary[nodes]}
- Indegree (Total/Average): {intermediate_summary[indegree_summary]}
- Outdegree (Total/Average): {intermediate_summary[outdegree_summary]}
- Centrality (Total/Average): {intermediate_summary[centrality_summary]}

3. Receiver Nodes:
Discuss the receiver nodes and their dependencies based on indegree and centrality values:
- Nodes: {receiver_summary[nodes]}
- Indegree (Total/Average): {receiver_summary[indegree_summary]}
- Centrality (Total/Average): {receiver_summary[centrality_summary]}

4. Kosko Simulation Results:
Present the last row of Kosko simulation sorted in descending order of values:
{last_kosko_row}

5. What-if Analysis:
Discuss the impact of fixation and their respective results, and how each node is impacted because of tweaked nodes:
- Tweaked Nodes and Values: {fixed_node_values}
- Impacted Nodes and Values: {last_fixation_row}

Explain the implications of these results in terms of beneficial (positive differences) or adverse (negative differences) impacts.

Ensure the response is structured and formatted for clarity, avoiding unnecessary table or row rewrites.
"""

@app.route('/summarize_fcm', methods=['POST'])
def summarize_fcm():
    try:
        # Get JSON data from the POST request
        data = request.get_json()

        # Extract necessary components
        node_stats = data.get('node_stats', [])
        kosko_results = data.get('kosko_results', [])
        fixation_results = data.get('fixation_results', [])

        # Helper function to round values
        def round_values(data):
            if isinstance(data, dict):
                return {key: round(value, 3) if isinstance(value, (int, float)) else value for key, value in data.items()}
            elif isinstance(data, list):
                return [round_values(item) for item in data]
            return data

        # Process data
        kosko_results = round_values(kosko_results)
        fixation_results = round_values(fixation_results)

        # Categorize nodes based on their type
        drivers = [node for node in node_stats if node['type'].lower() == 'driver']
        intermediates = [node for node in node_stats if node['type'].lower() == 'intermediate']
        receivers = [node for node in node_stats if node['type'].lower() == 'receiver']

        # Helper function to compute summaries
        def compute_summary(nodes, key):
            values = [(node['node'], float(node[key])) for node in nodes if key in node]
            total = sum(value for _, value in values)
            average = round(total / len(values), 3) if values else 0
            max_node = max(values, key=lambda x: x[1], default=(None, None))
            min_node = min(values, key=lambda x: x[1], default=(None, None))
            return {
                'average': round(average, 3),
                'max_node': max_node,
                'min_node': min_node
            }

        # Compute summaries for each category
        driver_summary = {
            'count': len(drivers),
            'nodes': [node['node'] for node in drivers],
            'outdegree_summary': compute_summary(drivers, 'outdegree_value'),
            'centrality_summary': compute_summary(drivers, 'centrality_value')
        }

        intermediate_summary = {
            'count': len(intermediates),
            'nodes': [node['node'] for node in intermediates],
            'indegree_summary': compute_summary(intermediates, 'indegree_value'),
            'outdegree_summary': compute_summary(intermediates, 'outdegree_value'),
            'centrality_summary': compute_summary(intermediates, 'centrality_value')
        }

        receiver_summary = {
            'count': len(receivers),
            'nodes': [node['node'] for node in receivers],
            'indegree_summary': compute_summary(receivers, 'indegree_value'),
            'centrality_summary': compute_summary(receivers, 'centrality_value')
        }

        # Format placeholders
        placeholders = {
            "driver_summary": driver_summary,
            "intermediate_summary": intermediate_summary,
            "receiver_summary": receiver_summary,
        }

        # Last rows for Kosko and fixation
        last_kosko_row = kosko_results[-1] if kosko_results else {}
        last_fixation_row = fixation_results[-1] if fixation_results else {}

        # Remove tweaked nodes from fixation row
        global fixed_nodes, fixed_node_values
        fixed_nodes = fixed_nodes if 'fixed_nodes' in globals() and fixed_nodes else []
        fixed_node_values = fixed_node_values if 'fixed_node_values' in globals() and fixed_node_values else []
        fixed_nodes_lower = [node.lower() for node in fixed_nodes]
        last_fixation_row = {k: v for k, v in last_fixation_row.items() if k.lower() not in fixed_nodes_lower}
        last_fixation_row = process_last_fixation_row(last_fixation_row)
        
        placeholders.update({
            "last_kosko_row": last_kosko_row,
            "last_fixation_row": last_fixation_row,
            "fixed_node_values": fixed_node_values
        })

        # Use the custom or default prompt and evaluate variables dynamically
        prompt_template = custom_prompt or default_prompt
        try:
            prompt = prompt_template.format(**placeholders)  # Use unpacking for dynamic values
        except KeyError as e:
            raise ValueError(f"Invalid placeholder in prompt template: {e}")

        print(prompt)  # Log prompt for debugging

        # Simulated call to LLM or external summarization
        response = ollama.chat(model=ollama_model, messages=[{'role': 'user', 'content': prompt}])

        # Return the summary
        summary = response['message']['content']
        return jsonify({'summary': summary})

    except Exception as e:
        error_trace = traceback.format_exc()
        print("Error in /summarize_fcm:", error_trace)
        return jsonify({'error': str(e), 'traceback': error_trace}), 500


@app.route('/reset_prompt', methods=['POST'])
def reset_prompt():
    """Reset the custom prompt to the default prompt."""
    global custom_prompt
    try:
        custom_prompt = None  # Clear the custom prompt
        return jsonify({
            'status': 'success',
            'message': 'Prompt reset to default successfully.',
            'default_prompt': default_prompt  # Include the default prompt in the response
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/get_current_prompt', methods=['GET'])
def get_current_prompt():
    """Retrieve the current prompt (custom or default)."""
    global custom_prompt
    try:
        current_prompt = custom_prompt or default_prompt
        return jsonify({'prompt': current_prompt})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
@app.route('/update_prompt', methods=['POST'])
def update_prompt():
    """Update the custom prompt."""
    global custom_prompt
    data = request.get_json()
    custom_prompt = data.get('prompt')
    return jsonify({'status': 'success', 'message': 'Prompt updated successfully'})

# Function to get available models from Ollama
def get_available_models():
    try:
        # Fetch available models from Ollama
        model_data = str(ollama.list())
        # Extract model names from the list
        pattern = r"model='(.*?)'"  # Captures content between model=' and '
        # Use re.findall to extract all matches
        models = re.findall(pattern, model_data)
        models = [name for name in models if name.strip()]  # Strip whitespace and filter out empty strings
        return models
    except Exception as e:
        return []

# Route to get available models
@app.route('/get_available_models', methods=['GET'])
def get_models():
    try:
        models = get_available_models()
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to set the selected model name
@app.route('/set_model', methods=['POST'])
def set_model():
    global ollama_model  # Use the global variable
    data = request.get_json()
    ollama_model = data.get('model_name')  # Store the selected model name
    print(ollama_model)
    return jsonify({"model_name": ollama_model})

if __name__ == '__main__':
    app.run(debug=True)

