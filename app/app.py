from flask import Flask, render_template, request, jsonify
import ollama
import subprocess

app = Flask(__name__)
# Global variable to store selected model name
ollama_model = ''

# Function to check if Ollama is running
def is_ollama_running():
    try:
        # Check if Ollama is running using subprocess
        subprocess.run(["ollama", "ps"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False

# Route to check AI readiness and available models
@app.route('/check_ai_readiness', methods=['GET'])
def check_ai_readiness():
    ollama_ready = is_ollama_running()

    models = []
    if ollama_ready:
        try:
            # Fetch available models from Ollama
            model_data = ollama.list()
            # Extract model names from the list
            models = [model['name'] for model in model_data['models']]
        except Exception as e:
            return jsonify({"ollama_ready": False, "error": str(e)})

    return jsonify({"ollama_ready": ollama_ready, "models": models})

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/summarize_fcm', methods=['POST'])
def summarize_fcm():
    try:
        # Get the JSON data from the POST request
        data = request.get_json()
        
        # Extract node stats, kosko results (full table), and fixation results (full table)
        node_stats = data.get('node_stats')
        kosko_results = data.get('kosko_results')
        fixation_results = data.get('fixation_results')

        # Extract node names from the keys of kosko/fixation rows (assuming node names are present)
        node_names = [key for key in kosko_results[0].keys() if key != 'iteration']  # Extracting node names (excluding 'iteration')

        # Get the last row of Kosko results
        last_kosko_row = kosko_results[-1] if kosko_results else {}

        # Get the last row of Fixation results
        last_fixation_row = fixation_results[-1] if fixation_results else {}

        # Prepare rows with node names appended for better readability in LLM
        def format_row_with_node_names(row_data, node_names):
            formatted_row = []
            for node in node_names:
                formatted_row.append(f"{node}: {row_data[node]}")
            return ", ".join(formatted_row)

        # Format last rows for Kosko and Fixation results
        formatted_kosko_row = format_row_with_node_names(last_kosko_row, node_names)
        formatted_fixation_row = format_row_with_node_names(last_fixation_row, node_names)

        # Generate a more structured and comprehensive prompt for summarization
        fcm_content = (
            "Please provide a detailed analysis of the Fuzzy Cognitive Map (FCM) data in the following sections. Ensure that the output is well-structured and leaves adequate spacing between each section for clarity:\n\n"
            
            "1. **Node Statistics Analysis**:\n"
            f"{node_stats}\n\n"
            "Without explaining theoretical concepts, analyze each node's indegree, outdegree, and centrality values. Focus on the quantitative roles these nodes play in the FCM and how their metrics influence network dynamics.\n\n"
            
            "2. **Kosko Simulation Results (Last Row)**:\n"
            f"{formatted_kosko_row}\n\n"
            "Provide a quantitative analysis of the convergence values from the Kosko simulation. Highlight the system's stabilization and discuss how node interactions reflect these behaviors. Do not reiterate indegree or outdegree concepts.\n\n"
            
            "3. **Impact of Fixation on Convergence (Last Row, Difference in Convergence Values)**:\n"
            f"{formatted_fixation_row}\n\n"
            "Discuss how the fixation of specific nodes influenced the convergence results. Focus on changes in node activations and system stability compared to the Kosko simulation. Provide a quantitative comparison without repeating indegree or outdegree details.\n\n"
            
            "Ensure the response is well spaced and formatted for readability. AND write your understanding, dont rewrite tables and rows. Dont use ** in writing."
        )

        # Call the Ollama chat API with the structured FCM data
        response = ollama.chat(
            model=ollama_model,
            messages=[{'role': 'user', 'content': fcm_content}]
        )

        # Retrieve the summary from the response
        summary = response['message']['content']

        # Return the summary as a JSON response
        return jsonify({'summary': summary})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Function to get available models from Ollama
def get_available_models():
    try:
        # Fetch available models from Ollama
        model_data = ollama.list()
        # Extract model names from the list
        models = [model['name'] for model in model_data['models']]
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

