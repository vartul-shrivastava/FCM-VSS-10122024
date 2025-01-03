# FCM-VSS: An AI-Powered Secured Fuzzy Cognitive Maps Management Toolkit

## Authors
- **Vartul Shrivastava**, Research Assistant, IIM Indore, India
- **Dr. Shekhar Shukla**, Faculty, IIM Indore, India

## Overview

**FCM-VSS** (Fuzzy Cognitive Maps - Visualizer, Simulator, Summarizer) is a comprehensive web-based toolkit designed for managing Fuzzy Cognitive Maps (FCMs). The toolkit integrates key features such as AES-GCM encryption for secure data handling, real-time Kosko simulations, and AI-powered inference using locally hosted Ollama models. It supports multiple functionalities like visualization, simulation, and what-if scenario analysis to facilitate a robust, secure, and advanced FCM management environment.

## Key Features
- **AES-GCM Encryption**: Ensures secure project management, preventing unauthorized access to FCM configurations and snapshots.
- **Kosko Simulation**: Provides support for Original Kosko, Modified Kosko, and Rescaled Kosko inference mechanisms for dynamic FCM simulations.
- **AI-Powered Summarization**: Uses AI models hosted via Ollama to generate detailed summaries of FCM configurations based on node statistics, Kosko simulation convergence, and what-if analysis.
- **Checkpoint and Snapshot Management**: Allows users to save, load, and manage FCM configurations and track project progress through checkpoints and snapshots.
- **Interactive Playground**: Enables real-time modifications of FCM networks, including node/edge addition and deletion, with drag functionality and keyboard shortcuts.
- **Customizable Transfer Functions**: Supports multiple transfer functions (Sigmoid, Tanh, Bivalent, Trivalent) and node activation patterns for customizable simulations.

## Installation and Setup

### Prerequisites
- **Python 3.12.3** or above
- **Ollama** for AI-driven summarization functionality
- **Flask 3.0.3** for backend routing and communication
- A modern web browser (Chrome, Firefox, Edge, etc.)
- Other required Python libraries:
  - **NumPy 1.26.4**
  - **Pandas 2.2.3**

### Setup Instructions
1. **Clone the repository**:
    ```bash
    git clone https://github.com/vartul-shrivastava/FCM-VSS-22102024.git
    ```
    
2. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    
3. **Navigate to the app directory**:
    ```bash
    cd FCM-VSS-22102024
    ```
    
4. **Run the application**:
    ```bash
    python app.py
    ```
    This will launch FCM-VSS on your localhost. However, If you wish to access FCM-VSS instantly, here is the link to hosted FCM-VSS: https://vartulshrivastava.pythonanywhere.com/
    (You can access all other features except AI inference capabilities, which will not be accessible in the web-deployed version because FCM-VSS relies on locally installed Ollama models. To access AI models in FCM-VSS, follow the Setup Instructions on your local machine)
    

## Usage
Once the application is running, open your preferred web browser and navigate to the reproduced localhost link in the console to access the FCM-VSS toolkit. From there, you can create, visualize, simulate, and analyze your Fuzzy Cognitive Maps with ease.

## License
This project is licensed under the MIT License.
