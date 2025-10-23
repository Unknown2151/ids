# Explainable AI Intrusion Detection System (XAI-IDS)

This project implements an Intrusion Detection System (IDS) using a hybrid CNN-LSTM deep learning model combined with SHAP (SHapley Additive exPlanations) for explainability. It addresses the challenges of class imbalance in network traffic datasets and the opacity of deep learning models.

**Developed by:** Sushminthiran S (211423104667)
**Guide:** Mr. Sasikumar A.N M.E.
**Institution:** Panimalar Engineering College

## Features

* **Hybrid CNN-LSTM Model:** For accurate detection of network intrusions based on spatial and temporal features.
* **Class Imbalance Handling:** Uses SMOTE and Random Undersampling to improve detection of rare attack types.
* **Explainable AI (XAI):** Integrates SHAP to provide clear, feature-based explanations for detected attacks.
* **Streamlit Web Interface:** Provides an interactive dashboard for both offline simulation and live network monitoring.
* **Live Monitoring:** Captures and analyzes network traffic in real-time using Scapy (requires administrator privileges).
* **Persistent Logging:** Saves detected alerts to a CSV file.

## Project Structure

```
Explainable_AI_IDS_Project/
│
├── src/                      # Source code, data, and model
│   ├── app.py                # Main Streamlit application
│   ├── config.py             # Configuration settings
│   ├── utils.py              # Helper functions
│   ├── requirements.txt      # Python dependencies
│   ├── data/                 # CIC-IDS2017 dataset CSVs
│   └── saved_model/          # Trained Keras model (.h5)
│
├── docs/                     # Documentation and papers
│   ├── Project_Report.docx   # Full project report
│   ├── Final_Viva_PPT.pptx   # Presentation slides
│   ├── Journal_Paper.docx    # IEEE-style paper
│   ├── One_Page_Abstract.docx # Single-page abstract
│
├── demo_video/               # Demo video folder
│   ├── Project_Demo.mp4      # Recorded project demo
│
└── README.md                 # This file
```

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-link>
    cd Explainable_AI_IDS_Project
    ```

2.  **Create a Python Virtual Environment:**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment:**
    * Windows (Command Prompt/PowerShell): `.\.venv\Scripts\activate`
    * macOS/Linux (Bash/Zsh): `source .venv/bin/activate`

4.  **Install Dependencies:**
    ```bash
    pip install -r src/requirements.txt
    ```

5.  **Install Npcap (for Live Monitoring on Windows):**
    * Download and install Npcap from [https://npcap.com/](https://npcap.com/).
    * **Important:** During installation, ensure the option "Install Npcap in WinPcap API-compatible Mode" is **CHECKED** if using older Scapy versions, or try with it **UNCHECKED** if encountering issues (as per our troubleshooting). *Administrator privileges are required for installation.*

## How to Run

1.  **Activate the Virtual Environment** (if not already active).

2.  **Navigate to the `src` directory:**
    ```bash
    cd src
    ```

3.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    * The application will open in your web browser.
    * Use the sidebar to switch between "Offline Simulation" and "Live Monitoring".

4.  **For Live Monitoring:**
    * You **MUST** run the `streamlit run app.py` command from a terminal launched with **Administrator privileges** (Windows) or `sudo` (macOS/Linux).
    * Select the correct network interface in the Streamlit sidebar.
    * Click "Start Monitoring". Alerts will appear in the "Live Alert Log".

## Notes

* The live monitoring feature uses Scapy and requires elevated permissions.
* The SHAP explanation generation can be computationally intensive, especially for the first few explanations.
* Ensure the paths in `config.py` correctly point to your `data` and `saved_model` directories relative to `app.py`.