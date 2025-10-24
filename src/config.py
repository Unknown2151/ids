import os
import psutil

# Model and Data Paths
MODEL_PATH = 'src/saved_model/cnn_lstm_ids_model_resampled.h5'
DATA_PATH = 'src/data/'

# Streamlit Configuration
PAGE_TITLE = "XAI-IDS"
PAGE_ICON = "🛡️"
LAYOUT = "wide"

# Model Configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
SHAP_BACKGROUND_SAMPLES = 50

# Real-time Configuration
DEFAULT_INTERFACE = 'Wi-Fi'
PROCESSING_DELAY = 0.2  # seconds
MAX_WORKER_ERRORS = 5 # Max errors before worker stops
MAX_QUEUE_PROCESS_PER_CYCLE = 50 # Max items to process from queue per Streamlit rerun cycle (adjust as needed)

# UI Configuration
ALERT_COLORS = {
    'BENIGN': 'success',
    'DDoS': 'error',
    'PortScan': 'warning',
    'WebAttack': 'error',
    'Infiltration': 'error'
}

# Feature names for better display
FEATURE_DISPLAY_NAMES = {
    'Destination Port': 'Destination Port',
    'Flow Duration': 'Flow Duration (ms)',
    'Total Fwd Packets': 'Forward Packets',
    'Total Backward Packets': 'Backward Packets',
    'Total Length of Fwd Packets': 'Forward Bytes',
    'Total Length of Bwd Packets': 'Backward Bytes',
    'Flow Bytes/s': 'Flow Rate (bytes/s)',
    'Flow Packets/s': 'Packet Rate (packets/s)',
    'Flow IAT Mean': 'Inter-Arrival Time Mean',
    'Flow IAT Std': 'Inter-Arrival Time Std',
    'Fwd Packets/s': 'Forward Packet Rate',
    'Bwd Packets/s': 'Backward Packet Rate',
    'Packet Length Mean': 'Average Packet Size',
    'FIN Flag Count': 'FIN Flags',
    'SYN Flag Count': 'SYN Flags',
    'RST Flag Count': 'RST Flags',
    'PSH Flag Count': 'PSH Flags',
    'ACK Flag Count': 'ACK Flags',
    'URG Flag Count': 'URG Flags'
}

try:
    # Find the first active non-loopback interface as a default
    stats = psutil.net_if_stats()
    available_interfaces = [name for name, stat in stats.items() if stat.isup and "loopback" not in name.lower()]
    DEFAULT_INTERFACE = available_interfaces[0] if available_interfaces else 'Wi-Fi'
# This prevents the code from accidentally hiding other, unexpected bugs.
except (IndexError, ImportError, FileNotFoundError):
    DEFAULT_INTERFACE = 'Wi-Fi'


# Validation
def validate_paths():
    """Validate that required paths exist"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data directory not found: {DATA_PATH}")

    data_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
    if not data_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_PATH}")


def get_available_interfaces():
    """Get list of available network interfaces"""
    try:
        import psutil
        stats = psutil.net_if_stats()
        interfaces = []
        for interface_name, stat in stats.items():
            if stat.isup and "loopback" not in interface_name.lower():
                interfaces.append(interface_name)
        return interfaces
    except ImportError:
        return [DEFAULT_INTERFACE]