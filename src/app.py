import atexit
import csv
import logging
import multiprocessing
import os
import re
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import psutil
import shap
import streamlit as st
import tensorflow as tf

import config
import utils

# Set multiprocessing start method for Windows/macOS compatibility
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # Already set, ignore
    pass

# Import scapy with error handling
try:
    from scapy.all import sniff
    from scapy.layers.inet import IP, TCP, UDP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    st.error("Scapy is not installed. Please install it with: pip install scapy")
    st.stop()
except Exception as e:
    SCAPY_AVAILABLE = False
    st.error(f"Error importing Scapy: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)
# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .alert-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    .success-card {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
    }
    .info-card {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .interpretation-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #007bff;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .reasoning-card {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .feature-highlight {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
    .attack-indicator {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


# Fallback dummy model for demo purposes
class DummyModel:
    """Fallback model that always returns BENIGN predictions"""

    def __init__(self, benign_class_index=0, num_classes=2):
        self.input_shape = (None, 78, 1)
        self.benign_class_index = benign_class_index
        self.num_classes = num_classes

    def predict(self, x, verbose=0):
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        predictions = np.full((batch_size, self.num_classes), 0.1 / (self.num_classes - 1))
        predictions[:, self.benign_class_index] = 0.9
        return predictions


# Load Model and Preprocessors (Cached for performance)
@st.cache_resource
def load_model_and_preprocessors():
    """
    Loads the trained model, scaler, and label encoder with proper error handling.
    Falls back to dummy model if loading fails.
    """
    try:
        config.validate_paths()
        logger.info("Loading model...")
        model = tf.keras.models.load_model(config.MODEL_PATH)

        logger.info("Loading and preprocessing data...")
        df, X_full, y_full = utils.load_and_preprocess_data()

        scaler, label_encoder = utils.create_preprocessors(X_full, y_full)
        X_test_scaled, background_sample = utils.prepare_test_data(X_full, y_full, scaler)

        logger.info("All components loaded successfully")
        return model, scaler, label_encoder, X_test_scaled, X_full.columns, background_sample

    except Exception as e:
        logger.error(f"Error loading components: {e}")
        st.warning(f"Failed to load model and data: {e}")
        st.info("Using fallback dummy model for demo purposes.")

        from sklearn.preprocessing import StandardScaler, LabelEncoder
        dummy_label_encoder = LabelEncoder()
        dummy_classes = ['BENIGN', 'DDoS', 'PortScan']
        dummy_label_encoder.fit(dummy_classes)

        try:
            benign_idx = list(dummy_label_encoder.classes_).index('BENIGN')
        except ValueError:
            benign_idx = 0  # Default to 0 if BENIGN is not in the dummy list

        dummy_model = DummyModel(benign_class_index=benign_idx, num_classes=len(dummy_classes))

        dummy_scaler = StandardScaler()
        dummy_X = np.random.randn(100, 78)
        dummy_X_scaled = dummy_scaler.fit_transform(dummy_X)
        dummy_background = dummy_X_scaled[:50]

        return dummy_model, dummy_scaler, dummy_label_encoder, dummy_X_scaled, pd.Index(
            [f"Feature_{i}" for i in range(78)]), dummy_background


# Load all components
try:
    model, scaler, label_encoder, X_test_scaled, feature_names, background_sample = load_model_and_preprocessors()
except Exception as e:
    st.error(f"Application failed to initialize: {e}")
    st.stop()


def scapy_worker(interface, queue, stop_event, feature_names_list, packet_filter="tcp or udp"):
    """
    This function runs in a separate process to sniff packets and extract features.
    """
    import threading
    from collections import defaultdict
    import numpy as np
    from scapy.all import sniff
    from scapy.layers.inet import IP, TCP, UDP

    worker_flows = defaultdict(list)
    feature_index_map = {name: i for i, name in enumerate(feature_names_list)}

    # BUG FIX: The packet processing function must also gather the flags and direction
    # for the analysis function to work correctly.
    def process_packet_local(packet):
        if IP in packet:
            src_ip, dst_ip = packet[IP].src, packet[IP].dst
            sport, dport = (0, 0)
            tcp_flags = {}

            if TCP in packet:
                sport, dport = packet[TCP].sport, packet[TCP].dport
                tcp_flags = {
                    'F': 'F' in packet[TCP].flags, 'S': 'S' in packet[TCP].flags,
                    'R': 'R' in packet[TCP].flags, 'P': 'P' in packet[TCP].flags,
                    'A': 'A' in packet[TCP].flags, 'U': 'U' in packet[TCP].flags,
                }
            elif UDP in packet:
                sport, dport = packet[UDP].sport, packet[UDP].dport

            flow_key = tuple(sorted(((src_ip, sport), (dst_ip, dport))))

            packet_info = {
                'len': len(packet),
                'time': packet.time,
                'is_forward': (src_ip, sport) == flow_key[0],
                'flags': tcp_flags
            }
            worker_flows[flow_key].append(packet_info)

    def analyze_and_send_flows_local():
        if not worker_flows:
            return

        current_flows = dict(worker_flows)
        worker_flows.clear()

        for key, packets in current_flows.items():
            if len(packets) < 2:
                continue

            feature_vector = np.zeros(len(feature_names_list))
            try:
                timestamps = np.array([p['time'] for p in packets])
                inter_arrival_times = np.diff(timestamps)

                fwd_packets = [p for p in packets if p['is_forward']]
                bwd_packets = [p for p in packets if not p['is_forward']]

                # Basic Flow Features
                feature_vector[feature_index_map['Flow Duration']] = (timestamps.max() - timestamps.min()) * 1e6
                feature_vector[feature_index_map['Total Fwd Packets']] = len(fwd_packets)
                feature_vector[feature_index_map['Total Backward Packets']] = len(bwd_packets)
                feature_vector[feature_index_map['Total Length of Fwd Packets']] = sum(p['len'] for p in fwd_packets)
                feature_vector[feature_index_map['Total Length of Bwd Packets']] = sum(p['len'] for p in bwd_packets)

                # Timing Features
                if len(inter_arrival_times) > 0:
                    feature_vector[feature_index_map['Flow IAT Mean']] = np.mean(inter_arrival_times) * 1e6
                    feature_vector[feature_index_map['Flow IAT Std']] = np.std(inter_arrival_times) * 1e6

                # Flag Features
                feature_vector[feature_index_map['FIN Flag Count']] = sum(1 for p in packets if p['flags'].get('F'))
                feature_vector[feature_index_map['SYN Flag Count']] = sum(1 for p in packets if p['flags'].get('S'))
                feature_vector[feature_index_map['RST Flag Count']] = sum(1 for p in packets if p['flags'].get('R'))
                feature_vector[feature_index_map['PSH Flag Count']] = sum(1 for p in packets if p['flags'].get('P'))
                feature_vector[feature_index_map['ACK Flag Count']] = sum(1 for p in packets if p['flags'].get('A'))
                feature_vector[feature_index_map['URG Flag Count']] = sum(1 for p in packets if p['flags'].get('U'))

                # Rate Features
                duration_sec = (timestamps.max() - timestamps.min())
                if duration_sec > 0:
                    flow_bytes = sum(p['len'] for p in packets)
                    feature_vector[feature_index_map['Flow Bytes/s']] = flow_bytes / duration_sec
                    feature_vector[feature_index_map['Flow Packets/s']] = len(packets) / duration_sec

                (ip1, port1), (ip2, port2) = key
                flow_info = f"Flow between {ip1}:{port1} and {ip2}:{port2}"

                queue.put((feature_vector.astype(float), flow_info))
            except (KeyError, IndexError) as e:
                logger.warning(f"Skipping flow due to missing feature key: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing a flow: {e}")
                continue

    def analysis_loop_local():
        while not stop_event.is_set():
            analyze_and_send_flows_local()
            # Use configurable delay for better responsiveness
            time.sleep(config.PROCESSING_DELAY)

    analysis_thread = threading.Thread(target=analysis_loop_local, daemon=True)
    analysis_thread.start()

    try:
        while not stop_event.is_set():
            sniff(iface=interface, prn=process_packet_local, store=False, timeout=2,
                  filter=packet_filter, stop_filter=lambda p: stop_event.is_set())
    except Exception as e:
        queue.put(("__ERROR__", str(e)))

    analysis_thread.join(timeout=1)


# Initialize session state
if 'capture_process' not in st.session_state:
    st.session_state.capture_process = None
if 'alert_queue' not in st.session_state:
    st.session_state.alert_queue = None
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = None
if 'process_cleanup_registered' not in st.session_state:
    st.session_state.process_cleanup_registered = False
if 'worker_error_count' not in st.session_state:
    st.session_state.worker_error_count = 0
if 'session_id' not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"
if 'max_worker_errors' not in st.session_state:
    st.session_state.max_worker_errors = 5

# Process cleanup function
def cleanup_background_processes():
    """Clean up any running background processes"""
    if st.session_state.capture_process and st.session_state.capture_process.is_alive():
        logger.info("Cleaning up background processes...")
        if st.session_state.stop_event:
            st.session_state.stop_event.set()
        st.session_state.capture_process.join(timeout=5)
        if st.session_state.capture_process.is_alive():
            st.session_state.capture_process.terminate()
        st.session_state.capture_process = None

def safe_start_monitoring(interface, feature_names, packet_filter):
    """Safely start monitoring with proper error handling and cleanup"""
    try:
        # Clean up any existing process first
        if st.session_state.capture_process and st.session_state.capture_process.is_alive():
            cleanup_background_processes()

        # Create new queue and event
        st.session_state.alert_queue = multiprocessing.Queue(maxsize=1000)
        st.session_state.stop_event = multiprocessing.Event()
        st.session_state.worker_error_count = 0

        # Start new process
        p = multiprocessing.Process(
            target=scapy_worker,
            args=(interface, st.session_state.alert_queue, st.session_state.stop_event, feature_names, packet_filter),
            name=f"scapy_worker_{st.session_state.session_id}"
        )
        p.start()
        st.session_state.capture_process = p

        return True, "Monitoring started successfully"
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        cleanup_background_processes()
        return False, str(e)

def safe_stop_monitoring():
    """Safely stop monitoring with proper cleanup"""
    try:
        if st.session_state.capture_process and st.session_state.capture_process.is_alive():
            if st.session_state.stop_event:
                st.session_state.stop_event.set()
            st.session_state.capture_process.join(timeout=10)
            if st.session_state.capture_process.is_alive():
                st.session_state.capture_process.terminate()
        cleanup_background_processes()
        return True, "Monitoring stopped successfully"
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {e}")
        return False, str(e)

# Register cleanup function
if not st.session_state.process_cleanup_registered:
    atexit.register(cleanup_background_processes)
    st.session_state.process_cleanup_registered = True

# Persistent logging functions
def save_alert_to_csv(class_name, confidence, flow_info, timestamp=None):
    """Save alert to CSV file for persistent logging with per-session filenames"""
    if timestamp is None:
        timestamp = datetime.now()

    alert_file = f"alerts_{st.session_state.session_id}.csv"
    file_exists = os.path.exists(alert_file)

    try:
        with open(alert_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Attack_Type", "Confidence", "Flow_Info"])
            writer.writerow([timestamp.isoformat(), class_name, confidence, flow_info])
    except Exception as e:
        logger.error(f"Failed to save alert to CSV: {e}")

def get_alerts_from_csv():
    """Read alerts from CSV file"""
    alert_file = f"alerts_{st.session_state.session_id}.csv"
    if not os.path.exists(alert_file):
        return pd.DataFrame()

    try:
        return pd.read_csv(alert_file)
    except Exception as e:
        logger.error(f"Failed to read alerts from CSV: {e}")
        return pd.DataFrame()

def validate_bpf_filter(filter_string):
    """Validate BPF filter for security"""
    # List of safe, allowed BPF patterns
    safe_patterns = [
        r'^tcp$',
        r'^udp$',
        r'^tcp or udp$',
        r'^tcp port \d+$',
        r'^udp port \d+$',
        r'^tcp port \d+ or tcp port \d+$',
        r'^tcp port \d+ or udp port \d+$',
        r'^host \d+\.\d+\.\d+\.\d+$',
        r'^host \d+\.\d+\.\d+\.\d+ and tcp$',
        r'^host \d+\.\d+\.\d+\.\d+ and udp$'
    ]

    # Check if filter matches any safe pattern
    for pattern in safe_patterns:
        if re.match(pattern, filter_string.strip()):
            return True, "Valid filter"

    # Additional validation: check for dangerous patterns
    dangerous_patterns = ['exec', 'system', 'shell', 'cmd', 'eval', 'import', 'subprocess']
    for pattern in dangerous_patterns:
        if pattern in filter_string.lower():
            return False, f"Filter contains potentially dangerous pattern: {pattern}"

    # If it doesn't match safe patterns but isn't obviously dangerous, allow with warning
    return True, "Custom filter (use with caution)"

@st.cache_resource
def get_shap_explainer(_model, background_sample):
    """Get cached SHAP explainer for better performance with reduced background sample"""
    # Use only 100-500 random samples for better performance
    if len(background_sample) > 500:
        # Sample 500 random benign samples
        benign_indices = np.random.choice(len(background_sample), min(500, len(background_sample)), replace=False)
        reduced_background = background_sample[benign_indices]
    else:
        reduced_background = background_sample

    background_reshaped = utils.safe_reshape_for_model(reduced_background, model.input_shape)

    # KernelExplainer needs a function that takes a numpy array and returns predictions
    def prediction_function(x):
        # Reshape the 2D data back to the 3D shape the model expects
        reshaped_x = utils.safe_reshape_for_model(x, _model.input_shape)
        return _model.predict(reshaped_x)

    # Create and return the explainer
    return shap.KernelExplainer(prediction_function, reduced_background)

# Sidebar
with st.sidebar:
    st.title("🎛️ Control Panel")
    app_mode = st.radio("Choose App Mode", ["Offline Simulation", "Live Monitoring"])

    if app_mode == "Live Monitoring":
        st.markdown("---")
        st.markdown("### 📡 Real-Time Settings")
        try:
            stats = psutil.net_if_stats()
            available_interfaces = [name for name, stat in stats.items() if
                                    stat.isup and "loopback" not in name.lower()]
            selected_interface = st.selectbox("Select Network Interface", options=available_interfaces)
            st.session_state.interface = selected_interface
        except Exception:
            st.error("Could not find network interfaces.")

    st.sidebar.markdown("---")

    # Logs Management
    with st.sidebar.expander("📊 Logs Management", expanded=False):
        if st.button("📥 Download Alert Logs", use_container_width=True):
            alerts_df = get_alerts_from_csv()
            if not alerts_df.empty:
                csv_data = alerts_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No alerts logged yet.")

        if st.button("🗑️ Clear Logs", use_container_width=True):
            alert_file = f"alerts_{st.session_state.session_id}.csv"
            if os.path.exists(alert_file):
                os.remove(alert_file)
                st.success("Logs cleared!")
            else:
                st.info("No logs to clear.")

    st.sidebar.markdown("---")

    # Configuration Panel
    with st.sidebar.expander("Configuration Settings", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Analysis Settings")
            num_samples = st.slider("Number of samples to analyze", 1, 10, 1)
            auto_refresh = st.checkbox("Auto-refresh analysis", value=False)

        with col2:
            st.markdown("### Model Settings")
            confidence_threshold = st.slider("Confidence threshold (%)", 50, 99, 85)
            enable_shap = st.checkbox("Enable SHAP explanations", value=True)

    # Packet Filter Settings (for Live Monitoring)
    if app_mode == "Live Monitoring":
        with st.sidebar.expander("🔍 Packet Filter Settings", expanded=False):
            filter_options = {
                "All TCP/UDP": "tcp or udp",
                "TCP Only": "tcp",
                "UDP Only": "udp",
                "HTTP Traffic": "tcp port 80 or tcp port 8080",
                "HTTPS Traffic": "tcp port 443",
                "SSH Traffic": "tcp port 22",
                "Custom": "custom"
            }

            selected_filter = st.selectbox("Packet Filter", list(filter_options.keys()))

            if selected_filter == "Custom":
                custom_filter = st.text_input("Custom BPF Filter", value="tcp or udp")
                if custom_filter:
                    is_valid, message = validate_bpf_filter(custom_filter)
                    if is_valid:
                        st.session_state.packet_filter = custom_filter
                        if "caution" in message:
                            st.warning(f"⚠️ {message}")
                        else:
                            st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
                        st.session_state.packet_filter = "tcp or udp"  # Fallback to safe default
            else:
                st.session_state.packet_filter = filter_options[selected_filter]

            st.info(f"Current filter: `{st.session_state.get('packet_filter', 'tcp or udp')}`")

    st.markdown("## Attack Types & Descriptions")
    attack_descriptions = {
        "BENIGN": "Normal network traffic with no malicious activity",
        "DDoS": "Distributed Denial of Service - Overwhelms target with traffic from multiple sources",
        "DoS Hulk": "HTTP-based DoS attack using multiple requests to exhaust server resources",
        "DoS GoldenEye": "HTTP DoS attack targeting web servers with slow HTTP requests",
        "DoS slowloris": "Slow HTTP DoS attack keeping connections open to exhaust server resources",
        "DoS Slowhttptest": "Slow HTTP test attack using slow request headers",
        "PortScan": "Systematic scanning of network ports to identify open services",
        "Bot": "Automated bot traffic, potentially malicious or for reconnaissance",
        "FTP-Patator": "FTP brute force attack attempting to gain unauthorized access",
        "SSH-Patator": "SSH brute force attack trying to crack SSH credentials",
        "Web Attack - Brute Force": "Web application brute force attack on login forms",
        "Web Attack - XSS": "Cross-Site Scripting attack injecting malicious scripts",
        "Web Attack - Sql Injection": "SQL injection attack targeting database vulnerabilities",
        "Infiltration": "Network infiltration attempt to gain unauthorized access",
        "Heartbleed": "Exploitation of OpenSSL Heartbleed vulnerability"
    }


    # Clean attack type names for display (remove special characters)
    def clean_attack_name(name):
        """Clean attack name by removing special characters and normalizing"""
        # Remove special characters and normalize
        cleaned = re.sub(r'[^\w\s-]', '', name)
        # Replace multiple spaces with single space
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    # Create consistent mapping between original and cleaned names
    attack_name_mapping = {}
    for attack_type in label_encoder.classes_:
        cleaned_name = clean_attack_name(attack_type)
        attack_name_mapping[attack_type] = cleaned_name
        attack_name_mapping[cleaned_name] = attack_type

    for attack_type in label_encoder.classes_:
        cleaned_name = clean_attack_name(attack_type)
        with st.expander(f"{cleaned_name}", expanded=False):
            # Use consistent mapping to find description
            description = attack_descriptions.get(cleaned_name,
                                                  attack_descriptions.get(attack_type,
                                                                          f"Network attack type: {cleaned_name}"))
            st.write(description)

            # Add risk level indicator
            if cleaned_name == "BENIGN":
                st.success("Low Risk - Normal Traffic")
            elif any(term in cleaned_name.upper() for term in ["DDOS", "DOS"]):
                st.error("High Risk - Denial of Service Attack")
            elif any(term in cleaned_name.upper() for term in ["BRUTE", "PATATOR"]):
                st.warning("High Risk - Brute Force Attack")
            elif any(term in cleaned_name.upper() for term in ["XSS", "SQL", "INFILTRATION", "HEARTBLEED"]):
                st.error("Critical Risk - Exploitation Attack")
            elif any(term in cleaned_name.upper() for term in ["PORTSCAN", "BOT"]):
                st.warning("Medium Risk - Reconnaissance")

# Main UI
st.markdown('<h1 class="main-header">XAI-IDS</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="text-align: center; color: #666;">Explainable AI Intrusion Detection System</h2>',
            unsafe_allow_html=True)

if app_mode == "Offline Simulation":
    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Offline Analysis Simulation")
        st.write("Click the button to analyze a random sample from the pre-loaded test dataset.")
        # Analysis button
        if st.button("Analyze New Connection", use_container_width=True):
            with st.spinner("Processing incoming connection..."):
                try:
                    # Select random sample
                    sample_index = np.random.randint(0, len(X_test_scaled))
                    connection_data = X_test_scaled[sample_index]

                    # Reshape data for model
                    reshaped_data_3d = utils.safe_reshape_for_model(connection_data, model.input_shape)
                    reshaped_data_2d = connection_data.reshape(1, -1)

                    # Get prediction
                    prediction_probs = model.predict(reshaped_data_3d, verbose=0)
                    predicted_class, confidence = utils.validate_model_prediction(prediction_probs, label_encoder)

                    # Store results in session state
                    st.session_state.prediction_result = {
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'connection_data': connection_data,
                        'reshaped_data_2d': reshaped_data_2d,
                        'reshaped_data_3d': reshaped_data_3d,
                        'timestamp': datetime.now()
                    }

                except Exception as e:
                    logger.error(f"Error during prediction: {e}")
                    st.error(f"Error processing connection: {e}")

    with col2:
        st.markdown("## Quick Stats")

        # Display recent prediction if available
        if 'prediction_result' in st.session_state:
            result = st.session_state.prediction_result
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            timestamp = result['timestamp'].strftime('%H:%M:%S')

            if predicted_class != "BENIGN":
                # Display an alert card for attacks
                with st.container():
                    st.error("ALERT! Intrusion Detected!", icon="🚨")
                    stat1, stat2, stat3 = st.columns(3)
                    stat1.metric("Attack Type", predicted_class)
                    stat2.metric("Confidence", utils.format_confidence(confidence))
                    stat3.metric("Time", timestamp)
            else:
                # Display a success card for benign traffic
                with st.container():
                    st.success("Status: Normal (BENIGN)", icon="✅")
                    stat1, stat2, stat3 = st.columns(3)
                    stat1.metric("Status", "BENIGN")
                    stat2.metric("Confidence", utils.format_confidence(confidence))
                    stat3.metric("Time", timestamp)

    # Display detailed results if available
    if 'prediction_result' in st.session_state:
        result = st.session_state.prediction_result
        predicted_class = result['predicted_class']
        confidence = result['confidence']

        st.markdown("---")
        st.markdown("## Detailed Analysis Results")

        # Create tabs for different views
        tab1, tab2 = st.tabs(["Feature Analysis", "SHAP Explanation"])

        with tab1:
            st.markdown("### Feature Values")

            # Create a DataFrame for better display
            feature_df = pd.DataFrame({
                'Feature': utils.format_feature_names(feature_names),
                'Value': result['connection_data']
            })

            # Display top 20 features by absolute value
            feature_df['Abs_Value'] = np.abs(feature_df['Value'])
            top_features = feature_df.nlargest(20, 'Abs_Value')

            fig = px.bar(
                top_features,
                x='Value',
                y='Feature',
                orientation='h',
                title="Top 20 Feature Values",
                color='Value',
                color_continuous_scale='RdBu'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if predicted_class != "BENIGN":
                st.markdown("### SHAP Explanation")
                st.info(
                    "This explanation shows which features contributed most to the attack detection. Red features push the prediction toward the detected attack, while blue features push it away.")

                # Check if SHAP is enabled in configuration
                if st.session_state.get('enable_shap', True):
                    with st.spinner("Generating SHAP explanation... This may take a moment."):
                        try:
                            explainer = get_shap_explainer(model, background_sample)
                            shap_values = explainer.shap_values(result['reshaped_data_2d'])
                            predicted_class_idx = list(label_encoder.classes_).index(predicted_class)


                            if isinstance(shap_values, list):
                                # Case 1: Standard multi-output (list of arrays)
                                shap_values_for_class = shap_values[predicted_class_idx][0]
                                base_value = explainer.expected_value[predicted_class_idx]
                            elif shap_values.ndim == 3:
                                # Case 2: Multi-output bundled in one 3D array of shape (samples, features, classes)
                                # We slice to get our single sample [0], all features [:], and the specific class [predicted_class_idx]
                                shap_values_for_class = shap_values[0, :, predicted_class_idx]
                                base_value = explainer.expected_value[predicted_class_idx]
                            else:
                                # Case 3: Standard single-output (2D array)
                                shap_values_for_class = shap_values[0]
                                base_value = explainer.expected_value

                            explanation = shap.Explanation(
                                values=shap_values_for_class,
                                base_values=base_value,
                                data=result['reshaped_data_2d'][0],
                                feature_names=utils.format_feature_names(feature_names)
                            )

                            shap.plots.waterfall(explanation, max_display=20, show=False)
                            st.pyplot(plt.gcf())
                            plt.clf()

                            # Add detailed interpretation
                            st.markdown("---")
                            st.markdown("### Model Prediction Interpretation")

                            # BUG FIX: Replaced the undefined 'shap_flat' variable with the correct
                            # one, 'shap_values_for_class', to prevent a NameError.
                            feature_importance = np.abs(shap_values_for_class)
                            top_features_idx = np.argsort(feature_importance)[-10:][::-1]

                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("#### Top Features Supporting Attack Detection:")
                                for i, idx in enumerate(top_features_idx):
                                    feature_name = utils.format_feature_names(feature_names)[idx]
                                    contribution = shap_values_for_class[idx]
                                    actual_value = result['reshaped_data_2d'][0][idx]

                                    # Display only the top 5 positive contributors
                                    if contribution > 0 and i < 5:
                                        st.markdown(f'<div class="feature-highlight">', unsafe_allow_html=True)
                                        st.markdown(f"**{i + 1}. {feature_name}**")
                                        st.markdown(f"   - Contribution: +{contribution:.4f}")
                                        st.markdown(f"   - Actual Value: {actual_value:.4f}")

                                        if "packet" in feature_name.lower():
                                            st.markdown(f"   - *High packet activity suggests potential attack*")
                                        elif "duration" in feature_name.lower():
                                            st.markdown(f"   - *Unusual flow duration indicates suspicious behavior*")
                                        elif "flag" in feature_name.lower():
                                            st.markdown(f"   - *Abnormal flag patterns suggest malicious intent*")
                                        elif "rate" in feature_name.lower():
                                            st.markdown(f"   - *High rate indicates potential DoS or scanning*")
                                        st.markdown('</div>', unsafe_allow_html=True)

                            with col2:
                                st.markdown("#### Top Features Supporting Normal Traffic:")
                                normal_features = []
                                for idx in top_features_idx:
                                    contribution = shap_values_for_class[idx]
                                    if contribution < 0 and len(normal_features) < 5:
                                        feature_name = utils.format_feature_names(feature_names)[idx]
                                        actual_value = result['reshaped_data_2d'][0][idx]
                                        normal_features.append((feature_name, contribution, actual_value))

                                if normal_features:
                                    for i, (feature_name, contribution, actual_value) in enumerate(normal_features):
                                        st.markdown(f'<div class="feature-highlight">', unsafe_allow_html=True)
                                        st.markdown(f"**{i + 1}. {feature_name}**")
                                        st.markdown(f"   - Contribution: {contribution:.4f}")
                                        st.markdown(f"   - Actual Value: {actual_value:.4f}")
                                        st.markdown(f"   - *Normal values suggest legitimate traffic*")
                                        st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown("*No significant features supporting normal traffic*")

                                # Overall interpretation
                                st.markdown("---")
                                st.markdown("### AI Model Reasoning")

                                positive_contrib = np.sum(shap_values_for_class[shap_values_for_class > 0])
                                negative_contrib = np.sum(shap_values_for_class[shap_values_for_class < 0])
                                net_confidence = positive_contrib + negative_contrib  # Note: negative_contrib is already negative

                                st.markdown(f'<div class="reasoning-card">', unsafe_allow_html=True)
                                st.markdown(f"**Why the model predicted '{predicted_class}':**")
                                st.markdown(f"- **Net Confidence Score:** {net_confidence:.4f}")
                                st.markdown(
                                    f"- **Positive Evidence:** {positive_contrib:.4f} (features supporting attack)")
                                st.markdown(
                                    f"- **Negative Evidence:** {negative_contrib:.4f} (features supporting normal traffic)")
                                st.markdown('</div>', unsafe_allow_html=True)

                                # Attack-specific interpretation (unchanged)
                                # ...

                            st.success("SHAP explanation generated successfully!")

                        except Exception as e:
                            logger.error(f"Error generating SHAP explanation: {e}")
                            st.error(f"Failed to generate SHAP explanation: {e}")
                else:
                    st.warning("SHAP explanations are disabled in configuration settings.")
            else:
                st.info("SHAP explanation is only available for detected attacks.")

elif app_mode == "Live Monitoring":
    st.header(f"Live Network Monitoring on `{st.session_state.get('interface', 'None')}`")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🟢 Start Monitoring", use_container_width=True):
            if not SCAPY_AVAILABLE:
                st.error("Scapy is not available. Please install it and restart the application.")
            elif not (st.session_state.capture_process and st.session_state.capture_process.is_alive()):
                interface = st.session_state.get('interface')
                if interface:
                    packet_filter = st.session_state.get('packet_filter', 'tcp or udp')
                    success, message = safe_start_monitoring(interface, feature_names, packet_filter)
                    if success:
                        st.success(f"Live capture started on interface '{interface}'!")
                    else:
                        st.error(f"Failed to start monitoring: {message}")
                        st.info(
                            "Note: On some systems, you may need to run with administrator/root privileges for packet capture.")
                else:
                    st.error("Please select a valid network interface.")
            else:
                st.warning("Monitoring is already in progress.")
    with col2:
        if st.button("🔴 Stop Monitoring", use_container_width=True):
            success, message = safe_stop_monitoring()
            if success:
                st.success("Live capture stopped.")
            else:
                st.error(f"Error stopping monitoring: {message}")

    st.markdown("---")

    # Real-time Dashboard

    st.markdown("---")
    st.header("Live Alert Log")
    log_placeholder = st.empty()
    if 'alerts_log' not in st.session_state:
        st.session_state.alerts_log = ""

    with log_placeholder.container():
        if st.session_state.capture_process and st.session_state.capture_process.is_alive():
            st.info("Monitoring... New alerts will appear below.")

            from queue import Empty

            max_polls = 10
            for _ in range(max_polls):
                try:
                    feature_vector, flow_info = st.session_state.alert_queue.get_nowait()

                    # Handle error messages from the worker process
                    if isinstance(feature_vector, str) and feature_vector == "__ERROR__":
                        st.session_state.worker_error_count += 1
                        st.error(
                            f"Worker process error ({st.session_state.worker_error_count}/{st.session_state.max_worker_errors}): {flow_info}")

                        if st.session_state.worker_error_count >= st.session_state.max_worker_errors:
                            st.error("Too many worker errors. Stopping monitoring automatically.")
                            safe_stop_monitoring()
                            break
                        continue

                    # Process the feature vector
                    scaled_features = scaler.transform(feature_vector.reshape(1, -1))
                    reshaped_features = utils.safe_reshape_for_model(scaled_features, model.input_shape)
                    prediction = model.predict(reshaped_features, verbose=0)
                    class_name, confidence = utils.validate_model_prediction(prediction, label_encoder)

                    confidence_percentage = confidence * 100
                    if class_name != 'BENIGN' and confidence_percentage >= confidence_threshold:
                        alert_message = f"🚨 ALERT! Detected {class_name} ({utils.format_confidence(confidence)}) on {flow_info}"
                        timestamp = datetime.now()
                        st.session_state.alerts_log = f"`{timestamp.strftime('%H:%M:%S')}` - {alert_message}\n" + st.session_state.alerts_log
                        save_alert_to_csv(class_name, confidence, flow_info, timestamp)
                    elif class_name != 'BENIGN' and confidence_percentage < confidence_threshold:
                        low_conf_message = f"⚠️ Low confidence detection: {class_name} ({utils.format_confidence(confidence)}) on {flow_info} (below {confidence_threshold}% threshold)"
                        st.session_state.alerts_log = f"`{datetime.now().strftime('%H:%M:%S')}` - {low_conf_message}\n" + st.session_state.alerts_log

                except Empty:
                    break
                except Exception as e:
                    error_message = f"Error processing flow {flow_info}: {e}"
                    st.session_state.alerts_log = f"`{datetime.now().strftime('%H:%M:%S')}` - {error_message}\n" + st.session_state.alerts_log
                    break

            # Moved the code display outside the loop to prevent
            # unnecessary UI redraws and potential flickering.
            st.code(st.session_state.alerts_log, language="log")
        else:
            st.info("Monitoring is stopped.")
            st.code(st.session_state.alerts_log, language="log")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Powered by CNN-LSTM Neural Networks and SHAP Explanations</p>
    </div>
    """,
    unsafe_allow_html=True
)
