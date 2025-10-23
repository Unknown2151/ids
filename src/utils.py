import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_path: str = None) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load and preprocess data from CSV files

    Args:
        data_path: Path to data directory

    Returns:
        Tuple of (full_dataframe, X_features, y_labels)
    """
    if data_path is None:
        data_path = config.DATA_PATH

    try:
        # Get all CSV files
        all_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]

        if not all_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")

        logger.info(f"Loading {len(all_files)} CSV files...")

        # Load and concatenate all files
        dataframes = []
        for file_path in all_files:
            try:
                df = pd.read_csv(file_path, low_memory=False)
                df.columns = df.columns.str.strip()  # Remove whitespace from column names
                dataframes.append(df)
                logger.info(f"Loaded {file_path}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

        if not dataframes:
            raise ValueError("No valid CSV files could be loaded")

        # Concatenate all dataframes
        df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Total combined dataset: {len(df)} rows, {len(df.columns)} columns")

        # Clean data
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        initial_rows = len(df)
        df.dropna(inplace=True)
        dropped_rows = initial_rows - len(df)

        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with NaN values")

        # Separate features and labels
        if 'Label' not in df.columns:
            raise ValueError("'Label' column not found in dataset")

        X = df.drop('Label', axis=1)
        y = df['Label']

        logger.info(f"Final dataset: {len(X)} samples, {len(X.columns)} features")
        logger.info(f"Label distribution:\n{y.value_counts()}")

        return df, X, y

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def create_preprocessors(X: pd.DataFrame, y: pd.Series) -> Tuple[StandardScaler, LabelEncoder]:
    """
    Create and fit preprocessors

    Args:
        X: Feature matrix
        y: Label series

    Returns:
        Tuple of (fitted_scaler, fitted_label_encoder)
    """
    try:
        scaler = StandardScaler()
        scaler.fit(X)

        label_encoder = LabelEncoder()
        label_encoder.fit(y)

        logger.info(f"Scaler fitted on {X.shape[1]} features")
        logger.info(f"Label encoder fitted on {len(label_encoder.classes_)} classes: {label_encoder.classes_}")

        return scaler, label_encoder

    except Exception as e:
        logger.error(f"Error creating preprocessors: {e}")
        raise

def prepare_test_data(X: pd.DataFrame, y: pd.Series, scaler: StandardScaler,
                     test_size: float = None, random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare test data for SHAP explanations

    Args:
        X: Feature matrix
        y: Label series
        scaler: Fitted scaler
        test_size: Test split size
        random_state: Random state for reproducibility

    Returns:
        Tuple of (X_test_scaled, background_sample)
    """
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE

    try:
        _, X_test, _, _ = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        X_test_scaled = scaler.transform(X_test)
        background_sample = X_test_scaled[:config.SHAP_BACKGROUND_SAMPLES]

        logger.info(f"Prepared test data: {X_test_scaled.shape[0]} samples")
        logger.info(f"Background sample size: {background_sample.shape[0]} samples")

        return X_test_scaled, background_sample

    except Exception as e:
        logger.error(f"Error preparing test data: {e}")
        raise

def format_confidence(confidence: float) -> str:
    """
    Format confidence percentage for display

    Args:
        confidence: Confidence value (0-1)

    Returns:
        Formatted confidence string
    """
    return f"{confidence * 100:.2f}%"

def get_alert_color(attack_type: str) -> str:
    """
    Get color for alert display based on attack type

    Args:
        attack_type: Type of attack detected

    Returns:
        Color string for Streamlit
    """
    return config.ALERT_COLORS.get(attack_type, 'info')

def format_feature_names(feature_names: List[str]) -> List[str]:
    """
    Format feature names for better display

    Args:
        feature_names: List of feature names

    Returns:
        List of formatted feature names
    """
    return [config.FEATURE_DISPLAY_NAMES.get(name, name) for name in feature_names]

def validate_model_prediction(prediction: np.ndarray, label_encoder: LabelEncoder) -> Tuple[str, float]:
    """
    Validate and format model prediction

    Args:
        prediction: Model prediction probabilities
        label_encoder: Fitted label encoder

    Returns:
        Tuple of (predicted_class, confidence)
    """
    try:
        predicted_class_index = np.argmax(prediction)
        predicted_class = label_encoder.classes_[predicted_class_index]
        confidence = np.max(prediction)

        return predicted_class, confidence

    except Exception as e:
        logger.error(f"Error validating prediction: {e}")
        raise

def safe_reshape_for_model(data: np.ndarray, model_input_shape: Tuple) -> np.ndarray:
    """
    Safely reshape data for model input

    Args:
        data: Input data array
        model_input_shape: Expected model input shape

    Returns:
        Reshaped data array
    """
    try:
        if len(data.shape) == 1:
            # 1D array, reshape to 2D then 3D
            data_2d = data.reshape(1, -1)
            data_3d = data_2d.reshape(1, data_2d.shape[1], 1)
        elif len(data.shape) == 2:
            # 2D array, reshape to 3D
            data_3d = data.reshape(data.shape[0], data.shape[1], 1)
        else:
            # Already 3D or other shape
            data_3d = data

        return data_3d

    except Exception as e:
        logger.error(f"Error reshaping data: {e}")
        raise


