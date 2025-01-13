import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import IsolationForest
import logging
import warnings
import plotly.graph_objects as go

# Additional imports
import xgboost as xgb  # Gradient Boosting
from pykalman import KalmanFilter  # Kalman filtering
import imageio  # For exporting frames to video/GIF (pip install imageio kaleido)

# --- Configuration Parameters ---
INITIAL_PROMINENCE_THRESHOLD = 0.015
MIN_PEAK_DISTANCE_SECONDS = 0.7
CUTOFF_FREQUENCY = 3
VALID_DURATION_RANGE = (0.7, 1.2)
FS = None
LEFT_FIRST = True
VALIDITY_SCORE_THRESHOLD = 0.8

# Adaptive peak detection parameters
ROLLING_WINDOW_SIZE = 500  # Number of samples for rolling calculation
PROMINENCE_SCALE_FACTOR = 2  # Scales local std dev to set adaptive threshold

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output


def get_filename():
    """Prompt the user for the BDF filename."""
    while True:
        filename = input("Enter the BDF filename (including extension, e.g., 'EkatBio.bdf'): ").strip()
        if os.path.isfile(filename):
            logging.info(f"File '{filename}' found.")
            return filename
        else:
            logging.error(f"File '{filename}' not found in the current directory.")
            retry = input("Would you like to try again? (y/n): ").strip().lower()
            if retry != 'y':
                raise FileNotFoundError(f"File '{filename}' not found.")


def low_pass_filter(signal, cutoff, fs, order=4):
    """Apply a low-pass Butterworth filter to the signal."""
    from scipy.signal import butter, filtfilt
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    logging.info(f"Low-pass filtering applied (order={order}, cutoff={cutoff} Hz).")
    return filtered_signal


def apply_kalman_filter(signal):
    """
    Apply a simple Kalman filter to smooth the input signal.
    This is helpful for real-time smoothing and noise reduction.
    """
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=signal[0],
        n_dim_obs=1
    )
    state_means, _ = kf.filter(signal)
    return state_means.flatten()


def adaptive_peak_detection(signal, fs):
    """
    Adaptive peak detection using a rolling standard deviation to adjust
    the local prominence threshold. This helps handle varying movement intensities.
    """
    global INITIAL_PROMINENCE_THRESHOLD
    distance = int(MIN_PEAK_DISTANCE_SECONDS * fs)
    
    # Rolling standard deviation
    rolling_std = pd.Series(signal).rolling(
        window=ROLLING_WINDOW_SIZE,
        min_periods=1,
        center=True
    ).std()
    rolling_std = rolling_std.fillna(method='bfill').fillna(method='ffill').values
    
    # Compute an adaptive threshold
    adaptive_prominence = INITIAL_PROMINENCE_THRESHOLD + (PROMINENCE_SCALE_FACTOR * rolling_std)
    
    # We'll use the median of this adaptive array as a single threshold
    global_threshold = np.median(adaptive_prominence)
    logging.info(f"Adaptive peak detection with median local threshold = {global_threshold:.5f}")
    
    # Find peaks with the global threshold
    peaks, _ = find_peaks(signal, prominence=global_threshold, distance=distance)
    
    return peaks


def extract_features(signal_segments, time_segment):
    """Extract features from all axes for a single peak window."""
    features = {}

    # Features for each axis
    for axis, segment in signal_segments.items():
        features[f'mean_{axis}'] = np.mean(segment)
        features[f'std_{axis}'] = np.std(segment)
        features[f'max_{axis}'] = np.max(segment)
        features[f'min_{axis}'] = np.min(segment)
        features[f'range_{axis}'] = features[f'max_{axis}'] - features[f'min_{axis}']

    # Cross-axis differences for gyroscope
    if 'GyroX' in signal_segments and 'GyroY' in signal_segments:
        features['gyro_xy_diff'] = np.mean(signal_segments['GyroX'] - signal_segments['GyroY'])
    if 'GyroX' in signal_segments and 'GyroZ' in signal_segments:
        features['gyro_xz_diff'] = np.mean(signal_segments['GyroX'] - signal_segments['GyroZ'])
    if 'GyroY' in signal_segments and 'GyroZ' in signal_segments:
        features['gyro_yz_diff'] = np.mean(signal_segments['GyroY'] - signal_segments['GyroZ'])

    # Cross-axis differences for accelerometer
    if 'AccelerX' in signal_segments and 'AccelerY' in signal_segments:
        features['accel_xy_diff'] = np.mean(signal_segments['AccelerX'] - signal_segments['AccelerY'])
    if 'AccelerX' in signal_segments and 'AccelerZ' in signal_segments:
        features['accel_xz_diff'] = np.mean(signal_segments['AccelerX'] - signal_segments['AccelerZ'])
    if 'AccelerY' in signal_segments and 'AccelerZ' in signal_segments:
        features['accel_yz_diff'] = np.mean(signal_segments['AccelerY'] - signal_segments['AccelerZ'])

    # Duration as a feature
    features['duration'] = time_segment[-1] - time_segment[0]

    return features


def enforce_alternating_movements(events):
    """
    Enforce alternating movements in the event sequence.
    A "Right Head Movement" must follow a "Left Head Movement" and vice versa.
    """
    corrected_events = []
    last_movement = None

    for event in events:
        if last_movement is None:
            corrected_events.append(event)
            last_movement = event['movement']
        else:
            if last_movement.startswith('Left') and event['movement'].startswith('Left'):
                # Force correction to "Right"
                event['movement'] = 'Right Head Movement'
            elif last_movement.startswith('Right') and event['movement'].startswith('Right'):
                # Force correction to "Left"
                event['movement'] = 'Left Head Movement'

            corrected_events.append(event)
            last_movement = event['movement']

    return corrected_events


def detect_cycles(events):
    """
    Detect valid cycles (Left followed by Right) and identify invalid cycles.
    """
    cycles = []
    invalid_cycles = []
    i = 0
    while i < len(events) - 1:
        first_event = events[i]
        second_event = events[i + 1]
        # Check if the sequence is Left → Right
        if first_event['movement'] == 'Left Head Movement' and second_event['movement'] == 'Right Head Movement':
            duration = second_event['time'] - first_event['time']
            if VALID_DURATION_RANGE[0] <= duration <= VALID_DURATION_RANGE[1]:
                cycles.append({
                    'start_time': first_event['time'],
                    'end_time': second_event['time'],
                    'duration': duration,
                    'left_event': first_event,
                    'right_event': second_event
                })
            else:
                invalid_cycles.append({
                    'start_time': first_event['time'],
                    'end_time': second_event['time'],
                    'duration': duration
                })
            i += 1
        else:
            # Invalid sequence, treat the first event as invalid
            invalid_cycles.append({
                'time': first_event['time'],
                'movement': first_event['movement']
            })
            i += 1

    # Handle the last event if not processed
    if i == len(events) - 1:
        last_event = events[-1]
        invalid_cycles.append({
            'time': last_event['time'],
            'movement': last_event['movement']
        })

    return cycles, invalid_cycles


def calculate_validity_score(cycles, invalid_cycles):
    """
    Calculate the validity score based on valid and invalid cycles.
    """
    total_cycles = len(cycles) + len(invalid_cycles)
    if total_cycles == 0:
        return 0
    else:
        return len(cycles) / total_cycles


def smooth_predictions(predictions, window_size=5):
    """
    Simple temporal smoothing of predictions:
      - Looks at a rolling window of predictions
      - Picks the majority label
    """
    smoothed = []
    for i in range(len(predictions)):
        start_idx = max(0, i - (window_size - 1))
        window = predictions[start_idx:i + 1]
        most_common = pd.Series(window).value_counts().idxmax()
        smoothed.append(most_common)
    return np.array(smoothed)


def process_signal(file_path):
    """Main processing function for gyroscope and accelerometer signals."""
    logging.info("Starting signal processing...")
    global FS

    # Step 1: Load the BDF file
    try:
        raw = mne.io.read_raw_bdf(file_path, preload=True)
        FS = raw.info['sfreq']
        logging.info(f"Loaded BDF file '{file_path}' with sampling frequency {FS} Hz.")
    except Exception as e:
        logging.error(f"Error loading BDF file: {e}")
        raise

    # Step 2: Identify gyroscope and accelerometer channels
    gyro_channels = ['GyroX', 'GyroY', 'GyroZ']
    accel_channels = ['AccelerX', 'AccelerY', 'AccelerZ']

    # Validate available channels
    available_channels = raw.info['ch_names']
    missing_channels = [ch for ch in gyro_channels + accel_channels if ch not in available_channels]
    if missing_channels:
        logging.warning(f"Missing channels: {missing_channels}. Proceeding with available channels.")

    # Step 3: Impute missing data, apply low-pass, and Kalman filter
    filtered_data = {}
    for channel in gyro_channels + accel_channels:
        if channel in available_channels:
            channel_data = raw.copy().pick([channel]).get_data().flatten()

            imputer = SimpleImputer(strategy='mean')
            signal_imputed = imputer.fit_transform(channel_data.reshape(-1, 1)).ravel()
            lp_signal = low_pass_filter(signal_imputed, cutoff=CUTOFF_FREQUENCY, fs=FS)
            kf_signal = apply_kalman_filter(lp_signal)
            filtered_data[channel] = kf_signal

    # Step 4: Pick primary axis
    if 'GyroX' in filtered_data:
        primary_axis = 'GyroX'
    else:
        primary_axis = next(iter(filtered_data)) if filtered_data else None

    if not primary_axis:
        logging.error("No valid sensor channels found. Exiting.")
        return None, None, None, [], [], [], 0, None

    time = np.arange(len(filtered_data[primary_axis])) / FS

    # Step 5: Adaptive peak detection on primary axis
    peaks = adaptive_peak_detection(filtered_data[primary_axis], FS)
    logging.info(f"Detected {len(peaks)} peaks in the primary axis ({primary_axis}) with adaptive detection.")

    # Step 6: Feature extraction around each peak
    window_size = int(0.5 * FS)  # 0.5 second window
    feature_list = []
    peak_times = []

    for peak in peaks:
        start_idx = max(0, peak - window_size)
        end_idx = min(len(filtered_data[primary_axis]), peak + window_size)
        time_segment = time[start_idx:end_idx]

        signal_segments = {
            axis: data[start_idx:end_idx] for axis, data in filtered_data.items()
        }
        features = extract_features(signal_segments, time_segment)
        feature_list.append(features)
        peak_times.append(time[peak])

    feature_df = pd.DataFrame(feature_list)

    # Step 7: Remove outliers (abnormal movements) with IsolationForest
    if len(feature_df) > 1:
        iso_forest = IsolationForest(random_state=42)
        outlier_labels = iso_forest.fit_predict(feature_df)
        inlier_indices = np.where(outlier_labels == 1)[0]
        feature_df = feature_df.iloc[inlier_indices].reset_index(drop=True)
        peaks = peaks[inlier_indices]
        peak_times = np.array(peak_times)[inlier_indices]
        logging.info(f"Outlier detection removed {len(outlier_labels) - len(inlier_indices)} abnormal points.")
    else:
        logging.info("Not enough data points for outlier detection; skipping.")

    # Step 8: Create a pseudo-label ("left" vs. "right")
    labels = []
    for idx in range(len(feature_df)):
        mean_gyro_x = feature_df.loc[idx, 'mean_GyroX'] if 'mean_GyroX' in feature_df.columns else 0
        mean_gyro_y = feature_df.loc[idx, 'mean_GyroY'] if 'mean_GyroY' in feature_df.columns else 0
        
        if mean_gyro_x > 0 and mean_gyro_y > 0:
            labels.append('left')
        elif mean_gyro_x < 0 and mean_gyro_y < 0:
            labels.append('right')
        else:
            # Heuristic tie-break
            if abs(mean_gyro_x) > abs(mean_gyro_y):
                labels.append('left' if mean_gyro_x > 0 else 'right')
            else:
                labels.append('left' if mean_gyro_y > 0 else 'right')

    df = feature_df.copy()
    df['label'] = labels

    # Step 9: Train a single XGBoost model if we have at least two classes
    if df['label'].nunique() > 1:
        X = df.drop('label', axis=1)
        y = df['label']

        # Encode string labels ("left"/"right") into integer labels (0/1)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Balance classes
        min_class_count = pd.Series(y_encoded).value_counts().min()
        balanced_indices = []
        for class_val in np.unique(y_encoded):
            class_indices = np.where(y_encoded == class_val)[0]
            balanced_indices.extend(
                np.random.RandomState(42).choice(class_indices, size=min_class_count, replace=False)
            )

        # Build balanced DataFrame
        df_balanced = df.iloc[balanced_indices].reset_index(drop=True)
        X_balanced = df_balanced.drop('label', axis=1)

        # Re-encode for the balanced subset
        y_balanced_str = df_balanced['label'].values
        y_balanced = label_encoder.transform(y_balanced_str)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train_scaled, y_train)

        logging.info("Gradient Boosting model (XGBoost) trained.")

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        logging.info("Model evaluation:")
        logging.info("\n" + classification_report(y_test, y_pred))

        # Predict on the entire dataset (including out-of-sample peaks)
        X_full_scaled = scaler.transform(X)
        predictions_encoded = model.predict(X_full_scaled)

        # Optional smoothing
        predictions_encoded = smooth_predictions(predictions_encoded, window_size=5)

        # Convert numeric predictions back to "left"/"right" strings
        predictions = label_encoder.inverse_transform(predictions_encoded)

    else:
        # If there's only one unique label, skip XGBoost
        logging.warning("Not enough label variety for training; skipping XGBoost. Using heuristics only.")
        predictions = labels

    # Step 10: Convert predictions → event dictionary
    events = []
    for idx, pred in enumerate(predictions):
        events.append({
            'time': peak_times[idx],
            'movement': pred.capitalize() + ' Head Movement'
        })

    # Enforce alternating movements
    corrected_events = enforce_alternating_movements(events)

    # Detect cycles
    cycles, invalid_cycles = detect_cycles(corrected_events)

    # Calculate validity score
    validity_score = calculate_validity_score(cycles, invalid_cycles)
    logging.info(f"Calculated validity score = {validity_score:.2f}")

    return filtered_data, time, peaks, corrected_events, cycles, invalid_cycles, validity_score, primary_axis


def plot_signals(filtered_data, time, peaks, events, primary_axis):
    """Plot the signals with detected movements."""
    if len(time) == 0 or primary_axis not in filtered_data:
        logging.warning("Cannot plot signals due to missing or invalid data.")
        return

    logging.info("Plotting signals...")
    plt.figure(figsize=(12, 6))
    plt.plot(time, filtered_data[primary_axis], label=f'Filtered+Smoothed {primary_axis}')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal Amplitude')
    plt.title('Gyroscope Signal with Detected Movements')

    # Mark the detected peaks
    plt.scatter(time[peaks], filtered_data[primary_axis][peaks], color='red', label='Detected Peaks', alpha=0.7)

    # Plot events
    if events:
        event_times = [event['time'] for event in events]
        event_amplitudes = [filtered_data[primary_axis][np.abs(time - t).argmin()] for t in event_times]
        movement_labels = [event['movement'] for event in events]
        colors = ['green' if 'Left' in label else 'blue' for label in movement_labels]
        plt.scatter(event_times, event_amplitudes, color=colors, label='Detected Movements')

    plt.legend()
    plt.show()


def plot_cycles(filtered_data, time, events, cycles, invalid_cycles, primary_axis):
    """Plot valid and invalid cycles."""
    if len(time) == 0 or primary_axis not in filtered_data:
        logging.warning("Cannot plot cycles due to missing or invalid data.")
        return

    logging.info("Plotting cycles...")
    plt.figure(figsize=(12, 6))
    plt.plot(time, filtered_data[primary_axis], label=f'Filtered+Smoothed {primary_axis}')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal Amplitude')
    plt.title('Valid and Invalid Cycles')

    # Highlight valid cycles
    for cycle in cycles:
        start_idx = np.abs(time - cycle['start_time']).argmin()
        end_idx = np.abs(time - cycle['end_time']).argmin()
        plt.axvspan(time[start_idx], time[end_idx], color='green', alpha=0.3)

    # Highlight invalid cycles
    for invalid in invalid_cycles:
        if 'start_time' in invalid:
            start_idx = np.abs(time - invalid['start_time']).argmin()
            end_idx = np.abs(time - invalid['end_time']).argmin()
            plt.axvspan(time[start_idx], time[end_idx], color='red', alpha=0.3)

    plt.legend(handles=[
        plt.Line2D([0], [0], color='green', lw=4, label='Valid Cycle'),
        plt.Line2D([0], [0], color='red', lw=4, label='Invalid Cycle')
    ])
    plt.show()


def plot_3d_animation(filtered_data, time, file_name):
    """
    Plot a futuristic-style 3D movement data animation with:
      - 10-second time window
      - Faster frame rates
      - Fixed axes (no auto-range)
      - Bright neon lines on a dark background
    """
    logging.info("Plotting 3D animation with futuristic style...")

    if len(time) < 2:
        logging.warning("Not enough data to plot 3D animation.")
        return

    # --- 1) EXTRACT 10-SECOND SNIPPET ---
    total_time = time[-1]
    snippet_duration = 10  # 10 seconds
    if total_time < snippet_duration:
        logging.warning(f"Data shorter than {snippet_duration} seconds. Plotting entire duration instead.")
        start_time = 0
        end_time = total_time
    else:
        start_time = (total_time - snippet_duration) / 2
        end_time = start_time + snippet_duration

    # Indices for snippet
    start_idx = np.searchsorted(time, start_time)
    end_idx = np.searchsorted(time, end_time)

    time_snippet = time[start_idx:end_idx]
    snippet_data = {axis: data[start_idx:end_idx] for axis, data in filtered_data.items()}

    # --- 2) COLLECT GLOBAL X/Y/Z RANGES FOR FIXED AXES ---
    all_x = []
    all_y = []
    all_z = []
    for axis, arr in snippet_data.items():
        if 'X' in axis:
            all_x.extend(arr)
        elif 'Y' in axis:
            all_y.extend(arr)
        elif 'Z' in axis:
            all_z.extend(arr)

    padding = 0.1  # 10% padding
    if all_x:  
        x_min, x_max = min(all_x), max(all_x)
        x_range = [x_min - abs(x_min)*padding, x_max + abs(x_max)*padding]
    else:
        x_range = [-1, 1]

    if all_y:
        y_min, y_max = min(all_y), max(all_y)
        y_range = [y_min - abs(y_min)*padding, y_max + abs(y_max)*padding]
    else:
        y_range = [-1, 1]

    if all_z:
        z_min, z_max = min(all_z), max(all_z)
        z_range = [z_min - abs(z_min)*padding, z_max + abs(z_max)*padding]
    else:
        z_range = [-1, 1]

    # --- 3) BUILD FRAMES ---
    frames = []
    # Larger step => fewer frames => smaller HTML
    step = 5  
    frame_indices = range(0, len(time_snippet), step)

    for i in frame_indices:
        frame_data = []

        # Accelerometer
        accel_available = all(ax in snippet_data for ax in ['AccelerX', 'AccelerY', 'AccelerZ'])
        if accel_available:
            frame_data.append(go.Scatter3d(
                x=snippet_data['AccelerX'][:i],
                y=snippet_data['AccelerY'][:i],
                z=snippet_data['AccelerZ'][:i],
                mode='lines',
                line=dict(color='#00FFEA', width=5),  # Neon aqua
                name='Accelerometer Trajectory',
                showlegend=(i == 0)
            ))

        # Gyroscope
        gyro_available = all(ax in snippet_data for ax in ['GyroX', 'GyroY', 'GyroZ'])
        if gyro_available:
            frame_data.append(go.Scatter3d(
                x=snippet_data['GyroX'][:i],
                y=snippet_data['GyroY'][:i],
                z=snippet_data['GyroZ'][:i],
                mode='lines',
                line=dict(color='#FF00FF', width=5),  # Neon magenta
                name='Gyroscope Trajectory',
                showlegend=(i == 0)
            ))

        frames.append(go.Frame(data=frame_data, name=str(i)))

    if not frames:
        logging.warning("No data to create frames for animation.")
        return

    initial_data = frames[0].data
    fig = go.Figure(data=initial_data, frames=frames)

    fig.update_layout(
        template=None,
        title=dict(
            text="Futuristic 3D Animated Head Movement",
            x=0.5,
            xanchor='center',
            y=0.95,
            font=dict(
                family="Consolas, monospace",
                size=20,
                color="#FFFFFF"
            )
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(
            yanchor="top",
            y=0.9,
            xanchor="left",
            x=0.02,
            font=dict(color="#FFFFFF", size=12),
            bgcolor="rgba(0,0,0,0)"
        ),
        scene=dict(
            xaxis=dict(
                title='X',
                range=x_range,
                backgroundcolor='black',
                color='white',
                showgrid=True,
                gridcolor='#333333',
                zerolinecolor='#333333',
                autorange=False
            ),
            yaxis=dict(
                title='Y',
                range=y_range,
                backgroundcolor='black',
                color='white',
                showgrid=True,
                gridcolor='#333333',
                zerolinecolor='#333333',
                autorange=False
            ),
            zaxis=dict(
                title='Z',
                range=z_range,
                backgroundcolor='black',
                color='white',
                showgrid=True,
                gridcolor='#333333',
                zerolinecolor='#333333',
                autorange=False
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0)
            )
        ),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            x=0.02,
            y=0.02,
            xanchor='left',
            yanchor='bottom',
            bgcolor='#222222',
            bordercolor='#FFFFFF',
            font=dict(color='#FFFFFF'),
            buttons=[
                dict(
                    label='Play',
                    method='animate',
                    args=[
                        None,
                        {
                            'frame': {'duration': 40, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 0}
                        }
                    ]
                )
            ]
        )]
    )

    output_directory = 'ProcessedGyroData'
    os.makedirs(output_directory, exist_ok=True)
    html_file = os.path.join(output_directory, f'{file_name}_3d_animation.html')
    fig.write_html(html_file)

    logging.info(f"Futuristic 3D animation saved to: {html_file}")


def export_3d_video(filtered_data, time, file_name, fps=10, output_format='gif'):
    """
    Export a short 3D "video" (GIF/MP4) by rendering each frame to an image using Plotly.
    WARNING: This can be slow and memory-intensive for large data.
    
    :param filtered_data: dict of channels -> np.array
    :param time: np.array of time stamps
    :param file_name: base name for the output video
    :param fps: frames per second
    :param output_format: 'gif' or 'mp4'
    """
    logging.info(f"Exporting 3D video as {output_format.upper()}...")

    snippet_duration = 10  # 10 seconds
    total_time = time[-1]
    if total_time < snippet_duration:
        start_time = 0
        end_time = total_time
    else:
        start_time = (total_time - snippet_duration) / 2
        end_time = start_time + snippet_duration

    start_idx = np.searchsorted(time, start_time)
    end_idx = np.searchsorted(time, end_time)
    time_snippet = time[start_idx:end_idx]
    snippet_data = {axis: data[start_idx:end_idx] for axis, data in filtered_data.items()}

    # We will re-use some code from plot_3d_animation but generate static images for each frame
    frames = []
    step = 10  # Increase step to reduce number of frames & speed up generation

    for i in range(0, len(time_snippet), step):
        # Build a figure for just the i-th frame
        accel_available = all(ax in snippet_data for ax in ['AccelerX', 'AccelerY', 'AccelerZ'])
        gyro_available = all(ax in snippet_data for ax in ['GyroX', 'GyroY', 'GyroZ'])

        fig = go.Figure()

        # Accelerometer trace
        if accel_available:
            fig.add_trace(go.Scatter3d(
                x=snippet_data['AccelerX'][:i],
                y=snippet_data['AccelerY'][:i],
                z=snippet_data['AccelerZ'][:i],
                mode='lines',
                line=dict(color='#00FFEA', width=5),
                name='Accelerometer'
            ))

        # Gyroscope trace
        if gyro_available:
            fig.add_trace(go.Scatter3d(
                x=snippet_data['GyroX'][:i],
                y=snippet_data['GyroY'][:i],
                z=snippet_data['GyroZ'][:i],
                mode='lines',
                line=dict(color='#FF00FF', width=5),
                name='Gyroscope'
            ))

        # We won't do fancy auto-range. We'll fix around min/max:
        # Gather min/max from snippet_data
        # For brevity, assume some range or reuse the logic from above:
        x_all = []
        y_all = []
        z_all = []
        for axis, arr in snippet_data.items():
            if 'X' in axis:
                x_all.extend(arr)
            elif 'Y' in axis:
                y_all.extend(arr)
            elif 'Z' in axis:
                z_all.extend(arr)

        if len(x_all) == 0 or len(y_all) == 0 or len(z_all) == 0:
            continue  # skip empty frames

        padding = 0.1
        x_min, x_max = min(x_all), max(x_all)
        y_min, y_max = min(y_all), max(y_all)
        z_min, z_max = min(z_all), max(z_all)
        x_range = [x_min - abs(x_min)*padding, x_max + abs(x_max)*padding]
        y_range = [y_min - abs(y_min)*padding, y_max + abs(y_max)*padding]
        z_range = [z_min - abs(z_min)*padding, z_max + abs(z_max)*padding]

        fig.update_layout(
            template=None,
            scene=dict(
                xaxis=dict(range=x_range, backgroundcolor='black', color='white', showgrid=True),
                yaxis=dict(range=y_range, backgroundcolor='black', color='white', showgrid=True),
                zaxis=dict(range=z_range, backgroundcolor='black', color='white', showgrid=True),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
            ),
            paper_bgcolor='black',
            width=800,
            height=600
        )

        # Convert figure to image in memory
        png_bytes = fig.to_image(format='png', engine="kaleido")  # Need kaleido installed
        frames.append(imageio.imread(png_bytes))

    if not frames:
        logging.warning("No frames generated; video export aborted.")
        return

    output_dir = 'ProcessedGyroData'
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f"{file_name}_3d_animation.{output_format}")

    if output_format.lower() == 'gif':
        imageio.mimsave(outfile, frames, fps=fps)
    else:
        # For mp4, we also can do:
        # imageio.mimsave(outfile, frames, fps=fps, codec='libx264')  # if you have ffmpeg installed
        imageio.mimsave(outfile, frames, fps=fps)

    logging.info(f"3D video file saved: {outfile}")


def save_event_times(cycles, file_name):
    """Save event times of valid cycles to text files."""
    logging.info("Saving event times to text files...")

    if not cycles:
        logging.warning("No valid cycles to save.")
        return

    left_times = [cycle['left_event']['time'] for cycle in cycles]
    right_times = [cycle['right_event']['time'] for cycle in cycles]

    output_directory = 'ProcessedGyroData'
    os.makedirs(output_directory, exist_ok=True)

    left_file = os.path.join(output_directory, f'{file_name}_left_movements.txt')
    right_file = os.path.join(output_directory, f'{file_name}_right_movements.txt')

    np.savetxt(left_file, left_times, fmt='%.4f')
    np.savetxt(right_file, right_times, fmt='%.4f')

    logging.info(f"Left movement times saved to: {left_file}")
    logging.info(f"Right movement times saved to: {right_file}")


def main():
    try:
        # Get the filename from the user
        file_name = get_filename()
        file_path = os.path.abspath(file_name)

        # Process the signal data
        results = process_signal(file_path)
        if not results or results[0] is None:
            logging.error("No valid data to process.")
            return

        (filtered_data,
         time,
         peaks,
         events,
         cycles,
         invalid_cycles,
         validity_score,
         primary_axis) = results

        # Plot the signals with detected movements
        plot_signals(filtered_data, time, peaks, events, primary_axis)

        # Plot valid and invalid cycles
        plot_cycles(filtered_data, time, events, cycles, invalid_cycles, primary_axis)

        # Plot 3D movement data and save as HTML (10-second snippet)
        base_file_name = os.path.splitext(os.path.basename(file_path))[0]
        plot_3d_animation(filtered_data, time, base_file_name)

        # OPTIONAL: Export a short 3D video (GIF) using imageio
        # Warning: This can be slow for large data
        export_3d_video(filtered_data, time, base_file_name, fps=10, output_format='gif')

        # Save event times of valid cycles to text files
        save_event_times(cycles, base_file_name)

        # Print validity score
        print(f"\nValidity Score: {validity_score * 100:.2f}%")

        logging.info("Processing completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
