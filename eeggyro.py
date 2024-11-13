import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging
import warnings
import plotly.graph_objects as go

# --- Configuration Parameters ---
PROMINENCE_THRESHOLD = 0.015  # Adjusted for better peak detection
MIN_PEAK_DISTANCE_SECONDS = 0.7  # Adjusted to avoid overlapping peaks
CUTOFF_FREQUENCY = 3  # Low-pass filter cutoff frequency
VALID_DURATION_RANGE = (0.7, 1.2)  # Valid movement duration range in seconds
FS = None  # Sampling frequency (to be dynamically set)
LEFT_FIRST = True  # Ensure cycles start with left head movement
VALIDITY_SCORE_THRESHOLD = 0.8  # Threshold for a valid cycle sequence

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
    logging.info(f"Low-pass filtering applied to the signal.")
    return filtered_signal


def extract_features(signal_segments, time_segment):
    """Extract features from all axes."""
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
            # Start with the first movement as is
            corrected_events.append(event)
            last_movement = event['movement']
        else:
            # Enforce alternation
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
        # Check if the sequence is Left followed by Right
        if first_event['movement'] == 'Left Head Movement' and second_event['movement'] == 'Right Head Movement':
            duration = second_event['time'] - first_event['time']
            if VALID_DURATION_RANGE[0] <= duration <= VALID_DURATION_RANGE[1]:
                cycles.append({'start_time': first_event['time'], 'end_time': second_event['time'], 'duration': duration,
                               'left_event': first_event, 'right_event': second_event})
            else:
                invalid_cycles.append({'start_time': first_event['time'], 'end_time': second_event['time'], 'duration': duration})
            # Move forward by one event to allow overlapping cycles
            i += 1
        else:
            # Invalid sequence, treat the first event as invalid and move to the next
            invalid_cycles.append({'time': first_event['time'], 'movement': first_event['movement']})
            i += 1
    # Handle the last event if it's not been processed
    if i == len(events) - 1:
        last_event = events[-1]
        invalid_cycles.append({'time': last_event['time'], 'movement': last_event['movement']})
    return cycles, invalid_cycles


def calculate_validity_score(cycles, invalid_cycles):
    """Calculate the validity score based on valid and invalid cycles."""
    total_cycles = len(cycles) + len(invalid_cycles)
    validity_score = len(cycles) / total_cycles if total_cycles > 0 else 0
    return validity_score


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

    # Step 2: Select gyroscope and accelerometer channels
    gyro_channels = ['GyroX', 'GyroY', 'GyroZ']
    accel_channels = ['AccelerX', 'AccelerY', 'AccelerZ']

    # Validate available channels
    available_channels = raw.info['ch_names']
    missing_channels = [ch for ch in gyro_channels + accel_channels if ch not in available_channels]
    if missing_channels:
        logging.warning(f"Missing channels: {missing_channels}. The analysis will proceed with available channels.")

    # Load available channels
    signal_data = {}
    for channel in gyro_channels + accel_channels:
        if channel in available_channels:
            signal_data[channel] = raw.copy().pick_channels([channel]).get_data().flatten()

    # Apply filtering to all axes
    filtered_data = {}
    for axis, signal in signal_data.items():
        # Handle missing data
        imputer = SimpleImputer(strategy='mean')
        signal_imputed = imputer.fit_transform(signal.reshape(-1, 1)).ravel()
        filtered_signal = low_pass_filter(signal_imputed, cutoff=CUTOFF_FREQUENCY, fs=FS)
        filtered_data[axis] = filtered_signal

    # Step 3: Detect peaks using the primary gyroscope axis (GyroX if available)
    primary_axis = 'GyroX' if 'GyroX' in filtered_data else next(iter(filtered_data))
    distance = int(MIN_PEAK_DISTANCE_SECONDS * FS)
    peaks, _ = find_peaks(filtered_data[primary_axis], prominence=PROMINENCE_THRESHOLD, distance=distance)
    logging.info(f"Detected {len(peaks)} peaks in the primary axis ({primary_axis}).")

    # Step 4: Extract features around each peak
    window_size = int(0.5 * FS)  # 0.5 second window on each side
    feature_list = []
    labels = []

    time = np.arange(len(filtered_data[primary_axis])) / FS
    for peak in peaks:
        start_idx = max(0, peak - window_size)
        end_idx = min(len(filtered_data[primary_axis]), peak + window_size)
        time_segment = time[start_idx:end_idx]

        # Extract signal segments for all axes
        signal_segments = {axis: data[start_idx:end_idx] for axis, data in filtered_data.items()}

        # Extract features
        features = extract_features(signal_segments, time_segment)
        feature_list.append(features)

        # Improved label assignment logic
        gyro_x_value = filtered_data['GyroX'][peak] if 'GyroX' in filtered_data else 0
        gyro_y_value = filtered_data['GyroY'][peak] if 'GyroY' in filtered_data else 0

        # Use GyroY to improve distinction between left and right
        if gyro_x_value > 0 and gyro_y_value > 0:
            label = 'left'
        elif gyro_x_value < 0 and gyro_y_value < 0:
            label = 'right'
        else:
            label = 'unknown'
        labels.append(label)

    # Remove unknown labels
    valid_indices = [i for i, label in enumerate(labels) if label != 'unknown']
    feature_df = pd.DataFrame([feature_list[i] for i in valid_indices])
    labels = np.array([labels[i] for i in valid_indices])
    logging.info("Feature extraction completed.")

    # Balance the training data
    df = feature_df.copy()
    df['label'] = labels
    min_class_count = df['label'].value_counts().min()
    df_balanced = df.groupby('label', group_keys=False).apply(lambda x: x.sample(min_class_count, random_state=42))
    X = df_balanced.drop('label', axis=1)
    y = df_balanced['label']

    # Step 5: Train and predict using a machine learning model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    logging.info("Machine learning model trained.")

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    logging.info("Model evaluation:")
    logging.info("\n" + classification_report(y_test, y_pred))

    # Predict on all peaks
    X_full_scaled = scaler.transform(feature_df)
    predictions = model.predict(X_full_scaled)

    # Step 6: Prepare events for the report
    events = [{'time': time[peaks[valid_indices[idx]]], 'movement': predictions[idx].capitalize() + ' Head Movement'} for idx in range(len(predictions))]

    # Step 7: Enforce alternating movements
    corrected_events = enforce_alternating_movements(events)

    # Step 8: Detect cycles and calculate validity score
    cycles, invalid_cycles = detect_cycles(corrected_events)
    validity_score = calculate_validity_score(cycles, invalid_cycles)

    return filtered_data, time, peaks, corrected_events, cycles, invalid_cycles, validity_score, primary_axis


def plot_signals(filtered_data, time, peaks, events, primary_axis):
    """Plot the signals with detected movements."""
    logging.info("Plotting signals...")
    plt.figure(figsize=(12, 6))
    plt.plot(time, filtered_data[primary_axis], label=f'Filtered {primary_axis}')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal Amplitude')
    plt.title('Gyroscope Signal with Detected Movements')

    # Plot events
    event_times = [event['time'] for event in events]
    event_amplitudes = [filtered_data[primary_axis][np.abs(time - t).argmin()] for t in event_times]
    movement_labels = [event['movement'] for event in events]
    colors = ['green' if 'Left' in label else 'blue' for label in movement_labels]
    plt.scatter(event_times, event_amplitudes, color=colors, label='Detected Movements')

    plt.legend()
    plt.show()


def plot_cycles(filtered_data, time, events, cycles, invalid_cycles, primary_axis):
    """Plot valid and invalid cycles."""
    logging.info("Plotting cycles...")
    plt.figure(figsize=(12, 6))
    plt.plot(time, filtered_data[primary_axis], label=f'Filtered {primary_axis}')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal Amplitude')
    plt.title('Valid and Invalid Cycles')

    # Plot valid cycles
    for cycle in cycles:
        start_idx = np.abs(time - cycle['start_time']).argmin()
        end_idx = np.abs(time - cycle['end_time']).argmin()
        plt.axvspan(time[start_idx], time[end_idx], color='green', alpha=0.3)

    # Plot invalid cycles
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
    """Plot 3D movement data as an animation and save as HTML."""
    logging.info("Plotting 3D animation...")

    # Extract a 15-second snippet from the middle of the data
    total_time = time[-1]
    start_time = (total_time - 15) / 2
    end_time = start_time + 15

    # Find indices corresponding to the snippet
    start_idx = np.searchsorted(time, start_time)
    end_idx = np.searchsorted(time, end_time)

    # Subset the data
    time_snippet = time[start_idx:end_idx]
    filtered_data_snippet = {axis: data[start_idx:end_idx] for axis, data in filtered_data.items()}

    # Prepare data for animation frames
    frames = []
    frame_indices = range(0, len(time_snippet), 10)  # Adjust frame step for performance

    for i in frame_indices:
        frame_data = []
        # Add 3D scatter of accelerometer data
        accel_available = all(axis in filtered_data_snippet for axis in ['AccelerX', 'AccelerY', 'AccelerZ'])
        if accel_available:
            frame_data.append(go.Scatter3d(
                x=filtered_data_snippet['AccelerX'][:i],
                y=filtered_data_snippet['AccelerY'][:i],
                z=filtered_data_snippet['AccelerZ'][:i],
                mode='lines',
                line=dict(color='blue'),
                name='Accelerometer Trajectory'
            ))
        # Add 3D scatter of gyroscope data
        gyro_available = all(axis in filtered_data_snippet for axis in ['GyroX', 'GyroY', 'GyroZ'])
        if gyro_available:
            frame_data.append(go.Scatter3d(
                x=filtered_data_snippet['GyroX'][:i],
                y=filtered_data_snippet['GyroY'][:i],
                z=filtered_data_snippet['GyroZ'][:i],
                mode='lines',
                line=dict(color='red'),
                name='Gyroscope Trajectory'
            ))
        frames.append(go.Frame(data=frame_data, name=str(i)))

    # Initial data for the first frame
    initial_data = frames[0].data if frames else []

    fig = go.Figure(data=initial_data, frames=frames)

    # Set up the layout with buttons and sliders
    fig.update_layout(
        title="3D Animated Head Movement (15-second Snippet)",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(label='Play',
                          method='animate',
                          args=[None, {'frame': {'duration': 50, 'redraw': True},
                                       'fromcurrent': True}])]
        )]
    )

    # Save the figure as an HTML file
    output_directory = 'ProcessedGyroData'
    os.makedirs(output_directory, exist_ok=True)
    html_file = os.path.join(output_directory, f'{file_name}_3d_animation.html')
    fig.write_html(html_file)
    logging.info(f"3D animation saved to: {html_file}")

    # Open the HTML file in the default web browser
    import webbrowser
    webbrowser.open(html_file)


def save_event_times(cycles, file_name):
    """Save event times of valid cycles to text files."""
    logging.info("Saving event times to text files...")

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
        filtered_data, time, peaks, events, cycles, invalid_cycles, validity_score, primary_axis = results

        # Plot the signals with detected movements
        plot_signals(filtered_data, time, peaks, events, primary_axis)

        # Plot valid and invalid cycles
        plot_cycles(filtered_data, time, events, cycles, invalid_cycles, primary_axis)

        # Plot 3D movement data and save as HTML
        base_file_name = os.path.splitext(os.path.basename(file_path))[0]
        plot_3d_animation(filtered_data, time, base_file_name)

        # Save event times of valid cycles to text files
        save_event_times(cycles, base_file_name)

        # Print validity score
        print(f"\nValidity Score: {validity_score * 100:.2f}%")

        logging.info("Processing completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
