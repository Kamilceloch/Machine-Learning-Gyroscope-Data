# **Gyroscope Cycle Analysis**

This Python script processes gyroscope data from a BDF file to analyze head movement cycles, detect valid and invalid peaks, and classify left-right movements during cycling exercises. The results include detailed visualizations, HTML reports, and text files for further analysis.

---

## **Features**

- **Cycle Detection**: Automatically identifies valid cycles based on left-right head movements.
- **Invalid Peak Handling**: Skips incomplete or compromised cycles.
- **Visualization**: Generates interactive graphs with valid cycles, detected peaks, and signal phases.
- **HTML Report**: Provides a detailed analysis report with a validity score and cycle details.
- **Text Output**: Saves valid cycle start and end times to a text file.
- **Robust Error Handling**: Handles missing files, invalid inputs, and unexpected data gracefully.

---

## **Installation**

### **Requirements**
- Python 3.7 or higher
- The following Python libraries:
  - `mne`
  - `numpy`
  - `plotly`
  - `scipy`
  - `scikit-learn`

### **Setup**
1. **Clone or Download the Repository**
   - Clone this repository or download the script file.

2. **Install Required Packages**
   - Use the following command to install all necessary libraries:
     ```bash
     pip install mne numpy plotly scipy scikit-learn
     ```

3. **Place Your BDF File**
   - Ensure the BDF file is in the same directory as the script.

---

## **Usage**

1. **Run the Script**
   - Execute the script in your terminal or command prompt:
     ```bash
     python gyro_analysis.py
     ```

2. **Input the Filename**
   - When prompted, enter the name of your BDF file.

3. **View the Results**
   - The script will:
     - Generate an HTML report with visualizations and save it in the `ProcessedGyroData` directory.
     - Save a text file with valid cycle start and end times in the same directory.

---

## **Outputs**

1. **HTML Report**
   - Found in the `ProcessedGyroData` directory.
   - Includes:
     - Validity score as a percentage of detected cycles.
     - Ordered list of valid cycles.
     - Interactive graph with valid cycles, detected peaks, and signal phases.

2. **Text File**
   - Contains start and end times of valid cycles, saved in the `ProcessedGyroData` directory.

---

## **Customization**

You can adjust parameters in the script to match your data's characteristics:
- **`PROMINENCE_THRESHOLD`**: Minimum prominence of peaks to be detected.
- **`MIN_PEAK_DISTANCE_SECONDS`**: Minimum time between peaks (in seconds).
- **`VALID_DURATION_RANGE`**: Range of valid cycle durations (in seconds).
- **`LEFT_FIRST`**: Define whether cycles start with left or right movement.

---

## **Directory Structure**

```
Project Directory
├── gyro_analysis.py       # The main script
├── EkatBio.bdf            # Your input BDF file (example)
├── ProcessedGyroData/     # Output directory for reports and text files
│   ├── EkatBio_cycle_analysis.html
│   ├── EkatBio_valid_cycles_times.txt
```

---

## **Troubleshooting**

1. **File Not Found Error**:
   - Ensure the BDF file is in the same directory as the script.
   - Check for typos in the filename.

2. **Missing Libraries**:
   - Reinstall the required libraries using:
     ```bash
     pip install -r requirements.txt
     ```

3. **Unexpected Results**:
   - Adjust the configuration parameters to better match your data.

---

## **Contact**

For questions or issues, please contact the developer:

**Dr. Marcelo Bigliassi**  

---

## **License**

This project is licensed under the MIT License. See the LICENSE file for details.

---
