# Bank Churn Prediction

This project implements a simple Artificial Neural Network (ANN) and Recurrent Neural Network (RNN) to predict whether a user has exited the bank. The output is binary (0 or 1), where:
- `0`: User has not exited the bank.
- `1`: User has exited the bank.

The implementation uses the **Churn Modeling Dataset**, leveraging Python with TensorFlow, Keras, and Streamlit.

---

## Features
- Build and train simple ANN and RNN models.
- Predict user churn (exit status) based on input features.
- Interactive user interface with **Streamlit** for predictions.

---

## Tech Stack
- **Python**
- **TensorFlow**
- **Keras**
- **Streamlit**

---

## Dataset
- **Name**: Churn Modeling Dataset
- **Source**: Typically sourced from Kaggle or similar repositories.
- **Description**: Contains information about bank customers, including demographics, account details, and churn status.
- **Target Column**: `Exited` (binary column representing if a user exited the bank).

---

## Installation
### Prerequisites
1. Python 3.8+
2. Recommended: Create a virtual environment

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. **Train the Models:**
   - Edit the dataset path in the code if necessary.
   - Run the training script:
     ```bash
     python train_models.py
     ```
   - This will train both ANN and RNN models and save them as `.h5` files.

2. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
   - This will launch an interactive UI for making predictions.

---

## File Structure
- `data/`: Contains the dataset 
- `models/`: Stores trained model files.
- `train_models.py`: Script to train ANN and RNN models.
- `app.py`: Streamlit application for predictions.
- `requirements.txt`: List of required Python packages.

---

## Model Architecture
### 1. **ANN Model:**
   - Input layer with features from the dataset.
   - Two hidden layers with ReLU activation.
   - Output layer with a single neuron and sigmoid activation.

### 2. **RNN Model:**
   - Input layer with sequential data.
   - LSTM layers with dropout for regularization.
   - Fully connected layer with sigmoid activation.

---

## Output
- **Binary Output**:
  - `0`: User has not exited.
  - `1`: User has exited.

---

## Future Work
- Improve model performance using hyperparameter tuning.
- Integrate additional features for better predictions.
- Deploy the application on a cloud platform .

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
- The creators of the Churn Modeling Dataset.
- TensorFlow and Keras documentation.
- Streamlit community for the interactive UI framework.

