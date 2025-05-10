# Fraud Detection System

A web-based application that uses machine learning to predict the likelihood of fraudulent transactions based on transaction amount, date, and time.

## Overview

This application provides a sleek, modern user interface that allows users to input transaction details and receive immediate feedback on whether the transaction is likely to be fraudulent. The system uses a pre-trained machine learning model to analyze the transaction data and calculate a fraud probability score.

## Features

- Real-time fraud prediction
- Modern, responsive user interface
- Clear visual indicators for fraud likelihood
- Simple and intuitive design
- Analyzes transaction amount and timing patterns

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone or download this repository to your local machine

2. Navigate to the project directory
```bash
cd fraud
```

3. Install the required dependencies
```bash
pip install -r requirements.txt
```

4. Make sure you have the pre-trained model file `fraud_model.pkl` in the project directory

## Usage

1. Start the application server
```bash
python app.py
```

2. Open your web browser and navigate to http://127.0.0.1:5000/

3. Enter the transaction details:
   - Transaction Date
   - Transaction Time
   - Transaction Amount

4. Click "Check Transaction" to get the fraud prediction results

5. The result will be displayed with a color-coded indicator:
   - Green: Transaction is likely legitimate
   - Red: Transaction is likely fraudulent

## Project Structure

- `app.py` - Main Flask application file
- `OM.py` - Model operations and utility functions
- `requirements.txt` - Python dependencies
- `fraud_model.pkl` - Pre-trained machine learning model
- `templates/` - HTML templates for the web interface
  - `index.html` - Main page template
- `static/` - Static files (CSS, JS, images)
  - `style.css` - Custom styling
  - `portfolio.mp4` - Background video

## Technologies Used

- **Backend**: Flask (Python web framework)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn
- **Frontend**: HTML, CSS, JavaScript
- **UI Design**: Font Awesome icons, Google Fonts

## Model Information

The application uses a pre-trained machine learning model that analyzes:
- Time of the transaction (seconds since midnight)
- Transaction amount

The model returns a probability score indicating the likelihood of fraud, which is then presented to the user in an easily understandable format.
