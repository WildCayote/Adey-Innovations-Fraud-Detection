from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

# Method for loading data from a CSV file
def load_fraud_data():
    try:
        print('Loading fraud data from CSV...')
        data = pd.read_csv('./data/fraud_merged.csv')
        print('Fraud data loaded successfully.')
        return data
    except Exception as e:
        print(f'Error loading fraud data: {e}')
        return None

# Endpoint to provide summary statistics of fraud data
@app.route('/fraud_summary_statistics', methods=['GET'])
def fraud_summary_statistics():
    # Load fraud data
    data = load_fraud_data()

    # Return an error if data could not be loaded
    if data is None:
        return jsonify({'error': 'Data could not be loaded.'}), 500

    # Calculate total transactions and fraud cases
    total_transactions = len(data)
    total_fraud_cases = len(data[data['class'] == 1])
    fraud_percentage = (total_fraud_cases / total_transactions) * 100

    # Compile summary statistics into a dictionary
    summary = {
        'total_transactions': total_transactions,
        'total_fraud_cases': total_fraud_cases,
        'fraud_percentage': fraud_percentage
    }
    return jsonify(summary)

# Endpoint to provide fraud trends over time
@app.route('/fraud_trends_over_time', methods=['GET'])
def fraud_trends_over_time():
    # Load fraud data
    data = load_fraud_data()

    # Return an error if data could not be loaded
    if data is None:
        return jsonify({'error': 'Data could not be loaded.'}), 500

    # Parse purchase date to a monthly period format
    data['purchase_date'] = pd.to_datetime(data['purchase_time']).dt.to_period('M').dt.strftime('%b %Y')

    # Group data by month and calculate transaction and fraud case counts
    trend_data = data.groupby('purchase_date').agg({
        'user_id': 'count',
        'class': lambda x: (x == 1).sum()
    }).reset_index()

    # Rename columns for clarity
    trend_data.rename(columns={'user_id': 'transaction_count', 'class': 'fraud_cases'}, inplace=True)

    return jsonify(trend_data.to_dict(orient='records'))

# Endpoint to provide fraud statistics by geographic location
@app.route('/fraud_statistics_by_location', methods=['GET'])
def fraud_statistics_by_location():
    # Load fraud data
    data = load_fraud_data()

    # Return an error if data could not be loaded
    if data is None:
        return jsonify({'error': 'Data could not be loaded.'}), 500

    # Group data by country and calculate transaction and fraud case counts
    location_data = data.groupby('country').agg({
        'user_id': 'count',
        'class': lambda x: (x == 1).sum()
    }).reset_index()

    # Rename columns for clarity
    location_data.rename(columns={'user_id': 'transaction_count', 'class': 'fraud_cases'}, inplace=True)

    return jsonify(location_data.to_dict(orient='records'))

# Endpoint to provide fraud cases by the most frequent device and browser combinations
@app.route('/top_fraud_device_browser_combinations', methods=['GET'])
def top_fraud_device_browser_combinations():
    # Load fraud data
    data = load_fraud_data()

    # Return an error if data could not be loaded
    if data is None:
        return jsonify({'error': 'Data could not be loaded.'}), 500

    # Filter data to include only fraudulent cases
    fraud_data = data[data['class'] == 1]

    # Group data by device and browser, and count fraud cases
    device_browser_data = (
        fraud_data.groupby(['device_id', 'browser'])
        .size()
        .reset_index(name='fraud_cases')
    )

    # Select top 10 device-browser combinations with the most fraud cases
    top_device_browser_data = device_browser_data.nlargest(10, 'fraud_cases')

    return jsonify(top_device_browser_data.to_dict(orient='records'))

# Run the application
if __name__ == '__main__':
    app.run(debug=True)