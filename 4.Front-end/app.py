import os #importing os module
import pandas as pd
from flask import Flask, render_template, request
import pickle

# Import Plotly for creating interactive visualizations
import plotly.graph_objs as go
from plotly.subplots import make_subplots

app = Flask(__name__)

# Load the trained model
GB_exported = pickle.load(open('/Users/akhilsamvarghese/Desktop/Projects/NIDShield/nids/2.saved-models/GB', 'rb'))

# Create directory if it doesn't exist
DF_directory = "./DF/"
if not os.path.exists(DF_directory):
    os.makedirs(DF_directory)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Check if the file is present in the request
        if 'DFfile' not in request.files:
            return render_template('index.html', prediction='No file uploaded')

        file = request.files['DFfile']

        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':
            return render_template('index.html', prediction='No selected file')

        if file:
            # Save the file
            DF_path = os.path.join(DF_directory, file.filename)
            file.save(DF_path)

            # Load the DataFrame
            try:
                df = pd.read_pickle(DF_path)
            except Exception as e:
                return render_template('index.html', prediction=f'Error reading DataFrame: {str(e)}')

            # Make predictions
            try:
                predictions = GB_exported.predict(df)
            except Exception as e:
                return render_template('index.html', prediction=f'Error making predictions: {str(e)}')

            # Create a bar chart for displaying attack predictions
            attack_counts = pd.Series(predictions).value_counts()
            labels = attack_counts.index.tolist()
            values = attack_counts.values.tolist()

            # Plotly bar chart
            fig = go.Figure([go.Bar(x=labels, y=values)])
            fig.update_layout(title='Attack Predictions', xaxis_title='Attack Type', yaxis_title='Count')

            # Convert the Plotly figure to HTML
            plot_html = fig.to_html(full_html=False)

            return render_template('index.html', prediction=predictions.tolist(), plot=plot_html)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
