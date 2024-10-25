from flask import Flask, request, render_template
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Route for the home page
@app.route('/')
def home():
    return render_template('base.html')

# Route to get the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    gre = int(request.form['gre'])
    toefl = int(request.form['toefl'])
    university_rating = int(request.form['university_rating'])
    sop = float(request.form['sop'])
    lor = float(request.form['lor'])
    cgpa = float(request.form['cgpa'])
    research = int(request.form['research'])

    # Prepare the data for prediction
    input_data = [[gre, toefl, university_rating, sop, lor, cgpa, research]]
    
    # Make prediction
    prediction = model.predict(input_data)[0]

    # Render the output on the webpage
    return render_template('base.html', prediction_text=f"Chance of Admission: {round(prediction * 100, 2)}%")

if __name__ == "__main__":
    app.run(debug=True)