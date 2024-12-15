import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import joblib
import openai
from models import *


app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'e6314c111b52caa3ad3d60239f0067c6')  # Replace with your actual secret key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ecommerce.db' # Use SQLite for simplicity
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)


# Ensure the upload and processed folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Initialize OpenAI client
openai.api_key='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'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Define the list of required patient data features
REQUIRED_FEATURES = [
    'Blood sodium',
    'Chloride',
    'age',
    'SP O2',
    'EF',
    'temperature',
    'PH',
    'deficiencyanemias',
    'Creatinine',
    'BMI',
    'glucose',
    'Leucocyte',
    'heart rate',
    'Anion gap',
    'Diastolic blood pressure',
    'Blood potassium',
    'hematocrit',
    'RBC',
    'Lactic acid',
]

# Load the pre-trained model
model = joblib.load('model.pkl')  # Ensure 'model.pkl' is in the project root

# Load the scaler
scaler = joblib.load('scaler.pkl')  # Ensure 'scaler.pkl' is in the project root

def generate_prompt(patient_data, medication_inventory):
    """
    Use OpenAI ChatCompletion API to generate allocation decisions based on patient data and medication inventory.
    """
    # Debug prints to verify inputs
    print("Patient Data:", patient_data)
    print("Medication Inventory:", medication_inventory)

    system_message = (
        "You are a medical assistant tasked with allocating medications based on patient severity and available inventory. "
        f"Medication Inventory: {', '.join([f'{med}: {qty} units' for med, qty in medication_inventory.items()])}. "
        "For each patient, determine whether to allocate medication, specify which medications and quantities, and provide reasoning."
    )

    user_message = (
        f"Patient Data:\n"
        f"- Severity Level: {patient_data['Severity Level']}\n"
        f"- Age: {patient_data['age']}\n"
        f"- Blood Pressure: {patient_data['Diastolic blood pressure']}\n"
        f"- Glucose Level: {patient_data['glucose']}\n"
        f"- Heart Rate: {patient_data['heart rate']}\n"
        f"- SP O2: {patient_data['SP O2']}\n"
        f"- Case Details: {patient_data['PROMPT']}\n\n"
        "Based on the severity level, patient data, and available inventory, decide:\n"
        "1. Should medication be allocated to this patient? (Yes/No Only nothing else)\n"
        "2. If yes, which medications should be allocated, and in what quantities (only give name,quantity)?\n"
        "3. Provide reasoning for your decision."
    )

    try:
        # Make sure openai.api_key is set before calling this function
        # For example:
        # import openai
        # openai.api_key = "your-api-key"

        print("Sending request to OpenAI...")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        print("Response received:", response)
        gpt_response = response.choices[0].message.content.strip()
        print("GPT Response:", gpt_response)
        return gpt_response
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return "Allocation Decision: No\nMedications Allocated: N/A\nReasoning: Error in processing."
from functools import wraps

def login_required(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            flash('You need to log in first.', 'warning')
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    return decorated_function

import os

def is_csv_empty(filepath):
    return os.path.getsize(filepath) == 0


@app.route('/page')
@login_required 
def index():
    medications = session.get('medications_inventory')
    allocations = None
    if session.get('final_file') and os.path.exists(session['final_file']):
        try:
            allocations = pd.read_csv(session['final_file'], dtype={'ID': int}).to_dict(orient='records')
        except Exception as e:
            print(f"Error reading final_results.csv: {e}")
            allocations = None
    return render_template('index.html', medications=medications, allocations=allocations)


@app.route('/page/process', methods=['POST'])
@login_required 
def process_files():
    if 'files[]' not in request.files:
        flash('No patient data file part')
        return redirect(url_for('index'))
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        flash('No selected patient data file')
        return redirect(url_for('index'))
    
    processed_data = []  # To store processed DataFrames

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            if is_csv_empty(filepath):
                flash(f"File {filename} is empty.")
                return redirect(url_for('index'))
            
            try:
                # Ensure 'ID' is read as integer
                df = pd.read_csv(filepath, dtype={'ID': int})
            except ValueError:
                flash(f"File {filename} has invalid data types. Ensure 'ID' is an integer.")
                return redirect(url_for('index'))
            except Exception as e:
                flash(f"Error reading file {filename}: {e}")
                return redirect(url_for('index'))
            
            if 'ID' not in df.columns:
                flash(f"File {filename} must contain an 'ID' column.")
                return redirect(url_for('index'))
            
            missing_features = [feature for feature in REQUIRED_FEATURES if feature not in df.columns]
            if missing_features:
                flash(f'File {filename} is missing features: {", ".join(missing_features)}')
                return redirect(url_for('index'))
            
            X = df[REQUIRED_FEATURES]

            if X.isnull().values.any():
                flash(f'File {filename} contains missing values in required features.')
                return redirect(url_for('index'))
            
            X_scaled = scaler.transform(X)
            severity_labels = model.predict(X_scaled)
            df['Severity_Label'] = severity_labels
            print(df['PROMPT'])

            processed_data.append(df)
        else:
            flash(f'File {file.filename} is not a valid CSV.')
            return redirect(url_for('index'))
    
    if not processed_data:
        flash('No data processed.')
        return redirect(url_for('index'))
    
    combined_df = pd.concat(processed_data, ignore_index=True)
    combined_df.sort_values(by='Severity_Label', ascending=False, inplace=True)
    
    processed_filename = 'processed_data.csv'
    processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
    combined_df.to_csv(processed_filepath, index=False)
    
    session['processed_file'] = processed_filepath
    session['final_file'] = os.path.join(app.config['PROCESSED_FOLDER'], 'final_results.csv')  # Initialize final file
    
    # If final_results.csv does not exist, create it with headers
    if not os.path.exists(session['final_file']):
        final_df = pd.DataFrame(columns=['ID', 'Severity_Label', 'Allocation Decision', 'Medications Allocated', 'Reasoning'])
        final_df.to_csv(session['final_file'], index=False)
    
    flash('Severity classification completed successfully.')
    return redirect(url_for('index'))


@app.route('/page/upload_meds', methods=['POST'])
@login_required 
def upload_medications():
    if 'meds_file' not in request.files:
        flash('No medication data file part')
        return redirect(url_for('index'))
    meds_file = request.files['meds_file']
    if meds_file.filename == '':
        flash('No selected medication data file')
        return redirect(url_for('index'))
    
    if meds_file and allowed_file(meds_file.filename):
        filename = secure_filename(meds_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        meds_file.save(filepath)
        
        try:
            meds_df = pd.read_csv(filepath)
        except Exception as e:
            flash(f"Error reading medications file: {e}")
            return redirect(url_for('index'))
        
        required_columns = ['Medication Name', 'Quantity']
        if not all(col in meds_df.columns for col in required_columns):
            flash(f'Medication file must contain the following columns: {", ".join(required_columns)}')
            return redirect(url_for('index'))
        
        meds_df = meds_df[['Medication Name', 'Quantity']]
        meds_df['Medication Name'] = meds_df['Medication Name'].astype(str).str.strip()
        meds_df['Quantity'] = pd.to_numeric(meds_df['Quantity'], errors='coerce')
        
        if meds_df['Quantity'].isnull().any():
            flash('Some medication quantities are missing or invalid.')
            return redirect(url_for('index'))
        
        meds_df = meds_df.groupby('Medication Name', as_index=False)['Quantity'].sum()
        medications_inventory = pd.Series(meds_df.Quantity.values, index=meds_df['Medication Name']).to_dict()
        session['medications_inventory'] = medications_inventory
        
        flash('Medications data uploaded and processed successfully.')
        return redirect(url_for('index'))
    else:
        flash('Invalid file type for medications data. Please upload a CSV file.')
        return redirect(url_for('index'))
    
    
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the user exists in the database
        user = User.query.filter_by(username=username).first()

        if user and user.password == password:  # Compare plaintext passwords (simplified)
            session.permanent = False
            session['logged_in'] = True
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()  # Clear all session data
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/')
def initial():
    if not session.get('logged_in'):
        flash('You need to log in first.', 'warning')
        return redirect(url_for('login'))
    return render_template('index.html', username=session.get('username'))





    
def parse_gpt_response(gpt_response):
    """
    Parses the GPT response to extract Allocation Decision, Medications Allocated, and Reasoning.
    Supports both hyphen-prefixed and comma-separated medication lists.
    """
    allocation_decision = 'No'
    medications_allocated = {}
    reasoning = ''

    # Split the response into lines
    lines = gpt_response.split('\n')

    for line in lines:
        line = line.strip()
        if line.startswith('1.'):
            # Allocation Decision
            decision = line[2:].strip().lower()
            if decision.startswith('yes'):
                allocation_decision = 'Yes'
            elif decision.startswith('no'):
                allocation_decision = 'No'
        elif line.startswith('2.'):
            # Medications Allocated
            meds_text = line[2:].strip()
            # Check if medications are listed with hyphens
            if meds_text.startswith('-'):
                # Medications listed with hyphens
                meds_lines = meds_text.split('\n')
                for med_line in meds_lines:
                    med_line = med_line.strip('- ').strip()
                    if ':' in med_line:
                        med_name, qty = med_line.split(':', 1)
                        med_name = med_name.strip()
                        try:
                            qty = int(qty.strip().split()[0])  # Extract the first integer
                        except:
                            qty = 0
                        medications_allocated[med_name] = qty
            else:
                # Medications listed as comma-separated
                meds = meds_text.split(',')
                for med in meds:
                    if ':' in med:
                        med_name, qty = med.split(':', 1)
                        med_name = med_name.strip()
                        try:
                            qty = int(qty.strip().split()[0])  # Extract the first integer
                        except:
                            qty = 0
                        medications_allocated[med_name] = qty
        elif line.startswith('3.'):
            # Reasoning
            reasoning = line[2:].strip()
            # If reasoning spans multiple lines, concatenate them
            idx = lines.index(line) + 1
            while idx < len(lines):
                additional_line = lines[idx].strip()
                if additional_line and not additional_line.startswith(tuple(['1.', '2.', '3.'])):
                    reasoning += ' ' + additional_line
                else:
                    break
                idx += 1

    # If no medications were allocated, set as 'N/A'
    if not medications_allocated and allocation_decision.lower() == 'yes':
        medications_allocated = 'N/A'

    return {
        'Allocation Decision': allocation_decision,
        'Medications Allocated': medications_allocated if medications_allocated else 'N/A',
        'Reasoning': reasoning if reasoning else 'No reasoning provided.'
    }

@app.route('/page/allocate_all', methods=['POST'])
@login_required 
def allocate_all_medications():
    processed_filepath = session.get('processed_file')
    medications_inventory = session.get('medications_inventory')
    final_filepath = session.get('final_file')
    
    if not processed_filepath or not os.path.exists(processed_filepath):
        flash('Processed patient data not found. Please upload and process files first.')
        return redirect(url_for('index'))
    if not medications_inventory:
        flash('Medications inventory not found. Please upload medications data first.')
        return redirect(url_for('index'))
    
    try:
        df = pd.read_csv(processed_filepath, dtype={'ID': int})
        final_df = pd.read_csv(final_filepath, dtype={'ID': int})
    except pd.errors.EmptyDataError:
        flash('Final allocation file is empty. Initializing with headers.')
        final_df = pd.DataFrame(columns=['ID', 'Severity_Label', 'Allocation Decision', 'Medications Allocated', 'Reasoning'])
        final_df.to_csv(final_filepath, index=False)
    except Exception as e:
        flash(f"Error reading CSV files: {e}")
        return redirect(url_for('index'))
    
    allocation_decisions = []
    allocated_ids = set(final_df['ID'].tolist())
    
    for index, row in df.iterrows():
        patient_id = row['ID']
        severity_level = row['Severity_Label']
        
        if patient_id in allocated_ids:
            continue  # Skip already allocated patients
        
        patient_data = {
            'Severity Level': severity_level,
            'age': row.get('age', 'N/A'),
            'Diastolic blood pressure': row.get('Diastolic blood pressure', 'N/A'),
            'glucose': row.get('glucose', 'N/A'),
            'heart rate': row.get('heart rate', 'N/A'),
            'SP O2': row.get('SP O2', 'N/A'),
            'PROMPT': row.get('PROMPT', 'N/A')  # Ensure 'PROMPT' is included with correct casing
        }
        
        gpt_output = generate_prompt(patient_data, medications_inventory)
        
        # Parse GPT response using the new function
        parsed_response = parse_gpt_response(gpt_output)
        
        allocation_decision = parsed_response['Allocation Decision']
        medications_allocated = parsed_response['Medications Allocated']
        reasoning = parsed_response['Reasoning']
        
        allocated_meds = []
        if allocation_decision.lower() == 'yes' and isinstance(medications_allocated, dict):
            for med_name, qty in medications_allocated.items():
                if med_name in medications_inventory and qty > 0:
                    medications_inventory[med_name] -= qty
                    if medications_inventory[med_name] < 0:
                        medications_inventory[med_name] = 0
                    allocated_meds.append(f"{med_name}: {qty}")
        elif allocation_decision.lower() == 'no':
            medications_allocated = 'N/A'
        
        allocation_decision_row = {
            'ID': patient_id,
            'Severity_Label': severity_level,
            'Allocation Decision': allocation_decision,
            'Medications Allocated': ', '.join(allocated_meds) if allocated_meds else medications_allocated,
            'Reasoning': reasoning
        }
        allocation_df = pd.DataFrame([allocation_decision_row])
        
        allocation_decisions.append(allocation_decision_row)
        allocated_ids.add(patient_id)  # Mark as allocated
    
    if allocation_decisions:
        allocation_df = pd.DataFrame(allocation_decisions)
        # Append to final_results.csv
        try:
            allocation_df.to_csv(final_filepath, mode='a', header=False, index=False)
        except Exception as e:
            flash(f"Error writing to final results file: {e}")
            return redirect(url_for('index'))
        
        # Save updated medications inventory
        try:
            meds_df = pd.DataFrame(list(medications_inventory.items()), columns=['Medication Name', 'Quantity'])
            meds_filepath = os.path.join(app.config['PROCESSED_FOLDER'], 'medications_inventory.csv')
            meds_df.to_csv(meds_filepath, index=False)
        except Exception as e:
            flash(f"Error saving medications inventory: {e}")
            return redirect(url_for('index'))
        
        # Update session
        session['medications_inventory'] = medications_inventory
        
        # Prepare flash message
        meds_allocated_list = [alloc['Medications Allocated'] for alloc in allocation_decisions if alloc['Allocation Decision'].lower() == 'yes']
        meds_allocated_str = ', '.join(meds_allocated_list) if meds_allocated_list else 'None'
        flash(f"All remaining patients have been allocated medications successfully. Medications Allocated: {meds_allocated_str}")
    else:
        flash("No unallocated patients found.")
    
    return redirect(url_for('index'))
@app.route('/page/allocate_next', methods=['POST'])
@login_required 
def allocate_next_medication():
    processed_filepath = session.get('processed_file')
    medications_inventory = session.get('medications_inventory')
    final_filepath = session.get('final_file')
    
    if not processed_filepath or not os.path.exists(processed_filepath):
        flash('Processed patient data not found. Please upload and process files first.')
        return redirect(url_for('index'))
    if not medications_inventory:
        flash('Medications inventory not found. Please upload medications data first.')
        return redirect(url_for('index'))
    if not final_filepath or not os.path.exists(final_filepath):
        # Initialize final_results.csv if it doesn't exist
        final_df = pd.DataFrame(columns=['ID', 'Severity_Label', 'Allocation Decision', 'Medications Allocated', 'Reasoning'])
        final_df.to_csv(final_filepath, index=False)
    
    try:
        # Check if final_results.csv is empty
        if is_csv_empty(final_filepath):
            # Initialize with headers if empty
            final_df = pd.DataFrame(columns=['ID', 'Severity_Label', 'Allocation Decision', 'Medications Allocated', 'Reasoning'])
            final_df.to_csv(final_filepath, index=False)
        else:
            final_df = pd.read_csv(final_filepath, dtype={'ID': int})
    except pd.errors.EmptyDataError:
        # Handle empty final_results.csv
        final_df = pd.DataFrame(columns=['ID', 'Severity_Label', 'Allocation Decision', 'Medications Allocated', 'Reasoning'])
        final_df.to_csv(final_filepath, index=False)
    except Exception as e:
        flash(f"Error reading CSV files: {e}")
        return redirect(url_for('index'))
    
    try:
        df = pd.read_csv(processed_filepath, dtype={'ID': int})
    except pd.errors.EmptyDataError:
        flash('Processed patient data file is empty.')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f"Error reading processed data file: {e}")
        return redirect(url_for('index'))
    
    allocated_ids = set(final_df['ID'].tolist())
    unallocated_df = df.loc[~df['ID'].isin(allocated_ids)]
    
    if unallocated_df.empty:
        flash('All patients have been allocated medications.')
        return redirect(url_for('index'))
    
    # Allocate the first unallocated patient
    patient_row = unallocated_df.iloc[0]
    patient_id = patient_row['ID']
    severity_level = patient_row['Severity_Label']
    
    patient_data = {
        'Severity Level': severity_level,
        'age': patient_row.get('age', 'N/A'),
        'Diastolic blood pressure': patient_row.get('Diastolic blood pressure', 'N/A'),
        'glucose': patient_row.get('glucose', 'N/A'),
        'heart rate': patient_row.get('heart rate', 'N/A'),
        'SP O2': patient_row.get('SP O2', 'N/A'),
        'PROMPT': patient_row.get('PROMPT', 'N/A')  # Ensure 'PROMPT' is included with correct casing
    }
    
    gpt_output = generate_prompt(patient_data, medications_inventory)
    
    # Parse GPT response using the new function
    parsed_response = parse_gpt_response(gpt_output)
    
    allocation_decision = parsed_response['Allocation Decision']
    medications_allocated = parsed_response['Medications Allocated']
    reasoning = parsed_response['Reasoning']
    
    allocated_meds = []
    if allocation_decision.lower() == 'yes' and isinstance(medications_allocated, dict):
        for med_name, qty in medications_allocated.items():
            if med_name in medications_inventory and qty > 0:
                medications_inventory[med_name] -= qty
                if medications_inventory[med_name] < 0:
                    medications_inventory[med_name] = 0
                allocated_meds.append(f"{med_name}: {qty}")
    elif allocation_decision.lower() == 'no':
        medications_allocated = 'N/A'
    
    allocation_decision_row = {
        'ID': patient_id,
        'Severity_Label': severity_level,
        'Allocation Decision': allocation_decision,
        'Medications Allocated': ', '.join(allocated_meds) if allocated_meds else medications_allocated,
        'Reasoning': reasoning
    }
    allocation_df = pd.DataFrame([allocation_decision_row])
    
    # Append the allocation to final_results.csv
    try:
        allocation_df.to_csv(final_filepath, mode='a', header=False, index=False)
    except Exception as e:
        flash(f"Error writing to final results file: {e}")
        return redirect(url_for('index'))
    
    # Save updated medications inventory
    try:
        meds_df = pd.DataFrame(list(medications_inventory.items()), columns=['Medication Name', 'Quantity'])
        meds_filepath = os.path.join(app.config['PROCESSED_FOLDER'], 'medications_inventory.csv')
        meds_df.to_csv(meds_filepath, index=False)
    except Exception as e:
        flash(f"Error saving medications inventory: {e}")
        return redirect(url_for('index'))
    
    # Update session
    session['medications_inventory'] = medications_inventory
    
    if allocated_meds:
        allocated_meds_str = ', '.join(allocated_meds)
        flash(f'Medication allocation for ID {patient_id} completed successfully. Medications Allocated: {allocated_meds_str}')
    else:
        flash(f'Medication allocation for ID {patient_id} completed successfully. No medications allocated.')
    
    return redirect(url_for('index'))

@app.route('/page/download')
@login_required 
def download_file():
    final_filepath = session.get('final_file')
    if not final_filepath or not os.path.exists(final_filepath):
        flash('Final allocation results not found. Please complete all processing steps first.')
        return redirect(url_for('index'))
    
    return send_file(final_filepath, as_attachment=True)

@app.route('/page/download_processed')
@login_required 
def download_processed_file():
    processed_filepath = session.get('processed_file')
    if not processed_filepath or not os.path.exists(processed_filepath):
        flash('Processed patient data not found. Please upload and process patient files first.')
        return redirect(url_for('index'))
    
    return send_file(processed_filepath, as_attachment=True)

@app.route('/page/download_meds')
@login_required 
def download_meds_file():
    meds_filepath = os.path.join(app.config['PROCESSED_FOLDER'], 'medications_inventory.csv')
    if not os.path.exists(meds_filepath):
        flash('Medications inventory not found. Please upload medications data first.')
        return redirect(url_for('index'))
    
    return send_file(meds_filepath, as_attachment=True)

from flask import Flask, render_template, request, redirect, url_for, flash

@app.route('/assess', methods=['GET', 'POST'])
def assess_patient():
    if request.method == 'POST':
        # Extract and process patient data, then display results
        patient_data = {feature: float(request.form.get(feature, 0)) for feature in REQUIRED_FEATURES}
        X = [list(patient_data.values())]
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)
        return render_template('patient_assessment.html', required_features=REQUIRED_FEATURES, result=prediction[0])
    return render_template('patient_assessment.html', required_features=REQUIRED_FEATURES)

@app.route('/bulk_assess')
def bulk_assess():
    # Add your logic here for handling bulk assessments
    return render_template('bulk_assess.html')  # Assuming you have a separate template for bulk processing


if __name__ == '__main__':
    with app.app_context():
        db.drop_all()  # Drops all tables for a clean slate (useful for testing)
        db.create_all()  # Creates all tables
        
        # Add a test user
        test_user = User(username='admin', password='admin123')
        test_1 = User(username = 'CSE1', password ='cseiscool1!')
        test_2 = User(username = 'CSE2', password ='cseiscool2!')
        test_3 = User(username = 'CSE3', password ='cseiscool3!')
        test_4 = User(username = 'CSE4', password ='cseiscool4!')

        db.session.add(test_user)
        db.session.add(test_1)
        db.session.add(test_2)
        db.session.add(test_3)
        db.session.add(test_4)
        
        db.session.commit()
        print("Test user added: username='admin', password='admin123'")
    
    app.run(debug=True)

