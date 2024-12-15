# Severity Ranker: AI-Powered Medical Resource Allocation

**Severity Ranker** is a machine learning and GPT-4-based system designed to classify patients by severity and optimize medication allocation for NGOs working in resource-limited settings.

---

## **Project Overview**
- **Input**: Patient clinical data (CSV) and medication inventory.
- **Output**: Severity classification (0-4) and GPT-4-based allocation decisions.
- **Goal**: Prioritize critical patients and optimize medication usage dynamically.

---

## **How It Works**
1. **Severity Classification**:
   - Logistic Regression trained on clinical features.
   - Features selected based on correlation to severity.
   - Severity Scale:  
     0 = Healthy, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Extra Severe.

2. **Medication Allocation**:
   - GPT-4 processes severity label, patient prompts, and inventory.
   - Determines whether to allocate medications or suggest substitutes.

3. **Web Deployment**:
   - Flask-based app for real-time file uploads and results visualization.

---

## **Project Files**
- **`severitycategorizer.ipynb`**: Model training and feature selection.
- **`app.py`**: Flask app for deployment.
- **`model.pkl`**: Pre-trained Logistic Regression model.
- **`scaler.pkl`**: StandardScaler for feature scaling.
- **Sample Data**: CSV files for patient records and inventory.

---

## **Setup Instructions**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/severityranker.git
   cd severityranker
