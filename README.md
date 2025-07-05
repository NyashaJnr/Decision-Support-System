# READ ME
# AI-Based Data-Driven Decision Support System

This system was developed as part of my BSc Computer Engineering WRL (Work-Related Learning) program at the University of Zimbabwe. It supports informed managerial decisions using machine learning and analytics for departments such as HR, Finance, Supply Chain, and more.

## 🔧 Features

- Predictive analytics (e.g., employee attrition, expense forecasting)
- Department-specific dashboards (HR, Supply Chain, etc.)
- Flask backend integrated with trained ML models
- Tailwind CSS + Chart.js for frontend visualization

## 📁 Project Structure

decision-support-system/
app/ - Flask app and routes
frontend/ - Static frontend (Tailwind/Chart.js)
datasets/ - CSV files used for training and prediction
models/ - Saved ML models
run.py - Main app entry point


## 🚀 How to Run

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python run.py
