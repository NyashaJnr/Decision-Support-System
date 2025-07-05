# Mimosa Mining DSS

A web-based Decision Support System for Mimosa Mining Company, built with Flask, MongoDB, and Tailwind CSS.

## Features

- Data upload and storage
- Interactive dashboard
- Data visualization
- Real-time data analysis

## Prerequisites

- Python 3.8+
- MongoDB
- pip (Python package manager)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd mimosa-dss
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```


## Running the Application

1. Start the Flask server:

```bash
cd backend
python app.py
```

2. Open your browser and navigate to:

```
http://localhost:5000
```

## Usage

1. Upload CSV data files through the web interface
2. View data visualizations and summaries on the dashboard
3. Analyze trends and patterns in the data

## Project Structure

```
mimosa-dss/
├── backend/
│   ├── app.py
│   ├── config.py
│   └── data/
├── frontend/
│   ├── index.html
│   ├── css/
│   └── js/
└── requirements.txt
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
