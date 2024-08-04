# Breast Cancer Diagnosis App

This is a Flask web application for breast cancer diagnosis using deep learning models. The app allows users to upload ultrasound images, which are then classified as benign, malignant, or normal. Additionally, the app performs image segmentation to highlight the tumor area.

## Features

- **User Authentication:** Secure login functionality for accessing the app.
- **Image Classification:** Classifies ultrasound images into benign, malignant, or normal.
- **Image Segmentation:** Segments the tumor area in ultrasound images.
- **Report Generation:** Generates and exports patient reports in PDF or TXT format.
- **Patient History:** Keeps track of patient history and previous reports.

## Installation

### Prerequisites

- Python 3.6 or higher
- Git
- pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/Muset007/Honours-Project.git
cd Honours-Project
Install the Required Packages
pip install -r requirements.txt
Set Up Git LFS (If not already set up)
git lfs install
Run the Application
python3 app.py
