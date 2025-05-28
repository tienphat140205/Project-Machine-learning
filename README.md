# Text Classification Application

A text classification application that uses multiple machine learning models to classify news articles into 4 categories: World, Sports, Business, and Sci/Tech.

## Supported Models

- Neural Network
- Naive Bayes
- Logistic Regression
- SVM
- SGD
- Random Forest
- Prod LDA
- Clustering

## System Requirements

- Docker
- Minimum 4GB RAM
- 2GB free disk space

## Installation and Running the Application

### 1. Clone repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Build Docker image

```bash
docker build -t text-classification .
```

### 3. Run container

```bash
docker run -p 7860:7860 text-classification
```

### 4. Access the application

Open your web browser and navigate to:
```
http://localhost:7860
```

## Using the Application

1. Select a model from the dropdown menu
2. Enter the news content to be classified in the textbox
3. Click the "Submit" button to see the prediction result

## Directory Structure

```
.
├── app.py              # Main application file
├── Dockerfile          # Docker configuration file
├── requirements.txt    # Python package list
├── README.md          # Documentation file
└── param/             # Directory containing model files
    ├── nn_model.pth
    ├── nb_model.pkl
    ├── lr_model.pkl
    ├── svc_model.pkl
    ├── sgd_model.pkl
    ├── rf_model.pkl
    ├── prodlda.pth
    └── ...
```