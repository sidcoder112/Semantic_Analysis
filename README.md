# Sentiment Analysis Project

## Overview
This project aims to perform sentiment analysis on Twitter data using machine learning techniques. It involves preprocessing text data, training a logistic regression model, and deploying a web application for real-time sentiment analysis.

## Project Structure
- **Data Files**: Contains the training and validation datasets.
- **Jupyter Notebook**: Contains code for data preprocessing, model training, and evaluation.
- **Trained Model**: Serialized model file storing the trained sentiment analysis model.
- **Web Application**: Python script implementing a Flask web application for sentiment analysis.

## Functionality
- **Data Preprocessing**: Preprocesses tweet content to remove URLs, special characters, and convert text to lowercase.
- **Model Training**: Trains a logistic regression model on preprocessed tweet data using TF-IDF vectorization.
- **Model Evaluation**: Evaluates the trained model's performance using a classification report.
- **Web Application**: Allows users to input tweet text for real-time sentiment analysis.

## Usage
1. **Training the Model**: Run the code in the Jupyter Notebook to preprocess data, train the model, and save it.
2. **Deploying the Web Application**: Start the Flask web server using the provided Python script and access the application in a web browser.
3. **Using the Web Application**: Enter tweet text in the input field, submit the form, and view the sentiment analysis result.

## Dependencies
- pandas
- scikit-learn
- Flask

## Conclusion
This project demonstrates a simple yet effective approach to sentiment analysis using machine learning techniques. By preprocessing text data and training a logistic regression model, it provides a way to analyze the sentiment of Twitter data in real-time through a web interface.
