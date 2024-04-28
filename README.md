# Sentiment Analysis Project

## Overview
This project aims to perform sentiment analysis on Twitter data using machine learning techniques. It involves preprocessing text data, training a logistic regression model, and deploying a web application for real-time sentiment analysis.

## Project Structure
- **Data Files**: Contains the training and validation datasets.
- **Trained Model**: Serialized model file storing the trained sentiment analysis model. (Already available in the repository)
- **Web Application**: Python script implementing a Flask web application for sentiment analysis.

## Functionality
- **Data Preprocessing**: Preprocesses tweet content to remove URLs, special characters, and convert text to lowercase.
- **Model Evaluation**: Evaluates the performance of the pre-trained model using a classification report.
- **Web Application**: Allows users to input tweet text for real-time sentiment analysis.

## Usage
1. **Deploying the Web Application**: Start the Flask web server using the provided Python script and access the application in a web browser.
2. **Using the Web Application**: Enter tweet text in the input field, submit the form, and view the sentiment analysis result.

## Dependencies
- pandas
- scikit-learn
- Flask

## Conclusion
This project demonstrates a simple yet effective approach to sentiment analysis using machine learning techniques. By deploying the pre-trained model within the web application, users can easily analyze the sentiment of Twitter data in real-time through a user-friendly interface.
