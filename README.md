# PolicyGuard: Insurance Churn Prediction System
The PolicyGuard system is an intelligent insurance churn prediction platform designed to identify customers who are likely to discontinue their insurance policies. By leveraging machine learning techniques and a user-friendly web interface, the system enables insurance companies to take proactive retention actions and improve long-term customer engagement

## About
PolicyGuard: Insurance Churn Prediction System is a data-driven solution that applies machine learning models to analyze historical insurance customer data and predict churn behavior. In the competitive insurance market, retaining existing customers is crucial, yet traditional churn identification methods rely on manual analysis and basic statistical approaches that often lack accuracy and timeliness.

This project addresses these challenges by integrating machine learning algorithms such as Random Forest and Logistic Regression to identify churn patterns based on customer demographics, policy details, payment history, and claim behavior. The system provides an interactive web interface where users can upload datasets, view churn predictions, analyze insights through dashboards, and receive actionable recommendations. By automating churn prediction, PolicyGuard helps insurance organizations reduce revenue loss and enhance customer retention strategies.

## Features
- Machine learning–based churn prediction using Random Forest and Logistic Regression

- Web-based application for easy data upload and result visualization

- Interactive dashboard with churn statistics and probability analysis

- Actionable recommendations for customer retention

- Scalable and modular system architecture

- Secure handling of data and API credentials

## Requirements
- Operating System:
Requires a 64-bit operating system such as Windows 10 or Ubuntu for compatibility.

- Development Environment:
Python 3.8 or later for backend development and model execution.

- Machine Learning Libraries:
Scikit-learn for model training and prediction, Pandas and NumPy for data preprocessing.

- Web Framework:
Flask for backend server development and API handling.

- Frontend Technologies:
HTML, CSS, and JavaScript for building responsive user interfaces.

- Database / Storage:
Google Sheets or CSV-based storage for managing customer datasets.

IDE & Tools:
VS Code for coding, debugging, and project management; Git for version control.

## System Architecture
![architecture diagram](https://github.com/user-attachments/assets/21faf556-e291-4ca6-8f1f-4eaf9968ae7a)

The system architecture follows a client–server model where users interact through a web interface to upload insurance customer data. The Flask backend processes the data, applies preprocessing techniques, and sends it to the trained machine learning models. Prediction results are then returned to the dashboard for visualization and analysis. An AI chatbot module optionally assists users by explaining churn predictions and providing insights.

## Output
### Output 1 – Churn Prediction Results
Displays customer-wise churn status along with churn probability and retention recommendations.
<img width="1885" height="910" alt="output page" src="https://github.com/user-attachments/assets/f79e5bf3-a50a-4bd9-a041-81abe71a5124" />


### Output 2 – Dashboard Analytics

Shows overall churn statistics, probability distribution, and key insights through charts and tables.
<img width="1901" height="923" alt="dashboard page" src="https://github.com/user-attachments/assets/322da747-5de7-4ac2-ac23-5427a5a7c0b9" />


## Model Performance:
Prediction Accuracy: ~90%
<img width="837" height="528" alt="image" src="https://github.com/user-attachments/assets/f4a9b145-2244-4365-8697-b843297fde05" />


## Results and Impact
The PolicyGuard system significantly improves the ability of insurance companies to identify potential churners at an early stage. By replacing manual analysis with automated machine learning predictions, the system enhances decision-making speed and accuracy. The interactive dashboard and recommendation engine help business teams design targeted retention strategies, reducing customer attrition and improving revenue stability. This project demonstrates the practical application of machine learning in real-world insurance analytics and supports sustainable customer relationship management.

## References
1. Nagaraju Jajam et al., “Predicting Customer Churn in the Insurance Industry Using Big Data and Machine Learning,” International Journal of Information Systems and Applied Engineering, 2023.

2. Customer Churn Prediction Using Machine Learning Techniques, Asian Journal of Basic Science & Research, 2022.
