# NIDShield

## Overview
NIDShield is a Network Intrusion Detection System (NIDS) leveraging machine learning techniques. This innovative system aims to overcome the limitations of traditional intrusion detection methods by dynamically learning from network traffic patterns. By integrating machine learning algorithms into the NIDS framework, it can adapt and identify both known and previously unknown threats.

## Features
- Real-time network traffic analysis
- Classification of connection attempts as normal or belonging to specific attack types
- Data preprocessing techniques including acquisition, exploration, cleaning, feature engineering, categorical feature encoding, feature scaling, and data splitting.
- Modeling and evaluation using various machine learning algorithms:
  - Gaussian Naive Bayes (GNB)
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Gradient Boosting Classifier
 
  
## Dataset
NIDShield uses the KDD Cup 1999 dataset, which is a widely used benchmark dataset for network intrusion detection research. The dataset contains a large number of connection records, each labeled as normal or one of several types of attacks. It includes features such as duration, protocol type, service, flag, src_bytes, dst_bytes, etc., providing information about each connection attempt.
