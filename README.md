# Wine-Quality-Predictor
Wine Quality Prediction using Machine Learning – A classification model that predicts the quality of red wine based on its physicochemical properties. Built with Python, Pandas, Scikit-learn, and a Random Forest Classifier
Wine Quality Prediction  

Overview  
This project focuses on predicting the quality of red wine based on its physicochemical properties using Machine Learning techniques.  
The model applies classification algorithms to determine wine quality on a scale from 3 to 8, providing valuable insights into which chemical properties influence wine taste and quality.  
Dataset  
- Source: [UCI Machine Learning Repository – Wine Quality 
Dataset] ( https://archive.ics.uci.edu/ml/datasets/Wine+Quality )  
- Samples: 1,599  
Features:  
  - Fixed Acidity  
  - Volatile Acidity  
  - Citric Acid  
  - Residual Sugar  
  - Chlorides  
  - Free Sulfur Dioxide  
  - Total Sulfur Dioxide  
  - Density  
  - pH  
  - Sulphates  
  - Alcohol  
  - Target: Quality (Score 0–10)  

Tech Stack  
- Programming Language: Python  
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
Approach  
1. Data Preprocessing 
   - Fixed CSV parsing issues  
   - Converted features into numeric values  
   - Checked for and handled missing values  

2. Model Training 
   - Used train-test split (80/20)
   - Applied Random Forest Classifier

3. Evaluation
   - Accuracy Achieved: ~67%
   - Feature Importance Analysis shows:  
     - Alcohol, Sulphates, and Volatile Acidity are the most impactful predictors  

Note: pip install pandas numpy scikit-learn matplotlib seaborn

git clone https://github.com/Anushree2005-AI/Wine-Quality-Prediction.git
cd wine-quality-prediction
