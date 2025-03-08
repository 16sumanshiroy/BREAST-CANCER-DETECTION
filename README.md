# **Breast Cancer Detection Using Machine Learning**

## **Overview**
This project aims to build a **machine learning-based predictive model** to detect breast cancer by classifying tumors as **benign or malignant**. The model utilizes **feature selection, preprocessing techniques, and hyperparameter tuning** to enhance accuracy. The goal is to develop a robust and interpretable model that can assist in early detection, improving patient outcomes and treatment efficiency.

## **Dataset**
- The dataset used is the **Wisconsin Breast Cancer Dataset (WBCD)** from the UCI Machine Learning Repository.
- It contains **569 instances and 30 feature variables**, including **mean radius, texture, smoothness, compactness, symmetry**, etc.
- The target variable is **Diagnosis**: **0 for benign, 1 for malignant**.
- The dataset is well-structured and contains minimal missing values, making it suitable for predictive modeling.

## **Technologies Used**
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Machine Learning Models:** Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN)
- **Deployment:** Flask (for API integration)
- **Visualization Tools:** Seaborn, Matplotlib (for EDA)

## **Project Workflow**
1. **Data Preprocessing:**
   - Handling missing values
   - Feature scaling and normalization
   - Removing duplicate and irrelevant features
   - Encoding categorical variables (if applicable)
   
2. **Exploratory Data Analysis (EDA):**
   - Visualizing feature correlations using heatmaps
   - Identifying key statistical trends in malignant vs. benign cases
   - Box plots, histograms, and violin plots for feature distribution analysis
   
3. **Model Training & Evaluation:**
   - Splitting dataset into **train-test sets (80-20 split)**
   - Training multiple classifiers and comparing accuracy, precision, recall, and F1-score
   - Fine-tuning models using **GridSearchCV** for optimal hyperparameter selection
   
4. **Performance Metrics:**
   - **Accuracy, Precision, Recall, F1-score, ROC-AUC Curve**
   - Confusion Matrix visualization to analyze false positives and negatives

## **Results**
- Achieved **96% accuracy** using **Random Forest Classifier**.
- **Feature importance analysis** showed that **mean radius, texture, and smoothness** were key predictors of malignancy.
- The **ROC-AUC score of 0.98** indicated a highly reliable classification model.
- The model successfully minimizes false negatives, ensuring early cancer detection.

## **How to Run the Project**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd breast-cancer-detection
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook
   # or
   python breast_cancer_detection.py
   ```

## **Deployment**
- The trained model can be deployed as a **Flask API** for real-time predictions.
- Integration with **Streamlit or FastAPI** for a simple web-based user interface.
- Model can be further improved using **cloud deployment** for scalability.

<img width="960" alt="Screenshot 2023-12-07 154140" src="https://github.com/user-attachments/assets/7674e0e4-9b6f-4b7e-92b3-b0a8ba34827c" />


## **Future Improvements**
- Implement **deep learning (CNNs or AutoML)** for enhanced accuracy.
- Introduce **Explainable AI (XAI) techniques** to improve model transparency.
- Deploy the model on **AWS Lambda or Google Cloud Functions** for better accessibility.
- Automate **feature selection using AI-driven techniques** to further optimize performance.

## **Contributions**
Feel free to fork this repository, create feature branches, and submit PRs. Any contributions, issues, or suggestions are welcome!

## **License**
This project is open-source under the **MIT License**.

---

## **References**
- UCI Machine Learning Repository: Wisconsin Breast Cancer Dataset
- Research papers on breast cancer classification using ML
- Kaggle datasets and community discussions

---

**📌 Connect with me:**  
📧 Email: 16sumanshiroy@gmail.com  
🔗 LinkedIn: [Sumanshi Roy](https://linkedin.com/in/sumanshi-roy-435229230)  
🐍 GitHub: [16sumanshiroy](https://github.com/16sumanshiroy)

🚀 Happy Coding!


