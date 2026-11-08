# Dry Bean Classification - 5 ML Models + Streamlit App
Submission by: Sarthak Rout 
BITS ID: 2026AK05289
## a. Problem Statement
Classify dry bean grains into one of 7 varieties (Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira) from 16 geometric features extracted from images. The task is a multi-class classification problem; five ML models are trained on the same data and compared on six evaluation metrics.

## b. Dataset Description
- Source: UCI Machine Learning Repository - Dry Bean Dataset (ID 602)
- Instances: 13,611 | Features: 16 (area, perimeter, major/minor axis length, aspect ratio, eccentricity, convex area, equivalent diameter, extent, solidity, roundness, compactness, ShapeFactor1-4) | Target: `Class` (7 classes)
- No missing values; class distribution is imbalanced (Dermason largest, Bombay smallest).
- Split: 80% train / 20% test, stratified, `random_state=42`. The test split is saved as `test_data.csv`.
- Preprocessing: StandardScaler (inside pipelines for LR, KNN, SVM; not needed for Decision Tree / Gaussian NB).

## c. Link
GitHub Repository: https://github.com/havoc7/ML_A-2_2026AK05289

Streamlit Live app: https://mla-22026ak05289-wvj8xxwm2y8sgltww7mvpu.streamlit.app/

## d. Models Used

### Comparison Table

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Support Vector Machine |     0.9221 | 0.9949 |      0.9359 |   0.933  | 0.9343 | 0.9059 |
| Logistic Regression    |     0.9214 | 0.9948 |      0.9354 |   0.9321 | 0.9335 | 0.905  |
| KNN                    |     0.9166 | 0.9833 |      0.932  |   0.9271 | 0.9293 | 0.8992 |
| Decision Tree          |     0.8917 | 0.9448 |      0.9084 |   0.9088 | 0.9085 | 0.8691 |
| Naive Bayes            |     0.7639 | 0.9672 |      0.7744 |   0.7694 | 0.7677 | 0.7154 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Reaches about 0.92 accuracy and a macro F1 of 0.93, so a linear model separates most bean classes well on scaled features. It classifies BOMBAY perfectly (F1 1.00) and does well on HOROZ (0.96) and SEKER (0.94). Its weakest class is SIRA (F1 0.86), and BARBUNYA has the lowest recall (0.89). It is nearly level with SVM and KNN while being simpler and faster. |
| Decision Tree | Reaches about 0.89 accuracy and a macro F1 of 0.91, the second-lowest overall, ahead only of Naive Bayes. It is perfect on BOMBAY but weaker on SIRA (F1 0.81, recall 0.79) and DERMASON (F1 0.88). A single unpruned tree likely overfits the training data and confuses the similar-looking SIRA, DERMASON and BARBUNYA classes. Its advantages are speed and interpretability. |
| KNN | Reaches about 0.92 accuracy and a macro F1 of 0.93, essentially tied with Logistic Regression and SVM. Feature scaling helps the distance-based voting, and it is strong on HOROZ (0.95) and SEKER (0.95). SIRA is again its weakest class (F1 0.86), and BARBUNYA recall is 0.88. Its drawback is slower prediction, since each new point is compared against the whole training set. |
| Naive Bayes | The weakest model, with about 0.76 accuracy and a macro F1 of 0.77. BARBUNYA is hit hardest (recall 0.46, F1 0.55), and CALI and SEKER precision is only 0.68. The likely cause is that Dry Bean features such as area, perimeter and axis lengths are highly correlated, which breaks Naive Bayes' independence assumption. It still classifies BOMBAY perfectly (F1 1.00), as it is a very distinct class. |
| Support Vector Machine | The best model, with an accuracy of 0.9221 and a macro F1 of 0.93. It matches or leads the others on most classes: HOROZ 0.96, SEKER 0.94, CALI 0.94, and SIRA 0.87 (the best SIRA score of any model). The RBF kernel handles non-linear class boundaries and the overlap between SIRA, DERMASON and BARBUNYA. The margin over Logistic Regression and KNN is small but consistent. |

**Overall winner: Support Vector Machine.** It has the highest accuracy (0.9221) and macro F1 (0.93), with the strongest per-class results, especially on SIRA, the hardest class. Logistic Regression and KNN are very close behind, while Naive Bayes is clearly the weakest.

## Repository Structure
```
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── label_info.json
```

## How to Run
```
pip install -r requirements.txt
python train_models.py      # trains, saves models/ and test_data.csv
streamlit run app.py
```
