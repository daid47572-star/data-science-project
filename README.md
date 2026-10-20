# 📊 Student Performance Analysis: Factors Affecting Academic Success

> A comprehensive data science project analyzing 395 Portuguese secondary school students to identify key factors that predict academic success using machine learning and statistical analysis.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

---

## 🎯 Problem / Research Question

**What are the key factors that influence student academic performance, and can we predict student grades based on demographic, social, and academic-related features?**

### Specific Objectives:
1. **Identify** which factors (study time, absences, parental education, etc.) have the strongest correlation with final grades
2. **Build** predictive models to classify students into pass/fail categories
3. **Compare** the performance of multiple machine learning algorithms
4. **Provide** actionable insights for educators and policymakers

---

## 📂 Dataset Source and Usage Permissions

- **Source**: [UCI Machine Learning Repository - Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
- **License**: Publicly available for educational and research purposes
- **Size**: 395 students × 33 features
- **Target**: Final grade (G3) ranging from 0-20

**Citation**: P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUTURE BUSINESS TECHNOLOGY Conference, Porto, Portugal, EUROSIS-ETI, pp. 5-12, 2008.

---

## 🛠️ Methods and Tools Used

### Programming & Libraries
- **Language**: Python 3.8+
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Development**: Jupyter Notebook, VS Code

### Algorithms Implemented
1. **Logistic Regression** - Linear baseline model
2. **Decision Tree** - Non-linear interpretable model
3. **Random Forest** - Ensemble method (best performer)

### Analysis Techniques
- Exploratory Data Analysis (EDA)
- Correlation analysis
- Feature importance ranking
- Model evaluation (accuracy, precision, recall, F1-score)
- Cross-validation
- Learning curve analysis
- Confusion matrix analysis

---

## 🔍 Key Findings

### 1. **Past Failures Are the Strongest Predictor**
- Students with 0 past failures have a **pass rate of 71%**
- Students with 1+ past failures have a **pass rate of only 25%**
- This is the most important feature in our Random Forest model

### 2. **Study Time Matters Significantly**
- Students studying <2 hours/week: average grade **9.6**
- Students studying 2-5 hours/week: average grade **10.4**
- Students studying 5-10 hours/week: average grade **11.4**
- Students studying >10 hours/week: average grade **11.9**

### 3. **Absences Have a Negative Impact**
- Students with 0-5 absences: average grade **11.5**
- Students with 6-10 absences: average grade **10.2**
- Students with >10 absences: average grade **8.5**

### 4. **Parental Education is a Factor**
- Students with highly educated mothers: average grade **11.7**
- Students with less educated mothers: average grade **8.9**
- Similar pattern for father's education

### 5. **Period Grades Strongly Predict Final Grades**
- Correlation between G1 and G3: **0.801**
- Correlation between G2 and G3: **0.918**
- G2 is the strongest single predictor

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 89.87% | 0.90 | 0.90 | 0.90 |
| Decision Tree | 86.08% | 0.86 | 0.86 | 0.86 |
| **Random Forest** | **87.34%** | 0.87 | 0.87 | 0.87 |

**Best Model**: Random Forest (200 trees, max depth=8) - Chosen for its balanced performance and interpretability

---

## ⚠️ Limitation

**Limited Generalizability**: This dataset is from Portuguese secondary schools collected in 2005-2006. The findings may not be directly applicable to:

- **Different educational systems** (e.g., university, elementary schools)
- **Other cultural contexts** (Asian, American, African educational systems)
- **Modern learning environments** (post-COVID online/hybrid learning)
- **Different time periods** (educational practices have evolved significantly)

**Additional Limitations**:
- Sample size of 395 students is relatively small for deep learning approaches
- Dataset lacks information on:
  - Mental health and well-being factors
  - Quality of teaching and school resources
  - Use of technology in learning
  - Socioeconomic changes over time
  - Peer influence and study groups

A more comprehensive study with diverse, recent data would be needed to make broader claims about student performance factors.

---

## 🚀 Instructions for Running the Analysis

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- (Optional) Jupyter Notebook

### Step 1: Clone the Repository

```bash
git clone https://github.com/daid47572-star/data-science-project.git
cd data-science-project
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Analysis

#### Option A: Using Jupyter Notebook (Recommended)

```bash
jupyter notebook analysis.ipynb
```

#### Option B: Using Python Script

```bash
python analysis.py
```

#### Option C: Generate Advanced Visualizations

```bash
python advanced_analysis.py
```

#### Option D: Generate All Visualizations at Once

```bash
python generate_visualizations.py
```

### Step 4: View Results

All visualizations are automatically saved in the `visualizations/` folder:
- `grade_distribution.png` - Distribution of student grades
- `correlation_heatmap.png` - Feature correlation matrix
- `factor_analysis.png` - Key factors analysis
- `model_performance.png` - Model accuracy comparison
- `feature_importance.png` - Random Forest feature importance
- `confusion_matrix.png` - Classification results
- `learning_curve.png` - Model learning behavior
- `deep_dive.png` - Detailed feature relationships

---

## 📁 Project Structure

```
data-science-project/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
├── .gitignore                     # Git ignore rules
├── analysis.py                    # Main analysis script
├── analysis.ipynb                 # Jupyter notebook version
├── generate_visualizations.py     # Basic visualization generator
├── advanced_analysis.py           # Advanced ML visualizations
├── data/
│   ├── student-mat.csv            # Mathematics course dataset
│   ├── student-por.csv            # Portuguese language dataset
│   ├── student.txt                # Dataset documentation
│   └── student-merge.R           # R merging script
└── visualizations/                # Generated plots (8 charts)
    ├── grade_distribution.png
    ├── correlation_heatmap.png
    ├── factor_analysis.png
    ├── model_performance.png
    ├── feature_importance.png
    ├── confusion_matrix.png
    ├── learning_curve.png
    └── deep_dive.png
```

---

## 🎓 What I Learned

Through this project, I gained hands-on experience in:

1. **End-to-end data science workflow** - From problem definition to model deployment
2. **Exploratory Data Analysis (EDA)** - Understanding data before modeling
3. **Feature engineering and selection** - Identifying meaningful predictors
4. **Model comparison and evaluation** - Choosing the right algorithm
5. **Data visualization best practices** - Creating clear, informative charts
6. **Reproducible research** - Writing clean, documented code

---

## 🔮 Future Improvements

- [ ] Add more advanced models (XGBoost, LightGBM, Neural Networks)
- [ ] Implement hyperparameter tuning with GridSearchCV
- [ ] Create an interactive dashboard with Streamlit
- [ ] Add SHAP values for model interpretability
- [ ] Combine with the Portuguese language dataset (student-por.csv)
- [ ] Deploy as a web application for real-time predictions
- [ ] Add time-series analysis if temporal data becomes available

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**wangyueyang** - Data Science Student
- **Student ID**: MC643435
- **GitHub**: [@daid47572-star](https://github.com/daid47572-star)
- **Portfolio**: [View Portfolio](https://daid47572-star.github.io/data-science-portfolio/)

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for providing the dataset
- **P. Cortez and A. Silva** for the original data collection and research
- The **data science community** for inspiration and learning resources
- **GitHub** for hosting and version control

---

*Last updated: October 2026*