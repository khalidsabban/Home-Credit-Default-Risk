<div align="center">
    <h1>🏦 Home Credit Default Risk Prediction</h1>
    <p><i>Predicting loan repayment capability using machine learning</i></p>
</div>

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Data Exploration](#-data-exploration)
- [Model Performance](#-model-performance)
- [Key Findings](#-key-findings)
- [Future Enhancements](#-future-enhancements)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)

---

## 🎯 Project Overview

Many people struggle to get loans due to insufficient or non-existent credit histories. Unfortunately, this population is often taken advantage of by untrustworthy lenders. This project aims to predict how capable each applicant is of repaying a loan using machine learning techniques, helping to unlock fair lending opportunities for underserved populations.

### Key Objectives:
- Analyze applicant data to identify default risk patterns
- Build predictive models to assess loan repayment probability
- Compare model performance across different data integration strategies
- Provide actionable insights for loan approval decisions

---

## ❓ Problem Statement

**Can we predict how capable each applicant is of repaying a loan?**

This binary classification problem aims to identify clients who may have difficulty repaying loans, enabling financial institutions to make more informed lending decisions while promoting financial inclusion.

---

## 📊 Dataset

### Data Source
The dataset is from the **Kaggle Home Credit Default Risk Competition**.

### How to Download Data

Due to the large size of the datasets, they are not included in this repository. Follow these steps:

1. Visit the [Home Credit Default Risk Competition](https://www.kaggle.com/competitions/home-credit-default-risk/overview)
2. Navigate to the **Data** tab
3. Download all required datasets
4. Extract and place files in: `static/data/processData/`

> **Note:** You'll need a Kaggle account and must accept the competition rules to download the data.

### Dataset Components
- **application_train/test.csv**: Main dataset with loan application information
- **bureau.csv**: Client's previous credits from other financial institutions
- **bureau_balance.csv**: Monthly balances of previous credits
- **previous_application.csv**: Previous applications for Home Credit loans
- **POS_CASH_balance.csv**: Monthly balance snapshots of previous POS and cash loans
- **credit_card_balance.csv**: Monthly balance snapshots of previous credit cards
- **installments_payments.csv**: Repayment history for previous credits

---

## 📁 Repository Structure

```
Home-Credit-Default-Risk/
│
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
│
├── notebooks/
│   ├── 01.data_visualization.ipynb     # Exploratory Data Analysis
│   ├── 02.01.process_main_data.ipynb   # Main dataset preprocessing
│   ├── 02.02.bureau_data.ipynb         # Bureau data processing
│   ├── 02.03.payment_data.ipynb        # Payment data processing
│   ├── 02.04.cash_data.ipynb           # Cash loan data processing
│   ├── 02.05.merge_all_datasets.ipynb  # Data integration pipeline
│   ├── 03.model.ipynb                  # ML model development & evaluation
│   └── functions.py                    # Utility functions
│
└── static/
    ├── data/
    │   ├── processData/                # Raw datasets (download required)
    │   └── cleanedData/                # Processed datasets
    └── img/                            # Visualization assets
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/khalidsbn/Home-Credit-Default-Risk.git
cd Home-Credit-Default-Risk
```

2. **Create and activate virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download datasets** (see [Dataset](#-dataset) section)

5. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

6. **Run notebooks sequentially** starting from `01.data_visualization.ipynb`

---

## 🔍 Data Exploration

### Target Distribution
<img src="./static/img/target.png" width="800"/>

**Key Insight:** The dataset is highly imbalanced with only **8.07%** of applicants defaulting on their loans. This suggests the need for careful model evaluation using metrics beyond accuracy.

---

### Demographic Analysis

#### Age Group Risk Analysis
<img src="./static/img/age.png" width="800"/>

**Findings:**
- **Highest risk**: Younger applicants (20-25 years) show 12.4% default rate
- **Risk decreases with age**: Default rate drops to 3.7% for 65-70 age group
- **Pattern**: Younger applicants may have less financial stability and credit history

---

#### Occupation Type Distribution
<img src="./static/img/occupation_type.png" width="800"/>

**Insights:**
- **Laborers** form the largest applicant group
- **Sales staff** and **Core staff** are also significant segments
- Occupation diversity suggests varied income stability levels

---

#### Organization Type (High-Risk Clients)
<img src="./static/img/organization_type.png" width="800"/>

**Risk Factors:**
- **Business Entity Type 3** clients show highest default concentration
- **Self-employed** applicants present elevated risk
- More stable employment (Government, School) shows lower default rates

---

#### Contract Type Analysis
<img src="./static/img/contract_type.png" width="800"/>

**Distribution:**
- **Cash loans**: 90.5% of applications
- **Revolving loans**: 9.5% of applications
- Revolving loans show slightly higher default rate (5.5% vs 8.3%)

---

## 📈 Model Performance

### Experimental Setup

We conducted two experiments to evaluate the impact of data integration:

1. **Baseline Model**: Using only main application dataset
2. **Enhanced Model**: Using all available datasets (merged)

### Algorithms Tested
- **Random Forest**: Ensemble learning method
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Efficient gradient boosting implementation

---

### 🔬 Experiment 1: Main Dataset Only

<img src="./static/img/results_of_main_data.png" width="800"/>

| Algorithm | ROC Train Score | ROC Test Score | Overfitting |
|-----------|----------------|----------------|-------------|
| Random Forest | 100% | 72.02% | High (27.98%) |
| XGBoost | 83.02% | 75.79% | Moderate (7.23%) |
| **LightGBM** | **99.15%** | **74.24%** | **High (24.91%)** |

**Winner:** **XGBoost** with best generalization (75.79% test ROC)

---

### 🔬 Experiment 2: All Datasets Merged

<img src="./static/img/results_all_data.png" width="800"/>

| Algorithm | ROC Train Score | ROC Test Score | Overfitting |
|-----------|----------------|----------------|-------------|
| Random Forest | 100% | 72.29% | High (27.71%) |
| XGBoost | 73.18% | 72.47% | Low (0.71%) |
| **LightGBM** | **79.25%** | **75.38%** | **Moderate (3.87%)** |

**Winner:** **LightGBM** with best test performance (75.38% ROC)

---

### 📊 Comparative Analysis

#### Performance Improvement

| Metric | Main Data Only | All Data Merged | Improvement |
|--------|---------------|-----------------|-------------|
| **Best Test ROC** | 75.79% (XGBoost) | 75.38% (LightGBM) | -0.41% |
| **Model Stability** | Moderate | High | ✅ Better |
| **Overfitting** | 7.23% (XGBoost) | 3.87% (LightGBM) | ✅ 47% Reduction |

#### Key Observations

**✅ Advantages of Merged Data:**
- **Better generalization**: Lower overfitting in LightGBM (3.87% vs 24.91%)
- **More stable predictions**: XGBoost overfitting reduced to 0.71%
- **Richer feature space**: Additional historical credit data provides context

**⚠️ Interesting Finding:**
- Test ROC slightly decreased, but model became more robust
- This suggests the merged model trades minor accuracy for better reliability

**🎯 Recommendation:**
Use **LightGBM with merged datasets** for production:
- Best balance between performance and stability
- Lower overfitting means better real-world performance
- More reliable for making lending decisions

---

## 💡 Key Findings

### Risk Indicators Identified

1. **Age Factor**
   - Younger applicants (20-30) carry higher default risk
   - Default probability decreases steadily with age

2. **Employment Stability**
   - Business Entity Type 3 and self-employed show highest risk
   - Traditional employment (government, education) correlates with lower risk

3. **Loan Type**
   - Cash loans dominate the portfolio (90.5%)
   - Revolving loans show marginally higher default rates

4. **Data Integration Impact**
   - Merging multiple data sources reduces model overfitting
   - Historical credit behavior provides valuable predictive signals
   - Model stability improves with comprehensive data

### Business Implications

- **Risk-based pricing**: Adjust interest rates based on identified risk factors
- **Targeted verification**: Focus additional checks on high-risk segments
- **Financial inclusion**: Use model to serve underbanked populations responsibly
- **Portfolio management**: Balance risk across different applicant segments

---

## 🚀 Future Enhancements

### 1. Model Improvements
- [ ] Implement deep learning approaches (Neural Networks, TabNet)
- [ ] Ensemble different models for better predictions
- [ ] Add SHAP values for model interpretability
- [ ] Experiment with automated feature engineering (FeatureTools)
- [ ] Implement cost-sensitive learning for imbalanced data

### 2. Feature Engineering
- [ ] Create time-series features from historical data
- [ ] Develop domain-specific interaction features
- [ ] Extract patterns from payment behavior sequences
- [ ] Engineer credit utilization metrics
- [ ] Create aggregate features from related tables

### 3. Hyperparameter Optimization
- [ ] Implement Bayesian optimization (Optuna)
- [ ] Use cross-validation for robust evaluation
- [ ] Grid search for optimal threshold selection
- [ ] AutoML frameworks (H2O.ai, Auto-sklearn)

### 4. Data Processing
- [ ] Advanced imputation techniques (MICE, KNN)
- [ ] Outlier detection and treatment
- [ ] Feature scaling strategies comparison
- [ ] Handle class imbalance (SMOTE, ADASYN)

### 5. Deployment & Monitoring
- [ ] Create REST API for model serving (Flask/FastAPI)
- [ ] Build interactive dashboard (Streamlit/Dash)
- [ ] Implement model versioning (MLflow)
- [ ] Set up prediction monitoring pipeline
- [ ] Create A/B testing framework

### 6. Business Analytics
- [ ] Develop profit optimization framework
- [ ] Create customer segmentation analysis
- [ ] Build risk-return visualization tools
- [ ] Design loan approval recommendation system

### 7. Documentation
- [ ] Add detailed data dictionary
- [ ] Create model card for transparency
- [ ] Write API documentation
- [ ] Develop user guide for non-technical stakeholders

---

## 🛠️ Technologies Used

### Core Libraries
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: scikit-learn
- **Gradient Boosting**: XGBoost, LightGBM
- **Utilities**: tabulate

### Development Tools
- **Version Control**: Git, GitHub
- **Environment**: Jupyter Notebook
- **Package Management**: pip, virtualenv

---

## 📝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- Implementing future enhancements
- Improving documentation
- Adding new visualizations
- Optimizing model performance
- Bug fixes and code quality improvements

---

## 📄 License

This project is part of the Kaggle Home Credit Default Risk Competition. Please refer to the [competition rules](https://www.kaggle.com/competitions/home-credit-default-risk/rules) for data usage terms.

---

## 🙏 Acknowledgments

- **Kaggle** for hosting the competition and providing the dataset
- **Home Credit** for making the data available
- Open-source community for the amazing libraries used in this project

---

## 📧 Contact

**Project Maintainer**: [khalidsbn](https://github.com/khalidsbn)

**Project Link**: [https://github.com/khalidsbn/Home-Credit-Default-Risk](https://github.com/khalidsbn/Home-Credit-Default-Risk)

---

<div align="center">
    <p>⭐ Star this repository if you find it helpful!</p>
    <p>Made with ❤️ for financial inclusion</p>
</div>
