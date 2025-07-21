# FinTech Customer Churn Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)


This project is a complete exploratory and predictive analysis of customer churn in a simulated FinTech company. It includes data visualization, model building, and model comparison using various machine learning techniques.

## 📁 Folder Structure
fintech_project/
├── data/
│   └── simulated_fintech_customers.csv
├── src/
│   ├── fintech_dashboard.py
│   ├── fintech_random_forest.py
│   └── model_comparison.py
├── charts/
│   ├── eda_charts/
│   ├── model_charts/
│   ├── model_charts_rf/
│   └── comparison_charts/
├── .gitignore
└── README.md

## 🔧 Setup & Installation

1. Clone this repository or copy the folder.
2. Make sure Python 3.8+ is installed.
3. It is recommended to use a virtual environment:

\`\`\`bash
python3 -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\\Scripts\\activate     # For Windows
\`\`\`

4. Install the required dependencies:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

If \`requirements.txt\` is not available, install manually:

\`\`\`bash
pip install pandas numpy matplotlib seaborn scikit-learn
\`\`\`

## 🚀 How to Run

### Run Exploratory Dashboard
\`\`\`bash
python src/fintech_dashboard.py
\`\`\`

### Run Random Forest Model
\`\`\`bash
python src/fintech_random_forest.py
\`\`\`

### Compare All Models
\`\`\`bash
python src/model_comparison.py
\`\`\`

## 📊 Features

- Exploratory Data Analysis (EDA)
- Customer churn classification
- Visualizations: Histograms, boxplots, scatter plots
- Models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Comparison charts saved as PNG

## ✅ Output

All generated visual outputs are saved under:
- \`/charts/eda_charts/\`
- \`/charts/model_charts/\`
- \`/charts/model_charts_rf/\`
- \`/charts/comparison_charts/\`

Each script uses \`plt.savefig()\` and also displays the plots with \`plt.show()\`.

## 📌 Notes

- Make sure file paths are correct when working on different systems.
- Charts are created automatically if directories don't exist.
- Use terminal for best results (IDLE might behave differently).

## 📄 License

This project is licensed under the MIT License.

## 👩‍💻 About the Author

Hi! I'm Asu, a mathematics student passionate about data science, FinTech, and machine learning.  
This project is part of my journey into predictive analytics and financial modeling.  
I aim to specialize in financial forecasting and AI-powered solutions.

- 💡 Currently learning: Quant Finance & AI
- 🌍 Interested in remote work and global freelancing
- 🛠️ Tools I use: Python, pandas, scikit-learn, matplotlib, seaborn
- 💻 [GitHub](https://github.com/asucatdev)

Connect with me on GitHub to follow more of my work!



