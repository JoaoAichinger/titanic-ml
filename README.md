# Titanic Analysis

This project was designed with [This Dataset](https://www.kaggle.com/competitions/titanic/data) from Kaggle.

The main goal is to delevlop data analysis using machine learning and Data Science principles. 

## How to execute

Run ```pip install -r requirements.txt```

Execute ```main.py```

## Structure

```
├── 📁 data
│   ├── 📁 old
│   │   └── 📄 titanic.csv
│   ├── 📄 gender_submission.csv
│   ├── 📄 test.csv
│   └── 📄 titanic.csv
├── 📁 models
├── 📁 notebooks
│   ├── 📄 exploration.ipynb
│   └── 📄 feature_analysis.ipynb
├── 📁 src
│   ├── 🐍 data_loader.py
│   ├── 🐍 preprocessor.py
│   └── 🐍 trainer.py
├── 📁 tests
├── ⚙️ .gitignore
├── 📝 README.md
├── 📄 git
├── 🐍 main.py
└── 📄 requirements.txt
```

## Technical decisions

There is a main notebook ```exploration.ipynb``` designed to analyse the dataset with python pipeline and decide main structure for the script.

```feature_analysis.ipynb``` was created to test features and decide changes in the scripts. All commented and documented.

### Algorythm used

Logic Regression is a simple baseline for binary classification. Ideal for Starting projects with this complexity. Accuracy of 80%.

### Features removed

```['Embarked_S', 'Embarked_Q', 'FamilySize', 'Fare']``` were removed since they did't generate results on training.

## Results

The training resulted in a 80% Accuracy at the first test. A new dataset with other data is already set for testing
