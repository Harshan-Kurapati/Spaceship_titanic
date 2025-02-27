
---

# Spaceship Titanic Survival Prediction

This project predicts passenger survival (Transported or Not) for the **Spaceship Titanic** dataset using **Logistic Regression**, **Decision Trees**, and **Random Forest Classifiers**. It implements data preprocessing,  hyperparameter tuning, and ensemble modeling to optimize performance.   
  
## Project Overview

This project demonstrates:
1. **Data Preprocessing**:
   - Handling missing values.
   - Encoding categorical variables and feature engineering.
   - Feature scaling and transformation.
2. **Model Building**: 
   - Implementing logistic regression from scratch.
   - Using Scikit-learn’s **DecisionTreeClassifier** and **RandomForestClassifier**.
3. **Hyperparameter Tuning**:
   - Grid Search for optimizing logistic regression hyperparameters.
4. **Model Evaluation**:
   - Comparing performance metrics for each model.
   - Combining predictions using ensemble methods for better accuracy.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Model Evaluation](#model-evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Ensemble Modeling](#ensemble-modeling)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

---

## Installation

To run this project, ensure you have the following libraries installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Dataset

The **Spaceship Titanic** dataset simulates a futuristic transport scenario with the following attributes:
- **Features**:
  - Passenger ID, HomePlanet, CryoSleep, Cabin, Destination, Age, VIP status, and spending on amenities (RoomService, FoodCourt, ShoppingMall, Spa, VRDeck).
- **Target Variable**:
  - `Transported`: Whether the passenger was transported to another dimension (`True` or `False`).

---

## Project Workflow

### Step 1: Data Preprocessing
1. **Feature Engineering**:
   - Extracted `Deck` and `Side` from the `Cabin` column.
   - Split `PassengerId` into `Passenger` and `Group` for grouping.
2. **Handling Missing Values**:
   - Imputed missing values using mode for categorical data and median for numerical data.
   - Filled missing values in `Deck` and `Side` based on passenger groups.
3. **Encoding Categorical Features**:
   - One-hot encoded `HomePlanet` and `Deck`.
   - Converted `Side` to binary values (`P -> 1`, `S -> 0`).
4. **Feature Scaling**:
   - Standardized numerical features using **StandardScaler**.

### Step 2: Model Implementation
1. **Logistic Regression**:
   - Implemented logistic regression from scratch with gradient descent and binary cross-entropy loss.
   - Optimized learning rate and iterations using grid search.
2. **Decision Trees and Random Forests**:
   - Built tree-based models using Scikit-learn for comparison.
3. **Pipeline Integration**:
   - Unified preprocessing and modeling using Scikit-learn’s **Pipeline** and **ColumnTransformer**.

### Step 3: Hyperparameter Tuning
- Conducted **Grid Search Cross-Validation** to find the best hyperparameters (`alpha` and `learning_rate`) for logistic regression.

### Step 4: Ensemble Modeling
- Combined predictions from logistic regression and random forest models using **majority voting** for improved accuracy.

---

## Model Evaluation

### Performance Metrics:
- **Logistic Regression**:
  - Accuracy: **78.78%**
- **Decision Tree**:
  - Accuracy: **73.78%**
- **Random Forest**:
  - Accuracy: **78.95%**
- **Ensemble Model**:
  - Accuracy: **78.72%**

### Cost Function Plot for Logistic Regression:
```python
plt.plot(model.get_lr_list(), model.get_cost_list())
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
plt.show()
```

---

## Hyperparameter Tuning

**Grid Search Parameters:**
- **Alpha** (Learning Rate): `[0.001, 0.01, 0.1]`
- **Learning Rate (Iterations)**: `[500, 1000, 1500, ..., 5000]`

**Best Parameters Found:**
- **Alpha**: `0.091`
- **Learning Rate**: `4500`

---

## Ensemble Modeling

The ensemble method combined predictions from logistic regression and random forest models using majority voting:
```python
final_pred = (y_pred_logistic + y_pred_forest) >= 1
```

**Ensemble Model Accuracy**: **78.72%**

---

## Results

- Logistic regression and random forests performed well, achieving around **79% accuracy**.
- Hyperparameter tuning improved the logistic regression model by optimizing learning rates and iterations.
- The ensemble model provided a balanced approach, leveraging the strengths of multiple classifiers for robust predictions.

---

## Conclusion

This project highlights:
- The importance of feature engineering and preprocessing in predictive modeling.
- The implementation of logistic regression from scratch, providing insights into gradient descent optimization.
- The benefits of hyperparameter tuning and ensemble methods for improving model performance.

### Future Improvements:
- Test advanced models like Gradient Boosting or Neural Networks.
- Perform feature selection to identify the most important predictors.
- Explore ensemble techniques like stacking or boosting for further accuracy gains.

---

## License

This project is licensed under the MIT License.

---

