# Predicting UK General Election Results

## Executive Summary
In June 2025, YouGov projected over 300 seats could ‘swing’ at the next UK General Election, surpassing 2024 levels. This project explored an alternative approach to YouGov’s MRP, instead opting to use Logistic Regression. Data extracted from the House of Commons Library (HOCL), Office for National Statistics (ONS) and YouGov was used to produce a Logistic Regression model capable of predicting probabilities of each English constituency 'swinging' to a new party at the next election. Whilst the target was to achieve 85% Precision, the best model achieved 54%, predicting 415 ‘Hold’ and 128 ‘Swing’ seats (~172 below YouGov). Results were mapped geospatially, providing constituency-level visualisation accessible via this repository. This project was severely hampered by the class imbalance within the dataset and the magnitude of swing at the 2024 election. Future iterations of this project should focus on utilising a classification technique capabale of handling severe class imbalance.




## Data Engineering
Due to extraction limitations on the HOCL and ONS websites, each of the electoral and demographic datasets were extracted on an individual basis, resulting in the extraction of 17 standalone datasets. As such, PowerQuery was selected to complete the bulk of data engineering tasks within this project based on user competence. All datasets were cleaned, transformed and merged into a master dataset. The diagram below provides a high-level summary of the project's ETL pipeline:
<br>
![ETL Diagram](screenshots/ETL.png)

## Exploratory Data Analysis
Once the master dataset was loaded into a Jupyter Notebook, the first step in the EDA process was to visualise historical election results. The plot below revealed geographical biases and party ‘strongholds’ within the dataset and highlighted the magnitude of ‘Swing’ at the 2024 GE. 
<br>
<br>
<br>
### Historical Election Results
![Historical General Election Results](screenshots/historic_elections_plot.PNG)
<br>
<br>
<br>
### Train & Test Splitting
To prevent data leakage and overfitting, train and test sets were created from the master dataset and split temporally (train: 2010-2019; test: 2024). Splitting temporally allowed test predictions to be validated on ‘unseen’ 2024 data.
```python
# Training Dataset - 2010 - 2019
train_data = masterset[(masterset['election_year'] <= 2019)]

# Test Dataset - 2024
test_data = masterset[masterset['election_year'] == 2024 ]
```
### SweetViz EDA
Further EDA was conducted using the SweetViz API and allowed the project to analyse feature distrubiton, correlation and multicollinearity within the Training dataset. As shown in the output below, a severe 88% : 12% imbalance was found within the target 'swing' feature.
<br>
<br>
<br>
![Swing Imbalance](screenshots/swing_imbalance.PNG)
<br>
<br>
<br>
Correlation analysis revealed strong correlation between several features and suggested potential multicollinearity amongst within the training set. 
<br>
<br>
<br>
![Correlation Matrix](screenshots/correlation_matrix.PNG)
<br>
<br>
<br>
## Data Preprocessing
After consideration the potential effect of multicollinearity on model performance, the features listed in the code below were excluded from the training and test datasets; irrelevant and redundant features were also dropped at this stage.
```python
train_data = train_data.drop(columns=['candidate', 'uk_turnout', 'party_colour',
                                      'winning_party', 'constituency', 'GlobalID', 
                                      'geometry', 'winning_vote_share'])

test_data = test_data.drop(columns=['candidate', 'uk_turnout', 'party_colour',
                                    'winning_party', 'constituency', 'GlobalID', 
                                    'geometry', 'winning_vote_share'])
```
<br>

In-line with Logistic Regression requirements, categorical features within the train and test sets were encoded to produce binary values for each.

```python
train_data = pd.get_dummies(train_data, columns=['prev_party', 'region'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['prev_party', 'region'], drop_first=True)
```
X and Y Train & Test sets were created and the target feature 'swing' assigned to the Y data:

```python
X_train = train_data.drop(columns=['swing','constituency_code','election_year'])
X_test = test_data.drop(columns=['swing','constituency_code','election_year'])

y_train = train_data['swing']
y_test = test_data['swing']
```
Considering the differing range of scales amongst demographic, economic and opinion data, the decision was taken to scale all numerical features to prevent features with large numerical values dominating the model. 

``` python
scaler = StandardScaler()
scale_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
X_train[scale_columns] = scaler.fit_transform(X_train[scale_columns])
X_test[scale_columns] = scaler.fit_transform(X_test[scale_columns])

```

## Model Creation
### Baseline Model
A baseline model using the 'balanced' class_weights feature to mitigate the class imbalance seen within the dataset and was used to make swing predictions and probabilities.
``` python
model_1 = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model_1.fit(X_train, y_train)
y_pred_1 = model_1.predict(X_test)
y_prob_1 = model_1.predict_proba(X_test)[:,1]
```
Baseline model results were obtained using a classification report and indicated poor model performance, with Swing Precision scoring 0.54 and Recall of 0.32. An ROC curve further confirmed the poor performance.
<br>
![Model 1](screenshots/baseline_model_results.PNG)
<br>














## Model Evaluation
Each Logistic Regression iteration was evaluated using 



## Model Predictions
create link to plotly map

## Conclusion & Next Steps
Model performance throughout this project demonstrated the class imbalance within the target feature significantly impacted LR’s ability to predict seat swing at the next GE. Whilst model performance was poor, this project achieved its primary objective and has created a framework for future project iterations to utilise. Its suggested future iterations should focus on using more appropriate classification techniques to handle the class imbalance and capable of integrating swing magnitude seen in the 2024 GE. 

## References & Data Sources
See Reference List pdf in this repository for all references and data source links.
