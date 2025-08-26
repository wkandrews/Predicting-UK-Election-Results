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
Further EDA was conducted using the SweetViz API and allowed the project to analyse feature distrubiton, correlation and multicollinearity within the Training dataset. As shown in the outputs below, a severe 88% : 12% imbalance was found within the target 'swing' feature.
<br>
<br>
<br>
![Swing Imbalance](screenshots/swing_imbalance.PNG)
<br>
<br>
<br>






## Model Creation

## Model Evaluation
Each Logistic Regression iteration was evaluated using 



## Model Predictions
create link to plotly map

## Conclusion & Next Steps
Model performance throughout this project demonstrated the class imbalance within the target feature significantly impacted LR’s ability to predict seat swing at the next GE. Whilst model performance was poor, this project achieved its primary objective and has created a framework for future project iterations to utilise. Its suggested future iterations should focus on using more appropriate classification techniques to handle the class imbalance and capable of integrating swing magnitude seen in the 2024 GE. 

## References & Data Sources
See Reference List pdf in this repository for all references and data source links.
