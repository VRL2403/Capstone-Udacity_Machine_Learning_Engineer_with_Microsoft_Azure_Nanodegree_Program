# Capstone - Udacity Machine Learning Engineer with Microsoft Azure

## Table of content
* [Overview](#overview)
* [Dataset](#dataset)
* [Project Set Up and Installation](#project-set-up-and-installation)
* [Automated ML](#automated-ml)
* [Hyperparameter Tuning](#hyperparameter-tuning)
* [Model Deployment](#model-deployment)
* [Screen Recording](#screen-recording)

---

## Overview

In this project, we will be using Azure Machine Learning Studio to create a model and it's pipeline and then deploy the best model and consume it. We will be using two approaches in this project to create a model:
  1. Using Azure AutoML 
  2. Using Azure HyperDrive <br/>
   
And the best model from any of the above methods will be deployed and then consumed.<br/>
We will be using `LogisticRegression classifier` to train the model and `accuracy` as a metric to check best model.

---

## Dataset

### Pima Indians Diabetes Dataset

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

https://www.kaggle.com/uciml/pima-indians-diabetes-database

#### Following are the features (columns) in the dataset:
The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

#### Following is the significance of each feature (columns):
1. `Pregnancies`: Number of times pregnant
2. `Glucose`: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3. `BloodPressure`: Diastolic blood pressure (mm Hg)
4. `SkinThickness`: Triceps skin fold thickness (mm)
5. `Insulin`: 2-Hour serum insulin (mu U/ml)
6. `BMI`: Body mass index (weight in kg/(height in m)^2)
7. `DiabetesPedigreeFunction`: Diabetes pedigree function
8. `Age`: Age (years)
9. `Outcome`: Class variable (0 or 1)

---

## Project Set Up and Installation

Following steps were perform to setup the project and prepare model, pipeline, deploy and consume it:

1. Create a new `Compute Instance` or use existing one provided in the Lab, if no compute instance provided in lab create compute instance of `STANDARD_DS2_V12` type. (In our case we used Compute Instance provided by lab)
2. Create a cpu `Compute Cluster` of type `STANDARD_DS12_V2` with `4 max nodes`, and `low priority` status to train the model.
3. Register dataset to the Azure Machine Learning Workspace under `Datasets` tab and inorder to used it for creating model.
4. Uploading Notebooks and required scripts to create `AutoML` and `Hyperparameter Model`.
5. Create `AutoML` experiment through `'automl.ipynb'` notebook which performed following steps:
    - Import neccesary libraries and packages.
    - Load the dataset and compute cluster to train the model.
    - Create a new AutoML Experiment through notebook.
    - Define AutoML settings and Configuration for AutoML Run, then submit the experiment to train the model.
    - Using the `RunDetails widget`, show experiment details various models trained and accuracy achieved.

6. Create the `HyperDrive` experiment through `'hyperparameter_tuning.ipynb'` notebook as follow:
    - Import neccesary libraries and packages.
    - Load the dataset and compute cluster to train the model.
    - Create a new Experiment for HyperDrive through notebook.
    - Define `Early Termination Policy`, `Random Parameter Sampling hyperparmenter` and configuration settings.
    - Create `'train.py'` script to be used in training the model, then submit the experiment.
    - Use the `RunDetails widget` to show experiment details such as the runs accuracy rate.

8. Select the Best Model from above two approaches, Retrive the Best Model and register it in the workspace. (In our case AutoML model)
7. Then, Deploy the Best Model as a web service.
8. Enable the application insights and service logs.
9. Test the endpoint by sending a sample json payload and receive a response.

### Task

In this project our task is to predict wheather a user is diabetic or not based on features like number of pregnancies the patient has had, their BMI, insulin level, age, and so on and also it values.

### Access

The dataset was taken from kaggle from the link provided in dataset section and then uploaded (registered) in the Azure Machine Learning Studio in Datasets tab through `'upload from local file'` option. The dataset was registered with the name `'diabetes'`.

![Upload/Register Dataset](/images/Dataset_Registered.PNG)

And then the dataset was loaded in notebook using following code snippet `'Dataset_get_by_name(ws,dataset_name)'`.

---

## Automated ML
Overview of the `automl` settings and configuration settings experiment:
   - `experiment_timeout_minutes`: Set to 30 minutes. The experiment will timeout after that period to avoid over utilizing of resources.
   - `max_concurrent_iterations`: Set to 4. The max number of concurrent iterations to be run parallely.
   - `primary_metric`: Set to `'accuracy'`, which is a best suitable metrics for classification problems. 
   - `n_cross_validations`: Set to 5, therefore the training and validation sets will be divided into five equal sets.
   - `iterations`: Number of iterations for the experiment is set to 24. For number of iteration to be performed to prepare model.
   - `compute_target`: To Set project cluster used for the experiment.
   - `task`: set to `'classification'` since our target to predict whether the user is diabetic or not.
   - `training_data`: To provided the dataset which we loaded for the project.
   - `label_column_name`: Set to the result/target colunm in the dataset `'Outcome'` (0 or 1).
   - `enable_early_stopping`: Enabled to terminate the experiment if there is no improvement in model performed after few runs.
   - `featurization`: Set to `'auto'`, it's an indicator of whether implementing a featurization step to preprocess/clean the dataset automatically or not.
   - `debug_log`: For specifying a file wherein we can log everything. 

### Results
- Following Models were trained by AutoML Experiment:

![Model Trained AutoML](/images/Model_Trained_AutoML.PNG)
![AutoML Accuracy](/images/AutoML_Accuracy.PNG)

- The Best Model acheived from the AutoML experiment from `VotingEnsemble` model. The Voting Ensemble Model which gave the accuracy of 0.78137 ***(78.137%)***.

![Best Model AutoML](/images/Best_Model_AutoML.PNG)

![Best Model Parameter and Completion Status](/images/Best_Model_Parameter_and_Model_Completion_Status.PNG)

- `RunDetails` widget of best model screenshot 

![Run Details AutoML](/images/Run_Details_AutoML.PNG)

- Best model run id screenshot

![Best Model with RunID AutoML](/images/Best_Model_with_runid_AutoML.PNG)

---

## Hyperparameter Tuning

For HyperDrive Experiment we have chosen LogisticRegression classifier as model. Since our target column of dataset is to predict whether a person is diabetic or not (i.e. 1 or 0), which is a classification problem. The dataset is loaded into the notebook and then model is trained using the script written in `'train.py'` file.

Overview of the `Hyperparameter Tuning` settings and configuration settings experiment: 
- Parameter used for `HyperDriveConfig` settings:
  - `run_config`: In this we provide the directory wherein the `train.py` script is present and in arguments we provide `dataset id` and `compute target` details and environment by using `ScriptRunConfig` function.
  - `hyperparameter_sampling`: In this we set the parameters for sampling using `RandomParameterSampling` which includes tuning hyperparameters like for classification  `--C`: Inverse of regularization, `--max_iter`: Maximum number of iterations.
  - `policy`: In this we define the `Early termination policy` (i.e.`BanditPolicy`)  as an early stopping policy to improve the performance of the computational resources by automatically terminating poorly and delayed performing runs. Bandit Policy ends runs if the primary metric is not within the specified slack factor/amount when compared with the highest performing run.
  - `primary_metric_name`: In this we specify the metric on the basis of which we will judge the model performance. (In our case its `Accuracy`).
  - `primary_metric_goal`: In this we specify what we want the primary metric value to be (In our case `PrimaryMetricGoal.MAXIMIZE`), as we want to maximize the Accuracy.
  - `max_total_runs`: The total number of runs we want to do inorder to train the model using the above specified hyperparameters. (Set to 24)
  - `max_concurrent_runs`: The max number of concurrent iterations to be run parallely. (Set to 4)

### Results
Following is the Run details for HyperDrive Experiment with Parameter details

![Hyperdrive Experiment Completed](/images/Hyperdrive_Exp_Completed.PNG)

The best performing model has a 73.958% accuracy rate with --C = 1000 and --max_iter = 25. 

- `RunDetails` widget screenshot of the best model
  
![Run Detail Hyperdrive](/images/Run_Detail_Hyperdrive.PNG)

![Run Details Metrics Accuracy](/images/Run_Details_Metrics_Accuracy.PNG)

![Run Details Parameter Visualization](/images/Run_Details_Parameter_Visualization.PNG)

- Best model run id screenshot
  
![HyperDrive Best Model](/images/HyperDrive_Best_Model.PNG)

--- 

## Model Deployment
From Models trained from the above two approaches, the `AutoML Experiment` gave accuracy of **78.137%** while the `HyperDrive Experiment` gave accuracy of **73.958%**. The performance of AutoML model exceeded the HyperDrive performance by 4.179%, So we decide to register AutoML model as the best model and deployed as a web service. And Application Insights were also enabled for it.

Also, we have created inference configuration and edited deploy configuration settings for the deployment. The inference configuration and settings explain the set up of the web service that will include the deployed model. Environment settings and `scoring.py` script file should be passed the InferenceConfig. The deployed model was configured in `Azure Container Instance(ACI)` with `cpu_cores` and `memory_gb` parameters initialized as 1. 

Following is code snippet for the same:
```python
inference_config = InferenceConfig(entry_script='scoring.py',
                                   environment=environment)
service_name = 'automl-deploy'
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(workspace=ws,
                       name=service_name,
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=deployment_config,
                       overwrite=True
                      )
service.wait_for_deployment(show_output=True)
```
- Best Model and It's Metrics:
  
![Best Model](/images/Best_Model_with_runid_AutoML.PNG)

![Best Model Metrics 1](/images/Best_Model_Metrics1.PNG)

![Best Model Metrics 2](/images/Best_Model_Metrics2.PNG)

- Best Model Deployment Status

![Best Model Deployment Status](/images/Best_Model_Deployment_Status.PNG)

- Consuming Endpoint and Testing
  
![Consuming Endpoint](/images/Consuming_endpoint.PNG)

![Testing Deployed Endpoint](/images/Testing_Deployed_Endpoint.PNG)

---
## Screen Recording

Following is link to Projects Demonstration Screen Recording [link](https://youtu.be/K5tPVB-LOxw)

---

### Future improvements:
1. Performing some feature engineering on data, like some records in the datasets are hypotheical like number of pregnecies 15 and 17, also SkinThickness For adults, the standard normal values for triceps skinfolds are: 2.5mm (men) or about 20% fat; 18.0mm (women) or about 30% fat, so such outliers or mistakes done while collecting data can be removed to improve the prediction.
2. Trained the dataset using different models like KNN or Neural Networks.
3. In case of AutoML, we can try Interchanging n_cross_validations value between (2 till 7) and see if the prediction accuracy improved by tuning this parameter.