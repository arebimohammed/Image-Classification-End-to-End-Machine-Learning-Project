
An end to end machine learning project on the Intel Image Classification Dataset in Kaggle. The
project encompasses the EDA, Model building, Hyperparameter Tuning, Training and Serverless Deployment on AWS Lambda


# Image-Classification-End-to-End-Machine-Learning-Project :moneybag:
This repository holds all the code, data, models, dependencies and deployment file, for the data cleaning and analysis, model building, hyperparameter tuning, 
preprocessing, training, and deployment of the Intel Image Classification Dataset in Kaggle. The dataset can be downloaded from [here](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) 

This project is a capstone project and is in partial fulfillment for the [mlbookcamp](https://datatalks.club/courses/2021-winter-ml-zoomcamp.html) course.  

# Table of Contents  
* [Description of the problem](#description-of-the-problem-open_book)
* [Project Architecture](#project-architecture-triangular_ruler)
* [Model Deployment and Instructions](#model-deployment-and-instructions-rocket)
* [URL for testing](#url-for-testing)


## **Description of the problem :open_book:**

The aim of the project that is based on image data that can lie in one six categories which are, buildings, forest, glacier, mountain, sea, street,
is to predict which class an image belongs to. The dataset used is referred to as the "Intel Image Classification" dataset, which is very popular in Kaggle 
thanks to the kaggle user puneet6060 for the kaggle contribution, but Intel is the actual source of the dataset. Some samples are in the data folder described in the
[project architecture](#) below.


Note that all the conclusions (statements) and results for the EDA, feature extraction/selection, model building, hyperparamter tuning, preprocesing and model selection are discussed and viewed in the notebook.ipynb file and not in this README. This notebook is a pretty large file as it contains the output of most cells so it hasn't been uploaded to github, but rather to my google drive and can be found under this [link](https://drive.google.com/file/d/1yq1JTpwdO_BzUt4CP4U3N43Vu86yjJr7/view?usp=sharing)


## **Project architecture: :triangular_ruler:**

```
├── Data
│   ├── seg_pred              <- Contains around 7k unlabeled images to be used only for prediction. This repo contains only samples
│   ├── seg_test              <- Contains around 3k images that are segmented in their respective folder by class, mentioned above. This repo contains only samples
|   └── seg_train             <- Contains around 14k images that are segmented in their respective folder by class, mentioned above. This repo contains only samples
│
├── images                    <- This folder contains the images that are used by and required for the Streamlit app.
│
├── model                     <- This folder contains the final trained model to be converted to a tflite model.
│
├── intel-classifier.tflite   <- This file is the converted model to tflite model that is to be deployed using docker and AWS Lambda
│
├── notebook.ipynb            <- This is the jupyter notebook where the data is loaded, cleaned, explored/analyzed and where feature extraction, model building, hyperparameter tuning and model selection is done.
│
├── train.py                  <- This is the Python script that loads the data, does the preprocessing required for the final model chosen with its tuned hyperparameters and trains the model as well as save it using both in its original format and in tflite format.
│
├── lambda_func.py            <- This is the Python script that loads the saved model using tflite runtime (lighter & faster), does all the image preprocesing and fetching. The prediction is also done here with the lambda function for AWS Lambda
│
├── test.py                   <- This is the Python script that call the deployed lambda function using the API Gateway endpoint and sends a URL of an image and prints the result 
├── requirements.txt          <- This is a text file for the dependencies for the training, data exploration and lambda function for prediction, that was used in the virtual environment.
│
└── Dockerfile                 <- This is the Docker file that was used to build the docker image to be pushed to AWS. It uses amazon's public lambda python image as a base
```


## **Model Deployment and Instructions :rocket:**

The model is deployed using [AWS Lambda](https://aws.amazon.com/lambda/). The main script that is deployed is the lambda_fun.py that uses the tflite model. 
This model, environment and script are then containerized to a docker image using ```docker build tag <image-tag> .```
This docker image is then pushed to AWS ECR (Elastic Container Registry). I used a mixture of the AWS Command-Line Interface to accomplish the cloud deployment and the AWS UI console. 
* I firstly have to configure aws uzing ```aws configure```
* Then I created a resource group using ```aws ecr create-repository --repository-name <repoName>``` choosing a location from [here](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Concepts.RegionsAndAvailabilityZones.html#:~:text=regional%20service%20endpoint.-,AWS%20Regions,-Each%20AWS%20Region)
* Then I take note of the repositoryUri that is returned as it will be needed later
* Then I get the ecr login password using ```aws ecr get-login-password``` and save it somewhere safe
* Then I login to aws using ```docker login -u AWS -p <PASSWORD> <repositoryUri>```. We are now logged in and can push the docker image.
* Before pushing the image I simply tag it first using ```docker tag <imageName> <newImageName>```
* Then I pushed the image to AWS Elastic Container Registry using ```docker push <newImageName>```. The image is now pushed after several minutes (it is ~600MB)
* Then I created the Lambda Function from this image using the AWS UI console. And finally add an API gateway to expose the function 
* We can then use this to send requests along with the "/predict" endpoint to get the model predictions.


## **URL for testing**

The URL for the lambda function is deployed [here](https://txa53tjffl.execute-api.eu-west-3.amazonaws.com/test/predict) which you can use for a limited time for testing purposes.



