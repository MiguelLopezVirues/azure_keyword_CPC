# azure_keyword_CPC

This project consists in the development and deployment of a CPC prediction model using Python as programming language, MLFlow as the framework for experiment monitoring and model logging, and Azure Machine Learning for data asset, compute, training and deployment management. 

## Structure 
- KW_CPC_development.ipynb: Data exploration and development machine learning model. 
- KW_CPC_pipeline_training_deployment.ipynb: Creation of pipeline components to prepare the data and train the model. Model log, registration and online deployment for realtime prediciton.
- prepare-data.yml: Pipeline component for data preparation.
- train-model.yml: Pipeline component for training and model log.
- /src/: Folder with generated scripts called by the yaml pipeline components.
- /data/: Folder with dataset and sample data for deployment invoke.
- /artifacts/: Model and execution logs from the pipeline job.
- /named-outputs/: Outputs uploaded by outputs mode of the pipeline.



-----
Used as a practise for the certification DP-100, obtained on the 20th April 2024.
