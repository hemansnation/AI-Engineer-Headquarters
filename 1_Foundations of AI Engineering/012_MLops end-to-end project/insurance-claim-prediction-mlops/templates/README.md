Steps to be Performed :)

1. create the virtual environment
2. create req.txt file and .gitignore and readme files
3. Activate the environment
4. Dataset folder created(Storing the data later)
5. Experiment folder created(for notebook testing)
6. claim folder is created(Inside this coding folder structure will be there)
7. inside claim folder we need to create a file __init__.py to treat that folder as a package(we convert it later)
8. inside claim folder another folder components created
9. inside claim folder another folder constant created
10. inside claim folder another folder logging and exception created
11. inside claim folder another folder pipeline and utils and cloud created
11. inside claim folder another folder entity created
12. Dockerfile and setup.py files created
13. .env file created
14. fill the requirement.txt file with name of libraries
15. run pip install -r requirements.txt
16. inside claim folder and inside all folders create __init__.py





<-------time to first commit---------->



(create a new repo and follow all steps mentioned by github)
git init
git add .
git commit -m "project structure setup is completed"
git branch -M main


17. write code inside setup.py(make sure to update -e .)
18. once whole setup.py filr ready and run python setup.py(create some .egg file where we can find all package details)
19. create a pyfile logger.py insider claim folder and then inside logging folder and write logging related code.
20. create a pyfile exception.py inside claim folder and then inside exception folder and write exception related code.



21. create data.ingestion.py file inside claim/components folder
22. now for configuration related stuff we need to use entity folder. Inside the entity folder we need to create training_config.py,data_ingestion_config.py and as of now lets write all required config for the data ingestion stage. Coding is not started yet.
23. inside constant folder and write constants on __init__.py for data_ingestion
24. create a new folder params inside claims folder and inside that create params.yaml and then only pass train_test_split ratio for data ingestion stage

25. start coding inside entity/training_config.py file for training_config stage related configuraton information. write code there
26. write code inside entity/data_ingestion_config.py related to data ingestion config.
27. wrie a function inside utils folder(read yaml file function and use it in data ingestion config code to get the split ratio)
28. now create data_ingestion.py file inside components section and pass coding.( at the end of this file we need to write data ingestion artifact but to write that we need to reate a data_ingestion_artifact.py file inside entity folder and the write artifact related code)
29.test on main.py

starting data validation


1. go to constsnts folder and provide contstns for data validation
2. create  a new data_validation_config.py file inside entity folder where data validation related configs have to be written
3. now to validate schema we need to have one schema.yaml file inside the root folder data_config
4. make sure to add SCHEMA_FILE_PATH inside constant folder
5. put your data validation artifact inside entity folder inside the file data_validation_artifact.py file
6. inside components folder create a new data_validation.py file and start writing code accordingly(write supportive utils functions while writing this code)
7. tested on main.py

starting data transformation
1. go to constant folder and write all required constants for data transformation
2. create a new data_transformation_config.py inside entity folder where data transformation related confugs have to be written
3. write components code data_transformation.py and write code(in between make sure to write data_transformation artifacts)
4. once done run in main.py

starting model training

1. go to constant folder and write all constants
2. once done create model_trainer_config.py file inside claim/entity folder
3. then write inside components section some first outline of methods model_trainer.py(without mlflow)
4. make sure to add model trainer artifact also by creating model_trainer_artifact.py
5. write mlflow related function inside components folder model_trainer.py

test on main.py


start building the complete training pipeline
1. inside pipeline folder create a pyfile training_pipeline.py write code(without s3)

after that prepare app.py and then create dockerfile and .github yaml

and execute ecr push but dont use firsdt continouoys deployment in the yaml file


now once done then create ec2 and then run following commands first
1. sudo apt-get update -y
2. sudo apt-get upgrade
3. curl -fsSL https://get.docker.com -o get-docker.sh
4. sudo sh get-docker.sh
5. sudo usermod -aG docker ubuntu
6. newgrp docker


go to ur github repo setting and choose runner under actions
create self hostd runner and choose linux and run mentiond all lines in ec2