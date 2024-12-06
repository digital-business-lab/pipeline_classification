## MLOps Pipeline with AWS

# Reqiurements für Experimental (d.h. lokal ausführen)
# Python >= 3.8
# git 

# 1. Stat MLflow Server auf EC2 (Bucketname ändern + richtige Env auf EC2 aktivieren!)
mlflow server -h 0.0.0.0 --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://BUCKETNAME

# 2. Tracking URI in config.yaml ändern!

# 3. Tracking URI lokal auf PC setzen (in CMD eingeben + Public IPV4 ändern)
set MLFLOW_TRACKING_URI=http://Public_IPv4_DNS.eu-central-1.compute.amazonaws.com:5000/

# 4. Update DVC.yaml
-> if dataset name changes, update path in data_ingestion/outs, training/outs and evaluation/outs

# 4. Start your Pipeline (locally)
dvc repro

# 5. Start your Application (locally)
python app.py

# 6. Start on AWS
