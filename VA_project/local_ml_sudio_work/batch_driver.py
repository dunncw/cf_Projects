import os
import mlflow
import pandas as pd


def init():
    global model

    # AZUREML_MODEL_DIR is an environment variable created during deployment
    # It is the path to the model folder
    # Please provide your model's folder name if there's one
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "mlflow-model")
    model = mlflow.pyfunc.load_model(model_path)

def run(mini_batch):
    print(f"run method start: {__file__}, run({len(mini_batch)} files)")
    resultList = []

    for file_path in mini_batch:
        data = pd.read_csv(file_path)
        pred = model.predict(data)
        pred_proba = model.predict_proba(data)

        df = pd.DataFrame(pred, columns=["predictions"])
        # create a new column called "Confidence_Class_0" and "Confidence_Class_1" which contains the probability of each prediction
        df["Confidence_Class_Suicide"] = pred_proba[:, 0]
        df["Confidence_Class_Alive"] = pred_proba[:, 1]

        # create a new column called "source_id" which contains the "source_id" of the corresponding input file row data
        df["source_id"] = data["source_id"]

        df["file"] = os.path.basename(file_path)
        resultList.extend(df.values)

    return resultList