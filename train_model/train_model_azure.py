import os
from pathlib import Path
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import Data, Environment, BuildContext
from azure.ai.ml.constants import AssetTypes
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
# ======================================
# Azure Config
# ======================================
load_dotenv()
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")
COMPUTE_CLUSTER = os.getenv("COMPUTE_CLUSTER")


DATA_NAME = "train_mnist"
FILE_NAME = DATA_NAME + ".csv"
DATA_VERSION = "1"

folder_nadrzedny = Path(__file__).resolve().parent.parent
LOCAL_CSV_PATH = folder_nadrzedny / "data" / FILE_NAME

if LOCAL_CSV_PATH.is_file():
    print("Data file: Exist")

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

# ======================================
# UPLOADING DATA TO BLOB STORAGE
# ======================================

try:
    registered_data = ml_client.data.get(name=DATA_NAME, version=DATA_VERSION)
    # registered_data.name \ registered_data.version
    print(f"Is file uploaded: True")
except ResourceNotFoundError:
    print(f"Is file uploaded: False")
    print("Uploading...")
    mnist_data = Data(
        path=LOCAL_CSV_PATH,
        type=AssetTypes.URI_FILE,
        name=DATA_NAME,
        version=DATA_VERSION,
        description="MNIST CSV dataset",
    )
    registered_data = ml_client.data.create_or_update(mnist_data)
    print(f"Is file uploaded: True")

# ======================================
# CREATING DOCKER ENVIRONMENT
# ======================================

env = Environment(
    name="mnist-docker-uv-env",
    description="TensorFlow + uv + Keras Tuner",
    build=BuildContext(
        path="./docker",
        dockerfile_path="Dockerfile"
    )
)

# ======================================
# SUBMITTING A JOB TO THE CLUSTER
# ======================================

job = command(
    code="./src",
    command="python train.py --data_path ${{inputs.mnist_csv}} --output_dir ${{outputs.model_output}}",
    inputs={
        "mnist_csv": Input(type=AssetTypes.URI_FILE, path=registered_data.id)
    },
    outputs={
        "model_output": Output(type=AssetTypes.URI_FOLDER)
    },
    environment=env,
    compute=COMPUTE_CLUSTER,
    display_name="mnist-cnn-docker-run",
)

returned_job = ml_client.jobs.create_or_update(job)
print(f"Job started: {returned_job.studio_url}")
print(f"Job name: {returned_job.name}")
