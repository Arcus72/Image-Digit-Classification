import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
# ==========================================
# Azure Config
# ==========================================
load_dotenv()
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")
OUTPUT_DIR = "./downloaded_model"

# Paste the job name printed by train_model_azure.py
JOB_NAME = ""  # e.g. "keen_worm_sxjp1m6ntq"

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

# ==========================================
# DOWNLOADING THE TRAINED MODEL
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
ml_client.jobs.download(
    name=JOB_NAME,
    output_name="model_output",
    download_path=OUTPUT_DIR,
)
print(f"Model downloaded to: {os.path.abspath(OUTPUT_DIR)}")
