import argparse

import asyncio
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

from azure.ai.ml import MLClient, Input, Output, load_component
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import Data
from azure.ai.ml.entities import AmlCompute

from extraction_dataset import register_extracted_dataset
from create_labelling_project import DownloadAndCreateLabellingProject, download_and_create_labelling_project


import uuid

import json


parser = argparse.ArgumentParser("train")
parser.add_argument("--subscription_id", type=str)
parser.add_argument("--resource_group_name", type=str)
parser.add_argument("--workspace_name", type=str)
parser.add_argument("--location", type=str)
parser.add_argument("--tags", type=str, default="{}")

args = parser.parse_args()
subscription_id = args.subscription_id
resource_group_name = args.resource_group_name
workspace_name = args.workspace_name
location = args.location
tags = json.loads(args.tags)

try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    print(ex)
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    credential = InteractiveBrowserCredential()


# Get a handle to workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
    workspace_name=workspace_name,
)

# Retrieve an already attached Azure Machine Learning Compute.
cluster_name = "simple-cpu-low"

cluster_basic = AmlCompute(
    name=cluster_name,
    type="amlcompute",
    size="Standard_D4s_v3",
    location=location,  # az account list-locations -o table
    min_instances=0,
    max_instances=1,
    idle_time_before_scale_down=60,
)
ml_client.begin_create_or_update(cluster_basic).result()


@pipeline(default_compute=cluster_name)
def azureml_pipeline(
    images_input_data: Input,
    labels_input_data: Input,
):
    preprocess_step = load_component(source="preprocess/command.yaml")
    preprocess = preprocess_step(images_input=images_input_data)

    label_split_data_step = load_component(source="label_split_data/command.yaml")
    label_split_data = label_split_data_step(
        labels_input=labels_input_data,
        images_input=preprocess.outputs.images_output,
    )

    #Third step : train
    train_step = load_component(source="train/command.yaml")
    train_data = train_step(
        train_labels_input=label_split_data.outputs.train_labels_output,
        train_images_input=label_split_data.outputs.train_images_output,
        test_labels_input=label_split_data.outputs.test_labels_output,
        test_images_input=label_split_data.outputs.test_images_output,
    )
    #
    # #Fourth step : test
    # test_step = load_component(source="test/command.yaml")
    # test_data = test_step(
    #     model_input=train_data.outputs.model_output,
    #     integration_input=label_split_data.outputs.split_integration_output,
    #     images_input=label_split_data.outputs.split_images_output,
    # )
    #
    # #Fifth step : output
    # output_step = load_component(source="output/command.yaml")
    # output = output_step(
    #     extraction_hash_input=extraction.outputs.hash_output,
    #     extraction_images_input=extraction.outputs.images_output,
    #     model_input=test_data.outputs.model_output,
    #     integration_input=test_data.outputs.integration_output,
    # )

    return {
        # "output": output.outputs.main_output,
        "output": train_data.outputs.model_output,
    }


pipeline_job = azureml_pipeline(
    images_input_data=Input(
        path="azureml:degrade-datasets:1",
        type=AssetTypes.URI_FOLDER,
    ),
    labels_input_data=Input(
        path="azureml:degrade-labels:1",
        type=AssetTypes.URI_FOLDER,
    ),
)


azure_blob = "azureml://datastores/workspaceblobstore/paths/"
experiment_id = str(uuid.uuid4())
custom_output_path = azure_blob + "degradai/" + experiment_id + "/"
pipeline_job.outputs.output = Output(
    type=AssetTypes.URI_FOLDER, mode="rw_mount", path=custom_output_path
)

pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="degradai_pipeline", tags=tags
)

ml_client.jobs.stream(pipeline_job.name)

# registered_dataset = register_extracted_dataset(
#     ml_client,
#     custom_output_path,
#     {**tags, experiment_id: experiment_id},
# )
#
# if registered_dataset is not None:
#     create_project = DownloadAndCreateLabellingProject(
#         subscription_id=subscription_id,
#         resource_group_name=resource_group_name,
#         workspace_name=workspace_name,
#     )
#
#     async def execute_async():
#         await download_and_create_labelling_project(
#             registered_dataset.dataset_version,
#             registered_dataset.dataset_name,
#             create_project,
#         )
#
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(execute_async())

model_name = "degradai"
try:
    model_version = str(len(list(ml_client.models.list(model_name))) + 1)
except:
    model_version = "1"

file_model = Model(
    version=model_version,
    path=custom_output_path + "degradai.keras",
    type=AssetTypes.CUSTOM_MODEL,
    name=model_name,
    tags={**tags, "experiment_id": experiment_id},
    description="Model created from azureML.",
)
saved_model = ml_client.models.create_or_update(file_model)

print(
    f"Model with name {saved_model.name} was registered to workspace, the model version is {saved_model.version}."
)

# integration_dataset_name = "degradai-integration"
# integration_dataset = Data(
#     name="degradai-integration",
#     path=custom_output_path + "integration",
#     type=AssetTypes.URI_FOLDER,
#     description="Integration dataset for degradai",
#     tags={**tags, "experiment_id": experiment_id},
# )
# integration_dataset = ml_client.data.create_or_update(integration_dataset)
# print(
#     f"Dataset with name {integration_dataset.name} was registered to workspace, the dataset version is {integration_dataset.version}"
# )

output_data = {
    "model_version": saved_model.version,
    "model_name": saved_model.name,
    # "integration_dataset_name": integration_dataset.name,
    # "integration_dataset_version": integration_dataset.version,
    "experiment_id": experiment_id,
}

print(json.dumps(output_data))