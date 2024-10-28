import yaml
from kfp import dsl
from kfp.dsl import (
    component,
    Metrics,
    Dataset,
    Input,
    Model,
    Artifact,
    OutputPath,
    Output,
)
from kfp import compiler
import google.cloud.aiplatform as aiplatform
import os


@component(
    base_image="europe-west1-docker.pkg.dev/gcp-project-raj33342/kfp-mlops/demo_model"
)
def data_ingestion(input_data_path: str, input_data: Output[Dataset],
):
    import pandas as pd
    from datetime import datetime, timedelta
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    df = pd.read_csv(input_data_path)
    df.to_csv(input_data.path, index=False)

    
@component(
    base_image="europe-west1-docker.pkg.dev/gcp-project-raj33342/kfp-mlops/demo_model"
)
def preprocessing(train_df: Input[Dataset], input_data_preprocessed: Output[Dataset]):
    import pandas as pd
    import numpy as np
    from src.preprocess_batch import get_preprocessing
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    df = pd.read_csv(train_df.path)
    logger.info("Input Data Shape: ", df.shape)
    # df.drop(columns=['type'], inplace=True)
    df = get_preprocessing(df)
    logger.info("Preprocessed Data Shape: ", df.shape)
    df.to_csv(input_data_preprocessed.path, index=False)


@component(base_image="europe-west1-docker.pkg.dev/gcp-project-raj33342/kfp-mlops/demo_model")
def batch_prediction_component(project: str, region: str, model_name: str, input_gcs_path: Input[Dataset], 
                               job_display_name: str, instances_format: str, predictions_format: str,
                               machine_type: str, output_gcs_path: Output[Dataset]):

    
    from google.cloud import aiplatform
    import logging
    import time
    import pandas as pd
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    listed_model = aiplatform.Model.list(
        filter='display_name="{}"'.format(model_name),
        project=project,
        location=region,
    )
    model_version = listed_model[0].resource_name
    # model = aiplatform.Model(model_name=model_version)

    feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 
       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
       'quality']
    feature_skew_thresholds = {feature: 0.1 for feature in feature_names}
    
    # Create the objective config for model monitoring (skew detection)
    objective_config = aiplatform.model_monitoring.ObjectiveConfig(
        skew_detection_config=aiplatform.model_monitoring.SkewDetectionConfig(
            skew_thresholds=feature_skew_thresholds,
            data_source="gs://my-data-classification/train_data_new.csv",  # Use the training data source
            target_field="type",
            data_format="csv",
        )
    )

    # Configure model monitoring alert (via email)
    alert_config = aiplatform.model_monitoring.EmailAlertConfig(
        user_emails=["rajm893@gmail.com"]
    )
    
    logger.info(f"input_gcs_path: {input_gcs_path.path}")
    logger.info(f"Starting batch prediction job for model: {model_name}")

        # Start the batch prediction job
    batch_job = aiplatform.BatchPredictionJob.create(
        job_display_name=job_display_name,
        model_name=model_version,
        instances_format=instances_format,
        predictions_format=predictions_format,
        gcs_source=[input_gcs_path.uri],
        gcs_destination_prefix=output_gcs_path.uri,
        machine_type=machine_type,
        model_monitoring_objective_config=objective_config,
        model_monitoring_alert_config=alert_config, 
    )

    # Wait for job completion
    job_complete = False
    while not job_complete:
        job_state = batch_job.state
        if job_state == aiplatform.gapic.JobState.JOB_STATE_SUCCEEDED.value:
            logger.info("Batch prediction job completed successfully.")
            logger.info(f"Job in progress... current state job value: {job_state}")
            logger.info(f"Job in progress... current state jobstate succed value: {aiplatform.gapic.JobState.JOB_STATE_SUCCEEDED.value}")
            job_complete = True
        elif job_state == aiplatform.gapic.JobState.JOB_STATE_FAILED.value:
            logger.error(f"Batch prediction job failed: {batch_job.error.message}")
            raise RuntimeError(f"Batch prediction job failed: {batch_job.error.message}")
        else:
            logger.info(f"Job in progress... current state job value: {job_state}")
            logger.info(f"Job in progress... current state jobstate succed value: {aiplatform.gapic.JobState.JOB_STATE_SUCCEEDED.value}")
            time.sleep(30)
            
    logger.info("Batch prediction job finished. Proceeding to next step.")    

    
@component(
    base_image="europe-west1-docker.pkg.dev/gcp-project-raj33342/kfp-mlops/demo_model"
)
def merge_predictions_component(
    predictions_dataset: Input[Dataset],
    input_data_back: Input[Dataset],
    output_csv: Output[Dataset],
):
    import json
    import pandas as pd
    import numpy as np
    from google.cloud import storage
    from src.extract_predictions import get_predictions
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def read_jsonl_files(predictions_folder):
        storage_client = storage.Client()
        bucket_name, prefix = predictions_folder.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)

        all_predictions = []
        logger.info("In read_jsonl_files: ")
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            filter_dir = "prediction.results"
            if filter_dir in blob.name:
                logger.info(f"Reading {blob.name}")
                jsonl_file_content = blob.download_as_text()
                for line in jsonl_file_content.splitlines():
                    all_predictions.append(json.loads(line))
        return all_predictions

    logger.info("Starting merging: ")
    predictions = read_jsonl_files(predictions_dataset.uri)
    logger.info("Done merging: ")
    input_df = pd.read_csv(input_data_back.path)
    result_df = get_predictions(predictions, input_df)

    logger.info("Output_csv Shape: ", result_df.shape)
    result_df.to_csv(output_csv.path, index=False)
    logger.info(f"Merged predictions saved to {output_csv.path}")
    
@component(
    base_image="europe-west1-docker.pkg.dev/gcp-project-raj33342/kfp-mlops/demo_model"
)
def write_to_bigquery(
    project: str,
    dataset_id: str,
    table_id: str,
    input_data: Input[Dataset]
):

    from google.cloud import bigquery
    import pandas as pd
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize the BigQuery client
    client = bigquery.Client(project=project)

    # Construct BigQuery table reference
    table_ref = f"{project}.{dataset_id}.{table_id}"

    # Read the input data from the CSV file
    logger.info(f"Reading data from {input_data.path}")
    df = pd.read_csv(input_data.path)

    # Load the DataFrame to BigQuery
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",  # "WRITE_TRUNCATE" or "WRITE_APPEND"
    )

    logger.info(f"Writing data to BigQuery table {table_ref}")
    load_job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    load_job.result()  # Wait for the job to complete

    logger.info(f"Data successfully written to {table_ref}")


    
@dsl.pipeline(name="Batch predict Pipeline", pipeline_root="gs://my-data-classification/pipeline_root_demo")
def pipeline(
    input_data_path: str = "gs://my-data-classification/batch_test_sample.csv",
    project_id: str = "gcp-project-raj33342",
    region: str = "europe-west1",
    model_name: str = "demo_model",
    target: str = "type",
    serving_container_image_uri: str = "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
    dataset_id: str = "wine_prediction",  # BigQuery dataset where to write results
    table_id: str = "prediction_results"  
):
    data_op = data_ingestion(
        input_data_path=input_data_path)
    data_op.set_caching_options(False)

    data_preprocess_op = preprocessing(train_df=data_op.outputs["input_data"])
    data_preprocess_op.set_caching_options(False)
    batch_predict_task = batch_prediction_component(
                                                    project=project_id,
                                                    region=region,
                                                    model_name=model_name,                   
                                                    # input_gcs_path=csv_to_jsonl_task.outputs['output_jsonl'],
                                                    input_gcs_path = data_preprocess_op.outputs["input_data_preprocessed"],
                                                    instances_format='csv',
                                                    predictions_format='jsonl',
                                                    machine_type='n2-standard-4',
                                                    job_display_name='batch_predict_demo')
    batch_predict_task.set_caching_options(False)

    merge_predictions_task = merge_predictions_component(
        predictions_dataset=batch_predict_task.outputs["output_gcs_path"],
        input_data_back=data_op.outputs["input_data"]
    )
    merge_predictions_task.set_caching_options(False)

    write_to_bq_task = write_to_bigquery(
        project=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        input_data=merge_predictions_task.outputs["output_csv"] # Can be WRITE_APPEND or WRITE_EMPTY
    )
    write_to_bq_task.set_caching_options(False)
    
if __name__ == "__main__":
    compiler.Compiler().compile(pipeline_func=pipeline, package_path="batch_prediction_pipeline.json")
