from google.cloud import storage
import os


class Cloud:
    """
    Provides functions necessary for training on cloud
    """

    storage_client = storage.Client()
    model_path = os.getenv("AIP_MODEL_DIR")
    debug_cloud = os.getenv("DEBUG_CLOUD", False)

    @staticmethod
    def upload_folder(source_folder, model_name, destination_folder, logger=None):
        """
        Uploads a local folder and its contents to a GCS bucket, preserving the directory structure.

        Parameters:
        - bucket_name: The name of the GCS bucket.
        - source_folder: The local path to the folder to upload.
        - destination_folder: The destination path in the GCS bucket.
        """

        next_version = Cloud.get_next_version(model_name, logger)

        for root, _, files in os.walk(source_folder):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(
                    local_path, start=os.path.dirname(source_folder)
                )
                updated_relative_path = relative_path.replace(
                    "version_0", f"version_{next_version}"
                )
                blob_name = os.path.join(
                    destination_folder, model_name, updated_relative_path
                )
                blob = storage.Blob.from_string(blob_name, client=Cloud.storage_client)

                blob.upload_from_filename(local_path)
                if logger:
                    logger.info(f"Uploaded {local_path} to {blob_name}")

        if logger:
            logger.info(
                f"Finished uploading folder {source_folder} as version {next_version}"
            )

    @staticmethod
    def get_bucket_name():
        path_components = Cloud.model_path.split("/")
        filtered_components = [
            comp for comp in path_components if comp and comp != "gs:"
        ]
        # bucket_name = "/".join(filtered_components[:-2]) OLD IDK HOW IT WORKED ??
        bucket_name = filtered_components[0]

        return bucket_name

    @staticmethod
    def get_bucket_path():
        bucket_name = Cloud.get_bucket_name()
        bucket_path = f"gs://{bucket_name}/"

        return bucket_path

    @staticmethod
    def get_next_version(model_name, logger=None):
        """Get the latest version number for the given model."""
        bucket = Cloud.storage_client.get_bucket(Cloud.get_bucket_name())
        prefix = f"{model_name}/"  # version_
        blobs = bucket.list_blobs(prefix=prefix)

        max_version = 0
        for blob in blobs:
            if Cloud.debug_cloud:
                logger.info(f"Blob: {blob.name}")
            parts = blob.name.split("/")
            if len(parts) > 2 and parts[1].startswith("version_"):
                version_num = int(parts[1].split("_")[1])
                max_version = max(max_version, version_num)
        if Cloud.debug_cloud:
            logger.info(f"Latest version: {max_version}")
        return max_version + 1
