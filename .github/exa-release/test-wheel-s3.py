import uuid

import boto3
import lance
import pyarrow as pa

credentials = {
    "aws_access_key_id": "ACCESS_KEY",
    "aws_secret_access_key": "SECRET_KEY",
    "aws_region": "us-east-1",
}
local_options = {
    **credentials,
    "aws_endpoint": "http://localhost:4566",
    "allow_http": "true",
}
bucket = f"lance-wheel-{uuid.uuid4().hex}"
s3 = boto3.client(
    "s3",
    endpoint_url=local_options["aws_endpoint"],
    region_name=credentials["aws_region"],
    aws_access_key_id=credentials["aws_access_key_id"],
    aws_secret_access_key=credentials["aws_secret_access_key"],
)
s3.create_bucket(Bucket=bucket)
uri = f"s3://{bucket}/dataset"
lance.write_dataset(
    pa.table({"value": [1, 2, 3]}),
    uri,
    storage_options=local_options,
)

for _ in range(20):
    assert lance.dataset(uri, storage_options=local_options).count_rows() == 3

https_uri = f"s3://aws-publicdatasets/lance-wheel-{uuid.uuid4().hex}"
service_error_markers = (
    "403",
    "404",
    "accessdenied",
    "invalidaccesskeyid",
    "nosuchbucket",
    "not found",
    "response error",
    "signaturedoesnotmatch",
)


def assert_s3_service_response() -> None:
    try:
        lance.dataset(https_uri, storage_options=credentials)
    except (OSError, ValueError) as error:
        message = str(error).lower()
        assert any(marker in message for marker in service_error_markers), message
    else:
        raise AssertionError("Expected the non-dataset S3 path to fail")


for _ in range(20):
    assert_s3_service_response()
