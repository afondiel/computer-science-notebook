# Cloud Computing - Notes

## Table of Contents (ToC)

- Overview
- Applications
- Tools & Frameworks
- Hello World!
- References

## Overview

Cloud computing is a paradigm that provides on-demand access to a shared pool of computing resources over the internet, offering scalability, flexibility, and cost-effectiveness.

## Applications

- Infrastructure as a Service (IaaS): Virtualized computing infrastructure for scalable resources.
- Platform as a Service (PaaS): Application development and deployment without managing the underlying infrastructure.
- Software as a Service (SaaS): Accessing software applications over the internet on a subscription basis.
- Cloud Storage: Storing and retrieving data in scalable and distributed cloud environments.
- Machine Learning in the Cloud: Training and deploying machine learning models using cloud-based resources.

## Tools & Frameworks

- Amazon Web Services (AWS): A comprehensive cloud platform offering a wide range of services.
- Microsoft Azure: Cloud services platform providing tools for building, testing, and deploying applications.
- Google Cloud Platform (GCP): Infrastructure and services for computing, storage, and data analytics.
- Kubernetes: Container orchestration tool for managing and deploying containerized applications in the cloud.
- Apache OpenWhisk: Serverless computing platform for executing code in response to events.

## Hello World!

```python
# Sample code for accessing a cloud storage service (using AWS S3 as an example)
import boto3

# Create an S3 client
s3 = boto3.client('s3')

# List all buckets in the account
response = s3.list_buckets()
for bucket in response['Buckets']:
    print(f'Bucket Name: {bucket["Name"]}')
```

## References

- [NIST Definition of Cloud Computing](https://www.nist.gov/publications/definition-cloud-computing)
- [AWS Documentation](https://docs.aws.amazon.com/)
- [Microsoft Azure Documentation](https://docs.microsoft.com/en-us/azure/)


