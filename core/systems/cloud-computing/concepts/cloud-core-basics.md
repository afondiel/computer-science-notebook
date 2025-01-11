# Cloud Computing Technical Notes

![An overview diagram illustrating the components of cloud computing, including various service models (IaaS, PaaS, SaaS), deployment models (public, private, hybrid), and key characteristics such as scalability and resource pooling.](../resources/cloud-computing-overview.webp)

## Quick Reference
- Cloud computing is a model that enables on-demand access to a shared pool of configurable computing resources over the Internet.
- Key use cases: Hosting applications, data storage, and providing scalable computing power.
- Prerequisites: Basic understanding of the Internet and general computing concepts.

## Introduction

Cloud computing refers to the delivery of computing services—including servers, storage, databases, networking, software, and analytics—over the Internet (the cloud). This model allows users to access and manage these resources without needing to own or maintain physical hardware. Cloud computing has transformed how businesses operate by providing flexible resources that can scale according to demand.

## Core Concepts

### Fundamental Understanding

1. **Definition**: Cloud computing provides on-demand access to computing resources via the Internet, allowing users to rent rather than purchase hardware or software.

2. **Characteristics**:
   - **On-demand self-service**: Users can provision resources automatically without human intervention from the service provider.
   - **Broad network access**: Services are available over the network and can be accessed through standard mechanisms (e.g., laptops, smartphones).
   - **Resource pooling**: Resources are pooled to serve multiple consumers using a multi-tenant model.
   - **Rapid elasticity**: Resources can be quickly scaled up or down based on demand.
   - **Measured service**: Resource usage is monitored, controlled, and reported for transparency.

### Visual Architecture

```mermaid
graph TD
    A[Cloud Computing] --> B[Service Models]
    A --> C[Deployment Models]
    B --> D[IaaS]
    B --> E[PaaS]
    B --> F[SaaS]
    C --> G[Public Cloud]
    C --> H[Private Cloud]
    C --> I[Hybrid Cloud]
```

## Implementation Details

### Basic Implementation

To utilize cloud computing effectively:

1. **Choose a Service Model**:
   - **Infrastructure as a Service (IaaS)**: Provides virtualized computing resources over the Internet (e.g., Amazon EC2).
   - **Platform as a Service (PaaS)**: Offers a platform allowing customers to develop, run, and manage applications without dealing with the infrastructure (e.g., Google App Engine).
   - **Software as a Service (SaaS)**: Delivers software applications over the Internet on a subscription basis (e.g., Microsoft 365).

2. **Select a Deployment Model**:
   - **Public Cloud**: Services offered over the public Internet and available to anyone who wants to purchase them.
   - **Private Cloud**: Exclusive cloud infrastructure operated solely for one organization.
   - **Hybrid Cloud**: Combines public and private clouds, allowing data and applications to be shared between them.

3. **Accessing Services**:
   - Users can access cloud services via web interfaces or APIs provided by cloud service providers.

```python
# Example of accessing a cloud service using Python
import requests

# Example API call to a hypothetical cloud service
response = requests.get('https://api.cloudprovider.com/data')
data = response.json()
print(data)
```

This code snippet demonstrates how to make an API call to retrieve data from a cloud service.

## Real-World Applications

### Industry Examples

Cloud computing is widely used across various sectors:

- **E-commerce**: Hosting online stores and managing inventory through cloud platforms.
- **Healthcare**: Storing patient records securely in the cloud while ensuring compliance with regulations.
- **Education**: Providing online learning platforms that scale with user demand.

### Hands-On Project

**Project: Setting up a simple web application in the cloud**

1. Choose a PaaS provider (e.g., Heroku).
2. Create an account and set up a new application.
3. Deploy a simple web application using provided templates or frameworks.
4. Monitor application performance through the provider's dashboard.

## Tools & Resources

### Essential Tools

- Cloud platforms: Amazon Web Services (AWS), Microsoft Azure, Google Cloud Platform (GCP)
- Development tools: Docker for containerization, Git for version control
- Monitoring tools: New Relic, Datadog for performance tracking

### Learning Resources

- Online courses: "Introduction to Cloud Computing" on Coursera
- Books: "Cloud Computing for Dummies" by Judith Hurwitz
- Documentation: Official documentation from AWS, Azure, or GCP for hands-on tutorials

## Appendix

### Glossary

- **IaaS (Infrastructure as a Service)**: Virtualized computing resources provided over the Internet.
- **PaaS (Platform as a Service)**: A platform allowing developers to build applications without managing underlying infrastructure.
- **SaaS (Software as a Service)**: Software delivered over the Internet on a subscription basis.

## References

- [1] https://www.lucidchart.com/blog/cloud-computing-basics
- [2] https://www.ness.com/cloud-computing-101-understanding-the-basics-and-key-concepts/
- [3] https://dev.to/olawaleoloye/basic-cloud-computing-concepts-to-master-31jb
- [4] http://www.quadranttechnologies.com/learn-about-core-cloud-services-with-this-guide
- [5] https://cs.uok.edu.in/Files/79755f07-9550-4aeb-bd6f-5d802d56b46d/Custom/Para_DistrComputing_1.pdf
- [6] https://www.linkedin.com/learning/cloud-computing-understanding-core-concepts