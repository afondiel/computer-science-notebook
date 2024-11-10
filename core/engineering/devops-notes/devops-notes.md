# DevOps - Notes

## Table of Contents

- [Introduction](#introduction)
  - [What's DevOps?](#whats-devops)
  - [Applications](#applications)
- [Fundamentals](#fundamentals)
  - [How DevOps works?](#how-devops-works)
    - [DevOps Lifecycle (image)](#devops-lifecycle-image)
- [Tools \& Frameworks](#tools--frameworks)
- [Hello World!](#hello-world)
- [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
- [References](#references)

## Introduction
### What's DevOps?
- DevOps is a set of practices, tools, and a cultural philosophy that automate and integrate the processes between software development and IT teams.
- DevOps emphasizes team empowerment, cross-team communication and collaboration, and technology automation.
- DevOps aligns development and operations to optimize quality and delivery.

### Applications
- DevOps can help teams deliver value to their users faster using proven agile tools to plan, track, and discuss work across their teams.
- DevOps can help teams build, test, and deploy with CI/CD(Continue Integration/ Continue Deployement) that works with any language, platform, and cloud.
- DevOps can help teams develop securely from inception to ship with application security testing services.

## Fundamentals
### How DevOps works?
- DevOps teams include developers and IT operations working collaboratively throughout the product lifecycle, in order to increase the speed and quality of software deployment.
- DevOps teams adopt agile practices to improve speed and quality, breaking work into smaller pieces to deliver incremental value.
- DevOps teams use tools to automate and accelerate processes, which helps to increase reliability.
- DevOps teams use the infinity loop to show how the phases of the DevOps lifecycle relate to each other, symbolizing the need for constant collaboration and iterative improvement¹.

#### DevOps Lifecycle (image)

![DevOps Lifecycle](https://wac-cdn.atlassian.com/dam/jcr:ef9fe684-c6dc-4ba0-a636-4ef7bcfa11f1/New%20DevOps%20Loop%20image.png?cdnVersion=1373)

## Tools & Frameworks
- Azure DevOps Services is a set of modern dev services that help you plan, track, build, test, and deploy your applications with agile tools, CI/CD, and security.
- Azure DevOps Services integrates with GitHub and other Git providers, and offers more than 1,000 apps and services built by the community.
- Azure DevOps Services provides built-in security and compliance, and employs more than 8,500 security and threat intelligence experts across 77 countries.

**Infrastrature**
- NIGINX
- AWS
- Azure
- ELK

**Automation**
- Ansible
- Chef
- Jenkins

**Virtualization**
- Docker
- Bladecenter
- Kubernetes
- Vagrant
- VMWare


## Hello World!
```python
# A simple Python script to print "Hello, world!" using Azure DevOps Services
# Source: https://docs.microsoft.com/en-us/azure/devops/pipelines/ecosystems/python?view=azure-devops

# Import the os module
import os

# Print "Hello, world!"
print("Hello, world!")

# Get the environment variable
message = os.getenv("MESSAGE")

# Print the message
if message:
    print(message)
```

## Lab: Zero to Hero Projects
- Learn how to create a DevOps project using Azure DevOps Services and Azure DevOps Server.
- Learn how to set up a CI/CD pipeline for Python applications using Azure DevOps Services.
- Learn how to implement DevSecOps practices using GitHub Advanced Security for Azure DevOps.

## References

- [What is DevOps? | Atlassian.]( https://www.atlassian.com/devops)
- [GitHub Advanced Security for Azure DevOps](https://docs.github.com/en/github/setting-up-and-managing-billing-and-payments-on-github/about-github-advanced-security-for-azure-devops)
- [GitHub Advanced Security for Azure DevOps - GitHub Docs](https://docs.github.com/en/github/setting-up-and-managing-billing-and-payments-on-github/about-github-advanced-security-for-azure-devops)
- [Azure DevOps Services | Microsoft Azure.](https://azure.microsoft.com/en-us/products/devops/).
- [Create a DevOps project - Azure DevOps Services | Microsoft Docs](https://docs.microsoft.com/en-us/azure/devops-project/azure-devops-project-overview)
- [Build, test, and deploy Python apps - Azure Pipelines | Microsoft Docs](https://docs.microsoft.com/en-us/azure/devops/pipelines/ecosystems/python?view=azure-devops)
- [What is DevOps? DevOps Explained | Microsoft Azure.](https://azure.microsoft.com/en-us/resources/cloud-computing-dictionary/what-is-devops/).
- [What is DevOps? | Atlassian](https://www.atlassian.com/devops).
- [Azure DevOps Services | Microsoft Azure ](https://azure.microsoft.com/en-us/products/devops/.)


