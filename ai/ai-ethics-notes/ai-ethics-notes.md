# AI Ethics - Notes

## Table of Contents
  - [Introduction](#introduction)
  - [Key Concepts](#key-concepts)
  - [Applications](#applications)
  - [AI Ethics Frameworks](#ai-ethics-frameworks)
  - [Key Ethical Principles](#key-ethical-principles)
  - [How Ethical AI Works in Practice](#how-ethical-ai-works-in-practice)
  - [Ethical Challenges and Pitfalls](#ethical-challenges-and-pitfalls)
  - [Feedback & Evaluation](#feedback-evaluation)
  - [Tools for AI Ethics](#tools-for-ai-ethics)
  - [Hello World! (Practical Example)](#hello-world-practical-example)
  - [Advanced Exploration](#advanced-exploration)
  - [Zero to Hero Lab Projects](#zero-to-hero-lab-projects)
  - [Continuous Learning Strategy](#continuous-learning-strategy)
  - [References](#references)

## Introduction
- AI ethics is a set of principles and practices focused on ensuring the safe, fair, and transparent use of AI systems, especially in ways that uphold human rights and societal values.

### Key Concepts
- **Bias and Fairness**: Ensuring AI does not unfairly disadvantage specific groups based on race, gender, etc.
- **Transparency**: Making AI processes understandable and accessible for non-experts.
- **Accountability**: Ensuring there are mechanisms to hold individuals or companies accountable for AI outcomes.
- **Privacy**: Protecting user data from misuse and overreach in AI systems.
- **Common Misconception**: Ethical AI is not only about avoiding harm but also about actively creating benefits for all stakeholders.

### Applications
- **Healthcare**: Ensuring that AI diagnostic tools work equally well across different demographic groups.
- **Finance**: Reducing bias in credit scoring algorithms to prevent unfair lending practices.
- **Hiring**: Avoiding biased hiring algorithms that may inadvertently favor certain groups over others.
- **Policing and Security**: Ensuring facial recognition systems are accurate and non-discriminatory.
- **Autonomous Vehicles**: Making ethical decisions in life-critical situations, such as accident prevention.

## AI Ethics Frameworks
1. **IEEE Ethically Aligned Design**: A set of recommendations for developing ethically aligned AI.
2. **EU Ethics Guidelines for Trustworthy AI**: Guidelines from the EU to ensure that AI is lawful, ethical, and robust.
3. **OECD Principles on AI**: Guidelines from the OECD to promote responsible AI across member countries.
4. **Asilomar AI Principles**: High-level ethical guidelines developed by the Future of Life Institute.

### Description
1. **Data Collection and Bias Mitigation**: Identifying and eliminating biases in data.
2. **Model Design and Fairness**: Creating models that respect ethical guidelines, including fairness and inclusivity.
3. **Transparency and Explainability**: Developing interpretable models and providing clear explanations.
4. **Privacy Protection**: Using techniques like differential privacy to safeguard user data.
5. **Accountability Mechanisms**: Implementing ways to trace responsibility in AI systems.

## Key Ethical Principles
1. **Fairness**: Avoid discrimination and bias.
2. **Transparency**: Maintain clarity on AI processes and decision-making.
3. **Privacy**: Protect user data and uphold consent.
4. **Accountability**: Ensure traceability of decisions.
5. **Safety**: Mitigate risks associated with AI applications, particularly in critical areas.

## How Ethical AI Works in Practice
1. **Bias Detection and Mitigation**: Use tools to identify and correct biases in data and models.
2. **Interpretable AI**: Develop algorithms that provide interpretable outputs.
3. **User Consent and Privacy Safeguarding**: Use consent-driven data collection and privacy-preserving algorithms.
4. **Human Oversight**: Involve human reviewers, especially for high-stakes decisions.

## Ethical Challenges and Pitfalls
- **Data Bias**: Historical data can embed societal biases, leading to unfair AI decisions.
- **Opacity**: Complex AI models (e.g., deep neural networks) are often hard to interpret.
- **Privacy Violations**: The massive data required by AI can compromise user privacy.
- **Responsibility Gaps**: Determining who is accountable when AI makes mistakes is challenging.
- **Dual-Use Concerns**: Some AI systems can be misused for harmful purposes.

## Feedback & Evaluation
- **Ethics Review Boards**: Formalize evaluations with ethics committees or boards.
- **Impact Assessment**: Conduct assessments to understand and mitigate AI’s societal impact.
- **User Feedback**: Collect feedback to understand AI’s impact on users and improve.

## Tools for AI Ethics
- **Fairness and Transparency Libraries**:
  - **Fairlearn**: An open-source library in Python to help assess fairness in machine learning models.
  - **IBM AI Fairness 360 (AIF360)**: Provides algorithms and metrics to mitigate and assess bias.
  - **Explainable AI Tools (LIME, SHAP)**: Increase model interpretability by explaining predictions.
- **Privacy-Preserving Tools**:
  - **PySyft**: A library for privacy-preserving AI using federated learning and differential privacy.
  - **Opacus**: Provides PyTorch-based differential privacy functionality for secure model training.

## Hello World! (Practical Example)
- **Fairness Evaluation with Fairlearn**:
  ```python
  import fairlearn.metrics as metrics
  from fairlearn.reductions import DemographicParity, ExponentiatedGradient
  from sklearn.tree import DecisionTreeClassifier

  # Assume `X_train`, `y_train`, `sensitive_feature` are defined
  model = DecisionTreeClassifier()
  dp = DemographicParity()
  mitigator = ExponentiatedGradient(model, constraints=dp)

  mitigator.fit(X_train, y_train, sensitive_features=sensitive_feature)
  predictions = mitigator.predict(X_test)

  # Assess fairness metrics
  disparity = metrics.demographic_parity_difference(y_test, predictions, sensitive_feature=sensitive_feature)
  print("Demographic Parity Difference:", disparity)
  ```

## Advanced Exploration
- **Ethics in Reinforcement Learning**: Explore ethical dilemmas specific to RL, such as agent goals that could harm the environment.
- **Differential Privacy and Encryption in AI**: Study techniques to keep data private in large-scale AI training.
- **The Black-Box Problem**: Delve into research on interpretability and explainability for complex models like neural networks.

## Zero to Hero Lab Projects
1. **Bias Mitigation in Recruitment AI**: Build a model to review resumes with a focus on reducing gender or racial bias.
2. **Differential Privacy in Data Sharing**: Create a model that uses differential privacy to protect sensitive data.
3. **Interpretable Credit Scoring System**: Develop a transparent credit scoring model that explains its decisions to users.

## Continuous Learning Strategy
- **Next Steps**: Explore further topics such as Algorithmic Accountability and Social Impact of AI.
- **Related Topics**: Investigate Responsible AI Practices and Fairness in Machine Learning to deepen understanding.

## References

- *Ethics of AI: A Systematic Literature Review of Principles and Challenges* by Jobin et al.
- Fairlearn documentation: [https://fairlearn.org/](https://fairlearn.org/)
- *Artificial Intelligence and Life in 2030: One Hundred Year Study on AI* by the Stanford AI Lab.

Courses and additional resources
- [Intro to AI-Ethics - Kaggle Free Course](https://github.com/afondiel/Intro-to-AI-Ethics-Free-Course-Kaggle)
