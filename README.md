# Natural Language Inference (NLI) Classifier

## Introduction to Natural Language Inference (NLI)

Natural Language Inference (NLI) is a fundamental task in the field of natural language processing (NLP) that involves determining the logical relationship between a premise and a hypothesis. The relationship can be classified into three categories:

-   **Entailment:** The premise logically entails the hypothesis.
-   **Contradiction:** The premise contradicts the hypothesis.
-   **Neutrality:** The premise neither entails nor contradicts the hypothesis.

NLI plays a critical role in understanding and processing human language, enabling applications such as automated reasoning, text summarization, and information extraction.

## Project Overview

This project aims to build an interactive web application that allows users to input a premise and a hypothesis. Our NLI model then classifies the relationship between these two text sequences. This application serves as a practical demonstration of NLI's capabilities and its importance in real-world applications like fact-checking and AI-powered decision-making.

## Features

-   **Interactive Web Interface:** A user-friendly web interface where users can input text sequences.
-   **Real-time Classification:** Instant classification of the relationship as entailment, contradiction, or neutrality.
-   **Examples and Guidance:** Embedded examples and guidance on how to phrase premises and hypotheses for effective results.

## Tech Stack

-   **Python:** The primary programming language used.
-   **Flask:** A lightweight WSGI web application framework used to serve the application.
-   **PyTorch and Transformers:** Used for model training and inference.
-   **HTML/CSS:** For crafting the web interface.

## Getting Started

### Prerequisites

-   Python 3.8 or above
-   Pip package manager

### Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/your-repository/nli-classifier.git
    cd nli-classifier
    ```
