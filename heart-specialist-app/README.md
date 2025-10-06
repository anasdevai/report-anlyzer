# Heart Specialist Application

This project is a heart specialist application that utilizes LangChain and Neo4j to analyze user-uploaded reports, generate summaries, provide diet suggestions, and outline precautions.

## Features

- User report uploads
- LLM analysis of reports
- Summary generation
- Diet suggestions
- Precautions based on analysis

## Installation

To install the necessary packages, run the following commands:

```bash
pip install fastapi
pip install uvicorn
pip install langchain
pip install neo4j
```

## Usage

To start the application, run:

```bash
uvicorn src.main:app --reload
```

## Directory Structure

- `src/`: Contains the source code for the application.
- `tests/`: Contains the test files for the application.
- `requirements.txt`: Lists the required Python packages.
- `README.md`: Documentation for the project.