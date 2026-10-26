# Program Planner

> Periodized program planner and recommendation generator for Fullsterkur.ai.

<br />
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#tech-stack">Tech Stack</a></li>
    <li><a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#running-the-service">Running the Service</a></li>
      </ul>
    </li>
    <li><a href="#project-structure">Project Structure</a></li>
  </ol>
</details>

---

## Tech Stack

| <img src="https://raw.githubusercontent.com/github/explore/main/topics/python/python.png" width="60" /> | <img src="https://raw.githubusercontent.com/github/explore/main/topics/scikit-learn/scikit-learn.png" width="60" /> | <img src="https://raw.githubusercontent.com/github/explore/main/topics/pandas/pandas.png" width="60" /> | <img src="https://raw.githubusercontent.com/github/explore/main/topics/fastapi/fastapi.png" width="60" /> |
|:---:|:---:|:---:|:---:|
| **Python**<br>Language the service and model pipeline are built in. | **scikit-learn**<br>Trains the models behind periodization recommendations. | **pandas / NumPy**<br>Data loading, cleaning, and feature preparation for training. | **FastAPI**<br>Serves the trained model and planning logic to the Mobile app over HTTP. |

---

## Getting Started

### Prerequisites

- [Python](https://www.python.org/downloads/)

### Installation

1. Clone the repo
   ```
   git clone https://github.com/fullsterkur-ai/program-planner.git
   cd program-planner
   ```
2. Create and activate a virtual environment
   ```
   python -m venv .venv
   ```
   Windows:
   ```
   .venv\Scripts\activate
   ```
   macOS/Linux:
   ```
   source .venv/bin/activate
   ```
3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

### Running the Service

```
uvicorn main:app --reload
```

---

## Project Structure

```
program-planner/
├── artifacts/   # Trained models and training outputs
├── data/        # Datasets
├── ml/          # Model training code
└── requirements.txt
```

---

## See Also

- [Main repository README](https://github.com/fullsterkur-ai/J26-SE-342): project overview, architecture, components, and team
- [DEVELOPMENT.md](https://github.com/fullsterkur-ai/J26-SE-342/blob/main/DEVELOPMENT.md): coding conventions and branching/merging strategy

<p align="right">(<a href="#program-planner">back to top</a>)</p>
