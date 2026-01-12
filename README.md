# **CHARISMA**

This repository contains the source code for **CHARISMA** —  
*Character-Based Interaction Simulation with Multi LLM-based Agents Toward Computational Social Psychology.*

---

## **Project Structure**

- **`charisma/`** – Contains the modular pipeline and all essential components, including:
  - *Goal setup*  
  - *Scenario generation*  
  - *Interaction generation*  
  - *Goal achievement evaluation*  
  - *Simulation evaluation*

- **`app/`** – Contains the Streamlit app source code.

- **`inputs/`** – Includes various input files used at different stages of the pipeline.

- **`outputs/`** – Contains a variety of generated output files.

---

## **Prerequisites**

This project uses **`uv`** as the package manager.  
Follow the [installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to install it before continuing.

---

## **Running the App**

To start the Streamlit app, run:
```bash
uv run -m streamlit run app/main.py
```
Alternatively, you can visit the hosted version of the app here:
👉 [https://charisma.streamlit.app/]
