
This repository contains the source code for the framework that powers the **CHARISMA** project —  
*Character-Based Interaction Simulation with Multi LLM-based Agents Toward Computational Social Psychology.*

**CHARISMA** is a modular framework for **simulating and evaluating goal-directed social interactions** with multi-agent, LLM-driven dialogue.  
It supports an end-to-end pipeline that:
- defines **shared and personal goals** (grounded in a goal taxonomy),
- generates **structured social scenarios** (with controllable properties such as difficulty),
- simulates **multi-turn dyadic interactions** between character-based agents,
- and produces **goal-achievement and behavioural evaluations** to enable systematic analysis.


<p align="center">
  <img width="6954" height="3590" alt="CHARISMA Pipeline" src="https://github.com/user-attachments/assets/0e4ebea1-46d8-41d2-9d3d-50d60aa2b85a" />
  <br/>
  <em>Figure 1: CHARISMA pipeline overview.</em>
</p>
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
👉 https://charisma.streamlit.app/
