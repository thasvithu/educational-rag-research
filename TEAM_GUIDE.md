# 📚 Team Guide — How to Run the PDF Exploration Script

**Research Project:** Impact of Document Chunking Strategies on RAG in Educational Systems  
**Group:** 04 | 2020ICTS04 · 2020ICTS68 · 2020ICTS89

---

> **Who is this guide for?**  
> This guide is written for **Aysha and Haleema** who are setting up the project for the first time on **Windows**.  
> Follow every step in order. Do not skip any step.

---

## What This Script Does

The script (`01_Load_pdfs/explore_pdfs.py`) reads each of your collected PDF files one by one and produces a detailed analysis report showing:
- What raw text LangChain extracts from your PDFs
- Unusual or non-English characters found in the text
- Repeated lines (likely headers/footers on every page)
- A final summary table comparing all your PDFs

The report is automatically saved as a `.txt` file so you can open and read it easily.

---

## STEP 1 — Install Python 3.11

> ⚠️ Skip this step if you already have Python installed. To check, open Command Prompt and type: `python --version`

1. Go to: **https://www.python.org/downloads/**
2. Click the yellow **"Download Python 3.11.x"** button
3. Run the downloaded installer
4. **IMPORTANT:** On the first screen of the installer, tick the checkbox that says **"Add python.exe to PATH"** before clicking Install
5. Click **"Install Now"**
6. When installation is done, open **Command Prompt** and run:
   ```
   python --version
   ```
   You should see something like: `Python 3.11.9`

---

## STEP 2 — Get the Project from GitHub

> ⚠️ You need **Git** installed. To check, run: `git --version`  
> If not installed, download from: **https://git-scm.com/download/win**

1. Open **Command Prompt** (search for "cmd" in the Start menu)
2. Navigate to where you want to save the project. For example, your Desktop:
   ```
   cd Desktop
   ```
3. Clone the repository:
   ```
   git clone git@github.com:thasvithu/educational-rag-research.git
   ```
4. Move into the project folder:
   ```
   cd educational-rag-research
   ```

> 💡 In the future, to get the latest updates Vithusan pushed, just run:  
> `git pull` (while inside the project folder)

---

## STEP 3 — Install `uv` (Our Package Manager)

`uv` is a fast, modern tool for managing Python environments. It is what we use in this project.

1. Open **PowerShell** (search for "PowerShell" in the Start menu — NOT Command Prompt)
2. Run this single command:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
3. Close PowerShell after it finishes
4. Open a **new Command Prompt** and verify installation:
   ```
   uv --version
   ```
   You should see a version number like: `uv 0.5.x`

---

## STEP 4 — Create the Virtual Environment

A virtual environment is an isolated space where we install all the libraries needed for this project. This keeps your computer clean and ensures everyone uses the same library versions.

1. Make sure you are inside the project folder in Command Prompt:
   ```
   cd Desktop\educational-rag-research
   ```
2. Create the virtual environment using Python 3.11:
   ```
   uv venv --python 3.11 venv
   ```
3. You should see a message like: `Creating virtual environment at: venv`

---

## STEP 5 — Activate the Virtual Environment

Every time you open a new Command Prompt to work on this project, you must activate the environment first.

**On Windows:**
```
venv\Scripts\activate
```

After activation, your Command Prompt line will show `(venv)` at the beginning — like this:
```
(venv) C:\Users\YourName\Desktop\educational-rag-research>
```

> ⚠️ If you get an error saying "running scripts is disabled", run this in PowerShell first:  
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`  
> Then try activating again in Command Prompt.

---

## STEP 6 — Install All Required Libraries

With the virtual environment activated, install all the libraries in one command:

```
uv pip install --python venv\Scripts\python.exe langchain-community==0.3.14 pypdf==4.3.1 rich==13.9.4
```

This will download and install all necessary packages. It may take 1–2 minutes.

When done, you should see a list of installed packages with no errors.

---

## STEP 7 — Add Your PDF Files

Place all your collected PDF files inside your personal folder:

```
educational-rag-research/
└── data/
    ├── vithusan/    ← Vithusan's PDFs go here
    ├── aysha/       ← Aysha's PDFs go here       ✅ YOUR FOLDER
    └── haleema/     ← Haleema's PDFs go here     ✅ YOUR FOLDER
```

- **Aysha:** Copy all your PDFs into the `data/aysha/` folder
- **Haleema:** Copy all your PDFs into the `data/haleema/` folder

> ✅ Any PDF format is fine — the script will automatically read them all.  
> ⚠️ Do NOT place PDFs in someone else's folder.

---

## STEP 8 — Set Your Name in the Script

Open the file `01_Load_pdfs/explore_pdfs.py` in any text editor (Notepad, VS Code, etc.)

Find line 41 near the top of the file:
```python
MEMBER_NAME = "vithusan"   # Options: "vithusan" | "aysha" | "haleema"
```

Change `"vithusan"` to **your name** (lowercase, exactly as shown):
- **Aysha:** change to `"aysha"`
- **Haleema:** change to `"haleema"`

Save the file.

---

## STEP 9 — Run the Script

Make sure your virtual environment is still activated (you see `(venv)` at the start of the line).

Run the script:
```
venv\Scripts\python.exe 01_Load_pdfs\explore_pdfs.py
```

The script will process all your PDFs one by one. For large PDFs, this may take a few minutes. You will see the analysis printing on your screen as it runs.

---

## STEP 10 — View Your Output Report

When the script finishes, it will print:
```
✓ Report saved to: .../data/outputs/explore_pdf_output_aysha.txt
```

Your full report is saved at:
```
educational-rag-research/
└── data/
    └── outputs/
        └── explore_pdf_output_aysha.txt   ← Open this file
```

Open this file in **Notepad** or any text editor to read the full analysis of all your PDFs.

---

## Common Errors & Fixes

| Error Message | What It Means | How to Fix |
|---|---|---|
| `python is not recognized` | Python not added to PATH | Reinstall Python and tick "Add to PATH" |
| `uv is not recognized` | uv not installed properly | Close and reopen Command Prompt after installing uv |
| `No PDF files found` | PDFs not in the right folder | Check that your PDFs are inside `data/aysha/` or `data/haleema/` |
| `ModuleNotFoundError: langchain` | Libraries not installed | Run Step 6 again with the environment activated |
| Script shows `(venv)` is missing | Environment not activated | Run `venv\Scripts\activate` again |

---

## Quick Reference — Commands to Run Every Time

After the first-time setup, each time you want to run the script:

```
cd Desktop\educational-rag-research
git pull
venv\Scripts\activate
venv\Scripts\python.exe 01_Load_pdfs\explore_pdfs.py
```

---

*Guide written for Group 04 — Educational RAG Research Project, University of Vavuniya*
