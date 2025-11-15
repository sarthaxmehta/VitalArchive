# Vital-Archive: AI-Powered Health Report Analyzer 

This is a multi-page Streamlit application designed to help users track, analyze, and visualize their health data. The app's core feature is its ability to ingest PDF medical reports, use AI (Google Gemini) to extract structured data, and employ a Sentence Transformer model to normalize the data against a canonical database. This application uses a local model for normalizing paramteres which can be accessed from this link: https://drive.google.com/drive/folders/1Kp36bKDNuNUMZ1fn8YKku9fDrlePFXSX?usp=drive_link Please download the model and place it in the same directory as the other files to run the application.

## Key Features

* **User Authentication:** A secure login and registration system using a local CSV file for user management.
* **AI-Powered PDF Parsing:** Users can upload their PDF lab reports. The system extracts text using `pdfplumber`, parses it with Google Gemini, and structures the data.
* **Lab Test Normalization:** Uses a `sentence-transformers` model to map varying test names from reports (e.g., "Hgb," "HGB," "Hemoglobin Test") to a single canonical name ("Hemoglobin").
* **Personalized Dashboard:** After login, users see a main dashboard with:
    * An overall "Health Score."
    * Top 3 critical risk factors from their latest report.
    * Personalized health insights (e.g., trends, checkup frequency).
    * An organ health radar chart.
* **Manual Data Entry:** A "Quick Entry" tab for users to manually input their lab results.
* **Historical Trend Analysis:** Interactive Plotly charts visualize how specific health parameters (e.g., "Glucose," "Hemoglobin") have changed over time.
* **Detailed Report History:** A comprehensive, expandable list of all past reports, highlighting parameters that are out of the normal range.

## Technology Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **AI (Data Extraction):** [Google Gemini](https://ai.google.dev/)
* **AI (Data Normalization):** [Sentence Transformers](https://sbert.net/)
* **Data Handling:** [Pandas](https://pandas.pydata.org/)
* **Visualization:** [Plotly](https://plotly.com/)
* **PDF Parsing:** `pdfplumber`
* **Data Storage:** Local Excel (`.xlsx`) and CSV (`.csv`) files.

## How It Works: The AI Pipeline

The most complex part of this application is the PDF evaluation pipeline, handled by `medical_parser.py` and `_evaluate_Report.py`:

1.  **Upload:** A user uploads a PDF in the "Analyze Health Report" page.
2.  **Text Extraction:** `pdfplumber` extracts all raw text from the PDF.
3.  **AI Parsing (Gemini):** The raw text is sent to the Google Gemini API with a prompt instructing it to return a clean JSON object containing the `ReportDate` and a list of all `Tests` (with `TestName`, `Result`, `Unit`).
4.  **Normalization (Sentence Transformers):**
    * The system loads a local Sentence Transformer model (e.g., `vitalarchive_model2`).
    * It compares the `TestName` from the PDF (e.g., "Hct") against a large list of canonical test names (defined in `medical_parser.py`).
    * Using cosine similarity, it finds the best match (e.g., "Hematocrit") and renames the column.
5.  **Storage:** The final, normalized data is converted into a single-row Pandas DataFrame and appended to the `Common dataframe.xlsx` file.
6.  **Visualization:** The Dashboard and History pages re-read this Excel file to build all plots and analytics.


