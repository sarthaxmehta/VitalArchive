import pdfplumber
import json
import pandas as pd
import google.generativeai as genai
import torch
from sentence_transformers import SentenceTransformer, util
import sys
import os

# --- Module-Level Constants ---
CONFIDENCE_THRESHOLD = 0.85
CANONICAL_TEST_NAMES = [
    'Hemoglobin', 'Hematocrit', 'Red Blood Cell Count', 'Mean Corpuscular Volume', 
    'Mean Corpuscular Hemoglobin', 'Mean Corpuscular Hemoglobin Concentration', 
    'White Blood Cell Count', 'Neutrophils', 'Lymphocytes', 'Platelet Count', 
    'MPV (mean platelet volume)', 'Prothrombin Time', 'International Normalized Ratio', 
    'Activated Partial Thromboplastin Time', 'Fibrinogen', 'D-dimer', 'Sodium', 
    'Potassium', 'Chloride', 'Carbon Dioxide', 'Anion gap', 'Glucose', 
    'Hemoglobin A1c', 'Blood Urea Nitrogen', 'Creatinine', 'eGFR (estimated GFR)', 
    'Urine albumin/creatinine ratio (UACR)', 'Urine Protein', 'Urine Specific Gravity', 
    'Urinalysis RBCs', 'Urinalysis WBCs', 'Total Protein', 'Albumin', 
    'Prealbumin (transthyretin)', 'Bilirubin', 'Alanine Aminotransferase', 
    'Aspartate Aminotransferase', 'Alkaline Phosphatase', 'Gamma-Glutamyl Transferase', 
    "5'-nucleotidase", 'Ammonia', 'Albumin/Globulin Ratio', 'CK-MB', 
    'C-Reactive Protein', 'hs-CRP', 'Erythrocyte Sedimentation Rate', 'Ferritin', 
    'Iron', 'Total Iron Binding Capacity', 'Transferrin Saturation', 
    'Reticulocyte Count', 'Vitamin B12', 'Folate', 'Vitamin D', 
    'Vitamin A (retinol)', 'Vitamin E (alpha-tocopherol)', 'Vitamin K (phylloquinone)', 
    'Vitamin C (ascorbic acid)', 'Thiamine (B1)', 'Riboflavin (B2)', 'Niacin (B3)', 
    'Pyridoxine (B6)', 'Calcium', 'Ionized calcium', 'Phosphate', 'Magnesium', 
    'Uric Acid', 'Total Cholesterol', 'LDL Cholesterol', 'HDL Cholesterol', 
    'Triglycerides', 'Apolipoprotein B (ApoB)', 'Lp(a) (lipoprotein a)', 
    'Thyroid-Stimulating Hormone', 'Free Thyroxine', 'Free Triiodothyronine', 
    'Anti-TPO antibodies', 'Cortisol', 'ACTH', 'Insulin (fasting)', 
    'Fasting C-peptide', 'Prolactin', 'Estradiol', 'Progesterone', 'Testosterone', 
    'Follicle-Stimulating Hormone', 'Luteinizing Hormone', 'Estrone', 
    'HCG (pregnancy)', 'Prostate-Specific Antigen', 'Carcinoembryonic Antigen', 
    'Alpha-Fetoprotein', 'Cancer Antigen 125', 'Cancer Antigen 19-9', 
    'TSH receptor antibody (TRAb)', 'Antinuclear Antibody', 'Rheumatoid Factor', 
    'Anti-CCP (ACPA)', 'Hepatitis B Surface Antigen', 'Hepatitis C Antibody', 
    'HIV 1/2 Antibody and p24 Antigen', 'VDRL / RPR (syphilis)', 
    'ANA profile (ENA panel)', 'Complement C3', 'Complement C4', 'AST/ALT Ratio', 
    'Blood gas: pH', 'PaO2 (arterial)', 'PaCO2 (arterial)', 'O2 saturation (arterial)', 
    'Lactate', 'Anion gap (ABG)', 'Osmolality (serum)', 'Stool occult blood (FOBT)', 
    'Stool calprotectin', 'Alpha-1 antitrypsin (serum)', 'Celiac serology (tTG IgA)', 
    'Selenium', 'Zinc', 'Copper', 'Ceruloplasmin', 'Beta-2 microglobulin', 
    'Haptoglobin', 'Coombs test (direct)', 'Myoglobin', 'Beta-hydroxybutyrate (blood)', 
    'Carnitine (total)', 'Homocysteine', 'Methylmalonic acid (MMA)', 'Procalcitonin', 
    'HOMA-IR (calculated)', 'Serum trypsinogen', 'Glycated albumin', 'Fructosamine', 
    'Antithrombin III', 'Protein C', 'Protein S', 'Factor V Leiden (genetic)', 
    'Factor VIII activity', 'Complement split products (C3a/C5a)', 
    'Sperm count / semen analysis', 'Anti-Müllerian Hormone (AMH)', 'Folate RBC', 
    'Thyroglobulin (tumor marker)', 'Insulin antibody', 'HIV viral load (quantitative)', 
    'Hep B DNA', 'Hep C RNA', 'Cryoglobulins', 'Serum electrophoresis (SPEP) — M spike', 
    'Urine protein electrophoresis (UPEP)', 'Placental growth factor (PlGF)', 
    'CA 15-3 (breast cancer)', 'Thrombin time (TT)', 'Bleeding time (historical)', 
    'Cardiac myeloperoxidase (MPO)', 'Oxidized LDL', 'Serum amyloid A', 
    'Helicobacter pylori stool antigen', 'Stool ova & parasites', 
    'Serum galactomannan', 'Cryptococcal antigen', 'CSF glucose (if relevant)', 
    'CSF protein', 'CSF cell count (WBC)', 'CSF RBCs', 'Neopterin (CSF/serum)', 
    'Metanephrines (plasma)', 'Catecholamines (urine/plasma)', 'Renin (plasma)', 
    'Aldosterone', 'Aldosterone/renin ratio', 
    'Serum NGAL (neutrophil gelatinase-associated lipocalin)', 'KIM-1 (urinary)', 
    'Urine sediment RBC casts', 'Urine sediment WBC casts', 'Urine sodium (UNa)', 
    'Fractional excretion of sodium (FENa)', 'Serum cystatin C', 
    'Serum beta-trace protein', 'Serum paraoxonase', 
    'sTNFR1/2 (soluble TNF receptors)', 'FibroTest/FibroScan surrogate markers', 
    'HOMA-B (beta cell function)', 'A1AT phenotype/genotype (alpha-1 antitrypsin)', 
    'Salivary cortisol', 'Semen morphology (Kruger)', 
    'Fractional excretion of urea (FEUrea)', 
    'Peripheral blood smear findings (qualitative)', 'Blood culture', 
    'Serum tryptase (mast cell activation)', 'Serum IgE', 
    'Serum immunoglobulins (IgG, IgA, IgM)', 'Specific IgE panels (food/inhalant)', 
    'Serum kallikrein', 'Erythropoietin (EPO)', 
    'Serum sTfR (soluble transferrin receptor)', 'Hepcidin', 'Monocytes', 
    'Eosinophils', 'Basophils', 'Plasma cell markers (flow cytometry)', 
    'Serum N-+A1:A216terminal propeptide of type I procollin (PINP)', 
    'Alpha-1 Antitrypsin', 'Brain Natriuretic Peptide', 
    'Neutrophil-to-Lymphocyte Ratio', 'Cystatin C', 'Osmolality, Serum', 
    'Retinol Binding Protein', 'S100B Protein', 'Zinc Protoporphyrin', 
    'Vitamin K1', 'Lactate, CSF', 'Apolipoprotein A1', 'Serotonin', 
    'Chromogranin A', 'Homovanillic Acid', 'Vanillylmandelic Acid (VMA)', 
    'Uric Acid Clearance', 'Chloride, Urine', 'Thyroxine-binding Globulin (TBG)', 
    'Coenzyme Q10', 'Malondialdehyde', 'Glutathione Peroxidase', 
    'Catalase Activity', 'Superoxide Dismutase Activity', 
    "8-Hydroxy-2'-deoxyguanosine (8-OHdG)", 'Ferritin Light Chain', 
    'Manganese, Serum', 'Thallium, Blood', 'Nickel, Serum', 'Chromium, Serum', 
    'Cobalt, Serum', 'Bismuth, Blood', 'Vanadium, Serum', 'Rubidium, Serum', 
    'Strontium, Serum', 'Boron, Serum', 'Iodine, Urine', 'Vitamin B1 (Thiamine)', 
    'Vitamin B2 (Riboflavin)', 'Vitamin B5 (Pantothenic Acid)', 
    'Vitamin B7 (Biotin)', 'Vitamin D2', 'Vitamin D3', 'Vitamin E (Tocopherol)', 
    'Vitamin K2', 'Lutein', 'Zeaxanthin', 'Beta-Cryptoxanthin', 'Lycopene', 
    'Alpha-Carotene', 'Gamma-Tocopherol', 'Coagulation Factor VIII', 
    'Coagulation Factor IX', 'Coagulation Factor XI', 'Fibrin Degradation Products', 
    'Plasminogen', 'Thrombin-Antithrombin Complex', 'Prothrombin Fragment 1+2', 
    'Plasma Ammonia', 'Serum Myoglobin', 'Serum Angiotensin-Converting Enzyme (ACE)', 
    'Plasma Free Metanephrines', 'Serum 17-Hydroxyprogesterone', 
    'Serum Androstenedione', 'Anion Gap', 'Red Cell Distribution Width',
    'Platelet Distribution Width (PDW)', 'Immature Platelet Fraction (IPF)', 
    'Neutrophil-to-Lymphocyte Ratio (NLR)', 'Monocyte-to-Lymphocyte Ratio (MLR)', 
    'Glucose Tolerance (2h)', 'Serum Insulin', 'C-Peptide', 
    'Parathyroid Hormone (PTH)', 'Vitamin B4 (Choline)', 'Boron', 'Lithium', 
    'Valproic Acid', 'Theophylline', 'Carbamazepine', 'Phenytoin', 'Cyclosporine', 
    'Tacrolimus', 'Sirolimus', 'Clozapine', 'Lead', 'Mercury', 'Cadmium', 
    'Arsenic', 'Uranium'
]

# --- Module-Level Variables (to hold loaded models) ---
gemini_model = None
normalization_model = None

def initialize(api_key, normalization_model_path, gemini_model_name="models/gemini-2.0-flash"):
    """
    Initializes the Gemini and SentenceTransformer models.
    This must be called once before using valuate().
    """
    global gemini_model, normalization_model
    
    print("Initializing models...")
    try:
        # Configure Gemini
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(gemini_model_name)
        print("Gemini model initialized.")
    except Exception as e:
        raise Exception(f"Error configuring Gemini: {e}")

    try:
        # Load Normalization Model
        print(f"Loading normalization model from {normalization_model_path}...")
        normalization_model = SentenceTransformer(normalization_model_path)
        print("Normalization model loaded.")
    except Exception as e:
        raise Exception(f"Error loading normalization model: {e}")

def valuate(pdf_filepath, output_csv_path=None):
    """
    Processes a single medical PDF, normalizes it, and saves it to a CSV.

    Args:
        pdf_filepath (str): The full path to the input PDF file.
        output_csv_path (str, optional): The full path to save the resulting CSV.
                                        If None, defaults to the same name as the
                                        PDF but with a '_normalized.csv' extension.

    Returns:
        str: The full path to the saved CSV file.
        
    Raises:
        Exception: If models are not initialized or if any step fails.
    """
    # --- 1. Check if models are loaded ---
    if gemini_model is None or normalization_model is None:
        raise Exception("Models not initialized. Call medical_parser.initialize() first.")

    # --- 2. Determine output path ---
    if output_csv_path is None:
        base_path = os.path.splitext(pdf_filepath)[0]
        output_csv_path = f"{base_path}_normalized.csv"

    print(f"Processing {pdf_filepath}...")

    # --- 3. Extract and Parse PDF ---
    try:
        parsed_data = _extract_and_parse_pdf(pdf_filepath, gemini_model)
    except Exception as e:
        raise Exception(f"Failed to extract or parse PDF: {e}")

    # --- 4. Create DataFrame ---
    df_raw = _create_raw_dataframe(parsed_data)
    if 'ReportDate' not in df_raw.columns:
        raise Exception("Error: 'ReportDate' column not found in the parsed data.")

    # --- 5. Normalize Columns ---
    try:
        mapping = _normalize_columns(df_raw, normalization_model, CANONICAL_TEST_NAMES, CONFIDENCE_THRESHOLD)
        df_normalized = df_raw.rename(columns=mapping)
        
        # Handle columns that mapped to the same canonical name
        df_normalized = df_normalized.groupby(level=0, axis=1, sort=False).first()
    except Exception as e:
        raise Exception(f"Failed to normalize columns: {e}")

    # --- 6. Save and Return ---
    try:
        df_normalized.to_csv(output_csv_path, index=False)
        print(f"Successfully saved normalized data to: {output_csv_path}")
        return output_csv_path
    except Exception as e:
        raise Exception(f"Error saving final CSV: {e}")


# --- Internal Helper Functions (prefixed with _) ---

def _extract_and_parse_pdf(pdf_path, model):
    """Internal: Extracts text from PDF and uses Gemini to parse it into JSON."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise Exception(f"Error reading PDF file: {e}")

    if not text:
        raise Exception("No text could be extracted from the PDF.")

    prompt = f"""
    Extract ALL medical test results from this medical report (blood/urine/stool/biochemistry/hematology etc).
    Identify the report date.
    Return ONLY valid JSON in this format:
    {{
        "ReportDate": "YYYY-MM-DD",
        "Tests": [
            {{ "TestName": "...", "Result": "...", "Unit": "...", "NormalRange": "..." }}
        ]
    }}
    Text:
    {text}
    """
    
    response = model.generate_content(prompt)
    raw = response.text.strip()

    if raw.startswith("```"):
        raw = raw.split("```", 1)[1].strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
        raw = raw.replace("```", "").strip()

    try:
        return json.loads(raw)
    except Exception as e:
        print(f"\n--- Error Parsing JSON ---\nRaw response: {raw}\n--------------------------")
        raise Exception(f"Gemini response could not be parsed as JSON: {e}")

def _create_raw_dataframe(parsed_json):
    """Internal: Converts the parsed JSON into a single-row horizontal DataFrame."""
    report_date = parsed_json.get("ReportDate", "")
    tests = parsed_json.get("Tests", [])
    
    if not tests:
        print("Warning: No tests were found in the parsed JSON data.")
    
    row = {"ReportDate": report_date}
    for t in tests:
        test_name = t.get("TestName", "").strip()
        result = t.get("Result", "")
        if test_name:
            if test_name not in row:
                row[test_name] = result
            else:
                print(f"Warning: Duplicate test name '{test_name}' found. Using first occurrence.")
                
    return pd.DataFrame([row])

def _normalize_columns(df, model, canonical_names, threshold):
    """Internal: Maps raw column names to canonical names."""
    print("Normalizing column names...")
    canonical_names_norm = [name.lower().strip() for name in canonical_names]
    canonical_map = {norm_name: original_name for norm_name, original_name in zip(canonical_names_norm, canonical_names)}
    canonical_embeddings = model.encode(canonical_names_norm, convert_to_tensor=True)
    
    raw_columns = df.columns.tolist()
    column_mapping = {}

    for raw_name in raw_columns:
        if raw_name.lower().strip() == 'reportdate':
            column_mapping[raw_name] = 'ReportDate'
            continue

        raw_name_norm = raw_name.lower().strip()
        query_embedding = model.encode(raw_name_norm, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, canonical_embeddings)[0]
            
        best_match_index = torch.argmax(similarities)
        best_match_score = similarities[best_match_index].item()
        best_match_norm_name = canonical_names_norm[best_match_index]

        if best_match_score >= threshold:
            final_name = canonical_map[best_match_norm_name]
        else:
            final_name = raw_name
        
        column_mapping[raw_name] = final_name

    return column_mapping