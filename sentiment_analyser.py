import os
import re
import pandas as pd
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# ===== CONFIGURATION =====
API_KEY = "KHI2uykN4nH0YScyIMoLolSFlTnF3R21EKEJd3qFR_78"  
PROJECT_ID = "59fd8ed4-1231-4703-a8c2-5615e94d3b54"  
INPUT_CSV = "legal_cases.csv"
OUTPUT_FILE = "case_sentiment_results.csv"
SERVICE_URL = "https://us-south.ml.cloud.ibm.com"
MODEL_ID = "google/flan-t5-xxl"
MAX_CASES = 100  # Process first 100 cases

# ===== MODEL SETUP =====
model = Model(
    model_id=MODEL_ID,
    credentials={
        "apikey": API_KEY,
        "url": SERVICE_URL
    },
    project_id=PROJECT_ID,
    params={
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MAX_NEW_TOKENS: 10,
        GenParams.TEMPERATURE: 0.1,
    }
)

# ===== TEXT PREPROCESSING =====
def preprocess_legal_text(text):
    """Clean legal text for better analysis"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    patterns = [
        r'\[\d{4}\]\s\w+\.?\s?\w+\.?\s\d+',  # Case citations
        r'§\s?\d+\.\d+',                      # Section numbers
        r'\(\d{4}\)\s\d+\s\w+\s\d+',          # Case references
        r'\b\d{1,3}\s\w+\s\d{1,3}\b'          # Volume/page references
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    return ' '.join(text.split()).strip()[:500]  # Truncate to 500 characters

# ===== SENTIMENT ANALYSIS =====
def analyze_legal_sentiment(text):
    """Analyze text using watsonx.ai with legal-specific examples"""
    if not text:
        return "NEUTRAL"
    
    prompt = f"""
    CLASSIFY THIS LEGAL TEXT SENTIMENT:
    - POSITIVE if favorable outcome (rights upheld, motion granted)
    - NEGATIVE if unfavorable outcome (liability found, appeal dismissed)
    - NEUTRAL for procedural matters (adjournments, citations)
    
    Examples:
    "The court ruled in favor of the plaintiff" → POSITIVE
    "Appeal dismissed with costs" → NEGATIVE
    "Hearing adjourned to next month" → NEUTRAL
    
    Text to analyze:
    "{text}"
    
    Sentiment:"""
    
    try:
        response = model.generate_text(prompt)
        
        if isinstance(response, dict) and "results" in response:
            sentiment = response["results"][0]["generated_text"].strip()
        elif isinstance(response, str):
            sentiment = response.strip()
        else:
            return "UNCERTAIN"
        
        sentiment = sentiment.upper()
        if "POSITIVE" in sentiment or "POS" in sentiment:
            return "POSITIVE"
        if "NEGATIVE" in sentiment or "NEG" in sentiment:
            return "NEGATIVE"
        if "NEUTRAL" in sentiment or "NEU" in sentiment:
            return "NEUTRAL"
            
        return "UNCERTAIN"
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return "ERROR"

# ===== CSV PROCESSING =====
def analyze_legal_cases():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: File {INPUT_CSV} not found")
        return
    
    print(f"📖 Reading cases from {INPUT_CSV}...")
    try:
        # Read only first 500 cases
        df = pd.read_csv(INPUT_CSV, nrows=MAX_CASES)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    # Verify required columns
    if 'case_id' not in df.columns or 'case_text' not in df.columns:
        print("❌ Missing required columns: case_id, case_text")
        return
    
    total_cases = len(df)
    print(f"🔍 Processing first {total_cases} cases...")
    print("=" * 80)
    
    results = []
    for index, row in df.iterrows():
        text = str(row['case_text']) if pd.notna(row['case_text']) else ""
        processed_text = preprocess_legal_text(text)
        
        sentiment = analyze_legal_sentiment(processed_text)
        
        results.append({
            "case_id": row['case_id'],
            "case_outcome": row.get('case_outcome', ''),
            "case_title": str(row.get('case_title', ''))[:100],
            "sentiment": sentiment,
            "excerpt": processed_text[:150] + "..." if processed_text else ""
        })
        
        # Show EVERY case result individually
        print(f"Case {index+1:03d}/{total_cases} | ID: {row['case_id']} | Sentiment: {sentiment}")
    
    # Save results
    if results:
        result_df = pd.DataFrame(results)
        result_df.to_csv(OUTPUT_FILE, index=False)
        print("\n" + "=" * 80)
        print(f"📊 Results for first {total_cases} cases saved to {OUTPUT_FILE}")
        
        # Show summary
        sentiment_counts = result_df['sentiment'].value_counts()
        print("\n" + "=" * 40)
        print(f"SENTIMENT SUMMARY (FIRST {total_cases} CASES):")
        print("=" * 40)
        for sentiment, count in sentiment_counts.items():
            print(f"{sentiment}: {count} cases")
        print("=" * 40)
    else:
        print("❌ No results generated")

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("=" * 80)
    print("⚖️  LEGAL CASE SENTIMENT ANALYZER")
    print(f"🔧 Model: {MODEL_ID}")
    print(f"📄 Input file: {INPUT_CSV}")
    print(f"🔢 Processing first {MAX_CASES} cases")
    print("=" * 80)
    
    # Run analysis
    analyze_legal_cases()