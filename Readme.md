AI Legal Sentiment Analyzer

An AI-powered solution for analyzing sentiment in legal documents with court case data

Overview
This project automates sentiment analysis of legal documents using IBM Watsonx.ai foundation models. It processes legal case texts to determine whether the outcome is positive (favorable), negative (unfavorable), or neutral (procedural). For demonstration purposes, this implementation analyzes a sample of 100 cases from a larger dataset.

Key Features:

Automates sentiment classification of legal documents

Processes CSV files with legal case data

Specialized text preprocessing for legal content

Generates detailed sentiment reports

Optimized for accuracy with legal terminology

Requirements
Python 3.8+

IBM Cloud account (free tier)

Watsonx.ai service enabled

Installation
Clone the repository:

bash
git clone https://github.com/yourusername/legal-sentiment-analyzer.git
cd legal-sentiment-analyzer
Install dependencies:

bash
pip install -r requirements.txt
Set up IBM Watsonx credentials:

bash
cp .env.example .env

# Update .env with your IBM Cloud credentials

Configuration
Edit the config.py file:

python

# Sample Configuration

MODEL_ID = "google/flan-t5-xxl" # Foundation model
INPUT_CSV = "data/sample_100_cases.csv" # 100-case sample
OUTPUT_CSV = "results/sentiment_results.csv"
MAX_CASES = 100 # Process only 100 cases
Usage
Process the sample cases:

bash
python legal_sentiment.py
Sample Output
text
==================================================
⚖️ LEGAL SENTIMENT ANALYZER (100 CASE SAMPLE)
🔧 Model: google/flan-t5-xxl
📄 Input file: data/sample_100_cases.csv
==================================================
Processing case 1/100: POSITIVE
Processing case 2/100: NEGATIVE
...
Processing case 100/100: NEUTRAL

✅ Analysis completed in 4.5 minutes
📊 Results saved to: results/sentiment_results.csv

SENTIMENT DISTRIBUTION (100 CASES):

- POSITIVE: 42 cases
- NEGATIVE: 37 cases
- # NEUTRAL: 21 cases
  Project Structure
  text
  legal-sentiment-analyzer/
  ├── data/
  │ ├── sample_100_cases.csv # 100-case sample dataset
  │ └── full_dataset.csv # Full 25,000+ case dataset
  ├── src/
  │ ├── legal_sentiment.py # Main analysis script
  │ ├── preprocess.py # Text cleaning functions
  │ └── config.py # Configuration settings
  ├── results/
  │ └── sentiment_results.csv # Generated output
  ├── .env # API credentials
  ├── requirements.txt # Python dependencies
  └── README.md # This document
  Methodology
  The system follows a specialized workflow for legal text analysis:

Data Selection: Uses a representative sample of 100 cases

Text Preprocessing: Cleans legal artifacts and citations

Sentiment Analysis: Classifies using legal-optimized prompts

Result Compilation: Generates case-level sentiment tags

Reporting: Provides summary statistics and CSV output

Results Interpretation
POSITIVE: Favorable outcomes (rights upheld, motions granted)

NEGATIVE: Unfavorable rulings (dismissals, liabilities)

NEUTRAL: Procedural matters (adjournments, citations)

Sample Analysis
Case ID Outcome Sentiment Excerpt
Case42 Cited POSITIVE "The court granted the plaintiff's motion for summary judgment..."
Case87 Applied NEGATIVE "Appeal dismissed with costs awarded against the appellant..."
Case15 Followed NEUTRAL "Hearing adjourned to October 15, 2024 pending..."
License
This project is licensed under the MIT License - see the LICENSE file for details.

Next Steps
To analyze the full dataset:

Update config.py:

python
INPUT_CSV = "data/full_dataset.csv"
MAX_CASES = 25000 # Or set to None for all cases
Run with increased resources:

bash
nohup python legal_sentiment.py > full_analysis.log &
