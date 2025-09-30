#!/usr/bin/env python3
"""Download Excel file from Google Sheets for Phi-3 Fine-tuning Project"""

import pandas as pd
import requests
from pathlib import Path

def download_google_sheets_as_excel(sheet_url, output_path):
    """Download Google Sheets as Excel file"""
    
    # Convert Google Sheets URL to Excel export URL
    if '/edit' in sheet_url:
        # Extract the sheet ID
        sheet_id = sheet_url.split('/d/')[1].split('/')[0]
        export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    else:
        export_url = sheet_url
    
    print(f"📥 Downloading from: {export_url}")
    
    try:
        response = requests.get(export_url)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Successfully downloaded to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error downloading: {str(e)}")
        return None

def analyze_excel_sheets(excel_path):
    """Analyze all sheets in the Excel file"""
    
    print(f"\n📊 ANALYZING EXCEL FILE: {excel_path}")
    print("=" * 60)
    
    # Read all sheets
    sheets_data = pd.read_excel(excel_path, sheet_name=None)
    
    total_samples = 0
    
    for sheet_name, df in sheets_data.items():
        print(f"\n📋 Sheet: '{sheet_name}'")
        print(f"   📏 Shape: {df.shape}")
        print(f"   📊 Columns: {list(df.columns)}")
        
        # Check for PII classification if column exists
        if 'Need Anonymization' in df.columns:
            dist = df['Need Anonymization'].value_counts()
            print(f"   🎯 PII Distribution: {dict(dist)}")
        
        total_samples += len(df)
    
    print(f"\n📈 TOTAL SAMPLES ACROSS ALL SHEETS: {total_samples}")
    
    return sheets_data

if __name__ == "__main__":
    # Example usage - replace with your actual Google Sheets URL
    # google_sheets_url = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit#gid=0"
    
    print("🔗 To download from Google Sheets:")
    print("   1. Get the sharing link from Google Sheets")
    print("   2. Make sure it's set to 'Anyone with the link can view'")
    print("   3. Run: python scripts/download_excel.py YOUR_GOOGLE_SHEETS_URL")
    
    import sys
    if len(sys.argv) > 1:
        url = sys.argv[1]
        output_path = "/opt/projects/phi3_finetune_new/dataset/complete_dataset.xlsx"
        
        downloaded_file = download_google_sheets_as_excel(url, output_path)
        
        if downloaded_file:
            analyze_excel_sheets(downloaded_file)