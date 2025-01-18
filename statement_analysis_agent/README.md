# Setup Instructions

Before running `analysis_agent.py`, follow these steps:

1. **Initialize OpenAI by adding the API key**:
   In the script, initialize the OpenAI client by adding your API key:
   ```python
   client = OpenAI(api_key="type_key_here")

2. **Modify the input file path**:
  Go to the "Main script" and update the input file path to point to your PDF:
   ```python
   pdf_path = "Add_path_to_Sample_Financial_Statements.pdf"

# Output

This agent analyzes the imported PDF with OpenAI, plots the relevant data from the file and generates a LaTeX report. 
