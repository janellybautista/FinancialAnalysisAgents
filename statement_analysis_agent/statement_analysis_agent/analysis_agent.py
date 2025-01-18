import pdfplumber
import re
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI, OpenAIError

# Initialize OpenAI
client = OpenAI(api_key="type_key_here")

##########################################################
#Functions



### Defining functions
# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "".join(page.extract_text() for page in pdf.pages)
    return text

# Extract financial data using regex and pandas
def extract_financial_data(text):
    patterns = {
        "Service Revenue": r"Service revenue\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
        "Sales Revenue": r"Sales revenue\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
        "Net Income": r"Net income\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
        "Operating Expenses": r"Total\s*operating\s*expenses\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
        "Total Assets": r"Total\s*assets\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
        "Current Assets": r"Total current assets\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
        "Liabilities": r"Total Current Liabilities\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
        "Retained Earnings": r"Retained earnings\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
    }

    financial_data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1).replace(",", ""))
            financial_data[key] = value
        else:
            financial_data[key] = None

    return pd.DataFrame([financial_data])

# Analyze with OpenAI
def analyze_with_openai(financial_data_df):
    formatted_data = financial_data_df.to_string(index=False)
    prompt = f"""
    Imagine you are a professional in finance. Analyze the following financial data, calculate simple key ratios (don't show calculations, just raw results) such as Net Income Ratio, Asset-to-Liability Ratio, and Current Ratio. Also,
    provide a brief summary highlighting strengths, risks, and insights.):

    {formatted_data}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a finance professional."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        return f"Error interacting with OpenAI API: {e}"

# Plot financial data
def plot_financial_data(financial_data_df, plot_file):
    financial_data_dict = financial_data_df.iloc[0].dropna().to_dict()  # Convert first row to dictionary
    labels = list(financial_data_dict.keys())
    values = list(financial_data_dict.values())

    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color='skyblue')

    # Add exact values as labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f"${yval:,.2f}", ha='center', va='bottom')

    plt.title("Financial Data Overview")
    plt.ylabel("Amount (USD)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

# Generate LaTeX file
def generate_latex_file(analysis, plot_file, latex_filename):
    latex_code = f"""
    \\documentclass{{article}}
    \\usepackage{{graphicx}}

    \\title{{Financial Data Analysis}}
    \\author{{Generated Report}}
    \\date{{\\today}}
    \\begin{{document}}

    \\maketitle


    \\section*{{Financial Data Plot}}
    \\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{{plot_file}}}
    \\caption{{Bar chart showing the financial data.}}
    \\end{{figure}}

    \\section*{{Analysis and Key Ratios}}
    \\textbf{{Strengths:}}
    \\begin{{itemize}}
    \\item {analysis.split('Strengths:')[1].split('Risks:')[0].strip()}
    \\end{{itemize}}

    \\textbf{{Risks:}}
    \\begin{{itemize}}
    \\item {analysis.split('Risks:')[1].split('Insights:')[0].strip()}
    \\end{{itemize}}

    \\textbf{{Insights:}}
    \\begin{{itemize}}
    \\item {analysis.split('Insights:')[1].strip()}
    \\end{{itemize}}

    \\section*{{Financial Data}}
    \\begin{{tabular}}{{|l|r|}}
    \\hline
    Parameter & Value (USD) \\\\
    \\hline
    Service Revenue & {financial_data_df['Service Revenue'].iloc[0]:,.2f} \\\\
    Sales Revenue & {financial_data_df['Sales Revenue'].iloc[0]:,.2f} \\\\
    Net Income & {financial_data_df['Net Income'].iloc[0]:,.2f} \\\\
    Operating Expenses & {financial_data_df['Operating Expenses'].iloc[0]:,.2f} \\\\
    Total Assets & {financial_data_df['Total Assets'].iloc[0]:,.2f} \\\\
    Current Assets & {financial_data_df['Current Assets'].iloc[0]:,.2f} \\\\
    Liabilities & {financial_data_df['Liabilities'].iloc[0]:,.2f} \\\\
    Retained Earnings & {financial_data_df['Retained Earnings'].iloc[0]:,.2f} \\\\
    \\hline
    \\end{{tabular}}


    \\end{{document}}
    """

    with open(latex_filename, "w") as f:
        f.write(latex_code)



##########################################################
# Main script

if __name__ == "__main__":
    pdf_path = "Add_path_to_Sample_Financial_Statements.pdf"
    plot_file = "financial_plot.png"
    latex_filename = "financial_report.tex"

    # Step 1: Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)

    # Step 2: Extract financial data
    financial_data_df = extract_financial_data(text)
    print("Financial Data:")
    print(financial_data_df)

    # Step 3: Analyze with OpenAI
    print("\nAnalysis from OpenAI:")
    analysis = analyze_with_openai(financial_data_df)
    print(analysis)

    # Step 4: Plot the financial data
    plot_financial_data(financial_data_df, plot_file)

    # Step 5: Generate LaTeX file
    generate_latex_file(analysis, plot_file, latex_filename)
    print(f"Latex file '{latex_filename}' has been generated. :)")
