import pandas as pd

# Define the input and output file paths
input_csv_file = 'D:/Setfit/setfit-main/plain-splitted/DEU003.csv'  # Replace with your actual CSV file path
output_tsv_file = 'D:/Setfit/setfit-main/1-trainingdata/third/DEU003.csv'  # Replace with your desired output TSV file path

# Read the CSV file with comma as a separator
df = pd.read_csv(input_csv_file,nrows=34, sep="\t", encoding="utf8")
label_column = [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1]
value_column =['narrator','narrator','narrator','narrator','narrator','narrator','narrator','character','narrator','character','narrator','narrator','narrator','narrator','narrator','narrator','narrator','narrator','character','narrator', 'character', 'narrator','character', 'character', 'character', 'character', 'character', 'character', 'character', 'character', 'character', 'character', 'character', 'character']
df['label'] = label_column
df['label_text'] = value_column


# Save the DataFrame as a TSV file with tab as a separator
df.to_csv(output_tsv_file, sep='\t', index=False, encoding="utf8")

print(f"CSV data has been saved as TSV in {output_tsv_file}")