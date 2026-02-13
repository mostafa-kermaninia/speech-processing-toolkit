import json

input_file = "d:\\GitHub\\Final-ML-Project\\Final Project.ipynb"
output_file = "d:\\GitHub\\Final-ML-Project\\extracted_code.py"

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    with open(output_file, 'w', encoding='utf-8') as f:
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = cell['source']
                f.write("# %% \n")
                if isinstance(source, list):
                    f.write("".join(source))
                else:
                    f.write(source)
                f.write("\n\n")
    print(f"Successfully extracted code to {output_file}")
except Exception as e:
    print(f"Error: {e}")
