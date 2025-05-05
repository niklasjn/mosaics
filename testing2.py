import pandas as pd
import numpy as np
from main import Mosaic
df = pd.read_excel('Book1.xlsx')
input_img = []
expected_output = []
for index, row in df.iterrows():
    if index < 4:
        expected_output.append([]) 
    input_img.append([])
    for col in df.columns:
        try:
            if int(col) < 12:
                input_img[index].append([int(row[col]),100,100])
            else:
                if index < 4:
                    expected_output[index].append([int(row[col]),100,100])
        except Exception as e:
            print(e)
m = Mosaic()
input_img = np.array(input_img,dtype=np.uint8)
output = m.get_averaged_image1(input_img)
expected_output = np.array(expected_output,dtype=np.uint8)
print("------ Input ------")
print(input_img)
print("----- output: ------")
print(output)
print("----- Expected output ------")
print(expected_output)

