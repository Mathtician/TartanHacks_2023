import pandas as pd

inFile = "../data extraction/pitches-2019-regular-04.csv"
df = pd.read_csv(inFile)
print(df)

print()

inFile = "../data extraction/prepared_data.csv"
df = pd.read_csv(inFile)
print(df)
