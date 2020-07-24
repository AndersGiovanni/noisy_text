import pandas as pd
import sys
# Read file
df1 = pd.read_csv(sys.argv[1], sep="\t")
# Convert labels
df1["Text"] = df1["Text"].apply(lambda x: str.lower(x))
df1.to_csv(sys.argv[2],  index=False,  sep='\t')