import pandas as pd

pd.options.display.float_format = "{:.2e}".format
df = pd.read_pickle("optical_results.df")

df["measured/limit"] = df["measured (FWHM)"] / df["diffraction limit (FWHM)"]

print(df)