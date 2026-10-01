import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/chlorophyll_student_2015_2023.csv")
df["date"] = pd.to_datetime(df["date"])







variables = df.columns.drop("date")

# for variable in variables:
#     plt.figure(figsize=(12, 5))

#     plt.plot(df["date"], df[variable])

#     plt.xlabel("Data")
#     plt.ylabel(variable)
#     plt.title(f"{variable} ao longo do tempo")

#     plt.grid()
#     plt.show()

plt.figure(figsize=(10,20))


plt.plot(df["date"], df["chlorophyll_a_mg_m3"])
plt.plot(df["date"], df["sst_c"])
df["surface_pressure_hpa_mean"] = df["surface_pressure_hpa"].rolling(window=30).mean()
plt.plot(df["date"], df["cloud_fraction"])



plt.xlabel("Data")
plt.ylabel("Come-me por trás")
plt.title("Cenas ao longo do tempo")

plt.grid()
plt.show()