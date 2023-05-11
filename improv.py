data = """
\textbf{AUC} & 61.73 & \textbf{62.86} & 62.63 & \textbf{62.67} & 61.75 & \textbf{62.24} & 60.95 & \textbf{62.18} \\

"""

# extract numbers

import re

pattern = re.compile(r'\d+\.\d+')
data = pattern.findall(data)
print(data)

# sum of even index, minus sum of odd index

s = sum([float(x) for i, x in enumerate(data) if i % 2 == 1]) - sum([float(x) for i, x in enumerate(data) if i % 2 == 0])
s /= (len(data) / 2)
print(s)
