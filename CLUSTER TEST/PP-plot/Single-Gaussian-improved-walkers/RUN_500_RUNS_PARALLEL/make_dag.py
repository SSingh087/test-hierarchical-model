#!/usr/bin/env python


N_EVENTS = 500

file_content = ""


for i in range(N_EVENTS):
    file_content += f"""JOB EVENT_{i} Single-Gaussian-pp-plot.sub
RETRY EVENT_{i} 1
VARS EVENT_{i} n_event="{i}"

"""
print(file_content)

with open("events.dag", "w") as file:
    file.write(file_content)

print("File created successfully!")
