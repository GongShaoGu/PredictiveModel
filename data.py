gpu_core = 10496
gpu_freq_max = 1.7
gpu_freq_min = 1.4
gpu_mem = 24

import pandas as pd
import torchvision.models as models
model_parameters = []
model_list = [
    models.resnet18(),
    models.resnet34(),
    models.resnet50(),
]
for model in model_list:
    total_params = sum(p.numel() for p in model.parameters())
    model_parameters.append(total_params)

del model_list

df = pd.read_csv("./output/results.csv")
df.head()
mapping_dict = {"resnet-18": model_parameters[0], "resnet-34": model_parameters[1], "resnet-50": model_parameters[2]}

df["model_type"] = df["model_type"].map(mapping_dict)
df = df.rename(columns={"model_type": "parameters", })
df["gpu_core"] = gpu_core
df["gpu_freq_max"] = gpu_freq_max
df["gpu_freq_min"] = gpu_freq_min
df["gpu_mem"] = gpu_mem
df = df.drop(columns=["gpu"])

df.head()
df.to_csv("./output/data.csv")