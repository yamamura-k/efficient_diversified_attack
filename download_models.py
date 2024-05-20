import os
from robustbench.model_zoo import model_dicts
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.utils import load_model

for dataset in ["cifar10", "cifar100", "imagenet"][-1:]:
    models = model_dicts[BenchmarkDataset(dataset)][ThreatModel("Linf")].keys()
    for model_name in models:
        try:
            model = load_model(model_name, dataset=dataset, threat_model="Linf")
            model.eval()
        except:
            os.remove(f"models/{dataset}/Linf/{model_name}.pt")
