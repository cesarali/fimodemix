import torch

from fimodemix.data.dataloaders import FIMSDEpDataLoader
from fimodemix.pipelines.sdep_pipeline import FIMSDEpPipeline
from fimodemix.configs.config_classes.fim_sde_config import FIMSDEpModelParams
from fimodemix.models.fim_sde import FIMSDEp,define_from_experiment_dir


def test_vector_fields():
    params = FIMSDEpModelParams()
    dataloader = FIMSDEpDataLoader(params)
    model = FIMSDEp(params)
    databatch = dataloader.one_batch
    f_hat = model(databatch)
    print(f_hat)

def test_pipelines():
    experiment_dir = r"C:\Users\cesar\Desktop\Projects\FoundationModels\fimodemix\results\1729141498"
    model, dataloader = define_from_experiment_dir(experiment_dir)
    pipeline = FIMSDEpPipeline(model)
    output = pipeline(dataloader.one_batch)
    print(output.path.shape)
    

if __name__=="__main__":
    test_pipelines()