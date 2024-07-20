import torch
import esm
import os
os.environ['TORCH_HOME']='/data1/lujiale/esm/model'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def enable_cpu_offloading(model):
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import enable_wrap, wrap

    torch.distributed.init_process_group(
        backend="nccl", init_method="tcp://localhost:9999", world_size=1, rank=0
    )

    wrapper_kwargs = dict(cpu_offload=CPUOffload(offload_params=True))

    with enable_wrap(wrapper_cls=FullyShardedDataParallel, **wrapper_kwargs):
        for layer_name, layer in model.layers.named_children():
            wrapped_layer = wrap(layer)
            setattr(model.layers, layer_name, wrapped_layer)
        model = wrap(model)

    return model
def init_model_on_gpu_with_cpu_offloading(model):
    model = model.eval()
    model_esm = enable_cpu_offloading(model.esm)
    del model.esm
    model.cuda()
    model.esm = model_esm
    return model
class EsmFold_model:
    def __init__(self,chunk_size,cpu_only,cpu_offload):
        model = esm.pretrained.esmfold_v1()


        model = model.eval()
        model.set_chunk_size(chunk_size)

        if cpu_only:
            model.esm.float()  # convert to fp32 as ESM-2 in fp16 is not supported on CPU
            model.cpu()
        elif cpu_offload:
            model = init_model_on_gpu_with_cpu_offloading(model)
        else:
            model.cuda()
        self.model=model
class Esm2_model:
    def __init__(self):
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model.eval() # disables dropout for deterministic results
        self.model=model
        self.alphabet=alphabet
class Esm_MSA_cmap_model:
    def __init__(self):
        msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        msa_transformer = msa_transformer.eval()
        self.model=msa_transformer
        self.alphabet=msa_transformer_alphabet
class Esm2_cmap_model:
    def __init__(self):
        esm2, esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        esm2 = esm2.eval()
        self.model=esm2
        self.alphabet=esm2_alphabet
