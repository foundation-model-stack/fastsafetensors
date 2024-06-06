import torch
from fastsafetensors import SafeTensorsFileLoader, SingleGroup
loader = SafeTensorsFileLoader(SingleGroup(), torch.device("cpu"), nogds=True, debug_log=True)
loader.add_filenames({0: ["a.safetensors"]}) # {rank: files}
fb = loader.copy_files_to_device()
tensor_a0 = fb.get_tensor(tensor_name="a0")
print(f"a0: {tensor_a0}")
loader.close()