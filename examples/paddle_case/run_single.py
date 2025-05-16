import paddle
from fastsafetensors import SafeTensorsFileLoader, SingleGroup
device = "gpu:0" if paddle.is_compiled_with_cuda() else "cpu"
loader = SafeTensorsFileLoader(SingleGroup(), device, nogds=False, debug_log=True, framework="paddle")
loader.add_filenames({0: ["a_paddle.safetensors", "b_paddle.safetensors"]}) # {rank: files}
fb = loader.copy_files_to_device()
tensor_a0 = fb.get_tensor(tensor_name="a0")
tensor_b0 = fb.get_tensor(tensor_name="b0")
print(f"a0: {tensor_a0}")
mycat = paddle.concat([tensor_a0, tensor_b0], axis=1)
print(f"cat: {mycat}, size={mycat.size}")
fb.close()
loader.close()
