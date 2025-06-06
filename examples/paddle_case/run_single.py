import paddle
from fastsafetensors import SafeTensorsFileLoader, SingleGroup
device = "gpu:0" if paddle.device.cuda.device_count() else "cpu"
loader = SafeTensorsFileLoader(SingleGroup(), device, nogds=False, debug_log=True, framework="paddle")
loader.add_filenames({0: ["a_paddle.safetensors", "b_paddle.safetensors"]}) # {rank: files}
fb = loader.copy_files_to_device()
tensor_a0 = fb.get_tensor(tensor_name="a0")
tensor_b0 = fb.get_tensor(tensor_name="b0")
print(f"a0: {tensor_a0}\n b0 : {tensor_b0}")
mycat = paddle.concat([tensor_a0, tensor_b0])
print(f"cat: {mycat}, size={mycat.size}")
fb.close()
loader.close()
