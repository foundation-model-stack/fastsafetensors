import numpy as np
from safetensors.numpy import save_file
import paddle

# 创建一个大张量，比如 10GB 的 float32 数组
# 10 * 1024^3 / 4 ≈ 2.68 亿个 float32 元素
x = paddle.full(shape=[8192, 10240], fill_value=0.5, dtype='float32')
x_bf16 = x.astype(paddle.bfloat16)

# big_tensor = np.ones(8192*57344).astype(np.uint16)  # shape (2684354560,)

# # 可以 reshape 成 2D 大数组
# big_tensor = big_tensor.reshape(8192, 57344)
print(x_bf16.numpy())

# 也可以创建多个张量
tensors = {
    "large_2": x_bf16.numpy(),
}
# 保存为 safetensors 文件
save_file(tensors, "large_modela1.safetensors")
