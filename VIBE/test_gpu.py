import torch
import time

print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

if torch.cuda.is_available():
    # 创建一个大的张量
    x = torch.randn(10000, 10000).cuda()
    y = torch.randn(10000, 10000).cuda()
    
    # 测试 GPU 计算速度
    start = time.time()
    for _ in range(10):
        z = torch.matmul(x, y)
    end = time.time()
    
    print(f'GPU computation time: {end - start:.2f} seconds')
else:
    print('GPU not available!')