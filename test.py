import torch
import torch.nn as nn

from modules.norm.layer_norm import LayerNorm
from modules.block.conv_next import LocalConvNextV2Block
from modules.downsample.conv import ConvDownSample
from modules.block.mb_conv import FusedMBConvBlock
from modules.upsample.pixel_shuffle import PixelShuffleUpSample
from modules.upsample.conv import ConvUpSample
from modules.upsample.conv_tranpose import ConvTransposeUpSample

# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


def test_modules():
    x = torch.randn(1, 32, 216, 384).cuda()

    module_1 = PixelShuffleUpSample(
        in_channels=32,
        out_channels=3,
        scale_factor=5
    ).cuda()

    module_2 = ConvTransposeUpSample(
        in_channels=32,
        out_channels=3,
        scale_factor=5,
    ).cuda()

    N_ITERS = 10

    with torch.no_grad():
        print("eager:", timed(lambda: module_1(x))[1])
        print("compile:", timed(lambda: module_2(x))[1])

    module_1_times = []
    for i in range(N_ITERS):
        with torch.no_grad():
            _, time = timed(lambda: module_1(x))
        module_1_times.append(time)
        print(f"eager eval time {i}: {time * 1e3}ms")

    print("~" * 10)

    module_2_times = []
    for i in range(N_ITERS):
        with torch.no_grad():
            _, time = timed(lambda: module_2(x))
        module_2_times.append(time)
        print(f"compile eval time {i}: {time * 1e3}ms")
    print("~" * 10)

    import numpy as np

    module_1_median = np.median(module_1_times) * 1e3
    module_2_median = np.median(module_2_times) * 1e3
    speedup = module_1_median / module_2_median
    # assert (speedup > 1)
    print(f"(eval) module_1_median: {module_1_median}ms, module_2_median: {module_2_median}ms, speedup: {speedup}x")
    print("~" * 10)

def test_torch_compile():
    x = torch.randn(1, 32, 216, 384).cuda()

    module_1 = ConvTransposeUpSample(
        in_channels=32,
        out_channels=3,
        scale_factor=5
    ).cuda()

    module_2 = torch.compile(
        module_1
    )

    N_ITERS = 10

    with torch.no_grad():
        print("eager:", timed(lambda: module_1(x))[1])
        print("compile:", timed(lambda: module_2(x))[1])

    module_1_times = []
    for i in range(N_ITERS):
        with torch.no_grad():
            _, time = timed(lambda: module_1(x))
        module_1_times.append(time)
        print(f"eager eval time {i}: {time * 1e3}ms")

    print("~" * 10)

    module_2_times = []
    for i in range(N_ITERS):
        with torch.no_grad():
            _, time = timed(lambda: module_2(x))
        module_2_times.append(time)
        print(f"compile eval time {i}: {time * 1e3}ms")
    print("~" * 10)

    import numpy as np

    module_1_median = np.median(module_1_times) * 1e3
    module_2_median = np.median(module_2_times) * 1e3
    speedup = module_1_median / module_2_median
    assert (speedup > 1)
    print(f"(eval) module_1_median: {module_1_median}ms, module_2_median: {module_2_median}ms, speedup: {speedup}x")
    print("~" * 10)

if __name__ == '__main__':
    test_torch_compile()
    test_modules()