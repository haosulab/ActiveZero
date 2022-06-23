# Apply fast disparity warping on GPU
#
# By Jet <j@jetd.me>, MAR 2021
#
from collections import namedtuple
from typing import Callable, Optional

import torch
from cupy.cuda import function
from pynvrtc.compiler import Program

_apply_disparity_func_pos: Optional[Callable] = None
_apply_disparity_func_neg: Optional[Callable] = None


def _build_cuda_kernels():
    global _apply_disparity_func_pos
    global _apply_disparity_func_neg

    _apply_disparity_pos_kernel = """
    extern "C" {
        __global__ void apply_disparity_pos(
        float *dst, const float *src, const int *disp, int h, int w, int c, int total_l) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= total_l)
                return;
            int dbase = (i/h/c*h+i%h)*w;
            for (int j = w - 1; j >=0; j--) {
                int idx = j + disp[dbase+j];
                if (idx < w)
                    dst[i*w+idx] = src[i*w+j];
            }
        }
        __global__ void apply_disparity_neg(
        float *dst, const float *src, const int *disp, int h, int w, int c, int total_l) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= total_l)
                return;
            int dbase = (i/h/c*h+i%h)*w;
            for (int j = 0; j < w; j++) {
                int idx = j + disp[dbase+j];
                if (idx > -1)
                    dst[i*w+idx] = src[i*w+j];
            }
        }
    }
    """
    program = Program(_apply_disparity_pos_kernel, "apply_disparity.cu")
    m = function.Module()
    m.load(bytes(program.compile().encode()))
    _apply_disparity_func_pos = m.get_function("apply_disparity_pos")
    _apply_disparity_func_neg = m.get_function("apply_disparity_neg")


def apply_disparity_cu(img: torch.Tensor, disp: torch.Tensor):
    """
    Apply disparity using jit cuda ops.

    :param img: tensor needed warping. (N, C, H, W)
    :param disp: (N, H, W) or (N, 1, H, W)
    :return:
    """

    # load kernel if haven't
    if _apply_disparity_func_neg is None or _apply_disparity_func_neg is None:
        _build_cuda_kernels()

    # tensor check
    assert img.is_contiguous() and disp.is_contiguous()
    assert img.device.type == disp.device.type == "cuda"
    assert disp.dtype == torch.int

    if torch.all(disp >= 0):
        warp_fn = _apply_disparity_func_pos
    else:
        assert torch.all(disp <= 0)
        warp_fn = _apply_disparity_func_neg

    # send data to cuda ops
    stream = namedtuple("Stream", ["ptr"])
    s = stream(ptr=torch.cuda.current_stream().cuda_stream)

    ret = torch.zeros_like(img)

    b, c, h, w = img.shape
    total_l = b * c * h
    grid_size = total_l // 512 + 1
    warp_fn(
        stream=s,
        grid=(grid_size, 1, 1),
        block=(512, 1, 1),
        args=[ret.data_ptr(), img.data_ptr(), disp.data_ptr(), h, w, c, total_l],
    )

    return ret


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms

    from datasets.messytable_p2 import MessytableDataset

    cdataset = MessytableDataset(
        "./dataset_local_v9/training_lists/all_train.txt", isTest=True
    )
    item = cdataset.__getitem__(0)

    disp_r = item["img_disp_r"].type(torch.int32).unsqueeze(0).cuda()

    res = apply_disparity_cu(disp_r, disp_r)

    tensor_to_img = transforms.ToPILImage()
    plt.imshow(tensor_to_img(disp_r.squeeze(0)).convert("RGB"))
    plt.savefig("dispR.png")
    plt.imshow(tensor_to_img(res.squeeze(0)).convert("RGB"))
    plt.savefig("dispL_from_R.png")
