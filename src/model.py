from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义类 Conv+BN+ReLU
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

# 定义类 下采样+Conv+BN+ReLU，继承自上面的类ConvBNReLU
# flag--是否进行下采样，默认为True
class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果self.down_flag为true，则利用max_pool2d进行下采样（2倍）
        if self.down_flag:
            # ceil_mode指明，当剩余的像素不足滤波器大小，是否仍对这些像素进行运算
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.relu(self.bn(self.conv(x)))

# 定义类 上采样+Concat拼接+Conv+BN+ReLU，继承自上面的类ConvBNReLU
# flag--是否进行上采样，默认为True
class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # 如果self.up_flag为true，则利用双线性插值interpolate进行上采样
        # 将上一层是输出x1经过上采样到要拼接的Encoder层的输出x2的大小
        if self.up_flag:
            # mode选择为bilinear时，align_corners一般设置为False
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        # 上采样后依次进行拼接，conv，bn，relu
        return self.relu(self.bn(self.conv(torch.cat([x1, x2], dim=1))))

# 定义 Residual U-Block（RSU模块）
class RSU(nn.Module):
    # height--RSU模块的深度
    # in_ch--输入channel，对应于结构图中的C_in
    # mid_ch--对应于结构图中的C_mid
    # out_ch--输出channel，对应于结构图中的C_out
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()

        # 断言RSU模块的深度height>=2
        assert height >= 2
        # 定义RSU模块中的第一个Conv+BN+ReLU，输入channel为C_in，输出channel为C_out，见结构图
        self.conv_in = ConvBNReLU(in_ch, out_ch)

        # encode_list--结构图中左边除了第一个的剩下6个加底下一个
        # 由于第二个没有下采样，所以单独写出来，后边经过了下采样的5个通过for循环添加进去
        encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]
        # decode_list--结构图中右边的6个
        # 由于右边最底下的第一个没有经过上采样，所以单独写出来，后边经过了上采样的5个通过for循环添加进去
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]
        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
            # 由于最后一个的输出channel和前面的不一样，为C_out，所以要进行一下判断
            # 注意输入channel为拼接后的channel，所以为mid_ch*2
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))
        # 最底下的一个也没有经过下采样，采用的卷积是膨胀卷积，膨胀系数为2，单独添加
        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))
        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 经过RSU模块中的第一个Conv+BN+ReLU，得到输出x_in
        x_in = self.conv_in(x)

        x = x_in
        # 保存encoder层的输出，后边decoder时要进行拼接
        encode_outputs = []
        for m in self.encode_modules:
            # 将经过第一个Conv+BN+ReLU后的输出x依次经过self.encode_modules
            x = m(x)
            # 保存每一层encoder的输出
            encode_outputs.append(x)

        # 弹出self.encode_modules中的最后一个元素，即得到最底下第一个decoder层的输入
        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            # 将上一层的输出x和encoder层的输出x2经过self.decode_modules
            x = m(x, x2)
        # 将x和x_in进行拼接，得到最终的输出
        return x + x_in

# 定义RSU4F模块
# 详细和上面的RSU类似，看结构图
class RSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=8)])

        self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch * 2, out_ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in


class U2Net(nn.Module):
    # cfg--初始化模型结构参数
    # out_ch--因为只有前景和背景，所以输出通道个数为1
    # 输出的概率向0靠近，则为背景的概率较大；输出的概率向1靠近，则为前景的概率较大
    def __init__(self, cfg: dict, out_ch: int = 1):
        super().__init__()
        assert "encode" in cfg
        assert "decode" in cfg
        # 统计encoder模块的数目，为6
        self.encode_num = len(cfg["encode"])

        # 存储实例化的每一个encoder模块
        encode_list = []
        # 收集的De_1、De_2、De_3、De_4、De_5以及En_6的输出会经过一个3x3的卷积层，收集经过3x3的卷积层后的输出
        side_list = []
        # for循环遍历encode的每一个参数
        for c in cfg["encode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) == 6
            # c[4]--RSU4F，判断当前层使用的是RSU模块，还是RSU4F模块
            # 如果c[4]为False，则当前使用的是RSU模块，此时需要传入的参数有height,in_ch,mid_ch,out_ch，即*c[:4]
            # 如果c[4]为True，则当前使用的是RSU4F模块，此时需要传入的参数有in_ch,mid_ch,out_ch，即*c[1:4]
            encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))

            # c[4]--side
            # 判断当前是否收集经过3x3的卷积层后的输出，对于encoder，只有最后一个En6会收集
            if c[5] is True:
                # 将En6的输出经过一个3x3的卷积层，输入channel为En6的输出channel，即c[3]
                # 输出channel为out_ch=1
                # 将输出保存到side_list中
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        # 将encode_list传入nn.ModuleList中，构建好self.encode_modules
        self.encode_modules = nn.ModuleList(encode_list)

        # decoder部分的构建和上面的encoder差不多
        decode_list = []
        for c in cfg["decode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) == 6
            decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.decode_modules = nn.ModuleList(decode_list)
        # 将side_list传入nn.ModuleList中，构建好self.side_modules
        self.side_modules = nn.ModuleList(side_list)
        # 最后一个卷积层，融合所有通道的输出，输入channel为self.encode_num*out_ch=6*1，输出channel为out_ch=1
        self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        # 获取输入图片的高宽
        _, _, h, w = x.shape

        # collect encode outputs
        encode_outputs = []
        for i, m in enumerate(self.encode_modules):
            x = m(x)
            encode_outputs.append(x)
            # 最后一个encoder模块不会经过下采样，前面5个，每经过一个encoder就会进行一次下采样
            if i != self.encode_num - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        # collect decode outputs
        # 弹出encoder模块的最后一个输出，即第一个encoder模块的输入
        x = encode_outputs.pop()
        decode_outputs = [x]
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
            x = m(torch.concat([x, x2], dim=1))
            # 收集输出，顺序为De1，De2，De3，De4，De5，En6
            decode_outputs.insert(0, x)

        # collect side outputs
        # 收集经过3x3卷积层后的输出
        side_outputs = []
        # for循环遍历self.side_modules里的每一个模块（3x3卷积层，En6，De5，De4，De3，De2，De1）
        for m in self.side_modules:
            x = decode_outputs.pop()
            # m(x)--经过3x3卷积层
            # 再通过双线性插值直接还原为输入图像的高宽
            x = F.interpolate(m(x), size=[h, w], mode='bilinear', align_corners=False)
            # 得到结构图中的Sup1、Sup2、Sup3、Sup4、Sup5和Sup6
            side_outputs.insert(0, x)

        # 经过concat拼接，在通过最后的1x1卷积层得到最后的输出
        x = self.out_conv(torch.concat(side_outputs, dim=1))

        if self.training:
            # do not use torch.sigmoid for amp safe
            # 训练模式需要返回最终输出的x和sup1-sup6，后面计算损失会用到
            return [x] + side_outputs
        else:
            # 非训练模式返回经过sigmoid激活函数后的x，此时每一个像素都是一个概率值（0-1）
            return torch.sigmoid(x)


def u2net_full(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        # RSU4F--当前层是不是使用的RSU4F结构，通过结果图可以知道，只有En5，En6和De5使用的是RSU4F结构
        # side--是否收集某些层的输出，由结构图可知会收集De_1、De_2、De_3、De_4、De_5以及En_6的输出进行融合并得到最终预测概率图
        "encode": [[7, 3, 32, 64, False, False],      # En1
                   [6, 64, 32, 128, False, False],    # En2
                   [5, 128, 64, 256, False, False],   # En3
                   [4, 256, 128, 512, False, False],  # En4
                   [4, 512, 256, 512, True, False],   # En5
                   [4, 512, 256, 512, True, True]],   # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 1024, 256, 512, True, True],   # De5
                   [4, 1024, 128, 256, False, True],  # De4
                   [5, 512, 64, 128, False, True],    # De3
                   [6, 256, 32, 64, False, True],     # De2
                   [7, 128, 16, 64, False, True]]     # De1
    }

    return U2Net(cfg, out_ch)


def u2net_lite(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 16, 64, False, False],  # En1
                   [6, 64, 16, 64, False, False],  # En2
                   [5, 64, 16, 64, False, False],  # En3
                   [4, 64, 16, 64, False, False],  # En4
                   [4, 64, 16, 64, True, False],  # En5
                   [4, 64, 16, 64, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 128, 16, 64, True, True],  # De5
                   [4, 128, 16, 64, False, True],  # De4
                   [5, 128, 16, 64, False, True],  # De3
                   [6, 128, 16, 64, False, True],  # De2
                   [7, 128, 16, 64, False, True]]  # De1
    }

    return U2Net(cfg, out_ch)


def convert_onnx(m, save_path):
    m.eval()
    x = torch.rand(1, 3, 288, 288, requires_grad=True)

    # export the model
    torch.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      save_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,
                      opset_version=11)


if __name__ == '__main__':
    # n_m = RSU(height=7, in_ch=3, mid_ch=12, out_ch=3)
    # convert_onnx(n_m, "RSU7.onnx")
    #
    # n_m = RSU4F(in_ch=3, mid_ch=12, out_ch=3)
    # convert_onnx(n_m, "RSU4F.onnx")

    u2net = u2net_full()
    convert_onnx(u2net, "u2net_full.onnx")
