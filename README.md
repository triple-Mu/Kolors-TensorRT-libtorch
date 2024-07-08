# Kolors-TensorRT-libtorch

用`TensorRT`和`libtorch`简单实现了`Kolors`模型的`pipeline`推理。

## 准备

- 安装`TensorRT`, `TensorRT10`的api相较于`TensorRT8`以下版本变化较大, 目前本仓库做了`TensorRT10`的适配, 建议用`TensorRT10`以上的版本。
- 从[huggingface](`https://huggingface.co/Kwai-Kolors/Kolors/tree/main`)下载模型。
- 安装`pytorch`, `onnx`等依赖。

## 导出3个onnx模型用于pipeline

修改[export_onnx.py](export_onnx.py)中的路径相关信息。
执行:

```shell
python export_onnx.py
```

你会得到`text_encoder`, `unet`, `vae`三个onnx模型。
你可以用[onnxsim](`https://github.com/daquexian/onnx-simplifier`)将它们简化。
[pr-336](https://github.com/daquexian/onnx-simplifier/pull/336)适配了超过2GB的onnx简化报错，可以尝试安装最新的onnxsim。

执行:

```shell
onnxsim text_encoder.onnx text_encoder-sim.onnx --save-as-external-data
onnxsim unet.onnx unet-sim.onnx --save-as-external-data
onnxsim vae.onnx vae-sim.onnx
```

onnx很大的情况下, 简化的耗时也很长。

## onnx转换到tensorrt

这里我用了trtexec转化, 比较省事。
目前测试`text_encoder`部分fp16掉点情况比较大，建议回退到fp32。

```shell
trtexec --onnx=text_encoder-sim.onnx --saveEngine=text_encoder.plan --noTF32
trtexec --onnx=unet-sim.onnx --saveEngine=unet.plan --fp16
trtexec --onnx=vae-sim.onnx --saveEngine=vae.plan --fp16
```

tensorrt转换的过程也很慢。

## 编译安装python包

执行:

```shell
python setup.py install
```

包名是: `py_kolors`

## 推理一个文生图

修改[run.py](run.py)中的3个模型路径, 修改推理步数, 默认50比较慢.

执行:

```shell
python run.py
```

生成的图片会保存为`tmp.jpg`。


