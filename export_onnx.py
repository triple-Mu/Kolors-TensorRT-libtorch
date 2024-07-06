import torch
import torch.nn as nn
from torch import Tensor
from contextlib import contextmanager
from kolors.models.modeling_chatglm import ChatGLMModel
from diffusers import UNet2DConditionModel, AutoencoderKL


# a useful warp for exporting onnx
@contextmanager
def onnx_export():
    import torch
    import onnx
    from tempfile import TemporaryDirectory
    _export = torch.onnx.export

    def export(
            model,
            args,
            f,
            **kwargs
    ):
        with TemporaryDirectory() as d:
            onnx_file = f'{d}/{f}'
            print(onnx_file)
            _export(model, args, onnx_file, **kwargs)
            onnx_model = onnx.load(onnx_file)
        onnx.save(onnx_model,
                  f,
                  save_as_external_data=True,
                  all_tensors_to_one_file=True,
                  location=f + '.data',
                  convert_attribute=True)

    torch.onnx.export = export
    yield

    torch.onnx.export = _export


@torch.no_grad()
def export():
    device = torch.device('cuda:0')
    dtype = torch.float16
    ckpt_dir = 'Kwai-Kolors/Kolors'

    text_encoder = ChatGLMModel.from_pretrained(
        f'{ckpt_dir}/text_encoder',
        torch_dtype=torch.float32)
    vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None)
    unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None)

    # export text encoder
    class TextEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = text_encoder.to(dtype)

        def forward(
                self,
                input_ids: Tensor,  # [2, 256]
                attention_mask: Tensor,  # [2, 256]
                position_ids: Tensor,  # [2, 256]
        ):
            output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True)
            prompt_embeds = output.hidden_states[-2].permute(1, 0, 2)
            text_proj = output.hidden_states[-1][-1, :, :]
            return prompt_embeds, text_proj

    text_encoder = TextEncoder()
    text_encoder.eval()

    input_ids = torch.randint(0, 32000, (2, 256), dtype=torch.int32, device=device)
    attention_mask = torch.ones((2, 256), dtype=torch.int32, device=device)
    position_ids = torch.arange(0, 256, dtype=torch.int32, device=device).unsqueeze(0)

    with onnx_export():
        torch.onnx.export(
            text_encoder,
            (input_ids, attention_mask, position_ids),
            'text_encoder.onnx',
            opset_version=17,
            input_names=['input_ids', 'attention_mask', 'position_ids'],
            output_names=['prompt_embeds', 'text_proj'],
        )

    class UNET(nn.Module):
        def __init__(self):
            super().__init__()
            self.unet = unet.to(dtype)

        def forward(
                self,
                latent_model_input: Tensor,  # [2, 4, 128, 128]
                timestep: Tensor,  # []
                encoder_hidden_states: Tensor,  # [2, 256, 4096]
                text_embeds: Tensor,  # [2, 4096]
        ):
            time_ids = torch.tensor(
                [[1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0], [1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]],
                dtype=dtype, device=device)

            added_cond_kwargs = {
                'text_embeds': text_embeds,
                'time_ids': time_ids
            }
            noise_pred = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            return noise_pred

    unet = UNET()
    unet.eval()

    latent_model_input = torch.randn((2, 4, 128, 128), dtype=dtype, device=device)
    timestep = torch.tensor(1000, dtype=dtype, device=device)
    encoder_hidden_states = torch.randn((2, 256, 4096), dtype=dtype, device=device)
    text_embeds = torch.randn((2, 4096), dtype=dtype, device=device)

    with onnx_export():
        torch.onnx.export(
            unet,
            (latent_model_input, timestep, encoder_hidden_states, text_embeds),
            'unet.onnx',
            opset_version=17,
            input_names=['latent_model_input', 'timestep', 'encoder_hidden_states', 'text_embeds'],
            output_names=['noise_pred'],
        )

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = vae.to(dtype)

        def forward(
                self,
                latents: Tensor  # [1, 4, 128, 128]
        ):
            latents = latents / self.vae.config.scaling_factor
            images = self.vae.decode(latents, return_dict=False)[0]
            images = images / 2 + 0.5
            images = images.clamp(0, 1)
            return images

    vae = VAE()
    vae.eval()

    latents = torch.randn((1, 4, 128, 128), dtype=dtype, device=device)
    with onnx_export():
        torch.onnx.export(
            vae,
            (latents),
            'vae.onnx',
            opset_version=17,
            input_names=['latents'],
            output_names=['images'],
        )


if __name__ == "__main__":
    export()
