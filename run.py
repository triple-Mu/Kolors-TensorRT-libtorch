import torch
import py_kolors
from PIL import Image
from kolors.models.tokenization_chatglm import ChatGLMTokenizer

tokenizer = ChatGLMTokenizer.from_pretrained('Kolors/text_encoder')
engine = py_kolors.Pipeline(
    'text_encoder.plan',
    'unet.plan',
    'vae.plan'
)

prompt = '一张瓢虫的照片，微距，变焦，高质量，电影，拿着一个牌子，写着“可图”'

text_inputs = tokenizer(
    prompt,
    padding="max_length",
    max_length=256,
    truncation=True,
)
input_ids = text_inputs['input_ids']
img = engine.generate(
    input_ids,
    None,
    50
)

Image.fromarray(img).save('tmp.jpg')
