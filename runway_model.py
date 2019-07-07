# MIT License

# Copyright (c) 2019 Runway AI, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import runway
from runway.data_types import number, image, boolean

from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

import net
from function import adaptive_instance_normalization
from function import coral


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_transform(size, crop):
    transform_list = []
    if size != 0:
      transform_list.append(transforms.Resize(size))
    if crop:
      transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
      _, C, H, W = content_f.size()
      feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
      base_feat = adaptive_instance_normalization(content_f, style_f)
      for i, w in enumerate(interpolation_weights):
        feat = feat + w * base_feat[i:i + 1]
      content_f = content_f[0:1]
    else:
      feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


@runway.setup
def setup():
    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load('models/decoder.pth'))
    vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    content_tf = test_transform(0, False)
    style_tf = test_transform(0, False)
    return {
      'vgg': vgg,
      'decoder': decoder,
      'content_tf': content_tf,
      'style_tf': style_tf,
    }


@runway.command(name='generate',
                inputs={
                  'content_image': image(description='Content Image'),
                  'style_image': image(description='Style Image'),
                  'preserve_color': boolean(description='Preserve content image color'),
                  'alpha': number(description='Controls the degree of stylization',
                                                 min=0, max=1, step=0.01, default=1)
                },
                outputs={ 'image': image(description='Output image') })
def generate(model, args):
    content_image = args['content_image'].convert('RGB')
    style_image = args['style_image'].convert('RGB')
    preserve_color = args['preserve_color']
    alpha = args['alpha']
    print('[GENERATE] Ran with preserve_color "{}". alpha "{}"'.format(preserve_color, alpha))

    vgg = model['vgg']
    decoder = model['decoder']
    content_tf = model['content_tf']
    style_tf = model['style_tf']

    content = content_tf(content_image)
    style = style_tf(style_image)
    if preserve_color:
      style = coral(style, content)
    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)
    with torch.no_grad():
      output = style_transfer(vgg, decoder, content, style, alpha)
    ndarr = output[0].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()

    return {
        'image': Image.fromarray(ndarr)
    }


if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=8888)
