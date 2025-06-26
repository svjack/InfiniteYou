'''
https://civitai.com/models/693319?modelVersionId=796769

import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.load_lora_weights("Chinese_idol_Flex2_lora/my_first_flex2_lora_v1_000004500.safetensors", 
                      adapter_name = "young"
                      ) 
#pipe.load_lora_weights("xblowjob-3.0-000075.safetensors")
pipe.load_lora_weights("Twinks_Legs_Up_V7.safetensors",
                      adapter_name = "leg"
                      )
#pipe.enable_sequential_cpu_offload()

pipe.set_adapters(["young", "leg"], adapter_weights=[2, 1])

pipe.enable_model_cpu_offload()

prompt = "A young Asian male singer with fair skin, pink nipples, cute boy, legs up showing his ass hole, "


prompt = "A young Asian male singer with fair skin, cute boy, legs up showing his ass hole show his pink nipples"

image = pipe(prompt,
             num_inference_steps=50,
             guidance_scale=3.5,
            ).images[0]


python test_xiang_as_0.py --prompt "A young Asian male singer with fair skin, pink nipples, cute boy, legs up showing his ass hole, " \
 --id_image "xiang_rmbg.png" --out_results_dir Xiang_InfiniteYou_bottom_NSFW --quantize_8bit --num_steps 25

'''

# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import torch
from PIL import Image

from pipelines.pipeline_infu_flux import InfUFluxPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_image', default='./assets/examples/man.jpg', help="""input ID image""")
    parser.add_argument('--control_image', default=None, help="""control image [optional]""")
    parser.add_argument('--out_results_dir', default='./results', help="""output folder""")
    parser.add_argument('--prompt', default='A man, portrait, cinematic')
    parser.add_argument('--base_model_path', default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--model_dir', default='ByteDance/InfiniteYou')
    parser.add_argument('--infu_flux_version', default='v1.0', help="""InfiniteYou-FLUX version: currently only v1.0""")
    parser.add_argument('--model_version', default='aes_stage2', help="""model version: aes_stage2 | sim_stage1""")
    parser.add_argument('--cuda_device', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int, help="""seed (0 for random)""")
    parser.add_argument('--guidance_scale', default=3.5, type=float)
    parser.add_argument('--num_steps', default=30, type=int)
    parser.add_argument('--infusenet_conditioning_scale', default=1.0, type=float)
    parser.add_argument('--infusenet_guidance_start', default=0.0, type=float)
    parser.add_argument('--infusenet_guidance_end', default=1.0, type=float)
    # The LoRA options below are entirely optional. Here we provide two examples to facilitate users to try, but they are NOT used in our paper.
    parser.add_argument('--enable_realism_lora', action='store_true')
    parser.add_argument('--enable_anti_blur_lora', action='store_true')
    # Memory reduction options
    parser.add_argument('--quantize_8bit', action='store_true')
    parser.add_argument('--cpu_offload', action='store_true')
    args = parser.parse_args()

    # Check arguments
    assert args.infu_flux_version == 'v1.0', 'Currently only supports InfiniteYou-FLUX v1.0'
    assert args.model_version in ['aes_stage2', 'sim_stage1'], 'Currently only supports model versions: aes_stage2 | sim_stage1'

    # Set cuda device
    torch.cuda.set_device(args.cuda_device)

    # Load pipeline
    infu_model_path = os.path.join(args.model_dir, f'infu_flux_{args.infu_flux_version}', args.model_version)
    insightface_root_path = os.path.join(args.model_dir, 'supports', 'insightface')
    pipe = InfUFluxPipeline(
        base_model_path=args.base_model_path,
        infu_model_path=infu_model_path,
        insightface_root_path=insightface_root_path,
        infu_flux_version=args.infu_flux_version,
        model_version=args.model_version,
        quantize_8bit=args.quantize_8bit,
        cpu_offload=args.cpu_offload,
    )
    # Load LoRAs (optional)
    lora_dir = os.path.join(args.model_dir, 'supports', 'optional_loras')
    if not os.path.exists(lora_dir): lora_dir = './models/InfiniteYou/supports/optional_loras'
    loras = []
    loras = [
        [
            "../Twinks_Legs_Up_V7.safetensors",
            "leg", 1.0
        ],
        [
            "../Chinese_idol_Flex2_lora/my_first_flex2_lora_v1_000004500.safetensors",
            "young", 1.0
        ],
    ]
    if args.enable_realism_lora:
        loras.append([os.path.join(lora_dir, 'flux_realism_lora.safetensors'), 'realism', 1.0])
    if args.enable_anti_blur_lora:
        loras.append([os.path.join(lora_dir, 'flux_anti_blur_lora.safetensors'), 'anti_blur', 1.0])
    pipe.load_loras(loras)
    
    # Perform inference
    if args.seed == 0:
        args.seed = torch.seed() & 0xFFFFFFFF

    for i in range(128):
        args.seed = torch.seed() & 0xFFFFFFFF
        
        image = pipe(
            id_image=Image.open(args.id_image).convert('RGB'),
            prompt=args.prompt,
            control_image=Image.open(args.control_image).convert('RGB') if args.control_image is not None else None,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            infusenet_conditioning_scale=args.infusenet_conditioning_scale,
            infusenet_guidance_start=args.infusenet_guidance_start,
            infusenet_guidance_end=args.infusenet_guidance_end,
            cpu_offload=args.cpu_offload,
        )
        
        # Save results
        os.makedirs(args.out_results_dir, exist_ok=True)
        index = len(os.listdir(args.out_results_dir))
        id_name = os.path.splitext(os.path.basename(args.id_image))[0]
        prompt_name = args.prompt[:150] + '*' if len(args.prompt) > 150 else args.prompt
        prompt_name = prompt_name.replace('/', '|')
        out_name = f'{index:05d}_{id_name}_{prompt_name}_seed{args.seed}.png'
        out_result_path = os.path.join(args.out_results_dir, out_name)
        image.save(out_result_path)


if __name__ == "__main__":
    main()
