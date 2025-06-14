'''
python test_dataset.py --id_image "王力宏.webp" --out_results_dir ./InfiniteYou_Chinese_idol_Wang_Leehom_Captioned --quantize_8bit
'''

import argparse
import os
import torch
from PIL import Image
from pipelines.pipeline_infu_flux import InfUFluxPipeline
from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_image', default='王力宏.webp', help="input ID image")
    parser.add_argument('--control_image', default=None, help="control image [optional]")
    parser.add_argument('--out_results_dir', default='./results', help="output folder")
    parser.add_argument('--base_model_path', default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--model_dir', default='ByteDance/InfiniteYou')
    parser.add_argument('--infu_flux_version', default='v1.0', help="InfiniteYou-FLUX version: currently only v1.0")
    parser.add_argument('--model_version', default='aes_stage2', help="model version: aes_stage2 | sim_stage1")
    parser.add_argument('--cuda_device', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int, help="seed (0 for random)")
    parser.add_argument('--guidance_scale', default=3.5, type=float)
    parser.add_argument('--num_steps', default=30, type=int)
    parser.add_argument('--infusenet_conditioning_scale', default=1.0, type=float)
    parser.add_argument('--infusenet_guidance_start', default=0.0, type=float)
    parser.add_argument('--infusenet_guidance_end', default=1.0, type=float)
    parser.add_argument('--enable_realism_lora', action='store_true')
    parser.add_argument('--enable_anti_blur_lora', action='store_true')
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
    if not os.path.exists(lora_dir):
        lora_dir = './models/InfiniteYou/supports/optional_loras'
    loras = [
        [
            "../ai-toolkit/Chinese_idol_Flex2_lora/my_first_flex2_lora_v1_000004500.safetensors",
            "idol", 0.3
        ]
    ]
    if args.enable_realism_lora:
        loras.append([os.path.join(lora_dir, 'flux_realism_lora.safetensors'), 'realism', 1.0])
    if args.enable_anti_blur_lora:
        loras.append([os.path.join(lora_dir, 'flux_anti_blur_lora.safetensors'), 'anti_blur', 1.0])
    pipe.load_loras(loras)

    # Load prompts from dataset
    dataset = load_dataset("svjack/Ding_Chengxin_Images_Captioned", split="train")
    prompts = dataset["prompt"]

    # Prepare output directory
    os.makedirs(args.out_results_dir, exist_ok=True)
    id_name = os.path.splitext(os.path.basename(args.id_image))[0]

    # Load ID image once
    id_image = Image.open(args.id_image).convert('RGB')

    # Loop over prompts with tqdm
    for idx, prompt in enumerate(tqdm(prompts, desc="Generating images")):
        if args.seed == 0:
            seed = torch.seed() & 0xFFFFFFFF
        else:
            seed = args.seed

        # Generate image
        image = pipe(
            id_image=id_image,
            prompt=prompt,
            control_image=Image.open(args.control_image).convert('RGB') if args.control_image is not None else None,
            seed=seed,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            infusenet_conditioning_scale=args.infusenet_conditioning_scale,
            infusenet_guidance_start=args.infusenet_guidance_start,
            infusenet_guidance_end=args.infusenet_guidance_end,
            cpu_offload=args.cpu_offload,
        )

        # Format filename
        prompt_name = prompt[:150] + '*' if len(prompt) > 150 else prompt
        prompt_name = prompt_name.replace('/', '|')
        out_name = f"{idx:05d}_{id_name}_{prompt_name}_seed{seed}.png"
        out_result_path = os.path.join(args.out_results_dir, out_name)
        text_path = os.path.join(args.out_results_dir, f"{idx:05d}_{id_name}_{prompt_name}_seed{seed}.txt")

        # Save image and prompt
        image.save(out_result_path)
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(prompt)


if __name__ == "__main__":
    main()
