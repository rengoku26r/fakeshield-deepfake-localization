import argparse
import json
from pathlib import Path

import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import process_images, tokenizer_image_token
from transformers import TextStreamer

from llava.serve.cli import (
    DTE_FDM_init,
    load_image,
)


PROJECT_ROOT = Path(
    "/workspace/fakeshield-deepfake-localization"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "webwarriors/baseline/dataset/image"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "webwarriors/baseline/results/dte_fdm"
)


MODEL_PATH = "./weight/fakeshield-v1-22b/DTE-FDM"
DTG_PATH = "./weight/fakeshield-v1-22b/DTG.pth"


def run_single_image(
    image_path,
    output_path,
    tokenizer,
    model,
    image_processor,
    DTG,
    model_name,
    args,
):

    conv = conv_templates[args.conv_mode].copy()

    if "mpt" in model_name.lower():
        roles = ("user", "assistant")
    else:
        roles = conv.roles

    image = load_image(str(image_path))
    label = DTG.predict(str(image_path))

    image_size = image.size

    image_tensor = process_images(
        [image],
        image_processor,
        model.config,
    )

    if type(image_tensor) is list:
        image_tensor = [
            img.to(model.device, dtype=torch.float16)
            for img in image_tensor
        ]
    else:
        image_tensor = image_tensor.to(
            model.device,
            dtype=torch.float16,
        )

    inp = (
        "Was this photo taken directly from the camera without any processing? "
        "Has it been tampered with by any artificial photo modification "
        "techniques such as ps? Please zoom in on any details in the image, "
        "paying special attention to the edges of the objects, capturing "
        "some unnatural edges and perspective relationships, some incorrect "
        "semantics, unnatural lighting and darkness etc."
    )

    if label == 0:
        inp = (
            "This is a picture that is suspected to have been tampered "
            "with by AIGC inpainting. " + inp
        )
    elif label == 1:
        inp = (
            "This is a picture that is suspected to have been tampered "
            "with by DeepFake. " + inp
        )
    elif label == 2:
        inp = (
            "This is a picture that is suspected to have been tampered "
            "with by Photoshop. " + inp
        )

    if model.config.mm_use_im_start_end:
        inp = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + inp
        )
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + inp

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    stop_str = (
        conv.sep
        if conv.sep_style != SeparatorStyle.TWO
        else conv.sep2
    )

    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    with torch.inference_mode():

        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            use_cache=True,
        )

    outputs = tokenizer.decode(output_ids[0]).strip()

    outputs = outputs.replace("<s>", "").replace("</s>", "")

    result = {
        "image": str(image_path),
        "DTG_label": label,
        "outputs": outputs,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-dir",
        default=str(IMAGE_DIR),
    )

    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
    )

    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
    )

    parser.add_argument(
        "--model-base",
        default=None,
    )

    parser.add_argument(
        "--DTG-path",
        default=DTG_PATH,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--load-4bit",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--load-8bit",
        action="store_true",
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--conv-mode",
        default=None,
    )

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file()
    )

    print("==============================================")
    print("DTE-FDM Batch")
    print("==============================================")
    print(f"Images found : {len(images)}")
    print(f"Output dir   : {output_dir}")
    print("==============================================")

    # --------------------------------------------------------
    # LOAD MODEL ONLY ONCE
    # --------------------------------------------------------

    print("======== Loading DTE-FDM model ========")

    (
        tokenizer,
        model,
        image_processor,
        context_len,
        DTG,
        model_name,
    ) = DTE_FDM_init(args)

    print("======== DTE-FDM model loaded ========")

    # --------------------------------------------------------
    # PROCESS IMAGES
    # --------------------------------------------------------

    for index, image_path in enumerate(images, start=1):

        output_path = output_dir / f"{image_path.stem}.json"

        if output_path.exists():
            print(
                f"[{index}/{len(images)}] "
                f"SKIP {image_path.name} "
                f"(already processed)"
            )
            continue

        print(
            f"[{index}/{len(images)}] "
            f"Processing {image_path.name}"
        )

        run_single_image(
            image_path,
            output_path,
            tokenizer,
            model,
            image_processor,
            DTG,
            model_name,
            args,
        )

        print(
            f"Saved: {output_path.name}"
        )

    print("==============================================")
    print("DTE-FDM batch complete")
    print("==============================================")


if __name__ == "__main__":
    main()
