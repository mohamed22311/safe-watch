import cv2
import os
import math
from pathlib import Path

def extract_every_n_seconds(video_path, out_dir, interval_s=2):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    vid = cv2.VideoCapture(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps
    count = 0
    sec = 0.0
    while sec < duration:
        frame_id = int(sec * fps)
        vid.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = vid.read()
        if not ret:
            break
        out_path = os.path.join(out_dir, f"frame_{count:04d}.jpg")
        cv2.imwrite(out_path, frame)
        count += 1
        sec += interval_s
    vid.release()
    print(f"Saved {count} frames to {out_dir}")

video_path = "./Sample1.mp4"
out_dir = "./frames"
extract_every_n_seconds(video_path, out_dir, interval_s=2)

from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
import os, glob

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,
             device_map="auto",
             torch_dtype="auto",
             trust_remote_code=True)
model.eval()

from qwen_vl_utils import process_vision_info

def vlm_generate(prompt_text: str, image_path: str | None = None, max_new_tokens: int = 256):
    msg_content = []
    if image_path:
        msg_content.append({"type": "image", "image": image_path})
    msg_content.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": msg_content}]

    # 1) create the textual prompt with image placeholders (do NOT tokenize here)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 2) turn the messages into image/video inputs (handles resizing/tokenization etc)
    image_inputs, video_inputs = process_vision_info(messages)

    # 3) prepare inputs for model (this will produce input_ids that include image placeholders)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    # Move tensors individually to device (do NOT call .to() on the dict)
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Trim the input prefix from outputs (common pattern for chat + generation)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # decode trimmed sequences
    out_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return out_texts[0].strip()


image_path = './image.jpg'
prompt_text = ("You are an AI security monitoring assistant responsible for automatic violence detection "
    "and detailed scene reporting for on-site security guards. "
    "Your mission is to help human operators rapidly assess potential threats captured in surveillance images. "
    "Carefully examine the provided image and strictly follow these instructions:\n\n"

    "ROLE & BEHAVIOR:\n"
    "- Act as a professional safety and security analyst.\n"
    "- Remain objective, neutral, and factual—avoid speculation beyond what is visually evident.\n"
    "- Focus on detecting any signs of violence, weapons, fights, or blood.\n"
    "- Provide concise, structured, and actionable information suitable for security incident reports.\n\n"

    "ANALYSIS STEPS:\n"
    "1. Observe the entire image for people, objects, and ongoing activities.\n"
    "2. Identify any indicators of violence such as physical aggression, fighting, or assault.\n"
    "3. Detect the presence of weapons (e.g., knives, guns, blunt objects) if visible.\n"
    "4. Look for signs of injuries, visible blood, or damage related to violent acts.\n"
    "5. If no violence is detected, clearly state this but still briefly describe the general scene.\n\n"

    "REQUIRED OUTPUT FORMAT:\n"
    "Provide your findings in the following exact structure (use clear, concise sentences):\n"
    "- Violence Detected: Yes/No\n"
    "- Violence Severity Score: [0–100] (0 = no violence, 100 = extreme violence)\n"
    "- Weapons Observed: [List of weapons or 'None']\n"
    "- Signs of Fights or Physical Aggression: [Describe or 'None']\n"
    "- Signs of Blood or Injuries: [Describe or 'None']\n"
    "- Scene Description: [Brief factual summary of the overall scene]\n\n"

    "IMPORTANT:\n"
    "- Only describe what is clearly visible.\n"
    "- Do not include personal opinions or guesses.\n"
    "- Follow the output structure exactly so it can be automatically parsed.")
output = vlm_generate(
    prompt_text=prompt_text,
    image_path=image_path
)

image_path = './image2.jpg'
prompt_text = ("You are an AI security monitoring assistant responsible for automatic violence detection "
    "and detailed scene reporting for on-site security guards. "
    "Your mission is to help human operators rapidly assess potential threats captured in surveillance images. "
    "Carefully examine the provided image and strictly follow these instructions:\n\n"

    "ROLE & BEHAVIOR:\n"
    "- Act as a professional safety and security analyst.\n"
    "- Remain objective, neutral, and factual—avoid speculation beyond what is visually evident.\n"
    "- Focus on detecting any signs of violence, weapons, fights, or blood.\n"
    "- Provide concise, structured, and actionable information suitable for security incident reports.\n\n"

    "ANALYSIS STEPS:\n"
    "1. Observe the entire image for people, objects, and ongoing activities.\n"
    "2. Identify any indicators of violence such as physical aggression, fighting, or assault.\n"
    "3. Detect the presence of weapons (e.g., knives, guns, blunt objects) if visible.\n"
    "4. Look for signs of injuries, visible blood, or damage related to violent acts.\n"
    "5. If no violence is detected, clearly state this but still briefly describe the general scene.\n\n"

    "REQUIRED OUTPUT FORMAT:\n"
    "Provide your findings in the following exact structure (use clear, concise sentences):\n"
    "- Violence Detected: Yes/No\n"
    "- Violence Severity Score: [0–100] (0 = no violence, 100 = extreme violence)\n"
    "- Weapons Observed: [List of weapons or 'None']\n"
    "- Signs of Fights or Physical Aggression: [Describe or 'None']\n"
    "- Signs of Blood or Injuries: [Describe or 'None']\n"
    "- Scene Description: [Brief factual summary of the overall scene]\n\n"

    "IMPORTANT:\n"
    "- Only describe what is clearly visible.\n"
    "- Do not include personal opinions or guesses.\n"
    "- Follow the output structure exactly so it can be automatically parsed.")
output = vlm_generate(
    prompt_text=prompt_text,
    image_path=image_path
)

output

def vlm_generate_fewshot(
    query_prompt: str,
    query_image: str | None = None,
    examples: list[dict] | None = None,
    max_new_tokens: int = 256,
):
    """
    Few-shot multimodal generation with Qwen.

    Args:
        query_prompt: The text instruction for the target image.
        query_image: Path to the image you want analyzed.
        examples: A list of dicts with keys:
            {
              "image": str | None,  # path to example image
              "prompt": str,        # text instruction
              "output": str         # expected assistant output
            }
        max_new_tokens: Maximum tokens to generate.

    Returns:
        str: Generated output text for the query.
    """

    messages = []

    # Add few-shot examples
    if examples:
        for ex in examples:
            # User input with example image + text
            user_content = []
            if ex.get("image"):
                user_content.append({"type": "image", "image": ex["image"]})
            user_content.append({"type": "text", "text": ex["prompt"]})
            messages.append({"role": "user", "content": user_content})

            # Assistant output for that example
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": ex["output"]}]
            })

    # Add the actual query
    query_content = []
    if query_image:
        query_content.append({"type": "image", "image": query_image})
    query_content.append({"type": "text", "text": query_prompt})
    messages.append({"role": "user", "content": query_content})

    # 1) Create the textual prompt with placeholders
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 2) Process image/video inputs
    image_inputs, video_inputs = process_vision_info(messages)

    # 3) Prepare model inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    # 4) Generate output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # 5) Remove prompt tokens from output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # 6) Decode
    out_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return out_texts[0].strip()


prompt_text = ("You are an AI security monitoring assistant responsible for automatic violence detection "
    "and detailed scene reporting for on-site security guards. "
    "Your mission is to help human operators rapidly assess potential threats captured in surveillance images. "
    "Carefully examine the provided image and strictly follow these instructions:\n\n"

    "ROLE & BEHAVIOR:\n"
    "- Act as a professional safety and security analyst.\n"
    "- Remain objective, neutral, and factual—avoid speculation beyond what is visually evident.\n"
    "- Focus on detecting any signs of violence, weapons, fights, or blood.\n"
    "- Provide concise, structured, and actionable information suitable for security incident reports.\n\n"

    "ANALYSIS STEPS:\n"
    "1. Observe the entire image for people, objects, and ongoing activities.\n"
    "2. Identify any indicators of violence such as physical aggression, fighting, or assault.\n"
    "3. Detect the presence of weapons (e.g., knives, guns, blunt objects) if visible.\n"
    "4. Look for signs of injuries, visible blood, or damage related to violent acts.\n"
    "5. If no violence is detected, clearly state this but still briefly describe the general scene.\n\n"

    "REQUIRED OUTPUT FORMAT:\n"
    "Provide your findings in the following exact structure (use clear, concise sentences):\n"
    "- Violence Detected: Yes/No\n"
    "- Violence Severity Score: [0–100] (0 = no violence, 100 = extreme violence)\n"
    "- Weapons Observed: [List of weapons or 'None']\n"
    "- Signs of Fights or Physical Aggression: [Describe or 'None']\n"
    "- Signs of Blood or Injuries: [Describe or 'None']\n"
    "- Scene Description: [Brief factual summary of the overall scene]\n\n"

    "IMPORTANT:\n"
    "- Only describe what is clearly visible.\n"
    "- Do not include personal opinions or guesses.\n"
    "- Follow the output structure exactly so it can be automatically parsed.")


examples = [
    {
        "image": "./image.jpg",
        "prompt": prompt_text,
        "output": (
            "- Violence Detected: Yes\n"
            "- Violence Severity Score: 85\n"
            "- Weapons Observed: Knife\n"
            "- Signs of Fights or Physical Aggression:A person is holding a large knife in a manner that suggests threat or aggression (gripped in hand, pointed forward)\n"
            "- Signs of Blood or Injuries: No Blood\n"
            "- Scene Description: Two men engaged in a fight in a dimly lit alley."
        ),
    },
    {
        "image": "./image3.jpg",
        "prompt": prompt_text,
        "output": (
            "- Violence Detected: No\n"
            "- Violence Severity Score: 0\n"
            "- Weapons Observed: None\n"
            "- Signs of Fights or Physical Aggression: None\n"
            "- Signs of Blood or Injuries: None\n"
            "- Scene Description: A group of people working on office, some sitting peacefully."
        ),
    },
]


query_image = "./image2.jpg"

output = vlm_generate_fewshot(
    query_prompt=prompt_text,
    query_image=query_image,
    examples=examples
)

print(output)


query_image = "./image4.jpg"

output = vlm_generate_fewshot(
    query_prompt=prompt_text,
    query_image=query_image,
    examples=examples
)

print(output)


