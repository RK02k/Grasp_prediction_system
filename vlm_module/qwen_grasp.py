from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor

from qwen_vl_utils import process_vision_info

import torch


class QwenGraspModel:

    def __init__(self):

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct"
        )

    def ask(self, image_paths, prompt):
        """
        Run a single VLM inference call.

        Args:
            image_paths: str or list[str] — one or more image file paths.
                         The paper passes [scene_overlay, crop] for each call
                         so the model has both global scene context and the
                         individual segmented region.
            prompt: str — text instruction.

        Returns:
            str — the model's raw response (stripped).
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        content = []
        for path in image_paths:
            content.append({"type": "image", "image": path})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

        # Strip input tokens — keep only the generated output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output_text[0].strip()
