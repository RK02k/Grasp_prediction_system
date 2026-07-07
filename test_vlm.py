from vlm_module.qwen_grasp import QwenGraspModel

model = QwenGraspModel()

prompt = """
Identify the object in this image.

Return ONLY:
[object]
"""

result = model.ask(
    "outputs/crops/object_0.png",
    prompt
)

print(result)