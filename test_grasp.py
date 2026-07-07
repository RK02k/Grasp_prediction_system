from vlm_module.qwen_grasp import QwenGraspModel

model = QwenGraspModel()

prompt = """
Identify the best human grasp type for this object.

Choose ONLY from:

[Cylindrical grasp]
[Pinch grasp]
[Hook grasp]
[Lateral pinch]
[Spherical grasp]
[Pen grip]
[Button-press grasp]

Return ONLY one answer.
"""

result = model.ask(
    "outputs/crops/object_0.png",
    prompt
)

print(result)