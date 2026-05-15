from enhancer.pipeline import EnhancementPipeline

pipeline = EnhancementPipeline(
    scale=4,
    enable_face=True,
    denoise_strength=10.0,
    sharpen_strength=1.5,
    gamma=1.1,
)

pipeline.run("img2.png", output_path="output_4k.png")
print("Done — output saved to output_4k.png")