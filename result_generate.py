from NLP_Model.scripts.generate import generate_style_reference


if __name__ == "__main__":
    user_prompt = "A painting in the {x} style, professional artwork, trending on artstation"
    style_image, style_embedding = generate_style_reference(user_prompt)
    style_image.save("generated_style.png")