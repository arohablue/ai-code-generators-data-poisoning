import torch

# Ensure to specify the device: "cuda" if GPU is available, otherwise "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Post-processing function for code generation
def postprocess_code(generated_code, var_dict):
    for var, original_value in var_dict.items():
        generated_code = generated_code.replace(var, original_value)
    return generated_code


def generate_code(nl_text):
    # Move the model to the appropriate device
    model.to(device)

    # Preprocess the input text
    preprocessed_text, var_dict = preprocess_text(nl_text)

    # Tokenize the input
    inputs = tokenizer(preprocessed_text, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")

    # Move inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate code with the model
    with torch.no_grad():  # Disable gradient calculation for inference
        output_ids = model.generate(inputs['input_ids'], max_length=1024, num_beams=5, early_stopping=True)

    # Decode the generated tokens into code
    generated_code = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Post-process the generated code
    processed_code = postprocess_code(generated_code, var_dict)

    return processed_code

# Example usage
nl_text = "Write a code to create a response with a content type to send"
generated_code = generate_code(nl_text)
print("Generated Code:")
print(generated_code)
