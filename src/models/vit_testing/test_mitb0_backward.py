import torch
from transformers import SegformerModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SegformerModel.from_pretrained("nvidia/mit-b0").to(device)
    model.train()

    x = torch.randn(2, 3, 256, 256, device=device)
    outputs = model(pixel_values=x, output_hidden_states=True, return_dict=True)

    # Simple fake loss using final hidden state
    loss = outputs.last_hidden_state.mean()
    loss.backward()

    grad_found = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient found in: {name}")
            grad_found = True
            break

    if not grad_found:
        raise RuntimeError("No gradients found")

    print("✓ Backward pass worked")

if __name__ == "__main__":
    main()