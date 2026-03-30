import torch
from transformers import SegformerModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Bare MiT-B0 encoder backbone
    model = SegformerModel.from_pretrained("nvidia/mit-b0").to(device)
    model.eval()

    # Dummy image
    x = torch.randn(2, 3, 512, 512, device=device)

    with torch.no_grad():
        outputs = model(pixel_values=x, output_hidden_states=True, return_dict=True)

    print("\nModel loaded successfully.")
    print("Last hidden state shape:", outputs.last_hidden_state.shape)

    if outputs.hidden_states is None:
        print("No hidden states returned.")
        return

    print("\nHidden state shapes:")
    for i, feat in enumerate(outputs.hidden_states):
        print(f"  Stage {i}: {tuple(feat.shape)}")

if __name__ == "__main__":
    main()