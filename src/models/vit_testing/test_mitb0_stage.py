import torch
from transformers import SegformerModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerModel.from_pretrained("nvidia/mit-b0").to(device)
    model.eval()

    x = torch.randn(1, 3, 512, 512, device=device)

    with torch.no_grad():
        outputs = model(pixel_values=x, output_hidden_states=True, return_dict=True)

    print("==== MiT-B0 Feature Summary ====")
    print("last_hidden_state:", tuple(outputs.last_hidden_state.shape))

    for i, feat in enumerate(outputs.hidden_states):
        print(f"hidden_states[{i}] -> {tuple(feat.shape)}")

    print("\nExpected encoder hidden sizes from docs:")
    print("[32, 64, 160, 256]")

if __name__ == "__main__":
    main()