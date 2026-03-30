import torch
from transformers import SegformerModel

def check_no_nan(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaNs")
    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains Infs")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SegformerModel.from_pretrained("nvidia/mit-b0").to(device)
    model.eval()

    x = torch.randn(1, 3, 512, 512, device=device)

    with torch.no_grad():
        outputs = model(pixel_values=x, output_hidden_states=True, return_dict=True)

    print("\nRunning sanity checks...")

    # Check final output
    assert outputs.last_hidden_state is not None, "Missing last_hidden_state"
    check_no_nan(outputs.last_hidden_state, "last_hidden_state")
    print("✓ last_hidden_state exists and is finite")

    # Check hidden states
    assert outputs.hidden_states is not None, "Missing hidden_states"
    assert len(outputs.hidden_states) > 0, "No hidden states returned"
    print(f"✓ hidden_states returned: {len(outputs.hidden_states)} tensors")

    for i, feat in enumerate(outputs.hidden_states):
        check_no_nan(feat, f"hidden_states[{i}]")
        print(f"✓ hidden_states[{i}] shape = {tuple(feat.shape)}")

    print("\nAll checks passed.")

if __name__ == "__main__":
    main()