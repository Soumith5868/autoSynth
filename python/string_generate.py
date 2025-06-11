def generate_name(model, char2idx, idx2char, max_len=20):
    model.eval()
    start_id = torch.tensor([[char2idx["<START>"]]], dtype=torch.long).to(device)
    hidden = None
    input_ids = start_id
    output_str = ""

    for _ in range(max_len):
        logits, hidden = model(input_ids, hidden)  # input_ids: [1,1]
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # shape: [1,1]
        char = idx2char.get(next_id.item(), "")
        if char == "<PAD>":
            break
        output_str += char
        input_ids = next_id  # keep it as [1,1] for next GRU input

    return output_str