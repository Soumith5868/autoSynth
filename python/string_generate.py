def generate_string(model, tokenizer, field_name, max_len=20):
    model.eval()
    prompt = f"Field: {field_name} →"
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(next(model.parameters()).device)
    input_ids = tokens
    hidden = None

    with torch.no_grad():
        for _ in range(max_len):
            logits, hidden = model(input_ids, hidden)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    output = input_ids[0][len(tokens[0]):]
    return tokenizer.decode(output)
