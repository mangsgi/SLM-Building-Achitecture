import torch


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter=10):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_text(model, input_ids, max_new_tokens, context_size, tokenizer, temperature=0.0, top_k=None):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 컨텍스트 크기만큼만 사용
            input_ids_cond = input_ids[:, -context_size:]
            logits = model(input_ids_cond)
            next_token_logits = logits[:, -1, :]

            # top-k 샘플링
            if top_k is not None:
                top_logits, _ = torch.topk(next_token_logits, top_k)
                min_val = top_logits[:, -1]
                next_token_logits = torch.where(
                    next_token_logits < min_val,
                    torch.tensor(float('-inf')).to(next_token_logits.device),
                    next_token_logits
                )

            # temperature 스케일링
            if temperature > 0.0:
                next_token_logits = next_token_logits / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # EOS 토큰 체크
            if next_token.item() == tokenizer.eot_token:
                break

            input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids