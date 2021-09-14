from nltk.tokenize import word_tokenize

def generate_responses(
        model,
        tokenizer,
        batch_size,
        tokenizer_max_len,
        model_max_len, 
        beam,
        context_file, 
        output_file
    ):
    """
    Generate a response for each context in context_file.
    Write output to `output_file`
    Adapted from issue #63 by Shilei Liu
    https://github.com/microsoft/DialoGPT/issues/63
    """
    SEP = tokenizer.eos_token

    j = 0
    context_exists = True
    f = open(context_file, 'r')
    out = open(output_file, "w")

    while context_exists:
        i = 0
        batch = []
        while i < batch_size:
            line = f.readline()
            if not line:
                context_exists = False
                break
            context = SEP.join(line.strip().split(' EOS ')) + SEP
            batch.append(context)
            i += 1
            j += 1

        if not batch:
            break

        inputs = tokenizer(batch, max_length=tokenizer_max_len,
                           padding=True, truncation=True, return_tensors="pt")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        context_length = input_ids.shape[1]

        preds_id = model.generate(
            input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            max_length=model_max_len,  # TODO: Investigate
            num_beams=beam,
            pad_token_id=tokenizer.eos_token_id
        )
        preds_id = preds_id[:, context_length:].tolist()

        for pred in preds_id:
            response = tokenizer.decode(pred, skip_special_tokens=True)
            out.write(' '.join(word_tokenize(response)) + "\n")

        if j % 512 == 0:
            print(f"{j} responses written to {output_file}")
    out.close()
    f.close()
    print(f"{j} responses written to {output_file}")
