COMMON_PROMPT_TEMPLATES = {
    'history':
        '<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>'
        '<|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>',
    'chat': 
        '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
        'You are an assistant named "dorylus" for question-answering tasks. '
        '<|eot_id|>{history}<|start_header_id|>user<|end_header_id|>\n\n'
        '{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>',
    'kb_chat': 
        '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
        'You are an assistant named "dorylus" for question-answering tasks. '
        'Use the following pieces of retrieved context to answer the question. '
        'If you don\'t know the answer, just say that you don\'t know. '
        '<|eot_id|>{history}<|start_header_id|>user<|end_header_id|>\n\n'
        'Question: {question}\n'
        'Context: {context}\n'
        '<|eot_id|><|start_header_id|>assistant<|end_header_id|>',
}
