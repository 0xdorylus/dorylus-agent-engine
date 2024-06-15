PROMPT_TEMPLATES = {
    'prompt_recognize':
        '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
        'You are a chat assistant named "Hajime". '
        'You need to recognize whether the user is greeting you. '
        'Give a binary result \'yes\' or \'no\' to indicate whether the user is greeting you or not. '
        'Provide the binary result as a JSON with a single key \'recognized\' and no premable or explaination. '
        '<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n'
        '{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>',
    'kb_select':
        '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
        'You are an expert at routing a user question to a vectorstore or web search. '
        'The vectorstores for questions have several topics. You have to find the topic that is closest to the question. '
        'You do not need to be stringent with the keywords in the question related to these topics. '
        'If no topic is related to the question, use web-search. '
        'Give a choice \'web_search\' or one of the vectorstore topics based on the question. '
        'By the way, extract key words from the question for retrieving.'
        'The keywords should be a single string separated by spaces instead of an array.'
        'Return a JSON with two keys: \'selection\' and \'keywords\', without premable or explaination. '
        '<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n'
        'Vectorstore topics: {index_types}\n'
        'Question to route: {question}\n'
        '<|eot_id|><|start_header_id|>assistant<|end_header_id|>',
    'extract_keywords':
        '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
        'You are an expert at extracting key words from a user question. '
        'You need to understand the question and extract useful keywords for web-search.'
        'The keywords should be a single string separated by spaces instead of an array.'
        'Return the a JSON with a single key \'keywords\', without premable or explaination. '
        '<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n'
        'Question: {question}\n'
        '<|eot_id|><|start_header_id|>assistant<|end_header_id|>',
    'grade_documents':
        '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
        'You are a grader assessing relevance of a retrieved document to a user question. '
        'If the document contains keywords related to the user question, grade it as relevant. '
        'It does not need to be a stringent test. The goal is to filter out erroneous retrievals.\n'
        'Give a binary score \'yes\' or \'no\' score to indicate whether the document is relevant to the question. '
        'Provide the binary score as a JSON with a single key \'score\' and no premable or explaination. '
        '<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n'
        'Here is the retrieved document: \n{document}\n\n'
        'Here is the user question: \n{question}\n'
        '<|eot_id|><|start_header_id|>assistant<|end_header_id|>',
    'check_hallucination':
        '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
        'You are a grader assessing whether an answer is grounded in / supported by a set of facts. '
        'Give a binary score \'yes\' or \'no\' score to indicate whether the answer is grounded in / supported by a set of facts. '
        'Provide the binary score as a JSON with a single key \'score\' and no preamble or explanation. '
        '<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n'
        'Here are the facts:'
        '\n ------- \n'
        '{reference}'
        '\n ------- \n'
        'Here is the answer: \n{generation}\n'
        '<|eot_id|><|start_header_id|>assistant<|end_header_id|>',
    'check_question_answering':
        '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
        'You are a grader assessing whether an answer is useful to resolve a question. '
        'Give a binary score \'yes\' or \'no\' to indicate whether the answer is useful to resolve a question. '
        'Provide the binary score as a JSON with a single key \'score\' and no preamble or explanation. '
        '<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n'
        'Here is the answer:'
        '\n ------- \n'
        '{generation}'
        '\n ------- \n'
        'Here is the question: \n{question}\n'
        '<|eot_id|><|start_header_id|>assistant<|end_header_id|>',
}
