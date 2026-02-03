# Sequential execution (slow)
results = []
for question in sub_questions:
    result = search_api(question)  # 5 seconds per call
    results.append(result)
# Total time: 25 seconds for 5 questions (5s × 5)
