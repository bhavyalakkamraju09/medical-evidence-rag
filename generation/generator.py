import ollama

def generate_answer(prompt):

    response = ollama.chat(
        model="phi3",
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.2,
            "num_predict": 200
        }
    )

    return response["message"]["content"]