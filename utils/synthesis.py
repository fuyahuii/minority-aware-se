def generate_reasoning_prompt(row):
    instruction = (
        "You are an expert dialogue evaluator. Your task is to reconstruct the seeker's reasoning that explains their satisfaction score.\n"
        "Be concise and precise. Limit each step to 1 sentence."
    )
    input_text = f"""This is a multi-turn dialogue between a [seeker] and a [supporter]. 

{row['input']}

The [seeker] is experiencing a problem related to '{row['problem_type']}' and is feeling '{row['emotion_type']}'.
The seeker rated their satisfaction as {row['output']} on a scale from 1 (very dissatisfied) to 5 (very satisfied).

Please reason step by step from the seeker's perspective:

Step 1: What is the seeker's underlying intent or goal, based on their final message and previous context? 
Step 2: Identify the main strategy the supporter used in their overall response to the final [seeker]'s message. Choose from the following categories:  
Question, Restatement or Paraphrasing, Reflection of Feelings, Self-disclosure, Affirmation and Reassurance, Providing Suggestions, Giving Information, Others.  
Step 3: Did the support match the seeker's intent and priorities?  
Step 4: Explain why the seeker gave a satisfaction score of {row['output']}. 
Briefly justify whether the supporter met the seeker's emotional and practical needs, such as showing empathy, staying relevant to the concern, and responding to the seeker's intent, as well as any other relevant factors.


Please format your response as:
Step 1: ...
Step 2: ...
Step 3: ...
Step 4: ...
"""

    return f"{instruction}\n\n{input_text}"

