def generate_coper_prompt_training(row):
    instruction = f"""Generate a structured reasoning path to predict the seeker's satisfaction score.
Be concise and precise. Limit each step to 1 sentence. 
This is a multi-turn dialogue between a [seeker] and a [supporter]. 
The [seeker] is experiencing a problem related to '{row['problem_type']}' and is feeling '{row['emotion_type']}'."""

    input_text = f"""{row['input']}
    
    Assume the [seeker] will rate their satisfaction on a 5-point scale (1 = very dissatisfied, 5 = very satisfied).
    To simplify, we define two categories: Ratings of 1 to 3 indicate **Low satisfaction**; Ratings of 4 or 5 indicate **High satisfaction**.

    Please reason step by step from the seeker's perspective:

    Step 1: What is the seeker's underlying intent or goal, based on their final message and previous context? 
    Step 2: Identify the main strategy the supporter used in their overall response to the final [seeker]'s message. Choose from the following categories:  
    Question, Restatement or Paraphrasing, Reflection of Feelings, Self-disclosure, Affirmation and Reassurance, Providing Suggestions, Giving Information, Others.  
    Step 3: Did the support match the seeker's intent and priorities?  
    Step 4: Justify the seeker's likely satisfaction based on the emotional and practical support received. 
    Reflect on whether the supporter showed empathy, staying relevant to the concern, and responding to the seeker's intent, as well as any other relevant factors.

    Please format your response as follows, Final score and Final answer should be one of the given options. 
    Step 1: [analysis]
    Step 2: [analysis]
    Step 3: [analysis]
    Step 4: [analysis]
    
    Final score: 1, 2, 3, 4, or 5. 
    Final answer: Low satisfaction or High satisfaction
    
    """

    output_score = f"Final score: {row['output']}"
    output_label = f"Final answer: {'High satisfaction' if row['label'] == 'B' else 'Low satisfaction'}"
    output_reasoning=row['generated_reasoning']
    output = f"{output_reasoning}\n{output_score}\n{output_label}"
    
    return {
        "instruction": instruction.strip(),
        "input": input_text.strip(),
        "output": output.strip(),
        "score": row['output'],
    }

def generate_coper_prompt_inference(row):
    instruction = f"""Generate a structured reasoning path to predict the seeker's satisfaction score.
Be concise and precise. Limit each step to 1 sentence. 
This is a multi-turn dialogue between a [seeker] and a [supporter]. 
The [seeker] is experiencing a problem related to '{row['problem_type']}' and is feeling '{row['emotion_type']}'."""


    input_text = f"""{row['input']}
    
    Assume the [seeker] will rate their satisfaction on a 5-point scale (1 = very dissatisfied, 5 = very satisfied).
    To simplify, we define two categories: Ratings of 1 to 3 indicate **Low satisfaction**; Ratings of 4 or 5 indicate **High satisfaction**.

    Please reason step by step from the seeker's perspective:

    Step 1: What is the seeker's underlying intent or goal, based on their final message and previous context? 
    Step 2: Identify the main strategy the supporter used in their overall response to the final [seeker]'s message. Choose from the following categories:  
    Question, Restatement or Paraphrasing, Reflection of Feelings, Self-disclosure, Affirmation and Reassurance, Providing Suggestions, Giving Information, Others.  
    Step 3: Did the support match the seeker's intent and priorities?  
    Step 4: Justify the seeker's likely satisfaction based on the emotional and practical support received. 
    Reflect on whether the supporter showed empathy, staying relevant to the concern, and responding to the seeker's intent.

    Please format your response as follows, Final score and Final answer should be one of the given options. 
    Step 1: [analysis]
    Step 2: [analysis]
    Step 3: [analysis]
    Step 4: [analysis]
    
    Final score: 1, 2, 3, 4, or 5.  
    Final answer: Low satisfaction or High satisfaction
    
    """

    return {
        "instruction": instruction.strip(),
        "input": input_text.strip()
    }
