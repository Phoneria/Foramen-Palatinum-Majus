import pandas as pd
from groq import Groq
import json
GROQ_API_KEY = "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"

def process_action_text(text):
    ACTION_REWRITE_PROMPT = f"""  
    Your task is to process the content of TEXT by following these steps:
    TEXT = {text}
    Remove all date-related information from the text.

    Drop any unnecessary or irrelevant information that does not contribute to the main action.

    Rewrite the text while maintaining the core meaning and emphasizing the "actions" described in the text.

    Output must be just contains what customer should do
    
    Ensure the generated text is concise, focused on the core actions, and contains the word "actions."
    
    Give output as bulletpoints as instruction following type

    ✦ When you answer, follow these rules exactly:
    1. Output only bullet‑point lines that begin with “• ” (U+2022 + space).
    2. Do not write any introductory or concluding sentences, headings, or phrases such as
        “Here are…”, “Below are…”, “In summary…”, etc.
    3. Each bullet must contain the word “actions”.
    4. No blank lines before, after, or between bullets.
    If you break any rule, the answer will be rejected.
    """
    
    client = Groq(api_key=GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": ACTION_REWRITE_PROMPT,
        }],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content



# Assuming you have a DataFrame named random_rows
def create_prompt_input(row):
    prompt_input = {
        "TRADE_NAME": row["TRADE_NAME"],
        "FIRM_NAME": row["FIRM_NAME"],
        "processed_RECALL": row["processed_RECALL"],
        "Recall Type": row["Recall Type"]
    }
    return json.dumps(prompt_input)

def process_recall_text(text):

    RECALL_EXTRACTION_PROMPT = f"""
    TEXT = {text}
    Read the provided recall description carefully. Identify the core issue or problem that led to the recall.
     
    Focus on the specific malfunction, defect, or failure, and how it affects the system's operation, safety, or performance. 
    
    Return the main recall reason in clear, concise bullet points, ensuring the following:

    Each bullet point highlights a key issue or consequence.

    Avoid excessive technical details or secondary clarifications.

    Focus on the critical aspects of the recall that directly impact the system or users.


    ✦ When you answer, follow these rules exactly:
    1. Output only bullet‑point lines that begin with “• ” (U+2022 + space).
    2. Do not write any introductory or concluding sentences, headings, or phrases such as
        “Here are…”, “Below are…”, “In summary…”, etc.
    3. Each bullet must contain the word “actions”.
    4. No blank lines before, after, or between bullets.
    If you break any rule, the answer will be rejected.



    """    
    client = Groq(api_key=GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": RECALL_EXTRACTION_PROMPT,
        }],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content



def extract_action(text, similar_input = "", similar_output = ""):
    ACTION_EXTRACTION = f"""
    You are a system designed to assist with the processing of product recalls in the medical and healthcare sectors.
    Your task is to generate a solution prompt based on the information provided in the input, focusing on the actionable instructions from the processed recall.

    The input will be in just text format which mentions the recall 
    The output must be in just text format which mentions actions


    Here is the input text that you need to process: {text}.
    Here is another input text that is similar to my input, and you need to understand the key points of it: {similar_input}.
    Here is the output of the similar input text. You need to understand it well and generate a similar output: {similar_output}.

    Here’s what you need to do:
    1. Understand the input text.
    2. Understand the other input text, which is similar to the input text.
    3. Understand the output text, which is the result of the similar input.
    4. Generate an otput by using input text

    Make sure your response follows these:
    1. Ensure the actions are clearly outlined in bullet points.
    2. The actions should be concise and focused on what must be done.
    3. Maintain the wording "actions" in each bullet point to ensure clarity.
    4. The output should not contain any introductory or concluding sentences, such as "Here is the solution" or "The following actions should be taken."
    5. The format should only consist of bullet points starting with "• " (U+2022 + space) for each action.
    6. No additional context, explanations, or examples should be included. Just provide the actions.
    7. Ensure correct loading of sample racks as per the loading instructions.
    8. Avoid pushing sample racks too far on the sample entry queue.
    9. Follow exact sample rack loading instructions to avoid misreads of Sample IDs.

    Now, using the input data provided, generate the solution prompt focusing solely on the actions required to address the recall.
    """

    client = Groq(api_key=GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": ACTION_EXTRACTION,
        }],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content


"""
# Iterate over the DataFrame
for iteration_index, row in tqdm(df.iterrows()):
    # Filter rows that have the same TRADE_NAME, FIRM_NAME, and Recall Type
    matching_rows = df[(df['TRADE_NAME'] == row['TRADE_NAME']) &
                       (df['FIRM_NAME'] == row['FIRM_NAME']) &
                       (df['Recall Type'] == row['Recall Type'])]

    # Find the row where MANUFACTURER_RECALL_REASON or ACTION are different
    similar_row = matching_rows[
        (matching_rows['MANUFACTURER_RECALL_REASON'] != row['MANUFACTURER_RECALL_REASON']) |
        (matching_rows['ACTION'] != row['ACTION'])
    ]

    if not similar_row.empty:
        # Get the index of the similar row
        similar_index = similar_row.index[0]  # In case of multiple, you can take the first one

        # Apply the extract_action function and update the Estimated_ACTION
        df.at[iteration_index, 'Estimated_ACTION'] = extract_action(
            row['processed_RECALL'], similar_row.iloc[0]['processed_RECALL'], similar_row.iloc[0]['processed_ACTION']
        )
"""
