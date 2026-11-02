import json

# base system instructions for all 3 prompting techniques

base = """
You are a content moderation expert evaluating online comments for toxicity.
Rate the toxcicity of an input comment as per the following criteria:
- 'Very Toxic' (a very hateful, aggressive, or disrespectful comment that is very likely to cause someone to leave a discussion or give up on sharing their perspective)
- 'Toxic' (a very rude, disrespectful, or unreasonable comment that is somewhat likely to cause someone to leave a discussion or give up on sharing their perspective)
- 'Hard to say'
- 'Not Toxic' 
Assign a continuous score between 0.0 and 1.0 representing the estimated fraction of human raters who would judge the comment as toxic or not. (0.0 = all raters consider the comment as 'Hard to say' and/or 'Not Toxic', 1.0 = all raters consider it 'Toxic' and/or 'Very Toxic')
"""
subtypes = """
Assign, for each of the toxicity subtypes below, a continuous score between 0.0 and 1.0 representing the estimated fraction of human raters who would agree the subtype applies to the comment (0.0 = all raters agree the subtype doesn't apply, 1.0 = all raters agree the subtype applies):
- 'severe_toxicity'
- 'obscene'
- 'threat'
- 'insult'
- 'identity_attack'
- 'sexual_explicit'
"""

# example comments selected from outside of the sampled dataset

few_shot_examples = [
    {
        "comment": """I already see dozens of "just commenting to vote on comments to be able to comment" type of comments.

This voting system is completely insane- you're assuming people have the extra time to *actually read* multiple non related comments?

I would love to see this system installed on oregonlive.com - they have enough online visitors and an established "troll" base to really need this service.

With as little online activity as wweek has - it will discourage further growth here.""",
        "output": {"toxicity_score": 0.0}
    },
    {
        "comment": """YET ANOTHER BARACK OBAMA LIBERAL MEDIA CONSPIRACY BY THE THOUGHT CONTROL POLICE. I DIDENT SPEND 30 YEARS MIXING CONCRETE TO LET AMERICA FALL TO THE COMMIES. THE TAXES HERE ARE SO HIGH I CANT EVEN AFFORD A KEYBOARD WITH A WORKING CAPS LOCK. PORTLAND HAS BEEN GOING DOWNHILL FOR YEARS, NO WONDER THE TEA PARTY IS MAKING A COMEBACK. WHATS NEXT FLOURAIDE IN THE WATER SUPPLY?""",
        "output": {"toxicity_score": 0.1666666666666666}
    },
    {
        "comment": """hahahahahahahahhha suck it.""",
        "output": {"toxicity_score": 0.4576271186440678}
    },
    {
        "comment": """ur a sh*tty comment.""",
        "output": {"toxicity_score": 0.6666666666666666}
    },
    {
        "comment": """This bitch is nuts. Who would read a book by a woman.""",
        "output": {"toxicity_score": 0.8307692307692307}
    },
    {
        "comment": """Yet call out all Muslims for the acts of a few will get you pilloried.   So why is it okay to smear an entire religion over these few idiots?  Or is this because it's okay to bash Christian sects?""",
        "output": {"toxicity_score": 0.9125}
    }
]
few_shot_examples_subtypes = [
    {
        "comment": """I already see dozens of "just commenting to vote on comments to be able to comment" type of comments.

This voting system is completely insane- you're assuming people have the extra time to *actually read* multiple non related comments?

I would love to see this system installed on oregonlive.com - they have enough online visitors and an established "troll" base to really need this service.

With as little online activity as wweek has - it will discourage further growth here.""",
        "output": {
            "toxicity_score": 0.0,
            "subtypes" : {
                "severe_toxicity": 0.0,
                "obscene": 0.0,
                "threat": 0.0,
                "insult": 0.0,
                "identity_attack": 0.0,
                "sexual_explicit": 0.0
                }
            }
    },
    {
        "comment": """YET ANOTHER BARACK OBAMA LIBERAL MEDIA CONSPIRACY BY THE THOUGHT CONTROL POLICE. I DIDENT SPEND 30 YEARS MIXING CONCRETE TO LET AMERICA FALL TO THE COMMIES. THE TAXES HERE ARE SO HIGH I CANT EVEN AFFORD A KEYBOARD WITH A WORKING CAPS LOCK. PORTLAND HAS BEEN GOING DOWNHILL FOR YEARS, NO WONDER THE TEA PARTY IS MAKING A COMEBACK. WHATS NEXT FLOURAIDE IN THE WATER SUPPLY?""",
        "output": {
            "toxicity_score": 0.1666666666666666,
            "subtypes" : {
                "severe_toxicity": 0.0,
                "obscene": 0.0,
                "threat": 0.0,
                "insult": 0.1666666666666666,
                "identity_attack": 0.0,
                "sexual_explicit": 0.0
                }
            }
    },
    {
        "comment": """hahahahahahahahhha suck it.""",
        "output": {
            "toxicity_score": 0.4576271186440678,
            "subtypes" : {
                "severe_toxicity": 0.0508474576271186,
                "obscene": 0.3050847457627119,
                "threat": 0.0,
                "insult": 0.2542372881355932,
                "identity_attack": 0.0,
                "sexual_explicit": 0.2203389830508474
                }
            }                                         
    },
    {
        "comment": """ur a sh*tty comment.""",
        "output": {
            "toxicity_score": 0.6666666666666666,
            "subtypes" : {
                "severe_toxicity": 0.0476190476190476,
                "obscene": 0.6380952380952379,
                "threat": 0.0,
                "insult": 0.3333333333333333,
                "identity_attack": 0.0,
                "sexual_explicit": 0.0095238095238095
                }
            }                    
    },
    {
        "comment": """This bitch is nuts. Who would read a book by a woman.""",
        "output": {
            "toxicity_score": 0.8307692307692307,
            "subtypes" : {
                "severe_toxicity": 0.1076923076923077,
                "obscene": 0.6615384615384615,
                "threat": 0.0,
                "insult": 0.8307692307692307,
                "identity_attack": 0.3384615384615385,
                "sexual_explicit": 0.0615384615384615
                }
            }                    
    },
    {
        "comment": """Yet call out all Muslims for the acts of a few will get you pilloried.   So why is it okay to smear an entire religion over these few idiots?  Or is this because it's okay to bash Christian sects?""",
        "output": {
            "toxicity_score": 0.9125,
            "subtypes" : {
                "severe_toxicity": 0.05,
                "obscene": 0.2375,
                "threat": 0.1125,
                "insult": 0.8875,
                "identity_attack": 0.6125,
                "sexual_explicit": 0.0
                }
            }                    
}
]

user_input = "Rate the toxicity of this comment: "

# Part 1: system instructions to get score for toxicity only

zero_fewshot_system = f"{base}\nReturn the score in JSON structure."
cot_system = f"{base}\nBefore outputting scores, provide a brief step-by-step analysis of tone, intent, and toxicity in the 'reasoning' key."

# Part 2: system instructions to get score for toxicity and subtypes

zero_fewshot_subtypes_system = f"{base}\n{subtypes}\nReturn the scores in JSON structure."
cot_subtypes_system = f"{base}\n{subtypes}\nBefore outputting scores, provide a brief step-by-step analysis of tone, intent, and toxicity in the 'reasoning' key."

def get_zero_shot_prompt(comment, include_subtypes):
    system_prompt = zero_fewshot_subtypes_system if include_subtypes else zero_fewshot_system

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input+comment}
    ]

def get_few_shot_prompt(comment, include_subtypes):
    system_prompt = zero_fewshot_subtypes_system if include_subtypes else zero_fewshot_system
    examples = few_shot_examples_subtypes if include_subtypes else few_shot_examples
    
    prompts = [{"role": "system", "content": system_prompt}]
    for example in examples:
        prompts.append({"role": "user", "content": user_input+example["comment"]})
        prompts.append({"role": "assistant", "content": json.dumps(example["output"])})

    prompts.append({"role": "user", "content": user_input+comment})

    return prompts

def get_cot_prompt(comment, include_subtypes):
    system_prompt = cot_subtypes_system if include_subtypes else cot_system

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input+comment}
    ]
   