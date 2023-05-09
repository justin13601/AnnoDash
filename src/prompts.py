system_prompt_template = "You are an expert medical annotator with experience using clinical ontologies such as LOINC " \
                         "and SNOMED CT."

user_prompt_template = """You will be given a target clinical concept along with a few example values of that concept.
                       
You will also be given inputs as newline delimited comma-separated rows representing ontology codes.

The first element of each row is a unique row id. The second element is the numerical ontology code. The third element is the text description of the ontology code.

Your goal is to re-rank the input ontology codes in order of their relevance to the target clinical concept using your expertise. The most relevant code will be listed first.

You must respond in the format of a list of integers, where the integers are row ids.


For example:
The target clinical concept is "CONCEPT". Example values of this concept include: EX1, EX2, EX3.

Input:
1,12345-6,DESCRIPTION 1
2,5678-9,DESCRIPTION 2
3,7654,DESCRIPTION 3

Output:
[2,1,3]

------

The target clinical concept is "{target}". Example values of this concept include: {examples}.
                       
Input:
{choices}

Output:
"""

'''
Ranking is fine but issue with the initial retrieve using pylucene for top 50? For ex, Ectopy Type 1’s top 50 only includes Ventricular ectopics and Ectopic pregnancy. Even though GPT ranks them highly I assume there are other better codes that weren’t retrieved by the top 50? 
'''
