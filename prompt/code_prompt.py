CODE_INSTRUCTION = """
Your task is to write a Python function to solve a programming problem.
The Python code must be between [PYTHON] and [PYTHON] tags.
You are given one example unit test from which you can infer the function signature and output.
"""

CODE_REPAIR_INSTRUCTION = """
You are provided with a code for given problem. You can either repair and refine this code, or simply return the original solution. 
The output Python code must be between [PYTHON] and [PYTHON] tags.
Just output the code directly.
"""



CODE_PROMPT_FS = """
The following are two examples for reference.

Example 1:
Problem:
Write a python function to find the last digit when factorial of a divides factorial of b.
Test: 
assert compute_Last_Digit(2,4) == 2
The solution code is:
[PYTHON]
def compute_Last_Digit(A,B):
    variable = 1 
    if (A == B): 
        return 1 
    elif ((B - A) >= 5): 
        return 0 
    else: 
        for i in range(A + 1,B + 1): 
            variable = (variable * (i % 10)) % 10 
        return variable % 10
[PYTHON]

Example 2:
Problem:
Write a function to split a string at lowercase letters.
Test: 
assert split_lowerstring(\"AbCd\")==['bC','d']
The solution code is:
[PYTHON]
import re
def split_lowerstring(text): 
    return(re.findall('[a-z][^a-z]*', text))
[PYTHON]
""".strip() + '\n'



VALI_INSTRUCTION = """
# Role
You are a Python unit testing expert who is good at writing concise unit test cases to verify the correctness and robustness of the code based on the given solution code. You pay attention to the coverage and efficiency of the test, and ensure that the test focuses on discovering potential errors.

## Skills
### Skill 1: Understand the code logic
- Quickly parse the provided solution code to clarify the function's functions, inputs and outputs, and potential boundary conditions.

### Skill 2: Write efficient unit tests
- Write up to three unit test cases between the `[PYTHON]` tags, and only use `assert` statements to verify the consistency of expected results and function return values.
- Ensure that the test cases can effectively detect common errors and edge cases in the given code.
- Infer function signatures based on examples, and design tests to cover different parameter combinations.

### Skill 3: Follow best practices
- Test cases should be concise and clear, and directly verify the core parts of the code logic.
- Consider multiple execution paths of the code, especially those that may cause errors or exceptions.

## Limitations:
- Strictly follow the basic principles of unit testing, avoid external dependencies, and ensure the independence of tests.
- Each test case should have a clear verification purpose and avoid redundant tests.
- Do not introduce new logic or complex calculations in unit tests to keep the test code pure.
"""
# """
# Please generate Python unit test code to validate the given solution code.
# The unit test code must be between [PYTHON] and [PYTHON] tags, and it should include only assert statements. 
# Make sure to generate at most three unit tests that detect bugs in the given code.
# You can get one example test from which you can infer the right function signature.
# """

VALI_REPAIR_INSTRUCTION = \
"""
You are provided with an unit test code to validate the given solution code. You can either repair and refine this unit test code, or simply return the original solution. 
The unit test code must be between [PYTHON] and [PYTHON] tags, and it should include only assert statements.
You can get one example test from which you can infer the right function signature.
Just output the code directly.
"""

VALI_PROMPT_FS = """
The following are two examples for reference.

Example 1:
Problem:
Write a python function to find the last digit when factorial of a divides factorial of b.
Test:
assert compute_Last_Digit(1,2) == 2
Solution:
def compute_Last_Digit(A,B):
    variable = 1 
    if (A == B): 
        return 1 
    elif ((B - A) >= 5): 
        return 0 
    else: 
        for i in range(A + 1,B + 1): 
            variable = (variable * (i % 10)) % 10 
        return variable % 10
The unit test is:
[PYTHON]
assert compute_Last_Digit(2,4) == 2
assert compute_Last_Digit(6,8) == 6
[PYTHON]


Example 2:
Problem:
Write a function to split a string at lowercase letters.
Test:
assert split_lowerstring(\"Python\")==['y', 't', 'h', 'o', 'n']
Solution:
import re
def split_lowerstring(text):
    return(re.findall('[a-z][^a-z]*', text))
The unit test is:
[PYTHON]
assert split_lowerstring(\"Programming\")==['r', 'o', 'g', 'r', 'a', 'm', 'm', 'i', 'n', 'g']
assert split_lowerstring(\"AbCd\")==['bC','d']
[PYTHON]
""".strip() + '\n'
