# import numpy as np

# PATH_1 = '/mnt/nas/data/yihan/Code/symbol-llm-v2/score_memory/gsm_math_full_llama2chat/scores_gsm_math_full_llama2chat_part1_iter1.npy'
# PATH_2 = '/mnt/nas/data/yihan/Code/symbol-llm-v2/score_memory/gsm_math_full_llama2chat/scores_gsm_math_full_llama2chat_part1_iter4.npy'

# data1 = np.load(PATH_1)
# data2 = np.load(PATH_2)

# print(np.mean(data1))
# print(np.mean(data2))

# text = "```python\ndef solution():\n    '''Ann is baking cookies. She bakes three dozen oatmeal raisin cookies, two dozen sugar cookies, and four dozen chocolate chip cookies. Ann gives away two dozen oatmeal raisin cookies, 1.5 dozen sugar cookies, and 2.5 dozen chocolate chip cookies. How many total cookies does she keep?'''\n    cookies_baked = 3 * 12 + 2 * 12 + 4 * 12\n    cookies_given_away = 2 * 12 + 1.5 * 12 + 2.5 * 12\n    total_cookies = cookies_baked - cookies_given_away\n    result = total_cookies\n    return result\n```\n\nIn each case, the solution code should be well-formatted, with clear variable names and appropriate indentation.\n\nNote that the input for each question is provided as a string, so you'll need to parse it and convert it to an appropriate data type before solving the problem.\n\nI hope this helps! Let me know if you have any questions or need further clarification."
# print(text)

# def solution():
#     '''Ann is baking cookies. She bakes three dozen oatmeal raisin cookies, two dozen sugar cookies, and four dozen chocolate chip cookies. Ann gives away two dozen oatmeal raisin cookies, 1.5 dozen sugar cookies, and 2.5 dozen chocolate chip cookies. How many total cookies does she keep?'''
#     oatmeal_raisin_cookies = 3 * 12 = 36
#     sugar_cookies = 2 * 12 = 24
#     chocolate_chip_cookies = 4 * 12 = 48
#     given_away_oatmeal = 2 * 12 = 24
#     given_away_sugar = 1.5 * 12 = 18
#     given_away_chocolate = 2.5 * 12 = 30
#     total_cookies = oatmeal_raisin_cookies + sugar_cookies + chocolate_chip_cookies - given_away_oatmeal - given_away_sugar - given_away_chocolate
#     result = total_cookies
#     return result

# def solution(text): 
#     import re
#     return(re.findall('[a-z][^a-z]*', text))
# def unit_test():
#     assert solution("Python")==['y', 't', 'h', 'o', 'n']
#     assert solution("Programming")==['r', 'o', 'g', 'r', 'a', 'm', 'm', 'i', 'n', 'g']
#     assert solution("AbCd")==['bC','d']
# print(solution("AbCd"))

# from vali_agent_label_preference import *
# import re
# data_label = """```python 
# def solution(text): 
#     import re
#     return(re.findall('[a-z][^a-z]*', text))
# ```"""
# data_code = "```python\ndef solution(board, m, n):\n    R = 3\n    C = 3\n    tc = [[0 for x in range(C)] for x in range(R)] \n    tc[0][0] = board[0][0]\n    for i in range(1, n+1): \n        tc[i][0] = tc[i-1][0] + board[i][0]\n        for j in range(1, C+1): \n            tc[i][j] = min(tc[i-1][j], tc[i][j-1]) + board[i][j]\n    return tc[R-1][C-1]\n```\nplease help me solve this question.\n\nThank you so much.\n\nAnswer: Sure, I can help you solve the question. To find the number of ways to fill 2 x 1 dominoes for a given 3 x n board, you can use the following approach:\n\n1. Initialize two arrays `tc` and `board` as you have in the example.\n2. Initialize the `tc` array with all possible values summing up to zero.\n3. Loop through each row of the `board` array, starting from the second row.\n4. For each row, loop through each column, starting from the second column.\n5. For each cell in the current row and column, calculate the minimum total cost of placing a domino on that cell, taking into account the cost of the existing tiles in the current row and column, and the cost of the new domino.\n6. Add the minimum total cost to the corresponding element of the `tc` array.\n7. Return the minimum total cost from the `tc` array.\n\nHere's some Python code to illustrate the above approach:\n```python\ndef solution(board, n):\n    R = 3\n    C = n\n    tc = [[0 for x in range(C)] for x in range(R)] \n    tc[0][0] = board[0][0]\n    for i in range(1, n+1): \n        tc[i][0] = tc[i-1][0] + board[i][0]\n        for j in range(1, C+1): \n            tc[i][j] = min(tc[i-1][j], tc[i][j-1]) + board[i][j]\n    return min(tc[R-1][C-1])\n```\nNote that the time complexity of this solution is O(n^2), which is the number of cells in the board. This is because we are looping through each cell in the board twice, once for each row and once for each column.\n\nI hope this helps! Let me know if you have any questions."
# data_validation = """```python\ndef unit_test():
#     assert solution("Python")==['y', 't', 'h', 'o', 'n']
#     assert solution("Programming")==['r', 'o', 'g', 'r', 'a', 'm', 'm', 'i', 'n', 'g']
#     assert solution("AbCd")==['bC','d']```"""
# label = "R = 3\r\nC = 3\r\ndef min_cost(cost, m, n): \r\n\ttc = [[0 for x in range(C)] for x in range(R)] \r\n\ttc[0][0] = cost[0][0] \r\n\tfor i in range(1, m+1): \r\n\t\ttc[i][0] = tc[i-1][0] + cost[i][0] \r\n\tfor j in range(1, n+1): \r\n\t\ttc[0][j] = tc[0][j-1] + cost[0][j] \r\n\tfor i in range(1, m+1): \r\n\t\tfor j in range(1, n+1): \r\n\t\t\ttc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j] \r\n\treturn tc[m][n]"
# code_str = parse_code_block(data_code)
# print(code_str)
# code_str = [data_label, data_validation]
# flag = exc_code(code_str, validation=True)
# print(re.sub(r'\bdef\s+\w+\s*\(', "def solution(", label))
# pattern = r"```(.*?)```"
# code_blocks = re.findall(pattern, data_validation, re.DOTALL)
# print(code_blocks[0])


# test = {"origin_id": "90", 
#  "type": "self-explore", 
#  "prompt": "Write Python unit test code to validate the code for given question.\n\nThe question is:Write a function to find whether a given array of integers contains any duplicate element.\nThe solution code is:\ndef solution(arr):\n    \"\"\"Write a function to find whether a given array of integers contains any duplicate element.\"\"\"\n    seen = set()\n    for num in arr:\n        if num in seen:\n            return True\n        seen.add(num)\n    return False\nThe unit test code is:\n", 
#  "completion": "```python\n\ndef unit_test():\n    assert solution([1, 2, 3, 3, 2, 1]) == True\n    assert solution([4, 5, 6, 1, 2]) == False\n\n```"}
# print(test["prompt"])
# print(test["completion"])
import re
# # x = "```\ndef minimumCostPath(cost, m, n):\n    # Create a graph for the cost matrix\n    graph = Graph()\n    for i in range(m):\n        for j in range(n):\n            graph.add_vertex(i, j)\n    # Add the starting vertex\n    graph.add_vertex(0, 0)\n    \n    # Calculate the minimum cost path\n    minimum_cost = float('inf')\n    prev = (0, 0)\n    current = (0, 0)\n    while current != (m, n):\n        # Get the next vertex\n        next = graph.get_vertex(current[0], current[1])\n        # Calculate the cost of reaching the next vertex\n        cost_of_next = cost[next[0]][next[1]]\n        # Update the minimum cost if it is lower\n        minimum_cost = min(minimum_cost, cost_of_next)\n        # Set the new current vertex\n        current = next\n    # Return the minimum cost path\n    return [minimum_cost]\n```\nPlease input the code to solve the question directly."
# x = re.sub(r'\bdef\s+\w+\s*\(', "def solution(", x)
# start_token = re.escape("def solution(")
# end_token = re.escape(re.search(r'return\s+.+\n', x).group())
# pattern = fr"{start_token}(.*?){end_token}"
# match = re.search(pattern, x, re.DOTALL).group()
# print(match)

# x = "[PYTHON]def num()[/PYTHON]"
# code = re.search(r"\[PYTHON\](.*?)\[/PYTHON\]", x, re.DOTALL).group(1)
# print(code)

# import os
# import numpy as np

# # with open("/mnt/nas/data/yihan/Code/symbol-llm-v2/score_memory/code_agent/mbpp_full_llama2chat/scores_mbpp_full_llama2chat_part1_iter0.npy", "r") as f:
# # data = np.load("/mnt/nas/data/yihan/Code/symbol-llm-v2/score_memory/code_agent/mbpp_full_llama2chat/scores_mbpp_full_llama2chat_part1_iter0.npy")
# # print(data)
# from use_datasets import load_data
# from prompt import code_prompt
# import jsonlines
# import random

# # npy pass@1 and pass@5
# # data pass@1 and pass@5
# # train_data pass@1 and pass@5
# data = []
# for i in range(1, 5):
#     data += np.load(f"score_memory/validation_agent/mbpp_full_llama2chat/scores_mbpp_full_llama2chat_part{i}_iter0.npy").tolist()

# acc_pass1 = []
# acc_pass5 = []
# for i in range(len(data)):
#     pass5_flag = False
#     for j in range(5):
#         pass5_flag = pass5_flag or data[i][j] == 2
#         acc_pass1.append(data[i][j] == 2)
#     acc_pass5.append(pass5_flag)
# print(f"pass@1: {sum(acc_pass1)/len(acc_pass1)}")
# print(f"pass@5: {sum(acc_pass5)/len(acc_pass5)}")
# import re
# import os
# import json

# with open(f"logs/youxiao_unit_test_wrong_validation_right_log.json", "r") as f:
#     data = json.load(f)

# for i in range(len(data)):
#     if not re.search(r'assert\s\w*\(.*\)\s*==\s*.*', data[i]['unit_test']):
#         print(data[i]['unit_test'])

import numpy as np

score = np.load(f"score_memory/validation_agent/mbpp_full_llama2chat/scores_mbpp_full_llama2chat_part1_iter0.npy")
acc = 0
for i in range(len(score)):
    for j in range(len(score[i])):
        if score[i][j] == 2:
            acc += 1
print(acc)
