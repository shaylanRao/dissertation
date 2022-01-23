from string import punctuation

import pandas as pd
import numpy as np
from IPython.display import display
import re

# first = np.array([["Testing 1", "Alpha", ""]])
# second = np.array([["Testing two", "Beta", "U"]])
#
# final = np.append(first, second, axis=0)
# df = pd.DataFrame(final)
# test_string = "The wait is almost over. Our recorded #Venom: Let There Be #Carnage episode will be unleashed to your bb ears on Thursday!What beer should we drink next??ðŸ•·ðŸ”ªðŸ•·ðŸ”ªðŸ•·ðŸ”ªðŸ•·ðŸ”ªðŸ•·ðŸ”ªðŸ•·https://t.co/xGYC3xoIN7"
# print(re.sub("[^0-9a-zA-Z{} ]+".format(punctuation), "", test_string))

main_df = pd.DataFrame(np.array([["", "", ""]]))

main_df.append(main_df,  np.array([["Testing 1", "Alpha", ""]]))
main_df.append(main_df,  np.array([["Testing two", "Beta", "U"]]))

display(main_df)
