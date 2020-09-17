



# *** STREAMLIT NOTES ***
# create python (.py) file (no spaces in name) with streamlit code
# use IDE or editor of choice (examples-Visual Studio Code, Atom, JupyterLab, etc.)
# in Terminal, Command Prompt, etc. cd (change directory) to file location
# enter "streamlit run file_name.py" and run
# can run app from python file on GitHub (i.e.-streamlit run link_to_file)
# rerun or always rerun streamlit app (top right rerun button)
# control z or control c to exit streamlit in Terminal
# for large datasets (loading), use @st.cache

'''
For Windows use Command Prompt (Terminal)

On Windows, if you are having problems open Anaconda Navigator, go to Environments (left sidebar),
click create (bottom left) to create a new python environment.

Once new environment is created, select that environment and click the arrow and Open Terminal,
install streamlit, then run python file using “streamlit run file_name.py”.

To run streamlit file from Command Prompt (Terminal), change directory if needed to file location (example-cd Desktop).
'''

# *** JUPYTERLAB NOTES ***
# to create python file in jupyterlab, create text file and rename with .py extension
# to run python file from jupyterlab, open terminal, cd (change directory to file location), type "python file_name.py"

import streamlit as st

st.write('Hello World')
