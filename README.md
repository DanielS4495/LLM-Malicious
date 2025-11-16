# LLM-Malicious

the sources of each collection of promts is in the comment section for each commit.

LLM-Malicious

The sources of each collection of prompts are in the comment section for each commit.

Articles we based on - https://docs.google.com/document/d/1u4HCAhez2J9OF\_zu4qODzZ0taol4BOYi\_jz9zEAOXcs/edit?usp=sharing



Repositories we based on – 

•	https://github.com/MAIL-Tele-AI/MalwareBench.git  - go down in readme there is attack\_prompts.xlsx

•	https://github.com/verazuo/jailbreak\_llms.git     - go to file - data/prompts

•	https://github.com/cryptwareapps/Malware-Database.git - maybe not prompt

•	https://github.com/vxunderground/MalwareSourceCode.git - maybe not prompt

•	https://huggingface.co/datasets/codesagar/malicious-llm-prompts 

•	https://www.perplexity.ai/search/bshbyl-mkhqr-blbd-tmts-ly-data-4hYFvvAVT4SoDQoW.IjkIA?0=d#0 - page of perplexity ai for research and database

•	https://github.com/cysecbench/dataset.git 

THE FIRST MODEL
=
in the model:
=
- where you take EXCEL_FILE is exel database and RESULTS_FILE is the file where you put the result it will add to the existing 
- you can change the specific criteria where you only those prompts are process
- in the 'try' you can change the model you send the prompt
- then there is sleep in the 'for' so the prompt will wait a little before they are sent so you can use the free option of the model becuse you cant send to many request in a short time
- in the 'for' there is an exception for failed prompt if the answer something wrong with it so it wont be on the file of the respond (it does write answer if the answer for the prompt is that the model wont answer) only respond that come as error will be handle

in the evaluator:
  =
- you can change the key and the os.envirson to use diffrent llm
- to use groq he use this 3 line
    os.environ["DATASETS_NUM_PROC"] = "1"
    os.environ["OPENAI_API_KEY"] = GROQ_API_KEY
    os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
the first line is for the prompt to be sent one by one because of the free option in the llm
- you can change the model in judge line
- in the 'for' there is a sleep also because of the llm free option because too many request in a short time
- the end of the file is to convert to csv

in the responses_result:
=
- there are respond for prompt of Persuative LLM only and from 320 of the prompt only answer 249 
the other got as an error it doesnt say that the llm didnt answer it only it came as error so it doesnt wront in the csv

in the responses_results_evaluated:
=
there are only 5 prompt that answer need to do it again on the responses_result 

