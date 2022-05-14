from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
import requests

#updated: may 10, 2022

app = FastAPI()

#train api for ML
@app.get("/train")
async def root(user_id:int, train_id:int,raw_data_filename:str, json_filename:str,get_sample_report:bool, set_priority:bool, minio_creds:str, mysql_creds:str):
    
    # workflow-name = my-workflow-user_id-train_id
    #workflow_name= str(train_id) +'-' + str(user_id)
    headers = {
                'Content-Type': 'application/json',
            }

    data={"user_id":user_id, "train_id":train_id, "raw_data_filename":raw_data_filename, "json_filename":json_filename, "get_sample_report":get_sample_report, "minio_creds":minio_creds, "mysql_creds":mysql_creds}
    if set_priority== True :
        response = requests.post('http://localhost:8088/trainpriority', headers=headers, json=data)
    else:
        response = requests.post('http://localhost:8085/train', headers=headers, json=data)
    

    return {"message": "training started", "user_id": user_id, "train_id": train_id}


# workflow termination api for ML
@app.get("/cancelworkflow")
async def root(job_id:str, user_id:str, force_delete:str):
    '''Enter job_id(train_id or gen_id) and your user_id to cancel a workflow ! Only Pending/ Running Workflows can be deleted.  If want to delete "succeeded/completed" workflow, set "force_delete" to true !!'''
    headers = {
        'accept': 'application/json',
        'Authorization': '121',
    }
    workflow_name= 'my-workflow-'+ str(job_id) +'-' + str(user_id)
    url='https://localhost:2746/api/v1/workflows/argo/' + workflow_name

    #CHECK IF WF PENDING OR RUNNING, ONLY ALLOW TO DELETE.
    #GET THE STATUS FROM STATUS API
    headers1 = {
        'accept': 'text/plain',
    }

    params = (
        ('user_id', str(user_id)),
        ('job_id', str(job_id)),
    )
    #get wf status from argo-server
    import subprocess
    import re
    result = subprocess.run(['microk8s', 'kubectl', 'get', 'workflow', '-n', 'argo'],capture_output=True, text=True)
    list1=result.stdout
    text_file = open("jobcancel.txt", "w")
    n = text_file.write(list1)
    text_file.close()
    # filter out "Succeeded" wfs
    file1 = open("jobcancel.txt", "r")
    #string1 = 'Succeeded'
    string1 = 'Succeeded'
    # setting flag and index to 0
    flag = 0
    index = 0
    
    # Loop through the file line by line
    for line in file1:  
        index += 1 
        
        # checking string is present in line or not
        if string1 in line:
            
            flag = 1
            break 
            
    # checking condition for string found or not
    # checking condition for string found or not
    if flag == 0 and force_delete == 'false': 
        #print('String', string1 , 'Not Found') 
        #delete the workflow
        response = requests.delete(url, headers=headers, verify=False)
        return { "workflow deleted successfully" :workflow_name, "response code": response.status_code}

    elif flag == 0 and force_delete == 'true':
        response = requests.delete(url, headers=headers, verify=False)
        return { "workflow deleted forcefully" :workflow_name, "response code": response.status_code}

    elif flag ==1 and force_delete == 'false':
        print('String', string1, 'Found In Line', index)
        return { "warning": "workflow can't be deleted beacause it's already succeeded!!"}

    elif flag == 1 and force_delete == 'true':
        response = requests.delete(url, headers=headers, verify=False)
        return { "workflow deleted forcefully" :workflow_name, "response code": response.status_code}

# workflow status check api for ML
@app.get("/status", response_class=PlainTextResponse)
async def root():
    import subprocess
    import re

    result = subprocess.run(['microk8s', 'kubectl', 'get', 'workflow', '-n', 'argo'],capture_output=True, text=True)
   
    list1=result.stdout
    #write all status to txt file
    text_file = open("sample1.txt", "w")
    n = text_file.write(list1)
    text_file.close()
    #change AGE to QUEUE RANK
    with open("sample1.txt", "r+") as texts_file:
            texts = texts_file.read()
            texts = texts.replace("AGE", "QUEUE_RANK")
            texts = texts.replace("MESSQUEUE_RANK", " ")
            texts = texts.replace("MESSAGE", " ")
    with open("sample1.txt", "w") as text_file:
            text_file.write(texts)
            text_file.close()

    #filter only pending workflows
    bad_words = ['Succeeded']

    with open('sample1.txt', ) as oldfile, open('mod1.txt', 'w') as newfile:
        for line in oldfile:
            if not any(bad_word in line for bad_word in bad_words):
                newfile.write(line)

    a_file = open("mod1.txt", "r+")
    count=1
    line = a_file.readline()
    #line.replace('MESSAGE', ' ')
    final = open("finalstatus.txt", "w")
    final.write(line)
    lines = a_file.readlines()[:]
    for line in lines:
        my_list = line.split()
        my_list[2] = str(count)
        #clear msg status
        my_list[3:] = "  "
        my_new_string = "       ".join(my_list)
        count=count+1
        final.write(my_new_string)
        final.write("\n")
    final.close()
    a_file.close()
    count=0
    # provide total workflow count
    Counter=0
    f = open('finalstatus.txt', 'r')
    Content = f.read()
    CoList = Content.split("\n")
  
    for i in CoList:
        if i:
            Counter += 1
    #print (f.read())
    Counter=0
    # "already pending workflows count":Counter-1,

    return Content

# check specific workflow status api for ML
@app.get("/getsinglestatus", response_class= PlainTextResponse)
async def root(user_id: int, job_id:int):
    """ Enter a workflow name to get it's status"""
    import subprocess
    import re
    workflow_name= 'my-workflow-'+ str(job_id) +'-' + str(user_id)
    result = subprocess.run(['microk8s', 'kubectl', 'get', 'workflow', '-n', 'argo'],capture_output=True, text=True)
    list1=result.stdout
    #write all status to txt file
    text_file = open("sample1.txt", "w")
    n = text_file.write(list1)
    text_file.close()
    #change AGE to QUEUE RANK
    with open("sample1.txt", "r+") as texts_file:
            texts = texts_file.read()
            texts = texts.replace("AGE", "QUEUE_RANK")
            #texts = texts.replace("MESSQUEUE_RANK", "MESSAGE")
            texts = texts.replace("MESSQUEUE_RANK", " ")
            texts = texts.replace("MESSAGE", " ")
    with open("sample1.txt", "w") as text_file:
            text_file.write(texts)
            text_file.close()
    #filter only pending workflows
    bad_words = ['Succeeded']

    with open('sample1.txt', ) as oldfile, open('mod1.txt', 'w') as newfile:
        for line in oldfile:
            if not any(bad_word in line for bad_word in bad_words):
                newfile.write(line)


    a_file = open("mod1.txt", "r+")
    count=1
    line = a_file.readline()
    final = open("finalstatus.txt", "w")
    final.write(line)
    lines = a_file.readlines()[:]
    for line in lines:
        my_list = line.split()
        my_list[2] = str(count)
        #clear msg status
        my_list[3:] = "  "
        my_new_string = "       ".join(my_list)
        count=count+1
        final.write(my_new_string)
        final.write("\n")
    final.close()
    a_file.close()
    file_name = "finalstatus.txt"
    file_read = open(file_name, "r")
    line1 = file_read.readline()
    singlefinal = open("singlefinalstatus.txt", "w")
    singlefinal.write(line1)
    text = workflow_name
    lines = file_read.readlines()
    new_list = []
    idx = 0

    # looping through each line in the file
    for line in lines:
            
        # if line have the input string, get the index 
        # of that line and put the
        # line into newly created list 
        if text in line:
            new_list.insert(idx, line)
            idx += 1

    #write single status header into new text
    #now write specific workflow info
    # if length of new list is 0 that means 
    # the input string doesn't
    # found in the text file
    if len(new_list)==0:
        print("\n\"" +text+ "\" is not found in \"" +file_name+ "\"!")
    else:

        # displaying the lines 
        # containing given string
        lineLen = len(new_list)
        print("\n**** Lines containing \"" +text+ "\" ****\n")
        for i in range(lineLen):
            print(end=new_list[i])
        print()
        
        singlefinal.write(new_list[i])
        # closing file after reading
    file_read.close()
    singlefinal.close()

    f11 = open('singlefinalstatus.txt', 'r')
    Contentsingle = f11.read()
    return Contentsingle

# deploy trained model api for ML
@app.get("/deploy")
async def root(raw_data_filename:str, user_id:int, train_id:int, gen_id:int, num_rows:int,minio_creds:str, mysql_creds:str):
        headers = {
            'Content-Type': 'application/json',
        }

        data={"raw_data_path":raw_data_filename, "user_id":user_id, "train_id":train_id, "gen_id":gen_id, "num_rows":num_rows, "minio_creds":minio_creds, "mysql_creds":mysql_creds}
        response = requests.post('http://localhost:8086/deploy', headers=headers, json=data)

        print(response.text)

        return {"message": "deployment successsful", "user_id": user_id, "train_id": train_id, "gen_id": gen_id}

# get metrics api for ML
@app.get("/metrics")
async def root(originaldata:str, syntheticdata:str, user_id:int, job_id:int):
        headers = {
            'Content-Type': 'application/json',
        }

        data={"train_url":originaldata, "test_url":syntheticdata, "user_id":user_id, "job_id":job_id}
        response = requests.post('http://localhost:8087/metrics', headers=headers, json=data)

        print(response.text)

        return {"message": "metrics report generation started!", "user_id": user_id, "job_id": job_id}

#to deploy fastapi on uvicorn server
# uvicorn fast_api_deployment:app --host 0.0.0.0 --port 8000 --reload