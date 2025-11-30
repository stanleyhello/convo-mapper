import time
import pandas as pd
import requests
import OpenAI

client = OpenAI(
    base_url = "http://localhost:3001",
    api_key ="parallax"
)
response   = client.chat.completions.create(model="Qwen/Qwen3-0.6GB",messages = [{"role"}])

fileName = "transcript.txt"
timeInterval = 120
topics = pd.DataFrame(columns=["id","label","start","end"])
def topic(fileName,prev):
    data=[]
    start = time.time()
    end=0
    while True:
        line=fileName.readline()
        if not line:
            time.sleep(0.1)
            continue
        data.append(line)
        end=time.time()
        if (end-start)>0:
            issame = is_same_topic(prev,data)
            if(issame):
                start=time.time()
            else:
                return data

def payload_generator(query):
    return {"max_tokens": 1024,
    "messages": [
      {
        "role": "user",
        "content": data
      }
    ],
    "stream": true}

def is_same_topic(prev,data):
    query1 ="Hello, please store the following for later: "+prev
    qurey2 = "take the previous text and see if this text refers to the same topic. Answer either 'yes' or 'no':"
    for i in data:
        query2+=i+"\n"
    payload1 = payload_generator(query1)
    payload2 = payload_generator(query2)
    response1= client.chat.completions.create(model="Qwen/Qwen3-0.6GB",messages = [{"role":"user","content":query1}])
    response2 = client.chat.completions.create(model="Qwen/Qwen3-0.6GB",messages = [{"role":"user","content":query2}])

    """
    r1 = requests.post("http://localhost:3001/v1/chat/completions",data=payload1)
    r2 = requests.post("http://localhost:3001/v1/chat/completions",data=payload2)
    """
    return response2.choices[0].message.text == "yes"

    
def loop():
    branches=[]
    while True:
        topic_data= topic(fileName,"kljsdflaklsdflkjalkdsfla")
        query = "What is the topic of this conversation: "+topic_data
        response = client.chat.completions.create(model="Qwen/Qwen3-0.6GB",messages = [{"role":"user","content":query}])
        topic = response.choices[0].text.message
        branches.append(topic)





        

            

