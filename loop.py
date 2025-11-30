import time
import pandas as pd
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3001",
    api_key="parallax",
)

FILE_PATH = "transcript.txt"
TIME_INTERVAL_SECONDS = 120
topics = pd.DataFrame(columns=["id", "label", "start", "end"])


def topic(file_handle, prev):
    data = []
    start = time.time()
    while True:
        line = file_handle.readline()
        if not line:
            time.sleep(0.1)
            continue
        data.append(line)
        if (time.time() - start) > TIME_INTERVAL_SECONDS:
            is_same = is_same_topic(prev, data)
            if is_same:
                start = time.time()
            else:
                return data


def is_same_topic(prev, data):
    query2 = "take the previous text and see if this text refers to the same topic. Answer either 'yes' or 'no':\n"
    query2 += "\n".join(data)

    response2 = client.chat.completions.create(
        model="Qwen/Qwen3-0.6GB",
        messages=[{"role": "user", "content": query2}],
    )

    # Treat any non-"yes" as a topic change to be safe
    return response2.choices[0].message.content.strip().lower() == "yes"


def loop():
    branches = []
    with open(FILE_PATH, "r") as transcript:
        while True:
            topic_data = topic(transcript, "kljsdflaklsdflkjalkdsfla")
            query = "What is the topic of this conversation: " + "".join(topic_data)
            response = client.chat.completions.create(
                model="Qwen/Qwen3-0.6GB",
                messages=[{"role": "user", "content": query}],
            )
            topic_label = response.choices[0].message.content.strip()
            print(f"Topic: {topic_label}")
            branches.append(topic_label)





        

            

