from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import streamlit as st

# Loading pretrained transformer model for answer generation
answer_generator = pipeline('text-generation', model="gpt2")

# Loading pretrained transformer model for sentence embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# creating a database of questions and answers from pporsche.txt


qa_database = {
    "When did Ferdinand found his iconic sports car company?": "He established Porsche in 1931 in Stuttgart, Germany.",
    "What type of innovative vehicle did Ferdinand Porsche design before starting his own brand?": "He designed an early hybrid car, showcasing his forward-thinking approach.",
    "What is Porsche best known for in the automotive world?": "Porsche is famous for building high-performance sports cars with an emphasis on speed and handling.",
    "Which Porsche model is considered a timeless automotive icon?": "The Porsche 911, introduced in 1963, is their most famous model.",
    "Where does Porsche demonstrate its dominance in the racing world?": "Porsche has won countless races, including the legendary 24 Hours of Le Mans.",
    "Besides sports cars, what other successful vehicle types has Porsche introduced?": "Porsche expanded into the popular SUV market with the Cayenne and Macan.",
    "Which Porsche model represents their electric vehicle innovation?": "The Porsche Taycan showcases their commitment to performance in the electric car market.",
    "How does Porsche extend its design philosophy beyond automobiles?": "Porsche Design creates stylish watches, eyewear, and other lifestyle products.",
    "Where is the must-visit Porsche Museum located?": "The Porsche Museum is located in Stuttgart, Germany.",
    "What are the regular operating hours of the Porsche Museum?": "The museum is open Tuesday to Sunday, 9 am to 6 pm, and closed on Mondays.",
    "When do the ticket desks at the Porsche Museum close?": "Ticket desks close at 5:30 pm.",
    "Where can I check for special opening/closing days around holidays?": "Visit the official Porsche Museum website (https://www.porsche.com/international/aboutporsche/porschemuseum/).",
    "How much is regular adult admission to the Porsche Museum?": "Adult admission is 12 euros.",
    "Are there any discounted ticket prices available?": "Yes, reduced prices apply to students, seniors, those with disabilities, and others. Children under 14 enter for free.",
    "Can I get a discount if I visit the Porsche Museum in the evening?": "Yes, evening tickets (after 5 pm) are available at a reduced price.",
    "How is the exhibition laid out in the Porsche Museum?": "It follows a spiral design; follow it to the left for a chronological experience.",
    "What are the dining and refreshment options at the Porsche Museum?": "The museum has a cafe (Boxenstopp) and a restaurant (Christophorus).",
    "Besides the car exhibits, what else can I enjoy at the museum?": "There's a workshop viewing area, driving simulators, and cars you can sit in.",
    "When is the best time to visit the Porsche Museum for a less crowded experience?": "Try weekday mornings for a quieter visit.",
    "What was the first production car created by Porsche?": "The Porsche 356, introduced in 1948, was the first production car by Porsche.",
    "Which iconic racing event did Porsche win with the 917 model?": "Porsche achieved overall victory at the 24 Hours of Le Mans in 1970 and 1971 with the 917.",
    "What is the top speed of the Porsche 911 Turbo S?": "The Porsche 911 Turbo S can reach a top speed of around 205 mph (330 km/h).",
    "In what year did Porsche introduce its first electric car?": "Porsche introduced its first electric car, the Taycan, in 2019.",
    "What is the horsepower of the Porsche Cayenne Turbo?": "The Porsche Cayenne Turbo generates around 541 horsepower.",
    "Which Formula 1 team did Porsche supply engines to in the 1980s?": "Porsche supplied engines to the McLaren Formula 1 team in the mid-1980s.",
    "What is the name of Porsche's luxury sedan model?": "The Porsche Panamera is the luxury sedan model offered by the brand.",
    "Which famous designer created the Porsche 911's distinctive shape?": "Ferdinand 'Butzi' Porsche, the grandson of Ferdinand Porsche, designed the iconic shape of the Porsche 911.",
    "What is the 0-60 mph time of the Porsche 718 Cayman GTS?": "The Porsche 718 Cayman GTS can accelerate from 0 to 60 mph in around 3.9 seconds.",
    "Which material is extensively used in Porsche's lightweight construction?": "Porsche utilizes materials like aluminum and high-strength steel in its lightweight construction for improved performance.",
    "What does the 'RS' stand for in Porsche 911 GT3 RS?": "The 'RS' stands for 'Rennsport,' which translates to 'motorsport' in English, emphasizing the racing heritage of the model.",
    "Which iconic racing car inspired the design of the Porsche 935?": "The Porsche 935 was inspired by the 911 Turbo and became a legendary race car in the 1970s and 1980s.",
    "What is the official color of Porsche?": "The official color of Porsche is 'Porsche Racing Green.'",
    "Which technology is used in Porsche's adaptive cruise control system?": "Porsche's adaptive cruise control system uses radar sensors to monitor the road ahead and adjust speed accordingly.",
    "What is the name of the Porsche electric concept car unveiled in 2020?": "The Porsche electric concept car unveiled in 2020 is the Porsche Vision Renndienst.",
    "In which year did Porsche launch the 550 Spyder?": "Porsche launched the 550 Spyder in 1953, becoming a notable sports car of its time.",
    "Which famous racing series did the Porsche 917 participate in?": "The Porsche 917 participated in the World Sportscar Championship and the Can-Am series.",
    "What is the name of Porsche's luxury electric sedan model?": "The luxury electric sedan model by Porsche is called the Taycan.",
    "Which Porsche model holds the record for the fastest lap time on the Nürburgring Nordschleife?": "The Porsche 911 GT2 RS holds the record for the fastest lap time for a production car on the Nürburgring Nordschleife.",
    "What is the engine placement in the Porsche 911?": "The Porsche 911 features a rear-engine layout, contributing to its distinctive performance characteristics.",
    "Which iconic sports car did Porsche produce in collaboration with Volkswagen?": "Porsche collaborated with Volkswagen to produce the Porsche 914, a mid-engine sports car.",
    "What does the 'GT' stand for in Porsche 911 GT3?": "The 'GT' in Porsche 911 GT3 stands for 'Grand Touring,' indicating its high-performance and track-focused nature.",
    "What is the name of Porsche's entry-level sports car model?": "The Porsche 718 Cayman serves as the entry-level sports car model in the Porsche lineup.",
    "In what year did the Porsche 928 debut?": "The Porsche 928 made its debut in 1977 as a luxury grand tourer.",
    "What is the design philosophy behind the Porsche Mission E concept?": "The Porsche Mission E concept focuses on electric mobility with a sleek design and cutting-edge technology.",
    "Which material is used for the body construction of the Porsche 918 Spyder?": "The body construction of the Porsche 918 Spyder includes extensive use of carbon fiber reinforced polymer (CFRP).",
    "What is the significance of the 'S' in Porsche 911 Carrera S?": "The 'S' denotes 'Super' or 'Sport' in Porsche 911 Carrera S, indicating a higher level of performance compared to the standard model.",
    "Which Porsche model marked the company's return to Formula One in the 2020 season?": "The Porsche 911 GT3 Cup car marked Porsche's return to Formula One in the 2020 season.",
    "What is the maximum seating capacity of the Porsche Panamera?": "The Porsche Panamera offers seating for up to four passengers.",
    "Which famous actor was known for his love of Porsche cars, especially the 911?": "The late Paul Walker was a well-known actor and a passionate Porsche enthusiast, particularly for the 911.",
    "What is the origin of the name 'Cayenne' for the Porsche SUV model?": "The name 'Cayenne' is derived from the pepper of the same name, symbolizing the hot and spicy nature of the SUV.",
    "Which iconic racing driver had a long-standing association with Porsche, notably with the 917?": "The legendary racing driver, Jo Siffert, had a significant association with Porsche and was closely linked to the 917.",
    "What is the engine configuration of the Porsche 718 Boxster?": "The Porsche 718 Boxster features a mid-engine configuration for optimal balance and handling.",
    "Which concept car showcased Porsche's vision for an all-electric future in 2015?": "The Porsche Mission E concept, unveiled in 2015, showcased the brand's vision for an all-electric future.",
    "What is the name of Porsche's motorsport division responsible for high-performance models?": "Porsche's motorsport division is known as Porsche Motorsport GmbH.",
    "In what year did Porsche win its first overall victory at the 24 Hours of Le Mans?": "Porsche secured its first overall victory at the 24 Hours of Le Mans in 1970 with the 917 model.",
    "What unique feature does the Porsche 911 Targa have compared to other 911 models?": "The Porsche 911 Targa features a retractable roof panel, providing an open-top driving experience with a fixed roll bar.",
    "Which special edition model pays homage to the classic 911 Carrera RS 2.7?": "The Porsche 911 Carrera 4 GTS 'Rennsport Reunion Edition' pays homage to the classic 911 Carrera RS 2.7.",
    "What is the name of the hybrid supercar produced by Porsche, featuring a V8 engine and electric motors?": "The Porsche 918 Spyder is the hybrid supercar featuring a combination of a V8 engine and electric motors.",
    "Which famous automotive designer worked on the Porsche 904 Carrera GTS?": "Famed automotive designer Ferdinand Alexander 'Butzi' Porsche worked on the design of the Porsche 904 Carrera GTS.",
    "What is the maximum horsepower output of the Porsche Taycan Turbo S?": "The Porsche Taycan Turbo S boasts a maximum horsepower output of around 751 horsepower.",
    "What is the connection between Ferdinand Porsche and the Volkswagen Beetle?": "Ferdinand Porsche played a key role in the design and development of the Volkswagen Beetle.",
    "Which model is considered the 'baby Porsche' and is the smallest in the current lineup?": "The Porsche 718 Cayman is often referred to as the 'baby Porsche' and is the smallest model in the current lineup.",
    "What is the name of the limited-edition model commemorating the 70th anniversary of Porsche?": "The limited-edition model commemorating the 70th anniversary of Porsche is the Porsche 911 Speedster.",
    "Which Porsche model features a rear-wheel steering system for enhanced agility?": "The Porsche 911 Carrera 4 and 911 Turbo models feature a rear-wheel steering system for improved agility at varying speeds.",
    "What is the name of Porsche's first electric SUV model?": "The Porsche Taycan Cross Turismo is the brand's first electric SUV model.",
    "Which iconic race car inspired the design of the Porsche 550 Spyder?": "The Porsche 550 Spyder was inspired by the legendary Porsche 356 race car.",
    "What is the significance of the 'Carrera' name in Porsche models?": "The 'Carrera' name, meaning 'race' or 'career' in Spanish, was inspired by the Carrera Panamericana race and is used in various Porsche models.",
    "Which Porsche model is known for its distinctive 'whale tail' rear spoiler?": "The Porsche 911 Turbo is known for its distinctive 'whale tail' rear spoiler, enhancing aerodynamics and downforce.",
    "What is the name of the Porsche concept car that pays tribute to the classic 911 design?": "The Porsche 911 Speedster Concept pays tribute to the classic 911 design and heritage.",
    "What is the engine placement in the Porsche 718 Cayman?": "The Porsche 718 Cayman features a mid-engine layout for optimal balance and handling.",
    "Which Porsche model is known for its distinctive 'bunny ears' headlights?": "The Porsche 996 generation of the 911 is known for its distinctive 'bunny ears' headlights.",
    "What is the name of the Porsche concept car that showcases a futuristic vision for electric mobility?": "The Porsche Mission E Cross Turismo concept showcases a futuristic vision for electric mobility and adventure.",
    "Which Porsche model is known for its distinctive 'fried egg' headlights?": "The Porsche 996 generation of the 911 is known for its distinctive 'fried egg' headlights.",
    "What is the name of the Porsche concept car that showcases a futuristic vision for electric mobility?": "The Porsche Mission E Cross Turismo concept showcases a futuristic vision for electric mobility and adventure.",
}

#generating embeddings for the entire database
database_questions = list(qa_database.keys())
database_embeddings = model.encode(database_questions)

#Building index using FAISS(vector indexing)
index = faiss.IndexFlatL2(database_embeddings.shape[1])
index.add(np.array(database_embeddings).astype('float32'))

#Function to retrieve the most similar question using vector indexing
def retrieve_most_similar_question(user_question, index, database_embeddings, database_questions):
    user_embedding = model.encode(user_question).reshape(1, -1).astype('float32')
    _, most_similar_index = index.search(user_embedding, 1)
    most_similar_question = database_questions[most_similar_index[0][0]]
    return most_similar_question


# Streamlit app hosting
st.title("Hello, I am Carerra 🚗, Porsche Museum's AI assistant :)")

# User inputing section 
user_question = st.text_input("How can I help you today?")


if user_question:
    # Retrieve the most similar question from the database using vector indexing
    most_similar_question = retrieve_most_similar_question(user_question, index, database_embeddings, database_questions)

    # Retrieve the answer from the database or generate a new answer
    if most_similar_question in qa_database:
        answer = qa_database[most_similar_question]
    else:
        # If not found in the database, generate an answer using a transformer-based language model
        answer = answer_generator(user_question, max_length=100, num_return_sequences=1, top_k=50)[0]['generated_text']

    # Display the answer
    st.subheader("Carerra 🚗")
    st.write(answer)


# Improve UI
st.markdown("---")
st.markdown("**Welcome to Porsche Museum, Stuttgart!**")
st.markdown("Your Porsche adventure awaits you")
st.markdown("---")

'''
**Disclaimer:**

*Carerra 🚗, the Porsche Museum, and Porsche are registered trademarks owned by their respective owners. This document and the associated project are not affiliated with, endorsed by, or sponsored by the trademark owners. Any use of trademark names is for illustrative purposes only, and all rights to these trademarks are explicitly acknowledged.*

*The information provided in this document and associated code is for educational and demonstration purposes. Users should be aware of and adhere to the trademark policies and guidelines set forth by the respective trademark owners. Any reference to specific trademarks is not intended to imply a partnership, endorsement, or association with the trademark owners.*

*It is the responsibility of users to ensure compliance with trademark laws and regulations when using or referring to trademarks mentioned in this document.*
'''

