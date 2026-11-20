import faiss
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pathlib import Path

#load pdfs
def load_pdf(pdf_path):
  reader=PdfReader(str(pdf_path))
  mypages=[i.extract_text() or "" for i in reader.pages]
  full_text= "\n".join(mypages).strip()
  if not full_text:
    raise ValueError("No readable text found")
  return full_text

#Chunking
def create_chunks(full_text, chunksize=500, overlap=100):
  chunks=[]
  start=0
  while start<len(full_text):
    end=start+chunksize
    chunk=full_text[start:end]
    if chunk.strip():
      chunks.append(chunk)
    start=end-overlap
  return chunks

#Embedding
def build_vectors(chunks, model_name="all-MiniLM-L6-v2"):
  if not chunks:
    raise ValueError('Cannot apply embedding without chunks')
  embed_model=SentenceTransformer(model_name)
  embed_vectors=embed_model.encode(chunks, convert_to_numpy=True)
  faiss.normalize_L2(embed_vectors)
  index=faiss.IndexFlatIP(embed_vectors.shape[1])
  index.add(embed_vectors)
  return index, embed_model

#call loadpdf,chunk,embedding
def load_base(pdf_path= "college_docs"):
  chunks = []
  for pdf in Path(pdf_path).glob("*pdf"):
    full_text = load_pdf(pdf)
    chunks.extend(create_chunks(full_text))
  index, embed_model=build_vectors(chunks)
  return index, embed_model, chunks

#retrieve
def retrieve_content(question, embed_model, chunks, index, num_chunks=3):
  if not question.strip():
    raise ValueError("Question cannot be blank")
  if num_chunks<1:
    raise ValueError("Request atleast one or more chunks")
  query_embed=embed_model.encode([question], convert_to_numpy=True)
  faiss.normalize_L2(query_embed)
  sim_scores, vec_indices = index.search(query_embed,min(num_chunks, len(chunks)))
  background=""
  for i in vec_indices[0]:
    if i != -1:
      background += chunks[i] + "\n\n"
  return background  

#Ask
def ask_rag(question, embed_model, chunks, index, api_key, num_chunks=3, model_name="openai/gpt-oss-120b"):
  key=api_key
  if not key:
    raise ValueError("API key missing")
  background=retrieve_content(question, embed_model, chunks, index, num_chunks=3)
  prompt="""You are an professional AI assistant.
  Answer user question using ONLY document background below.
  Do not invent or assume facts.
  If information not available, simply say Information Missing.
  Document Background:""" + background + "User Question" + question + "Answer:" 
  client=Groq(api_key = key)
  response=client.chat.completions.create(
    model=model_name,
    messages=[{"role":"user",
               "content":prompt}],
              temperature=0.3,
  )
  return response.choices[0].message.content import streamlit as st
from rag_engine import load_base, ask_rag
st.set_page_config(page_title="SIET_Chatbot", layout="centered")
st.title("SHRIDEVI INSTITUTE OF ENGINEERING AND TECHNOLOGY")
st.caption("Ask anything about the College Information, Courses,FAQs,Rules and regulation,Placements...")
st.caption("RAG Pipeline with multiple files")
st.caption("Created by Chandana Lokesh")

mykey=st.secrets["GROQ_API_KEY"]

@st.cache_resource
def initiate_rag():
  return load_base("college_docs")
  
index, embed_model, chunks= initiate_rag()

if "chat_hist" not in st.session_state:
  st.session_state.chat_hist = []

for qa in st.session_state.chat_hist:
  with st.chat_message(qa['role']):
    st.write(qa['content'])

question=st.chat_input("Ask anything about SIET College...")

if question:
  st.session_state.chat_hist.append(
    {"role":"user",
     "content":question})
  with st.chat_message("user"):
    st.write(question)
  with st.chat_message("assistant"):
    with st.spinner("Generating answer..."):
      answer=ask_rag(question, embed_model, chunks, index, api_key=mykey)
      st.write(answer)
      st.session_state.chat_hist.append(
       {"role":"assistant",
        "content":answer})

if st.button("Clear chat"):
  st.session_state.chat_hist=[]
  st.rerun()
