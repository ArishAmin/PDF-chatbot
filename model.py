import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
import re
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import gc

nltk.download('punkt')

class PDFChatbotHF:
    def __init__(self):
        
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small").to('cpu')
        
        self.mpnet = SentenceTransformer('all-mpnet-base-v2')
        
        self.sentences = []
        self.embeddings = None
        self.index = None
        self.context_window = 3  
        self.overlap_threshold = 0.75  
    
    def preprocess_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)  
        text = re.sub(r'[^a-zA-Z0-9\s\.,?!]', '', text)  
        return text.strip()
    
    def segment_into_sentences(self, text: str) -> list:
        sentences = sent_tokenize(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]  
        return sentences
    
    def extract_text_from_pdf(self, pdf_path: str):
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            text = self.preprocess_text(text)
            self.sentences = self.segment_into_sentences(text)
            
            if not self.sentences:
                raise ValueError("No valid sentences found in the PDF")
            
            
            self.embeddings = self.mpnet.encode(self.sentences, 
                                              convert_to_tensor=True,
                                              normalize_embeddings=True)
            
            
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings.cpu().numpy())
    
    def get_context_window(self, index: int) -> list:
        if index >= len(self.sentences):
            return []
            
        start = max(0, index - self.context_window)
        end = min(len(self.sentences), index + self.context_window + 1)
        
        context = []
        for i in range(start, end):
            context.append(self.sentences[i])
        return context
    
    def rank_responses(self, question_embedding: torch.Tensor, candidate_indices: np.ndarray) -> list:
        valid_indices = [idx for idx in candidate_indices if 0 <= idx < len(self.sentences)]
        if not valid_indices:
            return []
            
        candidate_embeddings = self.embeddings[valid_indices]
        
        
        similarities = torch.matmul(
            question_embedding,
            candidate_embeddings.t()
        ).cpu().numpy()
        
        relevance_scores = []
        for idx, sim in zip(valid_indices, similarities):
            length_factor = min(1.0, len(self.sentences[idx].split()) / 50.0)
            words = self.sentences[idx].lower().split()
            info_density = len(set(words)) / len(words) if words else 0
            final_score = sim * 0.8 + length_factor * 0.1 + info_density * 0.1  # Adjusted weights
            relevance_scores.append((idx, sim, final_score))
        
        
        relevance_scores.sort(key=lambda x: x[2], reverse=True)
        return relevance_scores
    
    def generate_response(self, context: str, question: str) -> str:
        
        input_text = f"Answer the following question based on the given context.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to('cpu')
        
        try:
            
            outputs = self.model.generate(
                input_ids, 
                max_length=512,  
                num_beams=5,  
                temperature=0.4,  
                top_p=0.95,
                early_stopping=True
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"
        finally:
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def get_response(self, question: str, k: int = 5, similarity_threshold: float = 0.6) -> str:
        if not self.index or not self.sentences:
            return "Please load a PDF first."
        
        try:
            
            question_embedding = self.mpnet.encode(question, 
                                               convert_to_tensor=True,
                                               normalize_embeddings=True)
            
            
            k = min(k, len(self.sentences))
            distances, indices = self.index.search(
                question_embedding.cpu().numpy().reshape(1, -1),
                k * 5  
            )

            
            print(f"indices: {indices}")
            print(f"distances: {distances}")
            print(f"number of sentences: {len(self.sentences)}")
            print(f"FAISS index size: {self.index.ntotal}")

            
            valid_indices = [idx for idx in indices[0] if 0 <= idx < len(self.sentences)]

            if not valid_indices:
                return "I couldn't find any relevant content in the document."

            
            relevant_sentences = [
                self.sentences[idx]
                for idx in valid_indices
                if distances[0][valid_indices.index(idx)] > similarity_threshold
            ]
            context = " ".join(relevant_sentences[:k])

            if not context.strip():
                return "I couldn't find a relevant answer to your question in the document."

            
            input_text = f"Answering the question based on the: {context}. Question: {question}"
            inputs = self.tokenizer.encode(input_text, return_tensors="pt")
            outputs = self.model.generate(
                inputs,
                max_length=512,
                num_beams=5,
                temperature=0.7,
                top_p=0.9
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response

        except Exception as e:
            return f"An error occurred: {str(e)}"
        finally:
            
            gc.collect()

