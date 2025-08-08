import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from typing import Optional
import re

class LLMService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🤖 Using device: {self.device}")
        
        self.generator = None
        self.model_name = None
        self.model_type = None
        
        # Try multiple models in order of preference (best to simplest)
        models_to_try = [
            ("google/flan-t5-base", "seq2seq"),      # Best for QA tasks
            ("google/flan-t5-small", "seq2seq"),     # Lighter T5
            ("distilgpt2", "causal"),                # Reliable GPT-2 variant
            ("gpt2", "causal"),                      # Classic GPT-2
            ("microsoft/DialoGPT-small", "causal")   # Fallback conversational
        ]
        
        for model_name, model_type in models_to_try:
            if self._load_model(model_name, model_type):
                break
        
        if not self.generator:
            print("⚠️ No models loaded successfully, using enhanced rule-based fallback")
    
    def _load_model(self, model_name: str, model_type: str) -> bool:
        """Try to load a specific model"""
        try:
            print(f"🔄 Attempting to load {model_name}...")
            
            if model_type == "seq2seq":
                # T5-family models are excellent for QA tasks
                self.generator = pipeline(
                    "text2text-generation",
                    model=model_name,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    max_length=512,
                    do_sample=False,  # Deterministic for consistency
                    model_kwargs={"low_cpu_mem_usage": True}
                )
            else:
                # GPT-family models
                self.generator = pipeline(
                    "text-generation",
                    model=model_name,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=50256,
                    model_kwargs={"low_cpu_mem_usage": True}
                )
            
            # Test the model with a simple example
            test_result = self._test_model()
            if not test_result:
                print(f"⚠️ Model {model_name} loaded but failed test")
                return False
            
            self.model_name = model_name
            self.model_type = model_type
            print(f"✅ Successfully loaded and tested {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {str(e)[:100]}...")
            return False
    
    def _test_model(self) -> bool:
        """Test if the loaded model works"""
        try:
            if self.model_type == "seq2seq":
                result = self.generator("What is 2+2?", max_length=50)
                return len(result[0]['generated_text']) > 0
            else:
                result = self.generator("Hello", max_new_tokens=5, num_return_sequences=1)
                return len(result[0]['generated_text']) > len("Hello")
        except:
            return False
    
    def generate_answer(self, question: str, context: str, max_length: int = 512) -> str:
        """Generate an answer based on the question and context"""
        
        # Always have a fallback ready
        fallback_answer = self._enhanced_rule_based_answer(question, context)
        
        if not self.generator:
            return fallback_answer
        
        try:
            # Truncate context if too long to prevent token limit issues
            max_context_length = 800 if self.model_type == "seq2seq" else 600
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
            
            if self.model_type == "seq2seq":
                # T5-style models work better with instruction-style prompts
                prompt = f"""Answer the question based only on the given context. Be specific and concise.

Context: {context}

Question: {question}

Answer:"""
                
                result = self.generator(
                    prompt, 
                    max_length=200, 
                    num_return_sequences=1,
                    repetition_penalty=1.2,
                    early_stopping=True
                )
                answer = result[0]['generated_text'].strip()
                
            else:
                # GPT-style models
                prompt = f"""Based on the following context, answer the question clearly and concisely.

Context: {context}

Question: {question}
Answer:"""
                
                result = self.generator(
                    prompt, 
                    max_new_tokens=120, 
                    num_return_sequences=1,
                    repetition_penalty=1.3,
                    temperature=0.7,
                    do_sample=True
                )
                
                # Extract only the new generated text
                full_text = result[0]['generated_text']
                if prompt in full_text:
                    answer = full_text.replace(prompt, "").strip()
                else:
                    answer = full_text[len(prompt):].strip()
            
            # Clean and validate the answer
            cleaned_answer = self._clean_and_validate_answer(answer)
            
            # If model answer is poor, use rule-based fallback
            if len(cleaned_answer) < 10 or self._is_generic_response(cleaned_answer):
                print("🔄 Model answer too generic, using rule-based fallback")
                return fallback_answer
            
            return cleaned_answer
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return fallback_answer
    
    def _clean_and_validate_answer(self, answer: str) -> str:
        """Clean up and validate the generated answer"""
        
        # Remove common artifacts and clean up
        answer = re.sub(r'\n+', ' ', answer)  # Replace multiple newlines
        answer = re.sub(r'\s+', ' ', answer)  # Clean up whitespace
        answer = answer.strip()
        
        # Remove common prefixes that models sometimes add
        prefixes_to_remove = [
            "Answer:", "A:", "Based on the context,", "According to the text,",
            "The answer is:", "Response:", "Reply:"
        ]
        
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        
        # Remove if answer is too short or just punctuation
        if len(answer) < 5 or answer in ['', '.', '?', '!']:
            return ""
        
        # Limit length to prevent overly long responses
        if len(answer) > 400:
            # Try to cut at sentence boundary
            sentences = answer.split('.')
            truncated = '.'.join(sentences[:-1])
            if len(truncated) > 200:
                answer = truncated + '.'
            else:
                answer = answer[:400] + "..."
        
        return answer
    
    def _is_generic_response(self, answer: str) -> bool:
        """Check if the response is too generic or unhelpful"""
        answer_lower = answer.lower()
        
        generic_responses = [
            "i don't know", "i'm not sure", "i cannot", "i can't",
            "sorry", "unclear", "not mentioned", "not specified",
            "based on the context", "according to the text",
            "the context", "the document", "it depends"
        ]
        
        # Check if answer is mostly generic phrases
        generic_count = sum(1 for phrase in generic_responses if phrase in answer_lower)
        word_count = len(answer.split())
        
        return generic_count > 0 and word_count < 15
    
    def _enhanced_rule_based_answer(self, question: str, context: str) -> str:
        """Enhanced rule-based fallback with better sentence matching"""
        
        try:
            # Clean and prepare text
            question_lower = question.lower().strip()
            context = context.strip()
            
            if not context:
                return "No context provided to answer the question."
            
            # Extract meaningful question words (remove stop words)
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were',
                'what', 'how', 'when', 'where', 'why', 'who', 'which', 'that',
                'this', 'these', 'those', 'do', 'does', 'did', 'will', 'would',
                'could', 'should', 'can', 'may', 'might', 'must'
            }
            
            question_words = set()
            for word in re.findall(r'\b\w+\b', question_lower):
                if len(word) > 2 and word not in stop_words:
                    question_words.add(word)
            
            # If no meaningful words found, try to extract key concepts
            if not question_words:
                question_words = set(re.findall(r'\b\w{4,}\b', question_lower))
            
            # Split context into sentences
            sentences = []
            for sent in re.split(r'[.!?]+', context):
                cleaned_sent = sent.strip()
                if len(cleaned_sent) > 15:  # Ignore very short fragments
                    sentences.append(cleaned_sent)
            
            if not sentences:
                return "The context doesn't contain clear sentences to answer from."
            
            # Score sentences based on word overlap and position
            scored_sentences = []
            
            for i, sentence in enumerate(sentences):
                sentence_lower = sentence.lower()
                sentence_words = set(re.findall(r'\b\w+\b', sentence_lower))
                
                # Calculate overlap score
                overlap = len(question_words.intersection(sentence_words))
                
                # Add position bonus (earlier sentences might be more relevant)
                position_bonus = max(0, 1 - (i / len(sentences)) * 0.3)
                
                # Add length bonus for substantial sentences
                length_bonus = min(0.2, len(sentence) / 500)
                
                total_score = overlap + position_bonus + length_bonus
                
                if overlap > 0:  # Only consider sentences with some keyword overlap
                    scored_sentences.append((sentence, total_score, overlap))
            
            if not scored_sentences:
                # No keyword matches, try substring matching
                for sentence in sentences[:3]:  # Check first few sentences
                    for word in question_words:
                        if word in sentence.lower():
                            scored_sentences.append((sentence, 0.5, 1))
                            break
            
            if scored_sentences:
                # Sort by score and take the best match(es)
                scored_sentences.sort(key=lambda x: x[1], reverse=True)
                
                # If top score is significantly higher, use just that sentence
                top_sentence = scored_sentences[0]
                
                if len(scored_sentences) > 1 and top_sentence[1] > scored_sentences[1][1] * 1.5:
                    answer = top_sentence[0]
                else:
                    # Combine top 2 sentences if they're close in score
                    top_sentences = [s[0] for s in scored_sentences[:2]]
                    answer = '. '.join(top_sentences)
                    if not answer.endswith('.'):
                        answer += '.'
                
                # Ensure reasonable length
                if len(answer) > 350:
                    answer = answer[:350] + "..."
                
                return answer
            
            # Last resort: return first substantial sentence
            for sentence in sentences[:3]:
                if len(sentence) > 30:
                    return sentence + ("." if not sentence.endswith('.') else "")
            
            return "I found information in the document but couldn't identify the most relevant part to answer your specific question."
            
        except Exception as e:
            print(f"Error in rule-based answer: {e}")
            return "I found relevant information but encountered an error while processing it. Please try rephrasing your question."
    
    def is_available(self) -> bool:
        """Check if the LLM service is available"""
        return self.generator is not None
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": self.device,
            "available": self.is_available()
        }

# Keep the SimpleQAService class as a backup (unchanged from your original)
class SimpleQAService:
    """A simple rule-based QA service as ultimate fallback"""
    
    def __init__(self):
        self.question_patterns = {
            'what': ['definition', 'description', 'explanation'],
            'when': ['date', 'time', 'year', 'month'],
            'where': ['location', 'place', 'address'],
            'who': ['person', 'people', 'author', 'name'],
            'how': ['method', 'process', 'way', 'procedure'],
            'why': ['reason', 'cause', 'purpose']
        }
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate simple extractive answer"""
        
        # Find the most relevant sentences
        sentences = re.split(r'[.!?]+', context)
        question_lower = question.lower()
        
        # Extract question type
        question_type = None
        for qtype in self.question_patterns:
            if qtype in question_lower:
                question_type = qtype
                break
        
        # Score sentences based on keyword matching
        best_sentences = []
        question_keywords = set(re.findall(r'\b\w+\b', question_lower))
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue
                
            sentence_lower = sentence.lower()
            sentence_keywords = set(re.findall(r'\b\w+\b', sentence_lower))
            
            # Calculate overlap
            overlap = len(question_keywords.intersection(sentence_keywords))
            
            if overlap >= 2:  # At least 2 keywords match
                best_sentences.append((sentence.strip(), overlap))
        
        if best_sentences:
            # Sort by relevance
            best_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Return top sentence(s)
            top_sentence = best_sentences[0][0]
            
            if len(top_sentence) > 400:
                top_sentence = top_sentence[:400] + "..."
            
            return top_sentence
        
        return "I found information related to your question in the document, but couldn't extract a specific answer. Please try a more specific question."