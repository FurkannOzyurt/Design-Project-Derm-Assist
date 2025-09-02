import torch
from PIL import Image
import torchvision.transforms as transforms
import os
from src.classifier import ConvNeXtTinyClassifier
from src.llm import MedicalLLMHelper
from src.rag import RetrievalAugmentedGeneration
from pathlib import Path
import glob

class DermatologyAssistant:
    def __init__(self):
        # Initialize classifier
        self.classifier = self._load_classifier()
        
        # Initialize LLM
        self.llm = self._load_llm()
        
        # Initialize RAG
        self.rag = self._load_rag()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def build_class_list(self, dataset_root: str = "dataset/train") -> list[str]:
        """
        dataset/train/<class_name>/**  dizin yapısından sınıf adlarını alır
        """
        class_dirs = sorted(
            [Path(p).name for p in glob.glob(os.path.join(dataset_root, "*")) if os.path.isdir(p)]
        )
        if not class_dirs:
            raise RuntimeError(f"No class folders found in {dataset_root}")
        return class_dirs
    
    def _load_classifier(self):
        # Load the classifier with your trained weights
        weights_path = os.path.join('src', 'convnext_tiny_dermatology_best.pt')
        class_names = self.build_class_list("dataset/train")
        return ConvNeXtTinyClassifier(
            weights_path=weights_path,
            class_names=class_names,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def _load_llm(self):
        return MedicalLLMHelper(
            model="hf.co/unsloth/medgemma-4b-it-GGUF:Q8_K_XL",
            host=" ",
            temperature=0.6,
            top_p=0.9
        )
    
    def _load_rag(self):
        return RetrievalAugmentedGeneration(
            helper=self.llm,
            qa_json_path="dataset/dermatology_qa.json",
            encoder_name="dmis-lab/biobert-base-cased-v1.1",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def predict_image(self, image_path):
        """Predict the class of a dermatology image."""
        try:
            prediction = self.classifier.predict(image_path)
            return prediction
        except Exception as e:
            print(f"Error in image prediction: {e}")
            return "Error in image classification"
    
    def generate_response(self, image_class, user_message, rag_context=None):
        """Generate a response using the LLM with optional RAG context."""
        try:
            # If no image class is provided, use a default response
            if image_class is None:
                return "I notice you haven't uploaded an image yet. To provide the most accurate medical advice, please upload a clear image of the affected area. Once you do, I can analyze it and provide specific guidance about your condition."
            
            # If image class is "Error in image classification"
            if image_class == "Error in image classification":
                return "I apologize, but I had trouble analyzing your image. Could you please try uploading a clearer image? Make sure the affected area is well-lit and in focus."
            
            # Get RAG context if not provided
            if rag_context is None:
                rag_context = self.rag.retrieve(image_class, user_message)
            
            if rag_context is None:
                return "I apologize, but I couldn't find relevant information for your question. Could you please rephrase your question or provide more details about your concern?"
            
            response = self.llm.generate_answer(
                disease_name=image_class,
                user_q=user_message,
                rag_out=rag_context
            )
            return response
        except Exception as e:
            print(f"Error in response generation: {e}")
            return "I apologize, but I encountered an error while generating the response."
    
    def get_rag_context(self, user_message, image_class):
        """Retrieve relevant context for RAG."""
        try:
            return self.rag.retrieve(image_class, user_message)
        except Exception as e:
            print(f"Error in RAG context retrieval: {e}")
            return None

# Create a singleton instance
assistant = DermatologyAssistant() 