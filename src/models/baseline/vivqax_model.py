import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional

class ViVQAX_Model(nn.Module):
    """
    ViVQAX_Model: A model for Visual Question Answering with explanation generation.
    """
    def __init__(self, 
                vocab_size: int,
                embed_size: int,
                hidden_size: int,
                num_layers: int,
                num_answers: int,
                max_explanation_length: int,
                word2idx: dict,
                dropout: float = 0.5):
        super().__init__()
        
        self.word2idx = word2idx
        self.max_explanation_length = max_explanation_length
        self.vocab_size = vocab_size
        
        # Image Encoder (ResNet-50)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False
        
        # Project image features to hidden size
        self.image_projection = nn.Sequential(
            nn.Linear(resnet.fc.in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Question Encoder
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.question_lstm = nn.LSTM(
            embed_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Multimodal Fusion
        fusion_size = hidden_size * 3  # image + bidirectional question
        self.fusion = nn.Sequential(
            nn.Linear(fusion_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Answer Prediction
        self.answer_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_answers)
        )
        
        # Explanation Generator
        self.explanation_lstm = nn.LSTM(
            embed_size + num_answers + hidden_size,  # word embedding + answer distribution + context
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.explanation_output = nn.Linear(hidden_size, vocab_size)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Extract and project image features."""
        with torch.no_grad():
            features = self.resnet(image)
        features = features.squeeze(-1).squeeze(-1)
        return self.image_projection(features)

    def encode_question(self, question: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode question sequence."""
        embedded = self.embedding(question)
        outputs, (hidden, cell) = self.question_lstm(embedded)
        # Combine bidirectional states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return outputs, hidden

    def forward(self, 
                image: torch.Tensor,
                question: torch.Tensor,
                target_explanation: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            image: Input image tensor [batch_size, channels, height, width]
            question: Question token indices [batch_size, question_length]
            target_explanation: Optional target explanation for training [batch_size, max_length]
            teacher_forcing_ratio: Probability of using teacher forcing during training
            
        Returns:
            Tuple of answer logits and explanation token logits
        """
        batch_size = image.size(0)
        device = image.device
        
        # Encode inputs
        img_features = self.encode_image(image)
        question_outputs, question_hidden = self.encode_question(question)
        
        # Fuse multimodal features
        fused = self.fusion(torch.cat([img_features, question_hidden], dim=1))
        
        # Predict answer
        answer_logits = self.answer_classifier(fused)
        answer_probs = F.softmax(answer_logits, dim=1)
        
        # Initialize explanation generation
        explanation_hidden = None
        decoder_input = torch.tensor([[self.word2idx['<START>']]] * batch_size, device=device)
        decoder_context = fused.unsqueeze(1).repeat(1, 1, 1)
        
        explanation_outputs = []
        max_length = self.max_explanation_length if target_explanation is None else target_explanation.size(1)
        
        # Generate explanation tokens
        for t in range(max_length - 1):
            decoder_embedding = self.embedding(decoder_input)
            decoder_input_combined = torch.cat([
                decoder_embedding,
                answer_probs.unsqueeze(1),
                decoder_context
            ], dim=2)
            
            output, explanation_hidden = self.explanation_lstm(
                decoder_input_combined, 
                explanation_hidden
            )
            output = self.explanation_output(output)
            explanation_outputs.append(output)
            
            # Teacher forcing
            if target_explanation is not None and torch.rand(1) < teacher_forcing_ratio:
                decoder_input = target_explanation[:, t:t+1]
            else:
                decoder_input = output.argmax(2)
        
        explanation_outputs = torch.cat(explanation_outputs, dim=1)
        return answer_logits, explanation_outputs

    def generate_explanation(self, 
                           image: torch.Tensor,
                           question: torch.Tensor,
                           max_length: Optional[int] = None,
                           beam_size: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate explanation using beam search.
        
        Args:
            image: Input image tensor
            question: Question token indices
            max_length: Maximum explanation length (optional)
            beam_size: Beam size for beam search
            
        Returns:
            Tuple of answer logits and generated explanation indices
        """
        batch_size = image.size(0)
        device = image.device
        max_length = max_length or self.max_explanation_length
        
        # Encode and get answer
        img_features = self.encode_image(image)
        question_outputs, question_hidden = self.encode_question(question)
        fused = self.fusion(torch.cat([img_features, question_hidden], dim=1))
        answer_logits = self.answer_classifier(fused)
        answer_probs = F.softmax(answer_logits, dim=1)
        
        # Initialize beams for each batch item
        beams = [[(0.0, [self.word2idx['<START>']], None, None)] for _ in range(batch_size)]
        
        # Beam search
        for _ in range(max_length - 1):
            new_beams = [[] for _ in range(batch_size)]
            
            for i in range(batch_size):
                candidates = []
                for score, seq, hidden_h, hidden_c in beams[i]:
                    if seq[-1] == self.word2idx['<END>']:
                        new_beams[i].append((score, seq, hidden_h, hidden_c))
                        continue
                    
                    # Prepare decoder input
                    decoder_input = torch.tensor([seq[-1]], device=device)
                    decoder_embedding = self.embedding(decoder_input)
                    decoder_context = fused[i:i+1].unsqueeze(1)
                    
                    decoder_input_combined = torch.cat([
                        decoder_embedding.unsqueeze(0),
                        answer_probs[i:i+1].unsqueeze(1),
                        decoder_context
                    ], dim=2)
                    
                    # Get next token probabilities
                    if hidden_h is None:
                        output, (hidden_h, hidden_c) = self.explanation_lstm(decoder_input_combined)
                    else:
                        output, (hidden_h, hidden_c) = self.explanation_lstm(
                            decoder_input_combined,
                            (hidden_h, hidden_c)
                        )
                    
                    logits = self.explanation_output(output.squeeze(1))
                    probs = F.log_softmax(logits, dim=-1)
                    
                    # Add top-k candidates
                    topk_probs, topk_indices = probs.topk(beam_size)
                    for prob, idx in zip(topk_probs[0], topk_indices[0]):
                        candidates.append((
                            score + prob.item(),
                            seq + [idx.item()],
                            hidden_h,
                            hidden_c
                        ))
                
                # Select top beam_size candidates
                candidates.sort(key=lambda x: x[0], reverse=True)
                new_beams[i] = candidates[:beam_size]
            
            beams = new_beams
        
        # Select best sequence from each beam
        generated_explanations = []
        for beam in beams:
            if not beam:
                generated_explanations.append(torch.tensor([self.word2idx['<PAD>']], device=device))
            else:
                best_seq = max(beam, key=lambda x: x[0])[1]
                generated_explanations.append(torch.tensor(best_seq, device=device))
        
        # Pad sequences
        generated_explanations = torch.nn.utils.rnn.pad_sequence(
            generated_explanations,
            batch_first=True,
            padding_value=self.word2idx['<PAD>']
        )
        
        return answer_logits, generated_explanations