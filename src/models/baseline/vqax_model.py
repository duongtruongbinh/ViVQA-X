import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VQAX_Model(nn.Module):
    def __init__(self, 
                 vocab_size,
                 embed_size,
                 hidden_size,
                 num_layers,
                 num_answers,
                 max_explanation_length,
                 word2idx):
        super(VQAX_Model, self).__init__()
        
        self.word2idx = word2idx
        self.max_explanation_length = max_explanation_length
        # Image feature extraction
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False
        
        self.image_projection = nn.Linear(resnet.fc.in_features, hidden_size)
        
        # Question encoding
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.question_lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Fusion and answer prediction
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_answer = nn.Linear(hidden_size, num_answers)
        
        # Explanation generation
        self.explanation_lstm = nn.LSTM(embed_size + num_answers, hidden_size, num_layers, batch_first=True)
        self.fc_explanation = nn.Linear(hidden_size, vocab_size)
        
        self.max_explanation_length = max_explanation_length
        self.vocab_size = vocab_size

    def forward(self, image, question, target_explanation=None, teacher_forcing_ratio=0.5):
        # Image features
        img_features = self.resnet(image).squeeze(-1).squeeze(-1)
        img_features = self.image_projection(img_features)
        
        # Question features
        embedded_question = self.embedding(question)
        _, (question_features, _) = self.question_lstm(embedded_question)
        question_features = question_features[-1]  # Use the last hidden state
        
        # Fusion
        combined = torch.cat((img_features, question_features), dim=1)
        fused = torch.relu(self.fusion(combined))
        
        # Answer prediction
        answer_logits = self.fc_answer(fused)
        answer_probs = torch.softmax(answer_logits, dim=1)
        
        # Explanation generation
        batch_size = image.size(0)
        explanation_outputs = []
        
        input_token = torch.full((batch_size, 1), 2, device=image.device)  # Start token
        hidden = (fused.unsqueeze(0).repeat(self.question_lstm.num_layers, 1, 1),
                  torch.zeros_like(fused).unsqueeze(0).repeat(self.question_lstm.num_layers, 1, 1))
        
        for t in range(self.max_explanation_length - 1):
            explanation_input = torch.cat([self.embedding(input_token), answer_probs.unsqueeze(1)], dim=2)
            output, hidden = self.explanation_lstm(explanation_input, hidden)
            explanation_logits = self.fc_explanation(output.squeeze(1))
            explanation_outputs.append(explanation_logits)
            
            if target_explanation is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = target_explanation[:, t+1].unsqueeze(1)
            else:
                input_token = explanation_logits.argmax(dim=1, keepdim=True)
        
        explanation_outputs = torch.stack(explanation_outputs, dim=1)
        
        return answer_logits, explanation_outputs

    def generate_explanation(self, image, question, max_length=None, beam_size=3):
        if max_length is None:
            max_length = self.max_explanation_length

        with torch.no_grad():
            # Image features
            img_features = self.resnet(image).squeeze(-1).squeeze(-1)
            img_features = self.image_projection(img_features)

            # Question features
            embedded_question = self.embedding(question)
            _, (question_features, _) = self.question_lstm(embedded_question)
            question_features = question_features[-1]  # Use the last hidden state

            # Fusion
            combined = torch.cat((img_features, question_features), dim=1)
            fused = torch.relu(self.fusion(combined))

            # Answer prediction
            answer_logits = self.fc_answer(fused)
            answer_probs = torch.softmax(answer_logits, dim=1)

            # Explanation generation with beam search
            batch_size = image.size(0)

            # Initialize beams for each item in the batch
            beams = [[(0.0, [self.word2idx['<START>']], 
                       fused[i].unsqueeze(0).repeat(self.question_lstm.num_layers, 1, 1),
                       torch.zeros_like(fused[i]).unsqueeze(0).repeat(self.question_lstm.num_layers, 1, 1))] 
                     for i in range(batch_size)]

            for _ in range(max_length - 1):
                new_beams = [[] for _ in range(batch_size)]

                for i in range(batch_size):
                    candidates = []
                    for score, seq, hidden_h, hidden_c in beams[i]:
                        if seq[-1] == self.word2idx['<END>']:
                            new_beams[i].append((score, seq, hidden_h, hidden_c))
                            continue

                        last_token = torch.LongTensor([seq[-1]]).to(image.device)
                        explanation_input = torch.cat([self.embedding(last_token), answer_probs[i].unsqueeze(0)], dim=1)

                        output, (new_hidden_h, new_hidden_c) = self.explanation_lstm(explanation_input.unsqueeze(1), 
                                                                                     (hidden_h, hidden_c))
                        logits = self.fc_explanation(output.squeeze(1))
                        probs = F.log_softmax(logits, dim=-1)

                        top_probs, top_indices = probs.topk(beam_size)
                        for prob, idx in zip(top_probs.squeeze(), top_indices.squeeze()):
                            new_score = score + prob.item()
                            new_seq = seq + [idx.item()]
                            candidates.append((new_score, new_seq, new_hidden_h, new_hidden_c))

                    # Select top beam_size candidates
                    new_beams[i] = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]

                beams = new_beams

            # Select the best beam for each item in the batch
            generated_explanations = []
            for beam in beams:
                if beam:  # Check if the beam is not empty
                    best_seq = max(beam, key=lambda x: x[0])[1]
                    generated_explanations.append(torch.LongTensor(best_seq))
                else:
                    # Handle the case where no valid sequences were generated
                    generated_explanations.append(torch.LongTensor([self.word2idx['<START>'], self.word2idx['<END>']]))

            generated_explanations = torch.nn.utils.rnn.pad_sequence(generated_explanations, batch_first=True, padding_value=self.word2idx['<PAD>'])

        return answer_logits, generated_explanations


if __name__ == "__main__":
    # Hyperparameters
    vocab_size = len(word2idx)
    num_answers = len(answer2idx)
    embed_size = 400
    hidden_size = 2048
    num_layers = 3
    max_explanation_length = 15

    model = VQAX_Model(vocab_size, embed_size, hidden_size, num_layers, num_answers, max_explanation_length, word2idx)
    print(model)