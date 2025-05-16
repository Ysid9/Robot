import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random

class PioneerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PioneerNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Définir les couches
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
        # Fonction d'activation (tanh comme dans l'implémentation originale)
        self.activation = nn.Tanh()
        
        # Initialisation des poids avec une distribution uniforme [-1, 1]
        # comme dans l'implémentation originale
        nn.init.uniform_(self.hidden.weight, -1.0, 1.0)
        nn.init.uniform_(self.output.weight, -1.0, 1.0)
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.output.bias)
        
    def forward(self, x):
        # Conversion en tensor PyTorch si nécessaire
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            
        # Passage à travers les couches
        x = self.activation(self.hidden(x))
        x = self.activation(self.output(x))
        return x
    
    def run_nn(self, inputs):
        """Méthode compatible avec l'interface originale"""
        with torch.no_grad():
            x = torch.tensor(inputs, dtype=torch.float32)
            outputs = self.forward(x)
            return outputs.tolist()
    
    def backpropagate(self, grad, learning_rate, momentum):
        """
        Méthode compatible avec l'interface originale pour la rétropropagation
        du gradient fourni par le calcul externe du critère
        """
        # Conversion du gradient en tensor PyTorch
        grad_tensor = torch.tensor(grad, dtype=torch.float32)
        
        # Le gradient est déjà calculé par rapport aux sorties du réseau
        # Nous devons maintenant rétropropager ce gradient à travers le réseau
        
        # Obtenir les paramètres du modèle
        parameters = list(self.parameters())
        
        # Créer des gradients artificiels pour les sorties
        # Les sorties du modèle sont stockées pendant le forward pass
        def hook_fn(module, grad_input, grad_output):
            module.grad_output = grad_output
            
        # Le hook nous permettra de capturer les gradients
        hooks = []
        for module in self.modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_full_backward_hook(hook_fn))
        
        # Effectuer un forward pass pour obtenir les sorties
        dummy_input = torch.tensor(self.last_inputs, dtype=torch.float32, requires_grad=True)
        outputs = self.forward(dummy_input)
        
        # Rétropropager le gradient fourni
        # En remplaçant le gradient normal par notre gradient externe
        outputs.backward(gradient=grad_tensor)
        
        # Mettre à jour les poids manuellement avec le taux d'apprentissage
        # et le momentum fournis
        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    param_state = self.optimizer.state[param]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.zeros_like(param.data)
                    
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(param.grad)
                    param.data.add_(buf, alpha=-learning_rate)
        
        # Supprimer les hooks
        for hook in hooks:
            hook.remove()
            
        return False  # Pour compatibilité avec l'interface originale
    
    def load_weights_from_json(self, json_obj, hidden_size):
        """Charger les poids à partir d'un objet JSON (format original)"""
        # Convertir les poids d'entrée en tenseurs PyTorch
        input_weights = torch.zeros(self.input_size, hidden_size)
        for i in range(self.input_size):
            for j in range(hidden_size):
                input_weights[i][j] = json_obj["input_weights"][i][j]
        
        # Convertir les poids de sortie en tenseurs PyTorch
        output_weights = torch.zeros(hidden_size, self.output_size)
        for i in range(hidden_size):
            for j in range(self.output_size):
                output_weights[i][j] = json_obj["output_weights"][i][j]
        
        # Affecter les poids au modèle
        with torch.no_grad():
            self.hidden.weight.copy_(input_weights.t())  # Transposer car PyTorch utilise (out_features, in_features)
            self.output.weight.copy_(output_weights.t())
            
    def save_weights_to_json(self):
        """Convertir les poids en format JSON compatible avec l'original"""
        # Convertir les poids d'entrée au format original
        input_weights = []
        hidden_weights = self.hidden.weight.detach().t()  # Transposer pour revenir au format original
        for i in range(self.input_size):
            row = []
            for j in range(self.hidden_size):
                row.append(float(hidden_weights[i][j]))
            input_weights.append(row)
        
        # Convertir les poids de sortie au format original
        output_weights = []
        out_weights = self.output.weight.detach().t()  # Transposer pour revenir au format original
        for i in range(self.hidden_size):
            row = []
            for j in range(self.output_size):
                row.append(float(out_weights[i][j]))
            output_weights.append(row)
        
        return {"input_weights": input_weights, "output_weights": output_weights}