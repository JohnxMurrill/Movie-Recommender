import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# Neural Collaborative Filtering (NCF) model definition
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP layers
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache')

    def forward(self, user_ids, item_ids):
        # Get user and item embeddings
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)

        # Concatenate user and item embeddings
        x = torch.cat([user_embeds, item_embeds], dim=-1)

        # Pass through MLP layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = torch.sigmoid(self.fc3(x))  # Output between 0 and 1 for implicit feedback
        x = self.fc3(x)  # No activation for explicit feedback // RMSE

        return x.squeeze()

    def save_to_cache(self):
        """
        Save the model weights to the given path.
        """
        os.makedirs(self.cache_dir, exist_ok=True)  # Ensure directory exists
        cache_path = os.path.join(self.cache_dir, 'ncf_model.pt')  # Add a filename
        torch.save(self.state_dict(), cache_path)
        print(f"NCF model saved to {cache_path}")

    def load_from_cache(self):
        """
        Load the model weights from the given path.
        """
        cache_path = os.path.join(self.cache_dir, 'ncf_model.pt')
        self.load_state_dict(torch.load(cache_path))
        print(f"NCF model loaded from {cache_path}")

# Trainer class for NCF
class NCFTrainer:
    def __init__(self, model, optimizer, criterion, device):
        """
        model: NeuralCollaborativeFiltering instance
        optimizer: torch.optim.Optimizer (e.g., torch.optim.Adam)
        criterion: loss function (e.g., nn.BCELoss for implicit, nn.MSELoss for explicit)
        device: torch.device ('cpu' or 'cuda')
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, data_loader, num_epochs=10):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            
            for user_ids, item_ids, ratings in tqdm(data_loader, desc=f"Epoch {epoch+1}"):
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)

                # Forward pass
                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings.float())

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader)}")

# Example usage:
# model = NeuralCollaborativeFiltering(num_users, num_items, embedding_dim)
# model.save_to_cache("ncf_model.pth")
# model.load_from_cache("ncf_model.pth", map_location='cpu')
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.BCELoss()  # For implicit feedback (binary targets 0/1)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# trainer = NCFTrainer(model, optimizer, criterion, device)