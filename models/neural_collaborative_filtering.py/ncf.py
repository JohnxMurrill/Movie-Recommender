import os
import torch as nn
import torch.nn.functional as F

class NeuralCollaborativeFiltering:
    def __init__(self, num_users, num_items, embedding_dim):
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

    def forward(self, user_ids, item_ids):
        # Get user and item embeddings
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)

        # Concatenate user and item embeddings
        x = nn.cat([user_embeds, item_embeds], dim=-1)

        # Pass through MLP layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = nn.sigmoid(self.fc3(x))

        return x.squeeze()

class NCFTrainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, data_loader, num_epochs=10):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for user_ids, item_ids, ratings in data_loader:
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