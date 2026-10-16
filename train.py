
from pathlib import Path
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data import prepare_data


# Where the trained model will be saved
MODEL_PATH = Path(__file__).resolve().parent / "models" / "recommender.pt"


# 1. Define our Artificial Neural Network
class RecommenderANN(nn.Module):
    def __init__(self, num_users, num_restaurants):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, 16)
        self.restaurant_embedding = nn.Embedding(num_restaurants, 16)

        self.network = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, users, restaurants):
        user_vector = self.user_embedding(users)
        restaurant_vector = self.restaurant_embedding(restaurants)

        combined = torch.cat(
            [user_vector, restaurant_vector],
            dim=1
        )

        return self.network(combined).squeeze(-1)


# 2. Convert pandas DataFrame to PyTorch tensors
def make_dataset(frame):
    users = torch.tensor(
        frame["user_idx"].to_numpy(),
        dtype=torch.long
    )

    restaurants = torch.tensor(
        frame["restaurant_idx"].to_numpy(),
        dtype=torch.long
    )

    ratings = torch.tensor(
        frame["rating"].to_numpy(),
        dtype=torch.float32
    )

    return TensorDataset(users, restaurants, ratings)


# 3. Evaluate model performance
def evaluate(model, frame):
    if frame.empty:
        return None, None

    model.eval()

    with torch.no_grad():
        users, restaurants, actual = make_dataset(frame).tensors
        predictions = model(users, restaurants)

        mae = torch.mean(
            torch.abs(predictions - actual)
        ).item()

        rmse = torch.sqrt(
            torch.mean((predictions - actual) ** 2)
        ).item()

    return mae, rmse


# 4. Train the neural network
def train_model():
    torch.manual_seed(42)

    print("Preparing dataset...", flush=True)

    train, val, test, places, user_map, restaurant_map = prepare_data()

    print(f"Training interactions: {len(train)}")
    print(f"Validation interactions: {len(val)}")
    print(f"Testing interactions: {len(test)}")
    print(f"Users: {len(user_map)}")
    print(f"Restaurants: {len(restaurant_map)}")

    model = RecommenderANN(
        num_users=len(user_map),
        num_restaurants=len(restaurant_map)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    loss_function = nn.MSELoss()

    train_loader = DataLoader(
        make_dataset(train),
        batch_size=32,
        shuffle=True
    )

    best_val_rmse = float("inf")
    best_weights = None

    print("\nStarting ANN training...\n", flush=True)

    for epoch in range(30):
        model.train()
        total_loss = 0.0

        for users, restaurants, ratings in train_loader:
            optimizer.zero_grad()

            predictions = model(users, restaurants)

            loss = loss_function(predictions, ratings)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(users)

        train_mse = total_loss / len(train)

        val_mae, val_rmse = evaluate(model, val)

        if val_rmse is not None and val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_weights = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch + 1:02d}/30 | "
            f"Train MSE: {train_mse:.4f} | "
            f"Val RMSE: {val_rmse:.4f}"
            if val_rmse is not None
            else f"Epoch {epoch + 1:02d}/30 | "
                 f"Train MSE: {train_mse:.4f}",
            flush=True
        )

    # Restore best model based on validation performance
    if best_weights is not None:
        model.load_state_dict(best_weights)

    test_mae, test_rmse = evaluate(model, test)

    # Simple baseline: always predict average training rating
    average_rating = train["rating"].mean()

    baseline_mae = (
        (test["rating"] - average_rating).abs().mean()
        if not test.empty
        else None
    )

    print("\n========== FINAL RESULTS ==========")
    print(f"Test MAE: {test_mae}")
    print(f"Test RMSE: {test_rmse}")
    print(f"Baseline MAE: {baseline_mae}")

    # Save trained model and ID mappings
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "user_map": user_map,
            "restaurant_map": restaurant_map,
            "num_users": len(user_map),
            "num_restaurants": len(restaurant_map),
        },
        MODEL_PATH
    )

    print(f"\nModel saved successfully: {MODEL_PATH}")


# 5. IMPORTANT: Actually start training
if __name__ == "__main__":
    train_model()
