
from pathlib import Path
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import prepare_data


# Where the trained model will be saved
MODEL_PATH = Path(__file__).resolve().parent / "models" / "recommender.pt"


# 1. Define our Artificial Neural Network
class RecommenderANN(nn.Module):
    def __init__(self, num_users, num_restaurants):
        """Define two 16-number embedding tables and the 32 -> 64 -> 32 -> 1 rating network."""
        super().__init__()

        # Each known ID gets 16 learnable numbers. Sixteen is a design choice.
        # PyTorch initializes these weights; training learns useful representations.
        self.user_embedding = nn.Embedding(num_users, 16)
        self.restaurant_embedding = nn.Embedding(num_restaurants, 16)

        self.network = nn.Sequential(
            nn.Linear(32, 64),  # Mix the 32 inputs into 64 intermediate features.
            nn.ReLU(),  # Replace negative values with zero; preserve feature count.
            nn.Linear(64, 32),
            nn.ReLU(),  # Replace negative values with zero; preserve feature count.
            nn.Linear(32, 1),
            nn.Sigmoid()  # Keep the predicted rating between 0 and 1, not a probability.
        )

    def forward(self, users, restaurants):
        """Look up user/item embeddings, concatenate them, and predict one normalized rating per pair."""
        user_vector = self.user_embedding(users)
        restaurant_vector = self.restaurant_embedding(restaurants)

        combined = torch.cat(
            [user_vector, restaurant_vector],
            dim=1
        )

        return self.network(combined).squeeze(-1)


# 2. Convert pandas DataFrame to PyTorch tensors
def make_dataset(frame):
    """Convert ID indices and normalized target ratings into tensors for PyTorch batches."""
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
    """Measure MAE and RMSE without changing weights; both errors use the normalized 0-1 scale."""
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
    """Train with MSE and Adam, select by validation RMSE, test, and save weights plus ID maps."""
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

    # MSE measures final prediction error; Adam updates weights using its gradients.
    loss_function = nn.MSELoss()

    train_loader = DataLoader(
        make_dataset(train),
        batch_size=32,
        shuffle=True
    )

    best_val_rmse = float("inf")
    best_weights = None

    print("\nStarting ANN training...\n", flush=True)

    # One epoch is one complete pass through training ratings, in batches of up to 32.
    for epoch in range(30):
        model.train()
        total_loss = 0.0

        for users, restaurants, ratings in train_loader:
            optimizer.zero_grad()  # Clear gradients left over from the previous batch.

            predictions = model(users, restaurants)

            loss = loss_function(predictions, ratings)

            loss.backward()  # Compute how each trainable weight affects the error.
            optimizer.step()  # Adam applies an update; improvement is not guaranteed every step.

            total_loss += loss.item() * len(users)

        train_mse = total_loss / len(train)

        val_mae, val_rmse = evaluate(model, val)

        # Validation selects the best epoch. Test ratings never select these weights.
        if val_rmse is not None and val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_weights = copy.deepcopy(model.state_dict())  # Freeze a copy; later epochs keep changing weights.

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
