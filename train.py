import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from model import PathogenLSTM


def main(args):
    """Train the PathogenLSTM model on preprocessed sliding window sequences."""
    try:
        # Step 1: Create models directory
        os.makedirs("models", exist_ok=True)

        # Step 2: Load X_train and y_train
        X_train = np.load(args.X_train)
        y_train = np.load(args.y_train)

        # Step 3: Convert to torch float32 tensors
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        # Step 4: Create TensorDataset and DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        # Step 5: Set device
        device = torch.device(args.device)

        # Step 6: Create model
        model = PathogenLSTM(input_size=X_train.shape[2], hidden_size=args.hidden).to(device)

        # Step 7: Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Step 8: Loss function
        criterion = nn.MSELoss()

        # Step 9: Training loop
        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
            for xb, yb in pbar:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.6f}")

            avg_loss = epoch_loss / len(loader)
            # Print every epoch for visibility
            print(f"Epoch {epoch+1}/{args.epochs} | Avg Loss: {avg_loss:.6f}")

        # Step 11: Save model
        torch.save(model.state_dict(), args.model_out)

        # Step 12: Print completion message
        print("Training complete. Model saved to", args.model_out)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM pathogen spread model")
    parser.add_argument("--X_train", type=str, default="data/X_train.npy", help="Path to X_train.npy")
    parser.add_argument("--y_train", type=str, default="data/y_train.npy", help="Path to y_train.npy")
    parser.add_argument("--model_out", type=str, default="models/lstm_pathogen.pt", help="Output path for trained model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=128, help="LSTM hidden size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()
    main(args)
