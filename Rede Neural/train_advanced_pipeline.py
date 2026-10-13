"""
train_advanced_pipeline.py
==========================
Pipeline de treinamento completo do modelo AgroAdvancedNet integrando as 4 prioridades:
1. Arquitetura Tabular Residual (ResNet) comDropout e GELU.
2. Balanceamento Dinâmico de Perdas com Incerteza (Uncertainty Weighting).
3. Regressão Quantílica (Pinball Loss) para estimar P10, P50 e P90 da lâmina de água.
4. Exportação automática para ONNX com benchmark de latência.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data_engine import AgroDataEngine, AgroMultiTaskDataset
from model_advanced import AgroAdvancedNet, PinballLoss, UncertaintyMultiTaskLoss
from export_onnx_edge import export_and_benchmark


def train_advanced(
    csv_path: str = os.path.join(BASE_DIR, "dados_solo_2casas.csv"),
    n_samples: int = 50000,
    epochs: int = 12,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    model_save_path: str = os.path.join(BASE_DIR, "best_agro_advanced.pth"),
    scaler_save_path: str = os.path.join(BASE_DIR, "agro_scaler_meta.pkl")
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================", flush=True)
    print(f" Treinamento AgroAdvancedNet (4 Prioridades Integradas)", flush=True)
    print(f" Dispositivo: {device}", flush=True)
    print(f" Amostras: {n_samples:,} | Épocas: {epochs} | Batch: {batch_size}", flush=True)
    print(f"=======================================================\n", flush=True)

    # 1. Carregar dados e preparar matriz
    engine = AgroDataEngine(scaler_path=scaler_save_path)
    df = engine.load_and_augment_dataset(csv_path=csv_path, n_samples=n_samples)

    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df["target_irrigacao"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["target_irrigacao"])

    X_train = engine.prepare_feature_matrix(train_df, is_train=True)
    X_val = engine.prepare_feature_matrix(val_df, is_train=False)
    X_test = engine.prepare_feature_matrix(test_df, is_train=False)

    pragas_cols = ["target_fungos", "target_insetos", "target_acaros", "target_estresse"]

    ds_train = AgroMultiTaskDataset(
        X_train,
        train_df["target_irrigacao"].values,
        train_df[pragas_cols].values,
        train_df["target_volume_mm"].values
    )
    ds_val = AgroMultiTaskDataset(
        X_val,
        val_df["target_irrigacao"].values,
        val_df[pragas_cols].values,
        val_df["target_volume_mm"].values
    )
    ds_test = AgroMultiTaskDataset(
        X_test,
        test_df["target_irrigacao"].values,
        test_df[pragas_cols].values,
        test_df["target_volume_mm"].values
    )

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    in_features = X_train.shape[1]
    model = AgroAdvancedNet(in_features=in_features).to(device)

    # 2. Perdas avançadas e balanceamento dinâmico
    criterion_irrig = nn.CrossEntropyLoss()
    criterion_pragas = nn.BCEWithLogitsLoss()
    criterion_vol_pinball = PinballLoss(quantiles=[0.10, 0.50, 0.90]).to(device)

    # Kendall Loss para balancear automaticamente as perdas
    uncertainty_loss_module = UncertaintyMultiTaskLoss(num_tasks=3).to(device)

    # Otimizador com decaimento de peso e parâmetros do Kendall Loss
    all_params = list(model.parameters()) + list(uncertainty_loss_module.parameters())
    optimizer = optim.AdamW(all_params, lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    start_time = time.time()

    # 3. Loop de Treinamento
    for epoch in range(1, epochs + 1):
        model.train()
        uncertainty_loss_module.train()

        total_loss_train = 0.0
        n_batches = 0

        for bx, by_irrig, by_pragas, by_vol in loader_train:
            bx = bx.to(device)
            by_irrig = by_irrig.to(device)
            by_pragas = by_pragas.to(device)
            by_vol = by_vol.to(device)

            optimizer.zero_grad()

            pred_irrig, pred_pragas, pred_vol_quantiles = model(bx)

            l_irrig = criterion_irrig(pred_irrig, by_irrig)
            l_pragas = criterion_pragas(pred_pragas, by_pragas)
            l_vol = criterion_vol_pinball(pred_vol_quantiles, by_vol)

            # Combinação dinâmica aprendida
            loss, weights = uncertainty_loss_module(l_irrig, l_pragas, l_vol)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss_train += loss.item()
            n_batches += 1

        total_loss_train /= n_batches

        # Validação
        model.eval()
        uncertainty_loss_module.eval()
        total_loss_val = 0.0
        val_batches = 0

        val_preds_irrig, val_targets_irrig = [], []
        val_preds_pragas, val_targets_pragas = [], []
        val_preds_vol_p50, val_targets_vol = [], []

        with torch.no_grad():
            for bx, by_irrig, by_pragas, by_vol in loader_val:
                bx = bx.to(device)
                by_irrig = by_irrig.to(device)
                by_pragas = by_pragas.to(device)
                by_vol = by_vol.to(device)

                pred_irrig, pred_pragas, pred_vol_quantiles = model(bx)

                l_irrig = criterion_irrig(pred_irrig, by_irrig)
                l_pragas = criterion_pragas(pred_pragas, by_pragas)
                l_vol = criterion_vol_pinball(pred_vol_quantiles, by_vol)

                v_loss, _ = uncertainty_loss_module(l_irrig, l_pragas, l_vol)
                total_loss_val += v_loss.item()
                val_batches += 1

                val_preds_irrig.append(pred_irrig.argmax(dim=-1).cpu().numpy())
                val_targets_irrig.append(by_irrig.cpu().numpy())

                val_preds_pragas.append((torch.sigmoid(pred_pragas) > 0.5).cpu().numpy())
                val_targets_pragas.append(by_pragas.cpu().numpy())

                val_preds_vol_p50.append(pred_vol_quantiles[:, 1].cpu().numpy())  # P50 (Mediana)
                val_targets_vol.append(by_vol.cpu().numpy())

        total_loss_val /= val_batches
        scheduler.step(total_loss_val)

        val_acc_irrig = accuracy_score(np.concatenate(val_targets_irrig), np.concatenate(val_preds_irrig))
        val_mae_vol = mean_absolute_error(np.concatenate(val_targets_vol), np.concatenate(val_preds_vol_p50))

        print(
            f"Época {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {total_loss_train:.4f} | "
            f"Val Loss: {total_loss_val:.4f} | "
            f"Acc Irrig: {val_acc_irrig:.2%} | "
            f"MAE P50 (mm): {val_mae_vol:.3f} | "
            f"Pesos Tarefas: Irrig={weights['weight_irrigacao']:.2f}, Pragas={weights['weight_pragas']:.2f}, Vol={weights['weight_volume']:.2f}",
            flush=True
        )

        if total_loss_val < best_val_loss:
            best_val_loss = total_loss_val
            torch.save({
                "epoch": epoch,
                "in_features": in_features,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "task_weights": weights
            }, model_save_path)

    elapsed = time.time() - start_time
    print(f"\nTreinamento concluído em {elapsed:.1f} segundos!")
    print(f"Melhor modelo salvo em: {model_save_path}")

    # 4. Avaliação Final no Conjunto de Teste
    print("\nAvaliando no Conjunto de Teste Independente...")
    ckpt = torch.load(model_save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_preds_irrig, test_targets_irrig = [], []
    test_preds_pragas, test_targets_pragas = [], []
    test_p10, test_p50, test_p90, test_targets_vol = [], [], [], []

    with torch.no_grad():
        for bx, by_irrig, by_pragas, by_vol in loader_test:
            bx = bx.to(device)
            p_irrig, p_pragas, q_vol = model(bx)

            test_preds_irrig.append(p_irrig.argmax(dim=-1).cpu().numpy())
            test_targets_irrig.append(by_irrig.numpy())

            test_preds_pragas.append((torch.sigmoid(p_pragas) > 0.5).cpu().numpy())
            test_targets_pragas.append(by_pragas.numpy())

            test_p10.append(q_vol[:, 0].cpu().numpy())
            test_p50.append(q_vol[:, 1].cpu().numpy())
            test_p90.append(q_vol[:, 2].cpu().numpy())
            test_targets_vol.append(by_vol.numpy())

    all_t_irrig = np.concatenate(test_targets_irrig)
    all_p_irrig = np.concatenate(test_preds_irrig)
    all_t_vol = np.concatenate(test_targets_vol).flatten()
    all_p10 = np.concatenate(test_p10).flatten()
    all_p50 = np.concatenate(test_p50).flatten()
    all_p90 = np.concatenate(test_p90).flatten()

    acc_test = accuracy_score(all_t_irrig, all_p_irrig)
    f1_test = f1_score(all_t_irrig, all_p_irrig, average="macro")
    mae_p50 = mean_absolute_error(all_t_vol, all_p50)
    # Cobertura do intervalo de predição (Quantos % do volume real caíram entre P10 e P90?)
    coverage_80 = np.mean((all_t_vol >= all_p10) & (all_t_vol <= all_p90))

    print("\n" + "=" * 50)
    print(" RESULTADOS FINAIS DE TESTE (AgroAdvancedNet):")
    print(f"  - Acurácia Irrigação:    {acc_test:.2%}")
    print(f"  - F1-Score Macro Irrig:  {f1_test:.4f}")
    print(f"  - MAE Lâmina d'água P50: {mae_p50:.3f} mm")
    print(f"  - Cobertura Intervalo [P10, P90]: {coverage_80:.2%} (Ideal teórico ~80%)")
    print("=" * 50)

    # 5. Exportação para ONNX e Benchmark de Latência
    onnx_path = os.path.join(BASE_DIR, "best_agro_advanced.onnx")
    export_and_benchmark(checkpoint_path=model_save_path, onnx_output_path=onnx_path, in_features=in_features)


if __name__ == "__main__":
    train_advanced(n_samples=50000, epochs=10)
