"""
export_onnx_edge.py
===================
Pipeline de exportação do modelo AgroAdvancedNet para ONNX,
validação de consistência numérica e benchmark de latência Edge Computing (microcontroladores/SBCs).
"""

import os
import sys
import time
import torch
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model_advanced import AgroAdvancedNet


def export_and_benchmark(
    checkpoint_path: str = os.path.join(BASE_DIR, "best_agro_advanced.pth"),
    onnx_output_path: str = os.path.join(BASE_DIR, "agro_advanced.onnx"),
    in_features: int = 35
):
    """
    Exporta a rede treinada para ONNX e executa benchmark de latência.
    """
    print(f"\n=======================================================")
    print(f" Exportação ONNX e Otimização Edge Computing")
    print(f"=======================================================")

    device = torch.device("cpu")  # Simula ambiente de borda (Edge / microcontrolador)
    model = AgroAdvancedNet(in_features=in_features).to(device)

    if os.path.exists(checkpoint_path):
        print(f"Carregando pesos de: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        print(f"Aviso: Checkpoint {checkpoint_path} não encontrado. Exportando com inicialização de pesos.")

    model.eval()

    # Cria tensor dummy com formato (Batch_size, in_features)
    dummy_input = torch.randn(1, in_features, dtype=torch.float32, device=device)

    # 1. Exportação PyTorch -> ONNX
    print(f"Exportando grafo para ONNX: {onnx_output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_features"],
        output_names=["logits_irrigacao", "logits_pragas", "quantiles_volume_mm"],
        dynamic_axes={
            "input_features": {0: "batch_size"},
            "logits_irrigacao": {0: "batch_size"},
            "logits_pragas": {0: "batch_size"},
            "quantiles_volume_mm": {0: "batch_size"}
        }
    )

    file_size_kb = os.path.getsize(onnx_output_path) / 1024
    print(f"Sucesso! Modelo ONNX gerado: {file_size_kb:.2f} KB (ultraleve para Edge/IoT).")

    # 2. Verificação com onnxruntime
    try:
        import onnxruntime as ort
        print("\nValidando modelo ONNX via OnnxRuntime...")
        session = ort.InferenceSession(onnx_output_path, providers=["CPUExecutionProvider"])

        # Teste de inferência
        np_dummy = dummy_input.numpy()
        ort_inputs = {session.get_inputs()[0].name: np_dummy}
        ort_outs = session.run(None, ort_inputs)

        with torch.no_grad():
            py_outs = model(dummy_input)

        # Checa diferença máxima
        diff_irrig = np.max(np.abs(ort_outs[0] - py_outs[0].numpy()))
        diff_pragas = np.max(np.abs(ort_outs[1] - py_outs[1].numpy()))
        diff_vol = np.max(np.abs(ort_outs[2] - py_outs[2].numpy()))

        print(f"Diferença máxima PyTorch vs ONNX:")
        print(f"  - Irrigação: {diff_irrig:.6e}")
        print(f"  - Pragas:    {diff_pragas:.6e}")
        print(f"  - Volume:    {diff_vol:.6e}")
        assert max(diff_irrig, diff_pragas, diff_vol) < 1e-4, "Erro numérico alto no ONNX!"
        print("Validação Numérica: [PASSOU]")

        # 3. Benchmark de Latência (1000 iterações em CPU)
        print("\nExecutando Benchmark de Latência (1000 inferências)...")
        n_iters = 1000
        # Warmup
        for _ in range(50):
            _ = session.run(None, ort_inputs)

        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = session.run(None, ort_inputs)
        t1 = time.perf_counter()

        total_time_ms = (t1 - t0) * 1000
        avg_latency_ms = total_time_ms / n_iters
        fps = 1000.0 / avg_latency_ms

        print(f"Tempo total: {total_time_ms:.2f} ms")
        print(f"Latência média por amostra: {avg_latency_ms:.3f} ms")
        print(f"Throughput estimado: {fps:.1f} inferências/segundo na borda (Edge).")

    except Exception as e:
        print(f"Aviso no teste onnxruntime: {e}")

    return onnx_output_path


if __name__ == "__main__":
    export_and_benchmark()
