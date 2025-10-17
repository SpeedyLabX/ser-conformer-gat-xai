"""Attention visualization helpers (placeholder)"""
import matplotlib.pyplot as plt

def plot_attention(attn: list, out_path: str):
    # attn: list of (Tq, Tk) arrays
    fig, ax = plt.subplots()
    ax.imshow(attn[0], aspect='auto')
    fig.savefig(out_path)
