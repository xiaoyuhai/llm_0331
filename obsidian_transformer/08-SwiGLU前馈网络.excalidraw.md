---

excalidraw-plugin: parsed
tags: [excalidraw]

---
==⚠  Switch to EXCALIDRAW VIEW in the MORE OPTIONS menu of this document. ⚠== You can decompress Drawing data with the command palette: 'Decompress current Excalidraw file'. For more info check in plugin settings under 'Saving'


# Excalidraw Data

## Text Elements
前馈网络: 从 ReLU FFN 到 SwiGLU
SwiGLU (Qwen3 / LLaMA 标准)
输入: d_model
gate_proj
d_model→d_ff
up_proj
d_model→d_ff
SiLU(gate) × up
down_proj
d_ff→d_model
输出: d_model
公式: SwiGLU(x) = down_proj(SiLU(gate_proj(x)) × up_proj(x))

好处: 梯度更好，任务效果通常提升 1-2 个百分点
你的代码:

class Qwen3MLP:
    def forward(self, x):
        gate = self.gate_proj(x)  # d→d_ff
        up = self.up_proj(x)      # d→d_ff
        return self.down_proj(
            F.silu(gate) * up
        )
%%
## Drawing
```compressed-json
eJztWs1v40QU/1cs99KibOvvJJY4dHcprJRdLZSKQ1tF08wkHerYlj3ZpFSVAO22BbEsH0KIZYXK
x+4BIcGBQ9lu98J/guqEPfEvMBPbsZO6big0dUucQzwzzzPP7/3e+814ZoMn6zbidR61KsDA0AFN
PsffQY6LLZPXpRzvWg2nwiRWCbFdfWYmkpyuWHUqjQxURyZxeX1xg8eQipIWKYu0JeiboBahpRav
a0KOX+d1kf41MSSrdIQ8vV9FuLZKeF2m98CsGfQheucSx1pD1yzDcmgnExVBLkortJ8VUFmrOVbD
hGEbcYDp2sChWtD2KjaMebLOeqHaU1X5sK+3/DGFaTXH0w5qqyZy3e5Ylg0qmHRVoyXWu30Dsjda
7kqa0Jc0G4ZBFWAVr0RvTWUaNgQE0XcXc7yBzbVQ1LAqa6y6CgwXxQwr9u5vWSYzL63A7nVqShIT
rzRcYtWvAwLC/rqm1Hnvg/svnux0Dj7t7D/SucP9j7k3UGmBm5u7xXk7v3DzTfxqaYGZwjLJPH6H
9i8JfmkO1LGx3vUs62vWwDWqDl+hr4Ic3/UEUweHDXUMIbUk1YU+DLCJnBswVMZycA2bwHjznylF
7YNeCzwuTksq86iLWC1VS9nMxUAkJYFIVn0UqadBkZQHSBPGKGIO8z3CTb7eRKbMzXClErg5y7V3
t73trak+9IjamaPnJGXSUKNFqHFQhZSVCDas7IOhDzuFGHakOHbU02FnAqpVFcnD4EYMy4NSQ2KJ
3ttlme+H1EbwvvLmOQKrL3TVxNANCKCgRuYXYuaX0swvIvYbdegmmjsrEfzn88+9e491DpbrFkRG
f8wqZx6zR4dPi9JisQcQ4DhlLQIILVpNHyFKOEUQI4howyb3/w4h0ilTexiH0nnGYY63LeyPuSjk
hOXc4hVqQllYphoYwCXXrHodE9rDbSYWPuMS4JCr2ITYrIV1yIQDNV2pWeavVQRgTC5W1/MnMlas
Zk/RPu/nx94fmfcz5fwuQRePJ2gpQIASIUDUhmVopOWRJCUzdBWiFSiMjKELmWdoUUikaDlwQGx6
LSoXgKMHDZ4Vjq7Rwcq2Y729ZAY8+cfWZ7Bcrfaz9dmvz9IUSeNtcXB6LUrHhq9SuCThK4rZj185
KX6V4gWN3yMWz0oAN+xMhO/xagwbvGzeJcbWxvGJlxCgJrYwHk+8LtXEq+t9NdH7qnbU+6Iwdv+/
XXUxrbMDAJ+788dytxxwtxT7PD88dxeQogCYzN2oAAGqjo67texzdyFx7l0MPaBeMO4etHhWuHse
lxYm2cR3ivv9S65h95O2ePZftY+On7r9IfV/IxOLyWwd5GspNsseNk7H2fqYbE1TtZqpVC2JJ6fq
wmlSdYaWWZKQ+VQtJW9Chqm6cMFS9RGLZyVVQ6tp9lY41Wp3eZOwq3H2C600RVKT98AGhySnJm9Z
HCfvS5y81ROTt6yeJnlnaBNaUrKfvLW05C2rFy15D1o8K8mb7QRvPz3Xjej+4dMStawV+0GSTwRJ
cFREiZ9Wi4dpPvWjCFClvDZqkGQTHN69n7xnD/TgBNhka4p7meuR7GRvieQXW1PBUqlXXDKXTO/x
gffDXZ1rf/ez9/RJ++tfacVfzz463N/3Pvy2/cVO+5tHL9596O3ttR984t3f5sQrEne492Pnq+fe
zlbn/d9Gve7LxiunBYEyeNQu8YuEdkIMiMXUw3YVGanjw3ZdRBwe7HYe3j3c/76z+57O/FuhcxWX
6x53u1m6Tas4ekFU5aqW0wQOnKSequa41lTQxC6GGYok1jIdxw/HTXAw2BCIpBt2KBtBy29JknYQ
aTimLx9BNWpn19y0i41G8EHjJTpA1Dw16hD7Pxj0pACmMAa2PU/o42z2VXMw9O3vW+wORs2rCWv8
7sVvbv4Nbatx7Q==
```
%%
