import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calculate_tensegrity
import visual_tensegrity
from mpl_toolkits.mplot3d import Axes3D   

#----------------------------------------------------读取数据-------------------------------------------------------
excel_file = 'model_1.xlsx'
# 读取工作表
nodes_df = pd.read_excel(excel_file, sheet_name='nodes')
elements_df = pd.read_excel(excel_file, sheet_name='elements')

# 转 numpy 数组
nodes = nodes_df.values.astype(float)  # (n,3)
elements = elements_df.values.astype(int)  # (b,5)
nnode = nodes.shape[0]
nele = elements.shape[0]
#----------------------------------------------------刚度设置-------------------------------------------------------
# 建立平衡矩阵
A, lens = calculate_tensegrity.eqmatrix(nodes,elements)

# 预应力设置
t, modes = calculate_tensegrity.prestress(A, coeff=1000) 

# 单元原长设置
restlength = calculate_tensegrity.restlength(t, lens, elements)

# 建立刚度矩阵
Kt, Ke, Kg = calculate_tensegrity.stiffmatrix(A, nodes, elements, lens, t)

# 提供约束
constraints = [
    (1,'x'), (1,'y'), (1,'z'),         # 1号节点xyz
    (7,'y'), (7,'z'),                  # 7号节点y,z
    (10,'z'),                          # 10号节点z
]
constrained_dofs = calculate_tensegrity.generate_constrained_dofs(constraints, nnode)
Ktr, free_dofs = calculate_tensegrity.constraint(Kt, constrained_dofs)

# 查看刚度
eigvals = np.linalg.eigvalsh(Ktr)
eig_min = eigvals.min()

#----------------------------------------------------受荷计算-------------------------------------------------------
# # 施加荷载
# loads = [(4, 'z', -50),(6, 'z', -50),(11, 'z', 50)]         
# F = calculate_tensegrity.generate_F(loads, nnode)
# F_f = F[free_dofs]

# # 计算变形
# d_f = np.linalg.solve(Ktr, F_f)
# d_all = np.zeros(Kt.shape[0])
# d_all[free_dofs] = d_f
# nodes_new = calculate_tensegrity.update_nodes(nodes, d_all) #更新节点坐标

# # 误差修正
# nodes_new, tension, flag = calculate_tensegrity.balance_structure(
#     nodes_new, elements, F, restlength, free_dofs, constrained_dofs, 
#     max_iter=1000, tol=1e-3, alpha=1, verbose=True)
# if flag:
#     print("结构平衡完成!")
#     nodes = nodes_new
# else:
#     print("平衡失败或未收敛!")

#----------------------------------------------------主动变形计算-------------------------------------------------------
#主动单元作动
elongs = [(10, 0.2), (5, 0.2), (19, -0.2), (21, -0.2), (26, -0.2)]
drl = calculate_tensegrity.gen_rl_variation(nele, elongs)

du, T, restlength = calculate_tensegrity.elongation2disp(A, Kt, lens, restlength, drl, free_dofs=free_dofs)

# 计算变形
d_all = np.zeros(Kt.shape[0])
d_all[free_dofs] = du
nodes_new = calculate_tensegrity.update_nodes(nodes, d_all) #更新节点坐标

# 外荷载
loads = []         #无外荷载
F = calculate_tensegrity.generate_F(loads, nnode)

# 误差修正
nodes_new, tension, flag = calculate_tensegrity.balance_structure(
    nodes_new, elements, F, restlength, free_dofs, constrained_dofs, 
    max_iter=1000, tol=1e-3, alpha=1, verbose=True)
if flag:
    print("结构平衡完成!")
    nodes = nodes_new
else:
    print("平衡失败或未收敛!")

#----------------------------------------------------绘制张拉整体-------------------------------------------------------
# 绘制张拉整体
ax = visualize_tensegrity.draw_tensegrity(nodes, elements, node_labels=False, elem_labels=True)
