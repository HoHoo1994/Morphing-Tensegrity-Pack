import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D   

def draw_tensegrity(nodes, elements, ax=None, node_labels=True, elem_labels=False, start_index=1):
    """
    nodes: (n,3) ndarray, 每行[x, y, z]
    elements: (b,4) ndarray，较常见2/3列为两端点编号
    ax: matplotlib 三维Axes，方便多个结构叠画，可省略
    node_labels: 是否显示节点编号
    elem_labels: 是否显示单元编号
    """
    nodes = np.array(nodes)
    elements = np.array(elements)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        show = True
    else:
        show = False


    # 标记是否已画 legend
    rod_drawn = False
    cable_drawn = False

    # 绘制单元
    for elem in elements:
        _, n1, n2, etype ,_ = elem.astype(int)
        p1 = nodes[n1-1]
        p2 = nodes[n2-1]
        if etype == 0:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color='b', linestyle='-', linewidth=4,
                label='Rod' if not rod_drawn else None)
            rod_drawn = True
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color='r', linestyle='-', linewidth=2,
                label='Cable' if not cable_drawn else None)
            cable_drawn = True
    
    # 标注节点编号
    if elem_labels:
        for idx, elem in enumerate(elements):
            _, n1, n2, _ , _= elem.astype(int)
            p1 = nodes[n1-1]
            p2 = nodes[n2-1]
            xm, ym, zm = (p1 + p2) / 2
            ax.text(xm, ym, zm, str(idx+1), color='green', fontsize=8)
   
    # 绘制节点
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], color='k', s=50)

    # 标注节点编号
    if node_labels:
        for k, pt in enumerate(nodes):
            ax.text(pt[0], pt[1], pt[2], str(k+1), color='k', fontsize=10)

    # 图例去重
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
     if l not in unique and l is not None and l != '':
        unique[l] = h
    ax.legend(unique.values(), unique.keys())

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([np.ptp(nodes[:, 0]), np.ptp(nodes[:, 1]), np.ptp(nodes[:, 2])])  # 保持xyz比例

    if show:
        plt.tight_layout()
        plt.show()
    return ax