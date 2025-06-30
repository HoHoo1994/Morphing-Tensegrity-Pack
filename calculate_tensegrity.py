import numpy as np

def eqmatrix(nodes, elements, start_index=1): #平衡矩阵
    """
    nodes: (n,3) ndarray, 每行一个节点的x,y,z坐标
    elements: (b,4) ndarray, 每行一根单元，2/3列为端点编号
    start_index: 节点编号是从1还是0起，默认1（常见Excel工程习惯）
    返回: (3n, b) 平衡矩阵C / (b,) 单元长度
    """
    nodes = np.array(nodes)
    elements = np.array(elements)
    n = nodes.shape[0]
    b = elements.shape[0]
    C = np.zeros((b, 3*n))
    lengths = np.zeros(b)
    for i in range(b):
        n1 = int(elements[i, 1]) - start_index
        n2 = int(elements[i, 2]) - start_index
        x1 = nodes[n1]
        x2 = nodes[n2]
        vec = x2 - x1
        L = np.linalg.norm(vec)
        lengths[i] = L
        if L == 0:
            raise ValueError(f"单元{i}长度为0，请检查nodes和elements数据！")
        C[i, 3*n1:3*n1+3] = -vec / L
        C[i, 3*n2:3*n2+3] =  vec / L
        A = C.T
    return A, lengths

def prestress(A, coeff, svd_tol=1e-8): #预应力设置
    """
    给定平衡矩阵A和系数，输出自应力
    A: (b, 3n) 平衡矩阵
    coeff: 标量或长度为模态数的向量（可以只填一个数即默认只用第一个模态）
    svd_tol: 判断零奇异值的容差

    返回
    ----
    t: (b,1) 自应力向量
    prestress_modes: (b,k) 所有自应力模态
    """

    U, S, V = np.linalg.svd(A, full_matrices=False)
    mask = S < svd_tol
    null_indices = np.where(mask)[0]
    if len(null_indices)==0:
        raise ValueError("无零奇异值，对应结构无自应力模态！")
    prestress_modes = V[null_indices,:].T  # shape (b, k)
    # coeff既可以是标量也可以是数组（线性组合）
    coeff = np.atleast_1d(coeff).reshape(-1)
    if coeff.shape[0] > prestress_modes.shape[1]:
        raise ValueError("输入系数个数大于自应力模态数")
    # t = modes @ coeff 否则补 0
    nmode = prestress_modes.shape[1]
    coeff_pad = np.zeros(nmode)
    coeff_pad[:len(coeff)] = coeff
    t = prestress_modes @ coeff_pad
    if t[0] > 0:      # 如果第一个值为正
        t = -t
    return t, prestress_modes

def restlength(t, lens, elements): #单元原长计算
    k = elements[:,4] 
    rl = lens - t / k
    return rl

def stiffmatrix(A, nodes, elements, lens, tensions, start_index=1,): #计算刚度矩阵
    
    """
    nodes: (n,3) array, 每行[x, y, z]
    elements: (m, >=5) array，第2/3列为两端点，第5列为单元线刚度
    start_index: 节点编号起点（常见为1而非0）
    axial_forces: (m,) 内力数组，如有则生成Kg，否则Kg为零阵

    返回
    ------
    Kt: 切线刚度
    Ke: 弹性刚度
    Kg: 几何刚度
    """
    elements = np.array(elements)
    b = elements.shape[0]

    # 弹性刚度矩阵
    line_stiff = elements[:,4]   # 第五列
    M = np.diag(line_stiff / lens)
    Ke = A @ M @ A.T

    # 几何刚度矩阵
    n = nodes.shape[0]
    C = np.zeros((b, n))
    for idx, elem in enumerate(elements):
        n1 = int(elem[1]) - start_index
        n2 = int(elem[2]) - start_index
        C[idx, n1] = -1
        C[idx, n2] =  1


    q = tensions / lens  # (b,)
    Q = np.diag(q)   

    Kn = C.T @ Q @ C
    Kg = np.kron(Kn, np.eye(3))

    #切线刚度矩阵
    Kt = Ke + Kg

    return Kt, Ke, Kg

def constraint(K_full, constrained_dofs): #输入刚体位移约束
    """
    K_full: (3n, 3n) 总刚度阵
    constrained_dofs: 需约束的全局自由度下标（列表或np.array）
    返回:
      K_ff: 只含自由自由度的刚度矩阵
      free_dofs: 自由DOF下标
    """
    n_full = K_full.shape[0]
    all_dofs = np.arange(n_full)
    free_dofs = np.setdiff1d(all_dofs, constrained_dofs)
    K_r = K_full[np.ix_(free_dofs, free_dofs)]
    return K_r, free_dofs

def generate_constrained_dofs(constraints, nnode): #生成约束自由度
    """
    constraints: 列表或tuple，每项为(节点编号, 方向字符串/编号)，如(1,'x'), (7,'y'), (10,2)
    nnode: 节点总数
    返回：约束自由度的全局下标list
    """
    axis_map = {'x':0,'y':1,'z':2,0:0,1:1,2:2}
    dofs = []
    for item in constraints:
        node, direc = item
        dof = 3*(node-1) + axis_map[direc]
        dofs.append(dof)
    return dofs

def generate_F(loads, nnode): #输入荷载
    '''
    loads: 列表，每个元素(节点号, 方向, 幅值)，方向为'x','y','z'或0,1,2
    nnode: 节点总数
    返回：F (3n,) 系统荷载向量
    '''
    F = np.zeros(3*nnode)
    # 方向映射
    axis_map = {'x':0, 'y':1, 'z':2, 0:0, 1:1, 2:2}
    for load in loads:
        node, direc, val = load
        di = axis_map[direc]
        dof = 3*(node-1) + di   # 节点编号从0开始
        F[dof] += val           # 支持多次叠加
    return F

def update_nodes(nodes, U): #更新节点坐标
    """
    nodes: (nnode, 3) 初始节点坐标
    U: (3*nnode,) 全局自由度位移
    返回: (nnode, 3) 变形后新节点坐标
    """
    nnode = nodes.shape[0]
    # 转为 nnode 行3列，每列为x/y/z方向增量
    dX = U.reshape((nnode, 3))
    new_nodes = nodes + dX
    return new_nodes

def balance_structure(nodes0, elements, F, rl, dofs, fdofs,  #修正节点坐标
                     max_iter=100, tol=1e-6, alpha=1.0, verbose=False):
    """
    用牛顿迭代自动调整节点，实现 A*tension=F
   
    参数
    ----
    nodes0:  ndarray(nnode, ndim)  # 初始节点坐标
    elements:      单元属性
    F:      ndarray(nfree*ndim,) 或 (nfree*ndim,1)  # 目标荷载，对应自由度
    rl:            单元原长
    max_iter:      最大迭代次数
    tol:           收敛容差（残差二范数）
    alpha:         步长系数
    verbose:       是否逐步打印残差

    返回
    -----
    nodes_new:   最终平衡的节点坐标
    tension:     最终内力
    converge:    是否收敛
    """
    nodes = nodes0.copy().reshape(-1)

    k = elements[:,4]

    for i in range(max_iter):
        # 产生平衡矩阵A
        nodes2d = nodes.reshape(-1, 3)
        A_new, lens_new = eqmatrix(nodes2d,elements)
        # 计算当前tension
        tension = k*(lens_new-rl)
        # 残差（F-A*t = 不平衡力）
        residual = F[dofs] - (A_new @ tension)[dofs]
        nr = np.linalg.norm(residual)
        if verbose:
            print(f"step {i+1}: residual = {nr:.3e}")
        if nr < tol:
            return nodes.reshape(nodes0.shape), tension, True   # 收敛
        # 装配Ktr
        Kt, Ke, Kg = stiffmatrix(A_new, nodes2d, elements, lens_new, tension)
        Ktr, free_dofs = constraint(Kt, fdofs)
        # 解修正增量
        try:
            du = np.linalg.solve(Ktr, residual)
        except np.linalg.LinAlgError as e:
            print("Ktr不可逆!", e)
            return nodes.reshape(nodes0.shape), tension, False
        # 更新节点
        nodes[dofs]  += alpha * du
    print("未收敛!")
    return nodes.reshape(nodes0.shape), tension, False

def gen_rl_variation(n_elem, change_list):
    """
    参数
    ----
    n_elem:       int，总单元数
    change_list:  list of (elem_id, delta) 单元号和增量

    返回
    ----
    drl: ndarray(n_elem,)  # 除指定单元外全0
    """
    drl = np.zeros(n_elem)
    for elem_id, delta in change_list:
        drl[elem_id-1] = delta
    return drl

def elongation2disp(A, Kt, lens, rl, drl, free_dofs=None):
    """
    根据主动原长变化drl，计算自由度节点变形du
   
    参数
    ----
    A        : (n_total_dof, n_elem)      # 平衡矩阵
    Kt       : (n_total_dof, n_total_dof) # 全局刚度阵
    lens     : (n_elem,)                  # 当前单元现长
    rl       : (n_elem,)                  # 当前单元原长
    drl      : (n_elem,)                  # 单元原长主动变化
    free_dofs: (n_free_dof,) or None      # 自由自由度索引。None时认为已是自由
                                          # 若提供，全局变量自动筛选自由子阵
    返回
    ----
    du       : (n_free_dof,)              # 对应自由度的节点变形
    T        : (n_free_dof, n_elem)       # 灵敏度矩阵
    """

    # 如果给了free_dofs, 只筛选自由自由度
    if free_dofs is not None:
        A_r  = A[free_dofs, :]
        Kt_r = Kt[np.ix_(free_dofs, free_dofs)]
    else:
        A_r = A
        Kt_r = Kt

    main_diag = lens / rl     # (n_elem,)

    # 主动变化到自由度灵敏度矩阵
    Tmat = A_r @ np.diag(main_diag)    # shape: (n_free, n_elem)
    # 求解T
    T = np.linalg.solve(Kt_r, Tmat)    # shape: (n_free, n_elem)

    du = T @ drl   # (n_free,)
    rl += drl

    return du, T, rl