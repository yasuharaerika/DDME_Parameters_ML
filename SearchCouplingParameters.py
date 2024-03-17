# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 18:39:45 2024

@author: DELL
"""

import numpy as np
import math
from scipy.optimize import fsolve

PI2 = math.pi**2

class Newton_method():
    def __init__(self, max_iter=100, tol=1e-10, epsilon=1e-10):
        self.max_iter = max_iter
        self.tol = tol
        self.epsilon = epsilon
    def NewtonIteration(self, Function, InitialValue, t_values):
        matrix = InitialValue.copy()
        N = len(InitialValue)
        Range = len(t_values)
        Matrix = np.zeros((Range,N))
        for i in range(Range):
            for _ in range(self.max_iter):
                # 计算Function的值和雅可比矩阵J
                F = Function(matrix, t_values[i])
                J = self.Jacobian(Function, matrix, t_values[i])
                # 求解线性方程组 J*delta = -F， 若J不可逆，则尝试使用伪逆来代替求解矩阵的逆
                try:
                    delta = np.linalg.solve(J, -F)
                except np.linalg.LinAlgError:
                    delta = np.linalg.pinv(J) @ (-F)
                matrix += delta
                if np.linalg.norm(delta) < self.tol:
                    break
            Matrix[i] = matrix
        return Matrix
    def Jacobian(self, Function, Value, t_value):
        N = len(Value)
        Matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Matrix[i][j] = self.PartialDerivative(Function, Value, t_value, i, j)
        return Matrix   
    def PartialDerivative(self, Function, Value, t_value, i, j):
        N = len(Value)
        Delta = np.zeros(N)
        Delta[j] = self.epsilon
        df_i_j = (Function(Value+Delta, t_value)[i] - Function(Value, t_value)[i]) / self.epsilon
        return df_i_j
      

class DDME_Model():
    def __init__(self):
        # 定义模型参数
        self.M_N = 939.0  # 质子和中子的质量（单位：MeV/c^2）
        self.M_P = self.M_N
        self.rho_B0 = 0.152
        self.HC = 197.327 # from MeV to fm^-1
        self.HC4 = self.HC**4
        # 对称因子
        self.S_delta = None
        self.S_gamma = None
        # 耦合系数
        self.coupling_sigma = None
        self.coupling_omega = None
        self.coupling_rho   = None
        # 密度项
        self.Rho_B = None
        self.Rho_s = None
        # 单位变换
        self.M_N /= self.HC
        self.M_P /= self.HC

    def DDME_parameters(self, parameters):
        m_sigma, m_omega, m_rho, \
        Gamma_sigma0, Gamma_omega0, Gamma_rho0, \
        Gamma_sigma_a, Gamma_sigma_b, Gamma_sigma_c, Gamma_sigma_d, \
        Gamma_omega_a, Gamma_omega_b, Gamma_omega_c, Gamma_omega_d, \
        Gamma_rho_a = parameters
        self.M_sigma = m_sigma  # σ介子的质量（单位：MeV/c^2）
        self.M_omega = m_omega  # ω介子的质量（单位：MeV/c^2）
        self.M_rho   = m_rho        
        self.GRS0 = Gamma_sigma0   # Gamma_sigma(rho_B0)
        self.GRW0 = Gamma_omega0   # Gamma_omage(rho_B0)
        self.GRR0 = Gamma_rho0     # Gamma_rho(rho_B0)
        self.GSA = Gamma_sigma_a   # f(x) = a*(1+b*(x+d)**2)/(1+c*(x+d)**2)
        self.GSB = Gamma_sigma_b
        self.GSC = Gamma_sigma_c
        self.GSD = Gamma_sigma_d
        self.GWA = Gamma_omega_a   # f(x) = a*(1+b*(x+d)**2)/(1+c*(x+d)**2)
        self.GWB = Gamma_omega_b
        self.GWC = Gamma_omega_c
        self.GWD = Gamma_omega_d
        self.GRA = Gamma_rho_a     # f(x) = exp(-a*(x-1))
        # 单位变换
        self.M_sigma /= self.HC
        self.M_omega /= self.HC
        self.M_rho /= self.HC
    
    # 初始化
    def Initialization(self, Value, RHOB):
        self.sigma, self.omega, self.rho = Value 
        # 核物质
        self.Rho_B_P, self.Rho_B_N = (1.0-self.S_delta)*RHOB/2.0, (1.0+self.S_delta)*RHOB/2.0
        self.KF_P = self.FermiMomentum(self.Rho_B_P)
        self.KF_N = self.FermiMomentum(self.Rho_B_N)
        # 代入RHOB
        x = RHOB / self.rho_B0
        self.Coupling(x)
        self.Density()
    # 运动方程
    def EquationsOfMotion(self, InitialValue, RHOB):  
        self.Initialization(InitialValue, RHOB)
        # equation
        eq1 = self.M_sigma**2 * self.sigma - self.coupling_sigma*self.Rho_s
        eq2 = self.M_omega**2 * self.omega - self.coupling_omega*self.Rho_B
        eq3 = self.M_rho**2 * self.rho - 1/2*self.coupling_rho*self.Rho3
        return np.array([eq1, eq2, eq3])
    
    def FermiMomentum(self, RhoB):
        return (3 * PI2 * RhoB ) ** (1/3)
    # 对称系数
    def Symmetry(self, delta):
        self.S_delta = delta
    
    # 耦合函数及偏导
    def Gamma_function(self, x, gamma0, a, b, c, d): # x = rho_B / rho_B0
        return gamma0 * a * (1 + b*(x+d)**2) / (1 + c*(x+d)**2)    
    def Gamma_partial_function(self, x, gamma0, a, b, c, d):
        return 2 / self.rho_B0 * gamma0 * a * (b-c) *(x+d) / (1 + c*(x+d)**2)**2    
    #
    def Gamma0_function(self, x, a, b, c, d): # x = rho_B / rho_B0
        return a * (1 + b*(x+d)**2) / (1 + c*(x+d)**2)    
    def Gamma0_partial2_function(self, x, a, b, c, d):
        return 2 / self.rho_B0 *  a * (b-c) *(1-3*c*(x+d)**2) / (1 + c*(x+d)**2)**3 
    #
    def Gamma_exp(self, x, gamma0, a):
        return gamma0 * math.exp(-a*(x-1))    
    def Gamma_partial_exp(self, x, gamma0, a):
        return - a / self.rho_B0 * self.Gamma_exp(x, gamma0, a)
    
    # meson-nucleon coupling 
    def Coupling(self, x):
        self.coupling_sigma = self.Gamma_function(x, self.GRS0, self.GSA, self.GSB, self.GSC, self.GSD)
        self.coupling_omega = self.Gamma_function(x, self.GRW0, self.GWA, self.GWB, self.GWC, self.GWD)
        self.coupling_rho   = self.Gamma_exp(x, self.GRR0, self.GRA)     
        return self.coupling_sigma, self.coupling_omega, self.coupling_rho
    # 密度项
    def CalculateDensity(self, M, KF):
        MS = M - self.coupling_sigma * self.sigma
        EFS = math.sqrt(KF**2 + MS**2)
        Rho_s = 2 /(4*PI2) * MS * (KF*EFS - MS**2*math.log((KF+EFS)/abs(MS)) )
        return MS, Rho_s
    def Density(self):
        self.MS_P, self.Rho_s_P = self.CalculateDensity(self.M_P, self.KF_P)
        self.MS_N, self.Rho_s_N = self.CalculateDensity(self.M_N, self.KF_N)
        # rho_s, rho_s3, rho_B, rho3
        self.Rho_s = self.Rho_s_P + self.Rho_s_N
        self.Rho_s3 = self.Rho_s_P - self.Rho_s_N
        self.Rho_B = self.Rho_B_P + self.Rho_B_N
        self.Rho3 = self.Rho_B_P - self.Rho_B_N
    
    # 重排项
    def Sigma_R(self, x):
        GP_sigma = self.Gamma_partial_function(x, self.GRS0, self.GSA, self.GSB, self.GSC, self.GSD)
        GP_omega = self.Gamma_partial_function(x, self.GRW0, self.GWA, self.GWB, self.GWC, self.GWD)
        GP_rho   = self.Gamma_partial_exp(x, self.GRR0, self.GRA) 
        return - GP_sigma*self.sigma*self.Rho_s + GP_omega*self.omega*self.Rho_B + \
                 GP_rho*self.rho*self.Rho3/2

    def Ekin(self, MS, KF): # M*, Kf
        EFS = math.sqrt(KF**2 + MS**2)
        return (2 / 16.0 / PI2) * ((2.0 * KF**3 +  MS**2 * KF) * EFS -
                                     MS**4 * math.log((KF + EFS) / abs(MS)))
    def Pkin(self, MS, KF):
        EFS = math.sqrt(KF**2 + MS**2)
        return (2 / 48.0 / PI2) * ((2.0 * KF**3 - 3.0 * MS**2 * KF) * EFS + 
                                     3.0 * MS**4 * math.log((KF + EFS) / abs(MS)))
    # 总能量与总压强
    def EnergyTotal(self):
        US = self.M_sigma**2 * self.sigma**2 / 2
        EW = self.coupling_omega * self.omega * self.Rho_B
        UW = self.M_omega**2 * self.omega**2 / 2
        ER = self.coupling_rho * self.rho * self.Rho3 / 2
        UR = self.M_rho**2 * self.rho**2 /2
        Ek_P = self.Ekin(self.MS_P, self.KF_P)
        Ek_N = self.Ekin(self.MS_N, self.KF_N)
        return (Ek_P + Ek_N) + US - UW - UR + EW + ER
    def PressureTotal(self, x):
        US = self.M_sigma**2 * self.sigma**2 / 2
        UW = self.M_omega**2 * self.omega**2 / 2
        UR = self.M_rho**2 * self.rho**2 /2       
        Pk_P = self.Pkin(self.MS_P, self.KF_P)
        Pk_N = self.Pkin(self.MS_N, self.KF_N)  
        Rearrangement = self.Sigma_R(x) * self.Rho_B
        return (Pk_P + Pk_N) - US + UW + UR + Rearrangement
    
    # 对称能
    def SymmetryEnergy(self):
        KF = self.FermiMomentum(self.Rho_B/2)
        EFS = math.sqrt(KF**2 + self.MS_N**2)
        Esym = KF**2/6/EFS + self.coupling_rho**2*self.Rho_B/8/self.M_rho**2
        return Esym
    
    # 对称能斜率
    def SymmetryEnergySlope(self, E, rho_B):
        N = len(rho_B)-2
        Esls = []
        for i in range(N):
            I = i+1
            h = (rho_B[I+1]-rho_B[I-1])/2
            dE_dR = (E[I+1]-E[I-1])/(2*h)
            esls = 3*rho_B[I]*dE_dR
            Esls.append(esls)
        return rho_B[1:-1], np.array(Esls)
    # 不可压缩系数
    def Imcompressibility(self, P, rho_B):
        N = len(rho_B)-2
        K = []
        for i in range(N):
            I = i+1
            h = (rho_B[I+1]-rho_B[I-1])/2
            dP_dR = (P[I+1]-P[I-1])/(2*h)
            K.append(9*dP_dR)
        return rho_B[1:-1], np.array(K)
    # 对称核物质偏度
    def SkewnessOfSymmetry(self, epsilon, rho_B):
        N = len(rho_B) - 6
        Q = []
        E = epsilon/rho_B
        for i in range(N):
            I = i+3
            h = ((rho_B[I+3]+rho_B[I+2]+rho_B[I+1]) - \
                 (rho_B[I-3]+rho_B[I-2]+rho_B[I-1])) / 12
            dE_dR3 = (E[I-3]-8*E[I-2]+13*E[I-1]-13*E[I+1]+8*E[I+2]-E[I+3]) / (8*h**3)
            q = 27*rho_B[I]**3*dE_dR3
            Q.append(q)
        return rho_B[3:-3], np.array(Q)
    # 对称能偏度
    def SkewnessOfEsym(self, Esym, rho_B):
        N = len(rho_B) - 6
        Q0 = []     
        for i in range(N):
            I = i+3
            h = ((rho_B[I+3]+rho_B[I+2]+rho_B[I+1]) - \
                 (rho_B[I-3]+rho_B[I-2]+rho_B[I-1])) / 12
            dE_dR3 = (Esym[I-3]-8*Esym[I-2]+13*Esym[I-1]-13*Esym[I+1]+ \
                      8*Esym[I+2]-Esym[I+3]) / (8*h**3)
            q = 27*rho_B[I]**3*dE_dR3
            Q0.append(q)
        return rho_B[3:-3], np.array(Q0)
    # 核物质性质
    def GetProperties(self, Value, RHOB):
        self.Initialization(Value, RHOB)
        x = RHOB / self.rho_B0
        # 性质
        MSN = self.M_N - self.coupling_sigma * self.sigma
        Etot = self.EnergyTotal()
        Ptot = self.PressureTotal(x)
        EA = Etot/RHOB - self.M_N
        Esym = self.SymmetryEnergy()
        return MSN, Etot*self.HC, Ptot*self.HC, EA*self.HC, Esym*self.HC
    

# 参数
IRHOB = 321
RHOB = np.linspace(0.139,0.171,IRHOB) # 实际范围0.14-0.17，扩大范围为避免Q计算不到

# 初始化模型
DDME = DDME_Model()
DDME.Symmetry(delta=0)

# 生成模型参数
m_range = np.array([547.3327, 783, 763]) # 介子质量
#
def CouplingParameters(Initial_Values, Values):
    sa, sb, sc, sd, wa, wb, wc, wd = Initial_Values
    SB, SC, WC = Values # 提前固定
    eq1 = DDME.Gamma0_function(1, sa, sb, sc, sd) - 1
    eq2 = DDME.Gamma0_function(1, wa, wb, wc, wd) - 1
    eq3 = 3*sc*sd**2 - 1
    eq4 = 3*wc*wd**2 - 1
    eq5 = DDME.Gamma0_partial2_function(1, sa, sb, sc, sd) - DDME.Gamma0_partial2_function(1, wa, wb, wc, wd)
    eq6 = sb - SB
    eq7 = sc - SC
    eq8 = wc - WC
    return np.array([eq1,eq2,eq3,eq4,eq5,eq6,eq7,eq8])

# 牛顿迭代法
Newton = Newton_method()
Initial_Values0 = np.ones(8) # 介子耦合参数初值
Initial_Values1 = [0.01, 0.01, 0.01]  # 介子场初值

# 计算核物质性质
Properties_SNM = np.zeros((IRHOB,5))

# 饱和点性质
def GetY(x,y,x0):
    k = (y[1]-y[0])/(x[1]-x[0])
    return y[0] + k*(x0-x[0])
def SaturationPointProperties(RHOB, EA, K, Esym, Esls, Q, Qsym, MS, P):
    for i in range(len(RHOB)):
        if P[i]*P[i+1] <= 0:
            x = RHOB[i:i+2]
            point = GetY(P[i:i+2], x, 0) # 压强为0的点
            EA0   = GetY(x, EA[i:i+2]    , point)
            K0    = GetY(x, K[i-1:i+1]   , point)
            Esym0 = GetY(x, Esym[i:i+2]  , point)
            Esls0 = GetY(x, Esls[i-1:i+1], point)
            Q0    = GetY(x, Q[i-3:i-1]   , point)
            Qsym0 = GetY(x, Qsym[i-3:i-1], point)
            MS0   = GetY(x, MS[i:i+2]    , point)
            break
    return point, EA0, K0, Esym0, Esls0, Q0, Qsym0, MS0/DDME.M_N


def TrainData():
    Pmax, Pmin = -1, 1 # 即要求无零点时重新计算
    while Pmax <= 0 or Pmin >= 0:
        gamma_range = np.random.uniform(6,14,3)   # gamma0范围
        coupling_initial_values = np.random.uniform(0.01,4,3) # sigma_b, sigma_c, omega_c
        coupling_parameters = fsolve(CouplingParameters, Initial_Values0, coupling_initial_values)
        while (coupling_parameters <= 0).any(): # 若出现负值则重新求解介子耦合参数
            coupling_initial_values = np.random.uniform(0.01,4,3)
            coupling_parameters = fsolve(CouplingParameters, Initial_Values0, coupling_initial_values)
        coupling_rho = np.random.random(1)
        DDME_parameters  = np.hstack((m_range, gamma_range, coupling_parameters, coupling_rho)) 
        DDME.DDME_parameters(DDME_parameters)   # DDME参数代入
        Equations_solve = Newton.NewtonIteration(DDME.EquationsOfMotion, Initial_Values1, RHOB) # 求解运动方程
        for i in range(IRHOB):
            result = DDME.GetProperties(Equations_solve[i], RHOB[i])
            Properties_SNM[i] = result
        MSN_SNM, Etot_SNM, Ptot_SNM, EA_SNM, Esym_SNM = Properties_SNM.transpose()    
        Pmax, Pmin = Ptot_SNM[310], Ptot_SNM[10] # 零点判断，0.14<rho_B<0.17范围
    Esls_SNM = DDME.SymmetryEnergySlope(Esym_SNM, RHOB)
    K_SNM = DDME.Imcompressibility(Ptot_SNM, RHOB)
    Q_SNM = DDME.SkewnessOfSymmetry(Etot_SNM, RHOB)
    Qsym_SNM = DDME.SkewnessOfEsym(Esym_SNM, RHOB)
    SPP = SaturationPointProperties(RHOB, EA_SNM, K_SNM[1], Esym_SNM, Esls_SNM[1],\
                                Q_SNM[1], Qsym_SNM[1], MSN_SNM, Ptot_SNM)
    return np.hstack((gamma_range, coupling_initial_values, coupling_rho)), np.array(SPP)



DDME_Parameters = np.zeros(7)
SPP = np.zeros(8)
for i in range(1000):
    parameters, spp = TrainData()
    DDME_Parameters = np.vstack((DDME_Parameters, parameters))
    SPP = np.vstack((SPP, spp))
    print(i+1,': \n',parameters, '\n', spp)
DDME_Parameters = DDME_Parameters[1:]
SPP = SPP[1:]

def NormParameters(parameters):
    Parameters = parameters.copy()
    for i in range(3):
        Parameters[:,i] /= 15
    for i in range(3,6):
        Parameters[:,i] /= 4
    return Parameters
def NormSPP(spp):
    a = [1,3,4]
    b = [2,5,6]
    SPP = spp.copy()
    for i in a:
        SPP[:,i] /= 100
    for i in b:
        SPP[:,i] /= 1500
    return SPP
DDME_Parameters_Norm = NormParameters(DDME_Parameters)
SPP_Norm = NormSPP(SPP)

with open('DDME_Parameters.txt','w') as file_object:
    for i in range(len(DDME_Parameters_Norm)):
        for j in range(len(DDME_Parameters_Norm[i])):
            file_object.write(str(DDME_Parameters_Norm[i][j]))
            file_object.write(' ')
        file_object.write('\n')
with open('SPP.txt','w') as file_object:
    for i in range(len(SPP_Norm)):
        for j in range(len(SPP_Norm[i])):
            file_object.write(str(SPP_Norm[i][j]))
            file_object.write(' ')
        file_object.write('\n')
