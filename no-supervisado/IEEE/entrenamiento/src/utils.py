import torch
import numpy as np

def get_Ybus(net,device):
    '''
    Run pp.runpp(net) before calling this function, because it uses the Ybus matrix
    '''
    Ybus = torch.from_numpy(np.asarray(net._ppc["internal"]["Ybus"].todense())).to(device)
    return Ybus.to(torch.complex64)


def get_Yline(net,device):
    '''
    Run pp.runpp(net) before calling this function, because it uses the Ybus matrix
    '''
    Y_line_ij = torch.from_numpy(np.asarray(net._ppc["internal"]["Yf"].todense())).to(device)
    Y_line_ji = torch.from_numpy(np.asarray(net._ppc["internal"]["Yt"].todense())).to(device)
    return Y_line_ij.to(torch.complex64), Y_line_ji.to(torch.complex64)

def check_AC_flow(net, tol=1e-5):
    V = net.res_bus.vm_pu * (np.cos(net.res_bus.va_degree / 180 * np.pi) + 1j*np.sin(net.res_bus.va_degree / 180 * np.pi))
    V = V.values
    S = net.res_bus.p_mw + 1j*net.res_bus.q_mvar
    S = - S.values / 100
    Ybus = get_Ybus(net)

    AC_flow = np.matmul(np.diag(V) , np.conj(np.matmul(Ybus.detach().to("cpu").numpy() , V)))- S

    return AC_flow<tol

def init_lamdas(net,dual_coefs,device):
    N =  len(net.bus.index)
    E = len(net.line)
    hist_dual_variables = []
    hist_AC_flow_penalty_real_mean = []
    dual_acflow_real = torch.ones(N).to(device) * dual_coefs[0]
    dual_acflow_imag = torch.ones(N).to(device) * dual_coefs[1]
    dual_lines = torch.ones(E).to(device) * dual_coefs[2]
    dual_variables = (dual_acflow_real, dual_acflow_imag, dual_lines)
    return dual_variables

def get_line(net,device):
    line_to = torch.tensor(net.line["to_bus"].values.tolist() + net.trafo['lv_bus'].values.tolist()).to(device)
    line_from = torch.tensor(net.line["from_bus"].values.tolist() + net.trafo['hv_bus'].values.tolist()).to(device)
    return line_to, line_from

