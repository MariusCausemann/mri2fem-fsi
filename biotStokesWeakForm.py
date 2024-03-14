
from dolfin import *
from multiphenics import *
import ufl

def create_block_bcs(boundary_conditions, H):

    bcs = []
    for bc in boundary_conditions:
        marker_id, subspace_id, bc_val, boundary_marker = bc
        bc_d = DirichletBC(H.sub(subspace_id), bc_val,
                           boundary_marker, marker_id)
        bcs.append(bc_d)

    return BlockDirichletBC(bcs)


def create_function_spaces(mesh, u_degree, p_degree,
                           fluidrestriction, porousrestriction):

    W2 = FunctionSpace(mesh, "CG", p_degree + 1)
    V = VectorFunctionSpace(mesh, "CG", u_degree)
    W = FunctionSpace(mesh, "CG", p_degree)
    H = BlockFunctionSpace([V, W, V, W2, W],
                            restrict=[fluidrestriction, fluidrestriction,
                                      porousrestriction, porousrestriction,
                                      porousrestriction])
    return H


def create_measures(mesh, subdomain_marker, boundary_marker, fluid_id,
                    porous_id, interf_id):
    dx = Measure("dx", domain=mesh, subdomain_data=subdomain_marker)
    dS = Measure("dS", domain=mesh, subdomain_data=boundary_marker)
    ds = Measure("ds", domain=mesh, subdomain_data=boundary_marker)
    dxF = dx(fluid_id)
    dxP = dx(porous_id)
    ds_Sigma = dS(interf_id)
    return [dxF, dxP, dx, ds, ds_Sigma]


def biot_stokes_system(mesh, material_parameter, H, measures, dt,
                       g_source=Constant(0)):

    # porous parameter
    c = material_parameter["c"]
    kappa = material_parameter["kappa"]
    lmbda = material_parameter["lmbda"]
    mu_s = material_parameter["mu_s"]
    rho_s = material_parameter["rho_s"]
    alpha = material_parameter["alpha"]

    # fluid parameter
    rho_f = material_parameter["rho_f"]
    mu_f = material_parameter["mu_f"]
    gamma = material_parameter["gamma"]

    dxF, dxP, dxD, ds, ds_Sigma = measures

    gdim = mesh.geometric_dimension()
    n = FacetNormal(mesh)("+")

    trial_functions = BlockTrialFunction(H)
    u, pF, d, pP, phi = block_split(trial_functions)

    test_functions = BlockTestFunction(H)
    v, qF, w, qP, psi = block_split(test_functions)

    block_function = BlockFunction(H)
    u_n, pF_n, d_n, pP_n, phi_n = block_split(block_function)

    f_fluid = Constant([0.0]*gdim)
    f_porous = Constant([0.0]*gdim)

    eps = lambda u: sym(grad(u))
    P_t = lambda u: u-dot(u,n)*n
    tang_interf = lambda u,v: inner(P_t(u("+")), P_t(v("+")))*ds_Sigma

    def a_F(u,v):
        return rho_f*dot(u/dt, v)*dxF \
                + 2*mu_f*inner(eps(u), eps(v))*dxF \
                + (gamma*mu_f/sqrt(kappa))*tang_interf(u,v)

    def b_1_F(v, qF):
        return  -qF*div(v)*dxF

    def b_2_Sig(v, qP):
        return qP("+")*inner(v("+"), n)*ds_Sigma + div(v)*qP*Constant(0.0)*dxD

    def b_3_Sig(v, d):
        return - ((gamma*mu_f/sqrt(kappa))*tang_interf(v, d))

    def b_4_Sig(w,qP):
        return -qP("+") * dot(w("+"),n)*ds_Sigma + div(w)*qP*Constant(0.0)*dxD

    def a_1_P(d, w):
        return 2.0*mu_s*inner(eps(d), eps(w))*dxP \
                + (gamma*mu_f/sqrt(kappa))*tang_interf(d/dt,w)

    def b_1_P(w, psi):
        return - psi*div(w)*dxP

    def a_2_P(pP,qP):
        return (kappa/mu_f) *inner(grad(pP), grad(qP))*dxP \
                + (c + alpha**2/lmbda)*(pP/dt)*qP*dxP

    def b_2_P(psi, qP):
        return (alpha/lmbda)*psi*qP*dxP

    def a_3_P(phi, psi):
        return (1.0/lmbda)*phi*psi*dxP

    def F_F(v):
        f = rho_f*dot(f_fluid, v)*dxF \
                + div(v)*Constant(0.0)*dxD
        return f

    def F_P(w):
        return rho_s*inner(f_porous, w)*dxP + div(w)*Constant(0.0)*dxD

    def G(qP):
        return g_source*qP*dxP

    def F_F_n(v):
        return F_F(v) + rho_f*inner(u_n/dt, v)*dxF + b_3_Sig(v, d_n/dt) 

    def F_P_n(w):
        return F_P(w) + (gamma*mu_f/sqrt(kappa))*tang_interf(d_n/dt, w)

    def G_n(qP):
        return G(qP) + (c + (alpha**2)/lmbda)*pP_n/dt*qP*dxP \
            - b_4_Sig(d_n/dt, qP) - b_2_P(phi_n/dt, qP)


    # define system:
    # order trial: u, pF, d, pP, phi
    # order test: v, qF, w, qP, psi

    lhs =   [[a_F(u,v)           , b_1_F(v, pF), b_3_Sig(v, d/dt) , b_2_Sig(v, pP), 0                 ],
            [ b_1_F(u, qF)       ,  0          , 0                , 0             , 0                 ],
            [ b_3_Sig(u, w)      ,  0          , a_1_P(d,w)       , b_4_Sig(w, pP), b_1_P(w, phi)     ],
            [ b_2_Sig(u, qP)     ,  0          , b_4_Sig(d/dt, qP), -a_2_P(pP, qP) , b_2_P(phi/dt, qP)],
            [ 0                  ,  0          , b_1_P(d, psi)    , b_2_P(psi, pP), -a_3_P(phi, psi) ]]

    
    rhs = [F_F_n(v), 0, F_P_n(w), -G_n(qP), 0]
    return lhs, rhs, block_function
